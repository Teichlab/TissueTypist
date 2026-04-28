"""
tissuetypist/features/spatial.py
=================================
Spatial feature functions: neighbour-max aggregation, edge detection,
distance-to-edge, and sliding-window pseudobulking.

Functions
---------
Public:
    sliding_window_pseudobulk        — pseudobulk via sliding windows
                                       (sum raw → normalize_total → log1p)
    include_neighbours               — k-NN neighbour-max features
    annotate_edge                    — edge detection + distance-to-edge
    build_neighbourhood_features_sd  — SD spot-level neighbourhood features
    build_neighbourhood_features_hd  — HD window-level neighbourhood features
    sliding_window_pseudobulk_hd     — HD cells → pseudobulk windows (raw counts)
    sliding_window_pseudobulk_cells  — Xenium/MERFISH/CosMx → pseudobulk (raw counts)

Internal helpers (called by the above):
    _calculate_window_corners
    _sliding_window
    _find_arithmetic_segments
    _find_technical_edge
    _distance_to_edge
    _calculate_window_corners_v2     — grid tiling (for HD/cell windowing)
    _validate_raw_counts             — raw-count sanity checks
"""

from __future__ import annotations

import logging
import warnings
from itertools import product
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Sliding-window pseudobulk
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_window_corners(
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    window_size: int,
    overlap: int = 0,
    drop_partial_windows: bool = False,
) -> pd.DataFrame:
    """
    Calculate the corner points of all windows covering the area from
    min_x to max_x and min_y to max_y, with specified window_size and overlap.

    From squidpy:
    https://github.com/scverse/squidpy/blob/bdd989983cd24aaa46bbf4751f1e904fe10ecc44/
    src/squidpy/tl/_sliding_window.py#L178
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if overlap >= window_size:
        raise ValueError("Overlap must be less than the window size.")

    x_step = window_size - overlap
    y_step = window_size - overlap

    x_starts = np.arange(min_x, max_x, x_step)
    y_starts = np.arange(min_y, max_y, y_step)

    starts = list(product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + window_size
    windows["y_end"] = windows["y_start"] + window_size

    if not drop_partial_windows:
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    else:
        valid_windows = (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        windows = windows[valid_windows]

    windows = windows.reset_index(drop=True)
    return windows[["x_start", "x_end", "y_start", "y_end"]]


def _sliding_window(
    adata: AnnData,
    library_key: str | None = None,
    window_size: int | None = None,
    overlap: int = 0,
    coord_columns: tuple[str, str] = ("globalX", "globalY"),
    sliding_window_key: str = "sliding_window_assignment",
    spatial_key: str = "spatial",
    drop_partial_windows: bool = False,
    copy: bool = False,
):
    """
    Divide a tissue slice into regularly shaped spatially contiguous
    regions (windows).  Based on squidpy's sliding_window, modified to
    output coordinates of each window.
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")

    if copy:
        adata = adata.copy()

    x_col, y_col = coord_columns
    if x_col in adata.obs and y_col in adata.obs:
        coords = adata.obs[[x_col, y_col]].copy()
    elif spatial_key in adata.obsm:
        coords = pd.DataFrame(
            adata.obsm[spatial_key][:, :2],
            index=adata.obs.index,
            columns=[x_col, y_col],
        )
    else:
        raise ValueError(
            f"Coordinates not found. Provide `{coord_columns}` in "
            f"`adata.obs` or specify a suitable `spatial_key` in `adata.obsm`."
        )

    if window_size is None:
        coord_range = max(
            coords[x_col].max() - coords[x_col].min(),
            coords[y_col].max() - coords[y_col].min(),
        )
        window_size = max(int(np.floor(coord_range // 3.95)), 1)

    if window_size <= 0:
        raise ValueError("Window size must be larger than 0.")

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    libraries = [None] if library_key is None else adata.obs[library_key].unique()

    sliding_window_df = pd.DataFrame(index=adata.obs.index)

    for lib in libraries:
        if lib is not None:
            lib_mask = adata.obs[library_key] == lib
            lib_coords = coords.loc[lib_mask]
        else:
            lib_mask = np.ones(len(adata), dtype=bool)
            lib_coords = coords

        min_x, max_x = lib_coords[x_col].min(), lib_coords[x_col].max()
        min_y, max_y = lib_coords[y_col].min(), lib_coords[y_col].max()

        windows = _calculate_window_corners(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            window_size=window_size,
            overlap=overlap,
            drop_partial_windows=drop_partial_windows,
        )

        lib_key = f"{lib}_" if lib is not None else ""

        for idx, window in windows.iterrows():
            x_start = window["x_start"]
            x_end = window["x_end"]
            y_start = window["y_start"]
            y_end = window["y_end"]

            if overlap == 0:
                mask = (
                    (lib_coords[x_col] >= x_start)
                    & (lib_coords[x_col] <= x_end)
                    & (lib_coords[y_col] >= y_start)
                    & (lib_coords[y_col] <= y_end)
                )
                obs_indices = lib_coords.index[mask]
                sliding_window_df.loc[obs_indices, sliding_window_key] = (
                    f"{lib_key}window_{idx}"
                )
                # Window centre coordinates (added to original squidpy function)
                sliding_window_df.loc[obs_indices, 'window_col'] = (
                    x_start + (window_size / 2)
                )
                sliding_window_df.loc[obs_indices, 'window_row'] = (
                    y_start + (window_size / 2)
                )
            else:
                mask = (
                    (lib_coords[x_col] >= x_start)
                    & (lib_coords[x_col] <= x_end)
                    & (lib_coords[y_col] >= y_start)
                    & (lib_coords[y_col] <= y_end)
                )
                obs_indices = lib_coords.index[mask]
                col_name = f"{sliding_window_key}_{lib_key}window_{idx}"
                sliding_window_df.loc[obs_indices, col_name] = True
                sliding_window_df.loc[:, col_name].fillna(False, inplace=True)

    if overlap == 0:
        sliding_window_df[sliding_window_key] = pd.Categorical(
            sliding_window_df[sliding_window_key],
            ordered=True,
            categories=sorted(
                sliding_window_df[sliding_window_key].unique(),
                key=lambda x: int(x.split("_")[-1]),
            ),
        )

    sliding_window_df[x_col] = coords[x_col]
    sliding_window_df[y_col] = coords[y_col]

    # Report totals
    n_windows = len(set(sliding_window_df[sliding_window_key]))
    print(f'### Total window number ###: {n_windows}')
    print(
        f'### Average cell number per window ###: '
        f'{round(sliding_window_df.shape[0] / n_windows, 1)}'
    )

    if copy:
        return sliding_window_df
    # Replace squidpy's _save_data with direct assignment
    for col_name, col_data in sliding_window_df.items():
        adata.obs[col_name] = col_data


def sliding_window_pseudobulk(
    adata: AnnData,
    section_col: str,
    window_size: int,
    coord_columns: Optional[Tuple[str, str]] = None,
    log_normalise: bool = True,
) -> AnnData:
    """
    Perform pseudobulk aggregation by sliding windows on spatial data.

    Sum raw counts per window, then optionally
    normalize_total(1e4) + log1p ("double normalisation").

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial coordinates.
    section_col : str
        Column in adata.obs indicating section/library grouping.
    window_size : int
        Size of the sliding window (in same units as coordinates).
    coord_columns : tuple of str, optional
        Names of the x, y coordinate columns in adata.obs.
        If None, coordinates are read from adata.obsm['spatial'].
    log_normalise : bool, default True
        If True, normalises counts to 10k per window and applies log1p.

    Returns
    -------
    AnnData
        Pseudobulked AnnData with aggregated counts in .X and
        window metadata in .obs.
    """
    if coord_columns is not None:
        _sliding_window(
            adata=adata,
            library_key=section_col,
            window_size=window_size,
            overlap=0,
            coord_columns=coord_columns,
            copy=False,
        )
    else:
        _sliding_window(
            adata=adata,
            library_key=section_col,
            window_size=window_size,
            overlap=0,
            coord_columns=("globalX", "globalY"),
            spatial_key='spatial',
            copy=False,
        )

    # Aggregate counts per window
    bdata = sc.get.aggregate(
        adata,
        by='sliding_window_assignment',
        func=['sum'],
    )
    bdata.X = bdata.layers['sum'].copy()

    # Transfer window metadata
    obs = adata.obs[
        [section_col, 'window_col', 'window_row', 'sliding_window_assignment']
    ].copy()
    obs = obs.drop_duplicates().set_index('sliding_window_assignment')
    bdata.obs[[section_col, 'window_col', 'window_row']] = (
        obs.loc[bdata.obs_names]
        [[section_col, 'window_col', 'window_row']]
        .values
    )
    # Ensure numeric columns are plain float (not object/categorical)
    # so that write_h5ad doesn't choke on non-string types
    bdata.obs['window_col'] = bdata.obs['window_col'].astype(float)
    bdata.obs['window_row'] = bdata.obs['window_row'].astype(float)

    # Optional log-normalisation ("double normalisation")
    if log_normalise:
        sc.pp.normalize_total(bdata, target_sum=1e4)
        sc.pp.log1p(bdata)
        print('Pseudobulk: summed per window and log-normalised.')
    else:
        print('Pseudobulk: summed per window without normalisation.')

    return bdata


# ═══════════════════════════════════════════════════════════════════════════
# Neighbour-max features
# ═══════════════════════════════════════════════════════════════════════════

def include_neighbours(
    data: pd.DataFrame,
    k: int = 6,
) -> pd.DataFrame:
    """
    Augment each spot's gene expression features with neighbour-max
    summary statistics.

    For each section, computes the maximum expression among k nearest
    neighbours for each gene.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by spot ID, containing 'x', 'y' coordinates,
        gene expression columns ending with '_own', and a 'section'
        column identifying sections.
    k : int, default 6
        Number of neighbours to consider (excluding self).

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with original and neighbour-max features,
        plus 'section', 'x', 'y', and optionally 'tissue' columns.
    """
    expression_features = [col for col in data.columns if col.endswith('_own')]

    augmented_df_list = []
    for section, group_df in data.groupby('section'):
        datapoint_ids = group_df.index
        coords = group_df[['x', 'y']].values
        X_gene = group_df[expression_features].values
        n_datapoints, n_genes = X_gene.shape

        k_section = k if n_datapoints > k + 1 else max(n_datapoints - 1, 1)

        nbrs = NearestNeighbors(n_neighbors=k_section + 1, algorithm='ball_tree')
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)

        neighbor_max = np.zeros_like(X_gene)
        for i in range(n_datapoints):
            neighbor_idx = indices[i][1:]  # Exclude self
            neighbor_values = X_gene[neighbor_idx, :]
            neighbor_max[i, :] = neighbor_values.max(axis=0)

        max_cols = [
            col.replace('_own', '_neighbour-max') for col in expression_features
        ]

        X_aug = np.hstack([X_gene, neighbor_max])
        all_cols = expression_features + max_cols
        section_df = pd.DataFrame(X_aug, columns=all_cols, index=datapoint_ids)

        section_df['section'] = section
        section_df[['x', 'y']] = group_df[['x', 'y']]
        if 'tissue' in group_df.columns:
            section_df['tissue'] = group_df['tissue'].values

        augmented_df_list.append(section_df)

    augmented_data = pd.concat(augmented_df_list)
    return augmented_data


# ═══════════════════════════════════════════════════════════════════════════
# Edge detection and distance-to-edge
# ═══════════════════════════════════════════════════════════════════════════

def _find_arithmetic_segments(series: pd.Series) -> List[Any]:
    """
    Identify element IDs belonging to arithmetic subsequences of
    length >= 5 within a Series.
    """
    values = series.values
    ids = series.index.to_numpy()
    n = len(values)
    i = 0
    arithmetic_ids: List[Any] = []

    while i < n - 1:
        diff = values[i + 1] - values[i]
        j = i + 1
        while j < n - 1 and (values[j + 1] - values[j] == diff):
            j += 1

        segment_length = j - i + 1
        if segment_length >= 5:
            arithmetic_ids.extend(ids[i : j + 1].tolist())

        i = j

    return arithmetic_ids


def _find_technical_edge(
    data: pd.DataFrame,
    coord_cols: Tuple[str, str] = ("x", "y"),
    min_segment_length: int = 5,
    max_inward_steps: int = 3,
    max_outward_fraction: float = 0.30,
) -> List[Any]:
    """
    Identify spot/window IDs lying along capture-area technical edges.

    For each of the four boundaries (x_max, x_min, y_max, y_min), walks
    inward up to ``max_inward_steps`` grid columns, searching for the first
    column whose perpendicular coordinates contain an arithmetic progression
    of length ≥ ``min_segment_length`` (via :func:`_find_arithmetic_segments`).
    The search stops as soon as a valid technical edge is found for that
    boundary.

    A candidate column is accepted only if the number of spots that lie
    *outward* of it (between it and the boundary, exclusive) is ≤
    ``max_outward_fraction`` × (number of spots in the candidate column).
    This prevents a genuine inward column from being mislabelled as technical
    when substantial tissue exists between it and the boundary.

    Parameters
    ----------
    data :
        DataFrame indexed by spot/window ID with numeric coordinate columns.
    coord_cols :
        Names of the (x, y) coordinate columns. Default ``("x", "y")``.
    min_segment_length :
        Minimum arithmetic-progression run length to consider a column a
        technical boundary candidate. Default 5.
    max_inward_steps :
        Maximum number of grid columns to walk inward from each boundary
        before giving up. Default 3.
    max_outward_fraction :
        Maximum allowed fraction of spots outward of a candidate column
        relative to the candidate column size.  Candidates with
        ``n_outward / n_candidate > max_outward_fraction`` are rejected.
        Default 0.30.

    Returns
    -------
    List[Any]
        IDs of spots/windows on capture-area technical edges.
    """
    x_col, y_col = coord_cols
    tech: List[Any] = []

    # Check both axes: left/right boundaries (search_col=x, arith tested in y)
    # and top/bottom boundaries (search_col=y, arith tested in x).
    for search_col, span_col in (
        (x_col, y_col),
        (y_col, x_col),
    ):
        unique_vals = np.sort(data[search_col].unique())

        # Walk inward from maximum boundary, then from minimum boundary.
        # ordered[0] is the boundary value; ordered[1] is one step inward, etc.
        for from_max in (True, False):
            ordered = unique_vals[::-1] if from_max else unique_vals

            for step in range(min(max_inward_steps, len(ordered))):
                val = ordered[step]
                ids = data.index[data[search_col] == val].tolist()

                if len(ids) < min_segment_length:
                    # Too sparse at this step — keep walking inward
                    continue

                # Test perpendicular coordinates for arithmetic progressions
                perp = data.loc[ids, span_col].sort_values()
                arith_ids = _find_arithmetic_segments(perp)
                if not arith_ids:
                    # No arithmetic structure — keep walking inward
                    continue

                # Validate: count spots outward of this candidate column
                # (grid values already visited / skipped, exclusive of this col)
                outward_vals = ordered[:step]
                n_outward    = int(data[search_col].isin(outward_vals).sum())
                n_candidate  = len(ids)
                if n_candidate > 0 and n_outward / n_candidate > max_outward_fraction:
                    # Too many spots outward — likely not the true boundary;
                    # keep walking inward
                    logger.debug(
                        "_find_technical_edge: step %d (val=%s) rejected — "
                        "%d outward spots > %.0f%% of %d candidate spots",
                        step, val, n_outward,
                        max_outward_fraction * 100, n_candidate,
                    )
                    continue

                # Accepted — flag arithmetic IDs and stop for this boundary
                tech.extend(arith_ids)
                logger.debug(
                    "_find_technical_edge: accepted step %d (val=%s) — "
                    "%d arithmetic IDs, %d outward spots",
                    step, val, len(arith_ids), n_outward,
                )
                break

    return list(set(tech))


def _distance_to_edge(
    data: pd.DataFrame,
    norm: bool = True,
    log1p: bool = True,
) -> pd.DataFrame:
    """
    Calculate each spot's distance to the nearest tissue edge.

    Edge spots get distance 0. Non-edge spots use kNN to find the
    nearest edge spot.  Optionally min-max normalises and applies
    log1p(distance * 10).
    """
    edge_mask = data['is_edge'] == True
    edge_df = data.loc[edge_mask, ['x', 'y']].copy()
    non_edge_df = data.loc[~edge_mask, ['x', 'y']].copy()

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nbrs.fit(edge_df.values)

    distances, _ = nbrs.kneighbors(non_edge_df.values)
    data.loc[~edge_mask, 'distance_to_edge'] = distances.flatten()
    data.loc[edge_mask, 'distance_to_edge'] = 0.0

    if norm:
        scaler = MinMaxScaler()
        data['distance_to_edge'] = scaler.fit_transform(
            data[['distance_to_edge']]
        ).flatten()

    if log1p:
        data['distance_to_edge'] = np.log1p(data['distance_to_edge'] * 10)

    return data


def annotate_edge(
    data: pd.DataFrame,
    tile_type: Literal['hexagon', 'square'],
    remove_technical_edge: bool,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Annotate spatial spot data with neighbour counts, edge flags, and
    distance-to-edge.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by spot ID containing 'section', 'x', 'y'.
    tile_type : {'hexagon', 'square'}
        Grid topology (6 neighbours for hexagon, 4 for square).
    remove_technical_edge : bool
        If True, exclude spots on technical edges from edge labelling.
    plot : bool, default False
        If True, plot distance-to-edge per section for QC.

    Returns
    -------
    pd.DataFrame
        Augmented with 'n_neighbours', 'is_edge', 'distance_to_edge'.
    """
    if tile_type not in ('hexagon', 'square'):
        raise ValueError("Invalid tile_type. Expected 'hexagon' or 'square'.")

    data['n_neighbours'] = np.nan
    data['is_edge'] = False
    data['distance_to_edge'] = np.nan

    for section, df in data.groupby('section'):
        coords = df[['x', 'y']].values

        n_neighbors = 7 if tile_type == 'hexagon' else 5
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        nbrs.fit(coords)
        distances, _ = nbrs.kneighbors(coords)

        # Reference distance is the smallest non-zero neighbour distance
        ref_distance = np.min(distances[:, 1])
        # Count neighbours within 10% of reference distance (excluding self)
        neighbour_counts = ((distances[:, 1:] / ref_distance) < 1.1).sum(axis=1)

        data.loc[df.index, 'n_neighbours'] = neighbour_counts

        # Flag edge spots based on neighbour threshold
        threshold = 4 if tile_type == 'hexagon' else 3
        edge_mask = neighbour_counts <= threshold
        data.loc[df.index, 'is_edge'] = edge_mask
        df_loc = df.copy()
        df_loc['is_edge'] = edge_mask

        # Optionally remove technical-edge spots
        if remove_technical_edge:
            tech_ids = _find_technical_edge(df_loc)
            data.loc[tech_ids, 'is_edge'] = False
            df_loc.loc[tech_ids, 'is_edge'] = False

        # Calculate distance to edge
        df_loc = _distance_to_edge(df_loc, norm=True, log1p=True)
        data.loc[df_loc.index, 'distance_to_edge'] = df_loc['distance_to_edge']

        if plot:
            sns.scatterplot(
                x='x', y='y', hue='distance_to_edge', data=df_loc,
                palette='rainbow_r', s=10,
            )
            plt.legend(
                title='Distance to edge\n(normalised)',
                bbox_to_anchor=(1, 1),
            )
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(section)
            plt.show()

    return data


# ═══════════════════════════════════════════════════════════════════════════
# Edge-detection QC plots
# ═══════════════════════════════════════════════════════════════════════════

def _save_edge_qc_plots(
    data: pd.DataFrame,
    save_dir: str,
    data_tag: str = "ref",
) -> None:
    """
    Save per-section edge-detection QC figures to *save_dir*.

    For each section produces 3 PNGs:
    1. ``edge_detection_{tag}_{section}.png``  — interior vs edge spots
    2. ``n_neighbours_{tag}_{section}.png``    — neighbour-count heatmap
    3. ``distance_to_edge_{tag}_{section}.png`` — distance-to-edge heatmap

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: x, y, section, is_edge, n_neighbours,
        distance_to_edge.
    save_dir : str
        Output directory (created if needed).
    data_tag : str
        Short label used in titles and filenames (e.g. "SD_3prime", "HD").
    """
    plot_dir = Path(save_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for section, sec_df in data.groupby("section"):
        # Sanitise section name for filenames
        safe_section = str(section).replace("/", "_").replace(" ", "_")
        prefix = f"{data_tag}_{safe_section}"

        x = sec_df["x"].values
        y = sec_df["y"].values

        # 1. Edge detection (interior vs edge)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        ax = axes[0]
        niche_col = "tissue" if "tissue" in sec_df.columns else "niche"
        if niche_col in sec_df.columns:
            categories = sec_df[niche_col].astype(str)
            for cat in sorted(categories.unique()):
                mask = categories == cat
                ax.scatter(x[mask], y[mask], s=3, alpha=0.5, label=cat)
            ax.legend(fontsize=6, bbox_to_anchor=(1, 1), loc="upper left")
        else:
            ax.scatter(x, y, s=3, alpha=0.5, c="grey")
        ax.set_title(f"[{data_tag}] {section} — Niche labels")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

        ax = axes[1]
        is_edge = sec_df["is_edge"].values.astype(bool)
        ax.scatter(x[~is_edge], y[~is_edge], s=3, alpha=0.3, c="lightgrey", label="interior")
        ax.scatter(x[is_edge], y[is_edge], s=5, alpha=0.8, c="red", label="edge")
        ax.legend(fontsize=8)
        ax.set_title(f"[{data_tag}] {section} — Edge detection (is_edge)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

        plt.tight_layout()
        fig.savefig(plot_dir / f"edge_detection_{prefix}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2. Neighbour count heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sc_plot = ax.scatter(x, y, c=sec_df["n_neighbours"].values,
                             cmap="viridis", s=3, alpha=0.7)
        plt.colorbar(sc_plot, ax=ax, label="n_neighbours")
        ax.set_title(f"[{data_tag}] {section} — Neighbour count")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(plot_dir / f"n_neighbours_{prefix}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 3. Distance to edge heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sc_plot = ax.scatter(x, y, c=sec_df["distance_to_edge"].values,
                             cmap="rainbow_r", s=3, alpha=0.7)
        plt.colorbar(sc_plot, ax=ax, label="distance_to_edge (norm + log1p)")
        ax.set_title(f"[{data_tag}] {section} — Distance to edge")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(plot_dir / f"distance_to_edge_{prefix}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("  [%s] Section '%s': 3 QC plots saved", data_tag, section)

    logger.info("[%s] All edge-detection QC plots saved to %s/", data_tag, plot_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Neighbourhood feature builders
# ═══════════════════════════════════════════════════════════════════════════
# These wrappers handle data preparation (extract genes, add coords,
# add section metadata) and call include_neighbours() and annotate_edge()
# to compute the final feature DataFrames consumed by the LR pipeline.

def build_neighbourhood_features_sd(
    adata: AnnData,
    genes: list[str],
    niche_col: Optional[str],
    section_col: str = "section_ID",
    coord_cols: Optional[Tuple[str, str]] = None,
    k_neighbours: int = 6,
    remove_technical_edge: bool = True,
    plot: bool = False,
    save_dir: Optional[str] = None,
    data_tag: str = "SD",
) -> pd.DataFrame:
    """
    Build neighbourhood-augmented feature DataFrame for SD spots.

    Drop-in replacement for ``neighbourhood.build_neighbourhood_features_sd``
    that wraps include_neighbours() and annotate_edge().

    Each spot gets:
    - ``<gene>_own``:            its own normalised expression
    - ``<gene>_neighbour-max``:  max expression among k nearest spots
    - ``n_neighbours``, ``is_edge``, ``distance_to_edge``
    - ``section``, ``x``, ``y``, ``niche`` metadata
    """
    genes_here = [g for g in genes if g in adata.var_names]
    if not genes_here:
        raise ValueError("No requested genes found in adata.var_names.")
    logger.info(
        "SD neighbourhood: %d / %d genes found in adata",
        len(genes_here), len(genes),
    )

    sub = adata[:, genes_here]
    X = sub.X
    if issparse(X):
        X = np.asarray(X.todense())

    data = pd.DataFrame(
        X.astype(np.float32),
        index=adata.obs_names,
        columns=[f"{g}_own" for g in genes_here],
    )

    # Coordinates — use array_col/array_row (integer grid indices) by default
    # for SD reference data, not obsm['spatial'] float pixel coords.
    if coord_cols is not None:
        data["x"] = adata.obs[coord_cols[0]].values.astype(float)
        data["y"] = adata.obs[coord_cols[1]].values.astype(float)
    elif "array_col" in adata.obs.columns and "array_row" in adata.obs.columns:
        # Default: integer grid coordinates
        data["x"] = adata.obs["array_col"].values.astype(float)
        data["y"] = adata.obs["array_row"].values.astype(float)
    else:
        # Fallback to obsm['spatial'] if array_col/row not available
        spatial = adata.obsm["spatial"]
        data["x"] = spatial[:, 0]
        data["y"] = spatial[:, 1]

    # Section + niche metadata
    # include_neighbours carries through "tissue" column specifically,
    # so we set it as "tissue" here, then rename to "niche" at the end
    # for compatibility with 02/04 scripts which expect "niche".
    data["section"] = adata.obs[section_col].values
    if niche_col is not None and niche_col in adata.obs.columns:
        data["tissue"] = adata.obs[niche_col].values
    else:
        data["tissue"] = "_query"

    # Neighbour-max
    logger.info("SD: computing neighbour-max (k=%d)...", k_neighbours)
    data = include_neighbours(data, k=k_neighbours)

    # Edge annotation — uses tile_type='square' for SD reference data.
    logger.info("SD: annotating edges (tile_type=square)...")
    data = annotate_edge(
        data,
        tile_type="square",
        remove_technical_edge=remove_technical_edge,
        plot=False,  # suppress plt.show(); save to file instead
    )

    # Save QC plots if save_dir is provided
    if save_dir is not None:
        _save_edge_qc_plots(data, save_dir=save_dir, data_tag=data_tag)

    # Rename "tissue" → "niche" for downstream consumers
    data = data.rename(columns={"tissue": "niche"})

    logger.info(
        "SD neighbourhood complete: %d spots × %d columns",
        len(data), len(data.columns),
    )
    return data


def build_neighbourhood_features_hd(
    adata_windows: AnnData,
    genes: list[str],
    niche_col: Optional[str],
    section_col: str = "section_ID",
    k_neighbours: int = 6,
    remove_technical_edge: bool = True,
    plot: bool = False,
    save_dir: Optional[str] = None,
    data_tag: str = "HD",
) -> pd.DataFrame:
    """
    Build neighbourhood-augmented feature DataFrame for HD pseudobulk windows.

    Drop-in replacement for ``neighbourhood.build_neighbourhood_features_hd``
    that wraps include_neighbours() and annotate_edge().

    Uses square-grid geometry (k=6 neighbours).
    Window centres are read from ``adata_windows.obs['window_col']`` /
    ``adata_windows.obs['window_row']``.

    Note: uses x/y (float window centres) for both kNN and edge detection
    rather than integer grid indices, to keep distance metrics on a single
    µm-scaled coordinate system.
    """
    for req in ("window_col", "window_row"):
        if req not in adata_windows.obs.columns:
            raise ValueError(
                f"adata_windows.obs must have '{req}' (window centres)."
            )

    genes_here = [g for g in genes if g in adata_windows.var_names]
    if not genes_here:
        raise ValueError("No requested genes found in adata_windows.var_names.")
    logger.info(
        "HD neighbourhood: %d / %d genes found in adata_windows",
        len(genes_here), len(genes),
    )

    sub = adata_windows[:, genes_here]
    X = sub.X
    if issparse(X):
        X = np.asarray(X.todense())

    data = pd.DataFrame(
        X.astype(np.float32),
        index=adata_windows.obs_names,
        columns=[f"{g}_own" for g in genes_here],
    )

    # Window centres as x/y (uses float coords, not int grid)
    data["x"] = adata_windows.obs["window_col"].values
    data["y"] = adata_windows.obs["window_row"].values

    # Section + niche metadata (use "tissue" for include_neighbours compatibility, rename later)
    data["section"] = adata_windows.obs[section_col].values
    if niche_col is not None and niche_col in adata_windows.obs.columns:
        data["tissue"] = adata_windows.obs[niche_col].values
    else:
        data["tissue"] = "_query"

    # Neighbour-max (k=6 default for HD square-grid windows)
    logger.info("HD: computing neighbour-max (k=%d)...", k_neighbours)
    data = include_neighbours(data, k=k_neighbours)

    # Edge annotation (square grid for HD)
    logger.info("HD: annotating edges (tile_type=square)...")
    data = annotate_edge(
        data,
        tile_type="square",
        remove_technical_edge=remove_technical_edge,
        plot=False,  # suppress plt.show(); save to file instead
    )

    # Save QC plots if save_dir is provided
    if save_dir is not None:
        _save_edge_qc_plots(data, save_dir=save_dir, data_tag=data_tag)

    # Rename "tissue" → "niche" for downstream consumers
    data = data.rename(columns={"tissue": "niche"})

    logger.info(
        "HD neighbourhood complete: %d windows × %d columns",
        len(data), len(data.columns),
    )
    return data


# ═══════════════════════════════════════════════════════════════════════════
# HD / Cell-level pseudobulk windowing
# (consolidated from degs.py — all windowing in one place)
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_window_corners_v2(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    window_size: float,
    overlap: float = 0,
    drop_partial_windows: bool = False,
) -> pd.DataFrame:
    """
    Calculate corner coordinates of a regular grid of windows covering
    the spatial domain [min_x, max_x] × [min_y, max_y].

    Adapted from squidpy's sliding window implementation.

    Parameters
    ----------
    min_x, max_x : X extent of the domain.
    min_y, max_y : Y extent of the domain.
    window_size  : Side length of each square window.
    overlap      : Overlap between consecutive windows. Default 0.
    drop_partial_windows : If True, discard border windows < window_size.

    Returns
    -------
    pd.DataFrame with columns: x_start, x_end, y_start, y_end.
    """
    from itertools import product as _product

    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= window_size:
        raise ValueError("overlap must be less than window_size.")

    step = window_size - overlap
    x_starts = np.arange(min_x, max_x, step)
    y_starts = np.arange(min_y, max_y, step)

    starts = list(_product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + window_size
    windows["y_end"] = windows["y_start"] + window_size

    if not drop_partial_windows:
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    else:
        windows = windows[
            (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        ]

    return windows[["x_start", "x_end", "y_start", "y_end"]].reset_index(drop=True)


def _validate_raw_counts(
    matrix,
    source: str,
    n_sample: int = 10_000,
) -> None:
    """
    Warn if a count matrix does not look like raw integer counts.

    Checks: non-integer values, suspicious median > 50, negative values.
    """
    import scipy.sparse as sp

    if sp.issparse(matrix):
        data = np.asarray(matrix.data, dtype=np.float64)
    else:
        data = np.asarray(matrix, dtype=np.float64).ravel()

    if len(data) > n_sample:
        rng = np.random.default_rng(seed=0)
        data = rng.choice(data, size=n_sample, replace=False)

    data = data[data != 0]

    if len(data) == 0:
        logger.warning(
            "Raw count check (%s): matrix appears to be all zeros — "
            "cannot validate.", source,
        )
        return

    fractional = np.abs(data - np.round(data))
    frac_nonzero = np.mean(fractional > 1e-6)
    if frac_nonzero > 0.01:
        logger.warning(
            "Raw count check (%s): %.1f%% of sampled non-zero values have "
            "fractional parts — does NOT look like raw integer counts.",
            source, frac_nonzero * 100,
        )

    median_val = np.median(data)
    if median_val > 50:
        logger.warning(
            "Raw count check (%s): median non-zero value is %.1f — "
            "unusually high for raw counts (expected < 10).",
            source, median_val,
        )

    if np.any(data < 0):
        logger.warning(
            "Raw count check (%s): negative values detected — "
            "impossible in raw counts.", source,
        )

    if frac_nonzero <= 0.01 and median_val <= 50 and not np.any(data < 0):
        logger.info(
            "Raw count check (%s): OK "
            "(median non-zero = %.1f, integer = %.1f%%)",
            source, median_val, (1 - frac_nonzero) * 100,
        )


def sliding_window_pseudobulk_hd(
    adata: AnnData,
    section_col: str,
    niche_col: str,
    library_col: str = "library",
    microns_per_pixel_map: dict[str, float] = None,
    target_spot_um: float = 55.0,
    min_cells_per_window: int = 10,
) -> AnnData:
    """
    Aggregate HD cells into spatial pseudo-spots by tiling each section
    into non-overlapping square windows.

    Window size is computed per section from its library's µm/px calibration:
    ``window_size_px = target_spot_um / microns_per_pixel``.

    Returns raw summed counts — normalisation is intentionally deferred to
    ``normalise_if_needed()`` at load time, so the gene-set denominator
    always matches (critical for panel-specific retraining).

    Parameters
    ----------
    adata : HD AnnData with obsm["spatial"], obs columns for section/niche/library/donor.
    section_col : obs column identifying each section.
    niche_col : obs column with niche labels for majority-vote assignment.
    library_col : obs column identifying the library/run. Default "library".
    microns_per_pixel_map : Dict mapping library ID → µm per pixel.
    target_spot_um : Target window size in microns. Default 55.0.
    min_cells_per_window : Windows with fewer cells are discarded. Default 10.

    Returns
    -------
    AnnData with raw summed counts in .X and window metadata in .obs.
    """
    import scipy.sparse as sp

    if microns_per_pixel_map is None:
        raise ValueError("microns_per_pixel_map must be provided.")

    if "raw_counts" in adata.layers:
        raw_source = "layers['raw_counts']"
        all_raw = adata.layers["raw_counts"]
        logger.info("Using raw counts from %s.", raw_source)
    else:
        raw_source = "adata.X"
        all_raw = adata.X
        logger.info(
            "layers['raw_counts'] not found — falling back to adata.X."
        )

    _validate_raw_counts(all_raw, source=raw_source)

    all_counts: list[np.ndarray] = []
    all_obs: list[dict] = []

    all_coords = adata.obsm["spatial"]
    all_niche = adata.obs[niche_col].values
    all_donor = adata.obs["donor"].values

    sections = adata.obs[section_col].unique()

    for section in sections:
        sec_idx = np.where(adata.obs[section_col].values == section)[0]

        library = adata.obs[library_col].iloc[sec_idx[0]]
        if library not in microns_per_pixel_map:
            raise ValueError(
                f"Library '{library}' (section '{section}') not found in "
                f"microns_per_pixel_map. Known: {list(microns_per_pixel_map.keys())}"
            )
        mpp = microns_per_pixel_map[library]
        window_size_px = target_spot_um / mpp

        coords = all_coords[sec_idx]
        x = coords[:, 0]
        y = coords[:, 1]
        niches = all_niche[sec_idx]
        donors = all_donor[sec_idx]

        min_x_sec, min_y_sec = x.min(), y.min()
        windows = _calculate_window_corners_v2(
            min_x=min_x_sec, max_x=x.max(),
            min_y=min_y_sec, max_y=y.max(),
            window_size=window_size_px,
        )

        n_win_before = 0
        for _, win in windows.iterrows():
            in_win = (
                (x >= win.x_start) & (x <= win.x_end) &
                (y >= win.y_start) & (y <= win.y_end)
            )
            n_cells = int(in_win.sum())
            if n_cells < min_cells_per_window:
                continue

            global_idx = sec_idx[in_win]
            sub_raw = all_raw[global_idx]
            summed = (
                np.asarray(sub_raw.sum(axis=0)).ravel()
                if sp.issparse(sub_raw)
                else np.asarray(sub_raw.sum(axis=0)).ravel()
            )
            all_counts.append(summed)

            niche_majority = pd.Series(niches[in_win]).mode().iloc[0]
            donor_majority = pd.Series(donors[in_win]).mode().iloc[0]
            col_idx = int(round((win.x_start - min_x_sec) / window_size_px))
            row_idx = int(round((win.y_start - min_y_sec) / window_size_px))
            all_obs.append({
                section_col:       section,
                niche_col:         niche_majority,
                library_col:       library,
                "donor":           donor_majority,
                "_n_cells":        n_cells,
                "window_col":      float(win.x_start) + window_size_px / 2,
                "window_row":      float(win.y_start) + window_size_px / 2,
                "window_col_idx":  col_idx,
                "window_row_idx":  row_idx,
            })
            n_win_before += 1

        logger.info(
            "  Section '%s' (library=%s, window=%.1fpx): %d windows",
            section, library, window_size_px, n_win_before,
        )

    if not all_counts:
        raise ValueError(
            "No windows survived min_cells_per_window filtering."
        )

    X = np.vstack(all_counts).astype(np.float32)
    obs_df = pd.DataFrame(all_obs).reset_index(drop=True)

    import anndata as ad
    pb = ad.AnnData(X=X, var=adata.var.copy(), obs=obs_df)

    logger.info(
        "HD sliding-window pseudobulk complete: %d windows from %d cells "
        "across %d sections (target_spot_um=%.1f) — raw counts saved.",
        pb.n_obs, adata.n_obs, len(sections), target_spot_um,
    )
    return pb


def sliding_window_pseudobulk_cells(
    adata: AnnData,
    section_col: str,
    target_spot_um: float = 55.0,
    coords_obsm_key: str = "spatial",
    niche_col: Optional[str] = None,
    min_cells_per_window: int = 10,
    sliding_window_key: Optional[str] = "sliding_window_assignment",
) -> AnnData:
    """
    Aggregate cell-level spatial transcriptomics data (Xenium, MERFISH, CosMx)
    into sliding-window pseudo-spots.

    Works natively in micrometre space — no µm/px calibration needed.

    Returns raw summed counts — normalisation deferred to normalise_if_needed().

    Parameters
    ----------
    adata : Cell-level AnnData with obsm[coords_obsm_key] in µm.
    section_col : obs column identifying each section.
    target_spot_um : Window size in µm. Default 55.0.
    coords_obsm_key : Key in obsm for (x, y) coordinates. Default "spatial".
    niche_col : obs column with niche labels (None to skip majority-vote).
    min_cells_per_window : Minimum cells per window. Default 10.
    sliding_window_key :
        Name of the obs column written *both* to the input ``adata`` (cells)
        and to the returned windowed AnnData. Each cell receives the window
        identifier of the window it falls in, or NaN if its window was
        dropped (< ``min_cells_per_window``) or the cell is outside the
        grid. Pass ``None`` to skip this side effect on the input adata.
        Default ``"sliding_window_assignment"`` — matches the HD pipeline,
        and lets ``predict_adata`` map window-level predictions back to
        individual cells via the ``sliding_window_col=`` argument.

    Returns
    -------
    AnnData with raw summed counts and window metadata. The input
    ``adata.obs`` is also modified in place: a ``sliding_window_key``
    column is added (unless ``sliding_window_key=None``).
    """
    import scipy.sparse as sp

    if coords_obsm_key not in adata.obsm:
        raise ValueError(
            f"obsm key '{coords_obsm_key}' not found. "
            f"Available: {list(adata.obsm.keys())}"
        )
    if section_col not in adata.obs.columns:
        raise ValueError(
            f"section column '{section_col}' not found in obs."
        )
    if niche_col is not None and niche_col not in adata.obs.columns:
        raise ValueError(
            f"niche column '{niche_col}' not found in obs."
        )

    if "raw_counts" in adata.layers:
        raw_source = "layers['raw_counts']"
        all_raw = adata.layers["raw_counts"]
        logger.info("Using raw counts from %s.", raw_source)
    else:
        raw_source = "adata.X"
        all_raw = adata.X
        logger.info(
            "layers['raw_counts'] not found — falling back to adata.X."
        )
    _validate_raw_counts(all_raw, source=raw_source)

    all_counts: list[np.ndarray] = []
    all_obs: list[dict] = []

    all_coords = adata.obsm[coords_obsm_key]
    all_niche = adata.obs[niche_col].values if niche_col is not None else None
    sections = adata.obs[section_col].unique()

    # Per-cell window assignment for round-trip projection of window-level
    # predictions back to cells. Cells in dropped windows / outside the grid
    # remain None → become NaN in the obs column.
    cell_window_assignment: Optional[np.ndarray] = (
        np.full(adata.n_obs, None, dtype=object)
        if sliding_window_key is not None
        else None
    )

    for section in sections:
        sec_idx = np.where(adata.obs[section_col].values == section)[0]
        coords = all_coords[sec_idx]
        x = coords[:, 0]
        y = coords[:, 1]

        min_x_sec, min_y_sec = x.min(), y.min()

        windows = _calculate_window_corners_v2(
            min_x=min_x_sec, max_x=x.max(),
            min_y=min_y_sec, max_y=y.max(),
            window_size=target_spot_um,
        )

        n_win_section = 0
        for _, win in windows.iterrows():
            in_win = (
                (x >= win.x_start) & (x <= win.x_end) &
                (y >= win.y_start) & (y <= win.y_end)
            )
            n_cells = int(in_win.sum())
            if n_cells < min_cells_per_window:
                continue

            global_idx = sec_idx[in_win]
            sub_raw = all_raw[global_idx]
            summed = (
                np.asarray(sub_raw.sum(axis=0)).ravel()
                if sp.issparse(sub_raw)
                else np.asarray(sub_raw.sum(axis=0)).ravel()
            )
            all_counts.append(summed)

            col_idx = int(round((win.x_start - min_x_sec) / target_spot_um))
            row_idx = int(round((win.y_start - min_y_sec) / target_spot_um))

            # Window identifier — used as the join key for cell ↔ window
            # round-trip projection. Section-namespaced so it's unique
            # globally even if window indices repeat across sections.
            window_id = f"{section}_window_{n_win_section}"

            obs_dict: dict = {
                section_col:      section,
                "_n_cells":       n_cells,
                "window_col":     float(win.x_start) + target_spot_um / 2,
                "window_row":     float(win.y_start) + target_spot_um / 2,
                "window_col_idx": col_idx,
                "window_row_idx": row_idx,
            }
            if sliding_window_key is not None:
                obs_dict[sliding_window_key] = window_id
                cell_window_assignment[global_idx] = window_id
            if niche_col is not None:
                obs_dict[niche_col] = pd.Series(all_niche[in_win]).mode().iloc[0]

            all_obs.append(obs_dict)
            n_win_section += 1

        logger.info(
            "  Section '%s' (window=%.1f µm): %d windows from %d cells",
            section, target_spot_um, n_win_section, len(sec_idx),
        )

    if not all_counts:
        raise ValueError(
            "No windows survived min_cells_per_window filtering."
        )

    X = np.vstack(all_counts).astype(np.float32)
    obs_df = pd.DataFrame(all_obs).reset_index(drop=True)

    import anndata as ad
    pb = ad.AnnData(X=X, var=adata.var.copy(), obs=obs_df)

    pb.obsm["spatial"] = obs_df[["window_col", "window_row"]].to_numpy(dtype=np.float64)

    # Write the cell → window mapping back to the input adata so callers can
    # project window-level predictions onto individual cells via
    # ``predict_adata(adata, hd_windows=pb, sliding_window_col=sliding_window_key)``.
    if sliding_window_key is not None:
        adata.obs[sliding_window_key] = cell_window_assignment
        n_assigned = int(sum(v is not None for v in cell_window_assignment))
        logger.info(
            "Wrote cell → window mapping to adata.obs['%s']: "
            "%d / %d cells assigned (rest fell in dropped windows or outside grid).",
            sliding_window_key, n_assigned, adata.n_obs,
        )

    logger.info(
        "Cell sliding-window pseudobulk complete: %d windows from %d cells "
        "across %d sections (target_spot_um=%.1f µm) — raw counts saved.",
        pb.n_obs, adata.n_obs, len(sections), target_spot_um,
    )
    return pb
