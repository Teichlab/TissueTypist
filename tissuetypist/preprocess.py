import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple, List, Any, Literal, Sequence
import warnings

from itertools import product
from scanpy import logging as logg
from squidpy.gr._utils import _save_data

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from . import load_data
from pathlib import Path

# a helper function for "_sliding_window"
# from squidpy (https://github.com/scverse/squidpy)
# https://github.com/scverse/squidpy/blob/bdd989983cd24aaa46bbf4751f1e904fe10ecc44/src/squidpy/tl/_sliding_window.py#L178
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
    Calculate the corner points of all windows covering the area from min_x to max_x and min_y to max_y,
    with specified window_size and overlap.

    Parameters
    ----------
    min_x: float
        minimum X coordinate
    max_x: float
        maximum X coordinate
    min_y: float
        minimum Y coordinate
    max_y: float
        maximum Y coordinate
    window_size: float
        size of each window
    overlap: float
        overlap between consecutive windows (must be less than window_size)
    drop_partial_windows: bool
        if True, drop border windows that are smaller than window_size;
        if False, create smaller windows at the borders to cover the remaining space.

    Returns
    -------
    windows: pandas DataFrame with columns ['x_start', 'x_end', 'y_start', 'y_end']
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if overlap >= window_size:
        raise ValueError("Overlap must be less than the window size.")

    x_step = window_size - overlap
    y_step = window_size - overlap

    # Generate starting points
    x_starts = np.arange(min_x, max_x, x_step)
    y_starts = np.arange(min_y, max_y, y_step)

    # Create all combinations of x and y starting points
    starts = list(product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + window_size
    windows["y_end"] = windows["y_start"] + window_size

    # Adjust windows that extend beyond the bounds
    if not drop_partial_windows:
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    else:
        valid_windows = (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        windows = windows[valid_windows]

    windows = windows.reset_index(drop=True)
    return windows[["x_start", "x_end", "y_start", "y_end"]]

# a helper function for "_sliding_window_psudobulk"
# hugely based on the sliding_function in squidpy (https://github.com/scverse/squidpy)
# https://github.com/scverse/squidpy/blob/afcb8d0e81085a1af03ae5a9f299c5df00e95d61/src/squidpy/tl/_sliding_window.py
# modified to output coordinates of each window
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
    Divide a tissue slice into regulary shaped spatially contiguous regions (windows).

    Parameters
    ----------
    %(adata)s
    window_size: int
        Size of the sliding window.
    %(library_key)s
    coord_columns: Tuple[str, str]
        Tuple of column names in `adata.obs` that specify the coordinates (x, y), e.i. ('globalX', 'globalY')
    sliding_window_key: str
        Base name for sliding window columns.
    overlap: int
        Overlap size between consecutive windows. (0 = no overlap)
    %(spatial_key)s
    drop_partial_windows: bool
        If True, drop windows that are smaller than the window size at the borders.
    copy: bool
        If True, return the result, otherwise save it to the adata object.

    Returns
    -------
    If ``copy = True``, returns the sliding window annotation(s) as pandas dataframe
    Otherwise, stores the sliding window annotation(s) in .obs.
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")

    # we don't want to modify the original adata in case of copy=True
    if copy:
        adata = adata.copy()

    # extract coordinates of observations
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
            f"Coordinates not found. Provide `{coord_columns}` in `adata.obs` or specify a suitable `spatial_key` in `adata.obsm`."
        )

    # infer window size if not provided
    if window_size is None:
        coord_range = max(
            coords[x_col].max() - coords[x_col].min(),
            coords[y_col].max() - coords[y_col].min(),
        )
        # mostly arbitrary choice, except that full integers usually generate windows with 1-2 cells at the borders
        window_size = max(int(np.floor(coord_range // 3.95)), 1)

    if window_size <= 0:
        raise ValueError("Window size must be larger than 0.")

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    libraries = [None] if library_key is None else adata.obs[library_key].unique()

    # Create a DataFrame to store the sliding window assignments
    sliding_window_df = pd.DataFrame(index=adata.obs.index)

    if sliding_window_key in adata.obs:
        logg.warning(f"Overwriting existing column '{sliding_window_key}' in adata.obs.")

    for lib in libraries:
        if lib is not None:
            lib_mask = adata.obs[library_key] == lib
            lib_coords = coords.loc[lib_mask]
        else:
            lib_mask = np.ones(len(adata), dtype=bool)
            lib_coords = coords

        min_x, max_x = lib_coords[x_col].min(), lib_coords[x_col].max()
        min_y, max_y = lib_coords[y_col].min(), lib_coords[y_col].max()

        # precalculate windows
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

        # assign observations to windows
        for idx, window in windows.iterrows():
            x_start = window["x_start"]
            x_end = window["x_end"]
            y_start = window["y_start"]
            y_end = window["y_end"]

            mask = (
                (lib_coords[x_col] >= x_start)
                & (lib_coords[x_col] <= x_end)
                & (lib_coords[y_col] >= y_start)
                & (lib_coords[y_col] <= y_end)
            )
            obs_indices = lib_coords.index[mask]

            if overlap == 0:
                mask = (
                    (lib_coords[x_col] >= x_start)
                    & (lib_coords[x_col] <= x_end)
                    & (lib_coords[y_col] >= y_start)
                    & (lib_coords[y_col] <= y_end)
                )
                obs_indices = lib_coords.index[mask]
                sliding_window_df.loc[obs_indices, sliding_window_key] = f"{lib_key}window_{idx}"
                
                ############ added to the original function in squidpy ############
                # to get coordinates of each window
                # center point
                sliding_window_df.loc[obs_indices, 'window_col'] = x_start+(window_size/2)
                sliding_window_df.loc[obs_indices, 'window_row'] = y_start+(window_size/2)
                ###################################################################

            else:
                col_name = f"{sliding_window_key}_{lib_key}window_{idx}"
                sliding_window_df.loc[obs_indices, col_name] = True
                sliding_window_df.loc[:, col_name].fillna(False, inplace=True)

    if overlap == 0:
        # create categorical variable for ordered windows
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
    
    ############ added to the original function in squidpy ############
    # report total window number and average cell number per window
    n_windows = len(set(sliding_window_df[sliding_window_key]))
    print(f'### Total window number ###: {n_windows}')
    print(f'### Average cell number per window ###: {round(sliding_window_df.shape[0]/n_windows,1)}')
    ###################################################################

    if copy:
        return sliding_window_df
    for col_name, col_data in sliding_window_df.items():
        _save_data(adata, attr="obs", key=col_name, data=col_data)

# this is a helper function for "preprocess"
def _sliding_window_psudobulk(
    adata: AnnData,
    section_col: str,
    window_size: int,
    coord_columns: Optional[Tuple[str, str]] = None,
    log_normalise: bool = True
) -> AnnData:
    """
    Perform pseudobulk aggregation by sliding windows on spatial data.

    Applies a sliding window assignment per spot based on coordinates,
    then aggregates expression data per window, and optionally log-normalises.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial coordinates in .obs or .obsm.
    section_col : str
        Column in adata.obs indicating section/library grouping.
    window_size : int
        Size of the sliding window (in same units as coordinates).
    coord_columns : tuple of str, optional
        Names of the x, y coordinate columns in adata.obs. If None,
        coordinates are read from adata.obsm['spatial'] into 'globalX', 'globalY'.
    log_normalise : bool, default True
        If True, normalises counts to 10k per window and applies log1p.

    Returns
    -------
    AnnData
        Pseudobulked AnnData with aggregated counts in .X and
        window metadata in .obs ('window_col', 'window_row', section_col).
    """
    # Assign sliding windows to data points
    if coord_columns!=None:
        _sliding_window(
            adata=adata,
            library_key=section_col,
            window_size=window_size,
            overlap=0,
            coord_columns=coord_columns,
            copy=False,  # we modify in place
        )
    else:
        _sliding_window(
            adata=adata,
            library_key=section_col,
            window_size=window_size,
            overlap=0,
            coord_columns=("globalX", "globalY"),
            spatial_key='spatial',
            copy=False,  # we modify in place
        )
    
    # Aggregate counts per window
    bdata = sc.get.aggregate(
        adata,
        by='sliding_window_assignment',
        func=['sum']
    )
    # Store summed counts in .X
    bdata.X = bdata.layers['sum'].copy()

    # Transfer window metadata to pseudobulk AnnData
    obs = adata.obs[[section_col,'window_col','window_row','sliding_window_assignment']].copy()
    obs = obs.drop_duplicates().set_index('sliding_window_assignment')
    bdata.obs[[section_col,'window_col','window_row']] = (
        obs.loc[bdata.obs_names]
        [[section_col, 'window_col', 'window_row']]
        .values
    )

    # Optional log-normalisation
    if log_normalise:
        sc.pp.normalize_total(bdata, target_sum=1e4)
        sc.pp.log1p(bdata)
        print('Pseudobulk: summed per window and log-normalised.')
    else:
        print('Pseudobulk: summed per window without normalisation.')
              
    return bdata

# a helper function for "prepare_dataframe"
def _include_neighbours(
    data: pd.DataFrame,
    k: int = 6
) -> pd.DataFrame:
    """
    Augment each spot's gene expression features by including neighbor-based summary statistics.
    For each section in the DataFrame, computes the maximum expression among k nearest neighbors
    for each gene, and returns an augmented DataFrame with original and neighbor features.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by spot ID, containing 'x', 'y' coordinates, gene expression columns
        ending with '_own', and a 'section' column identifying sections.
    k : int, default=6
        Number of neighbors to consider (excluding self).

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with original and neighbor-max features, plus 'section', 'x', 'y',
        and optionally 'tissue' columns.
    """
    
    # Identify expression feature columns
    expression_features = [col for col in data.columns if col.endswith('_own')]
    
    # Process each section separately
    augmented_df_list=[]
    for section, group_df in data.groupby('section'):
        # Extract spatial coordinates and gene expression for this section
        datapoint_ids = group_df.index
        coords = group_df[['x', 'y']].values
        X_gene = group_df[expression_features].values  # shape: (n_datapoints, n_genes)
        n_datapoints, n_genes = X_gene.shape

        # Set number of neighbors (excluding self); adjust if the section has too few spots
        k_section = k if n_datapoints > k + 1 else max(n_datapoints - 1, 1)

        # Find neighbors including self, then exclude self later
        nbrs = NearestNeighbors(n_neighbors=k_section + 1, algorithm='ball_tree')
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)

        # Compute max expression among neighbors for each spot
        neighbor_max = np.zeros_like(X_gene)
        for i in range(n_datapoints):
            neighbor_idx = indices[i][1:]  # Exclude the spot itself
            neighbor_values = X_gene[neighbor_idx, :]  # shape: (k_section, n_genes)
            neighbor_max[i, :] = neighbor_values.max(axis=0)
        
        # Prepare column names for neighbor features
        max_cols = [col.replace('_own', '_neighbour-max') for col in expression_features]

        # Combine original and neighbor-max features
        X_aug = np.hstack([X_gene, neighbor_max])      
        all_cols = expression_features + max_cols
        section_df = pd.DataFrame(X_aug, columns=all_cols, index=datapoint_ids)

        # Re-add metadata columns
        section_df['section'] = section
        section_df[['x', 'y']] = group_df[['x', 'y']]
        if 'tissue' in group_df.columns:
            section_df['tissue'] = group_df['tissue'].values
        
        # append
        augmented_df_list.append(section_df)
    
    # Concatenate all sections
    augmented_data = pd.concat(augmented_df_list)
    return augmented_data

# this is a helper function for "_find_technical_edge"
def _find_arithmetic_segments(series: pd.Series) -> List[Any]:
    """
    Identify element IDs that belong to arithmetic subsequences of length >= 5 within a Series.

    Scans the Series values where consecutive differences are equal (i.e., arithmetic sequences). Segments shorter than 5 are ignored.

    Parameters
    ----------
    series : pd.Series
        Input Series indexed by element IDs, containing numeric values.

    Returns
    -------
    List[Any]
        List of element IDs (from the Series index) that are part of any arithmetic
        segment with length at least 5.
    """
    values = series.values
    ids = series.index.to_numpy()
    n = len(values)
    i = 0
    arithmetic_ids: List[Any] = []
    
    # Identify all maximal arithmetic segments
    while i < n - 1:
        # Compute initial difference for potential segment
        diff = values[i+1] - values[i]
        j = i + 1
        # Extend segment while difference remains constant
        while j < n - 1 and (values[j+1] - values[j] == diff):
            j += 1

        segment_length = j - i + 1
        if segment_length >= 5:
            arithmetic_ids.extend(ids[i:j+1].tolist())

        # Move to the next potential start
        i = j

    return arithmetic_ids

# this is a helper function for "_annotate_edge"
def _find_technical_edge(data: pd.DataFrame) -> List[Any]:
    """
    Identify spot IDs that lie along technical edges of the spatial dataset.

    Technical edges are defined as spots on the minimum or maximum x or y boundaries
    that also form arithmetic progressions in the perpendicular coordinate of length >= 5.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by spot ID, containing numeric 'x' and 'y' coordinate columns.

    Returns
    -------
    List[Any]
        List of spot IDs that belong to arithmetic segments on any of the four spatial edges.
    """
    # Identify spots on each boundary
    x_max_ids = data.index[data['x']==data['x'].max()]
    x_min_ids = data.index[data['x']==data['x'].min()]
    y_max_ids = data.index[data['y']==data['y'].max()]
    y_min_ids = data.index[data['y']==data['y'].min()]
    # Find arithmetic segments (length >=5) along each edge
    xmax_ap_ids = _find_arithmetic_segments(data.loc[x_max_ids,'y'].sort_values())
    xmin_ap_ids = _find_arithmetic_segments(data.loc[x_min_ids,'y'].sort_values())
    ymax_ap_ids = _find_arithmetic_segments(data.loc[y_max_ids,'x'].sort_values())
    ymin_ap_ids = _find_arithmetic_segments(data.loc[y_min_ids,'x'].sort_values())
    # return
    technical_edge_ids = xmax_ap_ids + xmin_ap_ids + ymax_ap_ids + ymin_ap_ids
    
    return technical_edge_ids

# this is a helper function for "_annotate_edge"
def _distance_to_edge(
    data: pd.DataFrame,
    norm: bool = True,
    log1p: bool = True
) -> pd.DataFrame:
    """
    Calculate and annotate each spot's distance to the nearest tissue edge.

    For spots marked as 'is_edge' in the DataFrame, distance is zero. For others,
    uses kNN to find the nearest edge spot, computes Euclidean distance on 'x', 'y',
    then optionally applies min-max normalization and a log1p transform with scaling.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by spot ID, containing 'x', 'y' coordinates and a boolean
        'is_edge' column indicating technical edge spots.
    norm : bool, default True
        If True, applies min-max normalization to the raw distances.
    log1p : bool, default True
        If True, scales distances by 10 and applies log1p transform.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with a new 'distance_to_edge' column of float distances.
    """
    # Split edge and non-edge spots
    edge_mask = data['is_edge'] == True
    edge_df = data.loc[edge_mask, ['x', 'y']].copy()
    non_edge_df = data.loc[~edge_mask, ['x', 'y']].copy()
    
    # Build kNN using edge spots
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nbrs.fit(edge_df.values)
    
    # Compute nearest-edge distance for non-edge spots
    distances, _ = nbrs.kneighbors(non_edge_df.values)
    data.loc[~edge_mask, 'distance_to_edge'] = distances.flatten()
    data.loc[edge_mask, 'distance_to_edge'] = 0.0
    
    # Min-max normalization
    if norm:
        scaler = MinMaxScaler()
        data['distance_to_edge'] = scaler.fit_transform(
            data[['distance_to_edge']]
        ).flatten()
        
    # Optional log1p scaling: emphasize small epicardial distances
    if log1p:
        data['distance_to_edge'] = np.log1p(data['distance_to_edge'] * 10)
        
    return data

# this is a helper function for "preprocess"
def _annotate_edge(
    data: pd.DataFrame,
    tile_type: Literal['hexagon', 'square'],
    remove_technical_edge: bool,
    plot: bool = False
) -> pd.DataFrame:
    """
    Annotate spatial spot data with neighbor counts, edge flags, and distances to edge.

    For each section, computes the number of direct neighbors per spot based on grid topology,
    flags spots on the technical edge, optionally removes technical-edge spots,
    calculates distance to edge, and can plot distance maps.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by spot ID containing 'section', 'x', 'y' coordinate columns.
    tile_type : {'hexagon', 'square'}
        Grid topology determining neighbor counts (6 for hexagon, 4 for square).
    remove_technical_edge : bool
        If True, excludes spots on technical edges from being labeled as edges.
    plot : bool, default False
        If True, generates a scatter plot of distance-to-edge per section for QC.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with columns:
        - 'n_neighbours': int number of direct neighbors
        - 'is_edge': bool flag for edge spots
        - 'distance_to_edge': float distance metric to nearest edge
    """
    # Validate tile_type
    if tile_type not in ('hexagon', 'square'):
        raise ValueError("Invalid tile_type. Expected 'hexagon' or 'square'.")
    
    # Initialize columns
    data['n_neighbours'] = np.nan
    data['is_edge'] = False
    data['distance_to_edge'] = np.nan
    
    # Process each section separately
    for section, df in data.groupby('section'):
        coords = df[['x', 'y']].values
        
        # Determine neighbor count including self
        n_neighbors = 7 if tile_type == 'hexagon' else 5
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        nbrs.fit(coords)
        distances, _ = nbrs.kneighbors(coords) # first column will be the distance to self (0) and (n+1)th column has the distances to the n-th closest data point
        
        # Reference distance is the smallest non-zero neighbor distance
        ref_distance = np.min(distances[:, 1])
        # Count neighbors within 10% of reference distance (excluding self). To accept minor variation of the distances to direct neighbours.
        neighbour_counts = ((distances[:, 1:] / ref_distance) < 1.1).sum(axis=1)
        
        # Annotate neighbor counts
        data.loc[df.index, 'n_neighbours'] = neighbour_counts
        
        # Flag edge spots based on neighbor threshold
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
        
        # Calculate distance to edge and annotate
        df_loc = _distance_to_edge(df_loc, norm=True, log1p=True)
        data.loc[df_loc.index, 'distance_to_edge'] = df_loc['distance_to_edge']
        
        # Plot for quality control
        if plot:
            sns.scatterplot(
                x='x', y='y', hue='distance_to_edge', data=df_loc,
                palette='rainbow_r', s=10
            )
            plt.legend(
                title='Distance to edge\n(normalised)',
                bbox_to_anchor=(1, 1)
            )
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(section)
            plt.show()

    return data

def preprocess(
    adata: AnnData,
    section_col: str,
    coord_columns: Optional[Tuple[str, str]] = None,
    pseudobulk: bool = False,
    pseudobulk_window_size: Optional[int] = None,
    tile_type: Literal['hexagon', 'square'] = 'square',
    remove_technical_edge: bool = True,
    plot: bool = False
) -> pd.DataFrame:
    """
    Prepare spatial transcriptomics data for analysis.

    Steps:
    1. Optionally pseudobulk by sliding window aggregation.
    2. Select key tissue genes and extract per-spot expression.
    3. Include neighbor-based summary expression.
    4. Annotate each spot/window with edge flags and distances.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData containing gene expression and metadata.
    section_col : str
        Column in adata.obs indicating section/library grouping.
    coord_columns : tuple of str, optional
        Column names for x, y coordinates in adata.obs. If None,
        spatial coordinates are taken from adata.obsm['spatial'].
    pseudobulk : bool, default False
        Whether to aggregate counts into pseudospots via sliding windows.
    pseudobulk_window_size : int, optional
        Window size for pseudobulk aggregation; required if pseudobulk=True.
    tile_type : {'hexagon', 'square'}, default 'square'
        Grid topology for neighbor and edge calculations.
    remove_technical_edge : bool, default True
        Whether to exclude technical-edge spots from edge labeling.
    plot : bool, default False
        If True, plot distance-to-edge maps for quality control.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by spot or window ID, containing:
        - <gene>_own: original expression
        - <gene>_neighbour-max: neighbor max expression
        - section: section identifier
        - x, y: coordinates
        - n_neighbours: number of direct neighbors
        - is_edge: edge flag
        - distance_to_edge: transformed distance metric
    """
    # Step 1: Pseudobulk aggregation if requested
    if pseudobulk:
        if pseudobulk_window_size is None:
            raise ValueError("pseudobulk_window_size must be provided when pseudobulk=True.")
        print("Making pseudobulk per sliding window...")
        bdata = _sliding_window_psudobulk(
            adata=adata,
            section_col=section_col,
            window_size=pseudobulk_window_size,
            coord_columns=coord_columns,
            log_normalise=True
        )
        coord_columns = ('window_col', 'window_row')
    else:
        bdata = adata.copy()
    
    # Step 2: Prepare expression data
    print("Preparing expression data...")
    key_tissue_genes = load_data.key_tissue_genes()
    shared_genes = [g for g in key_tissue_genes if g in bdata.var_names]
    if not shared_genes:
        raise ValueError("No shared genes between AnnData and key_tissue_genes.")
    print(f"{len(shared_genes)} genes will be used.")
    data = bdata[:, shared_genes].to_df()
    data.columns = [f"{g}_own" for g in data.columns]
    
    # Step 3: Assemble DataFrame with metadata
    # Section IDs
    data['section'] = bdata.obs[section_col].values
    # Coordinates
    if coord_columns is None:
        data[['x', 'y']] = bdata.obsm['spatial']
    else:
        data[['x', 'y']] = bdata.obs[list(coord_columns)].values
        
    # Step 4: Include neighbor summaries
    data = _include_neighbours(data, k=6)
    
    # Step 5: Annotate edge and distance metrics
    print("Calculating distance from tissue edge...")
    data = _annotate_edge(
        data,
        tile_type=tile_type,
        remove_technical_edge=remove_technical_edge,
        plot=plot
    )

    return data

def preprocess_builtin_reference(
    gene_panel: Optional[Sequence[str]] = None,
    plot: bool = False
) -> pd.DataFrame:
    """
    Load and preprocess a built-in reference dataset.

    Steps:
    1. Load reference AnnData object from disk.
    2. Optionally subset to genes in `gene_panel`.
    3. Filter and log-normalise expression data.
    4. Run main `preprocess` pipeline on reference.
    5. Annotate with tissue labels and remove unassigned spots.

    Parameters
    ----------
    gene_panel : sequence of str, optional
        List of gene names to subset the reference dataset. If None, uses all genes.
    plot : bool, default False
        If True, generates QC plots during preprocessing.

    Returns
    -------
    pd.DataFrame
        Preprocessed reference DataFrame indexed by spot ID, including gene expression,
        neighbor features, edge annotations, and tissue labels.
    """
    # Step 1: Load reference data
    print("Downloading reference adata to the current directory")
    sc.readwrite._download("https://www.dropbox.com/scl/fi/hdqr79hym4aj1nn4qb2ip/visiumsd_oct_raw.h5ad?rlkey=oqwsy2s8lx7r27cgqzoi8gjo3&st=szmb6xw5&dl=1", Path("visiumsd_oct_raw.h5ad"))
    print("Reading in")
    _adata_ref: AnnData = sc.read_h5ad("visiumsd_oct_raw.h5ad")
    
    # Step 2: Subset to gene panel if provided
    if gene_panel is not None:
        shared_genes = list(set(_adata_ref.var_names).intersection(gene_panel))
        if not shared_genes:
            raise ValueError("No shared genes between reference and gene_panel.")
        _adata_ref = _adata_ref[:, shared_genes]
        
    # Step 3: Filter and log-normalise
    print("Log-normalising reference data...")
    sc.pp.normalize_total(_adata_ref, target_sum=1e4)
    sc.pp.log1p(_adata_ref)
    
    # Step 4: Main preprocessing pipeline
    data = preprocess(
        adata=_adata_ref,
        section_col='sample',
        coord_columns=('array_col', 'array_row'),
        pseudobulk=False,
        pseudobulk_window_size=None,
        tile_type='square',
        remove_technical_edge=True,
        plot=plot
    )
    
    # Step 5: Annotate tissue labels and filter
    data.loc[_adata_ref.obs_names,'tissue'] = _adata_ref.obs['annotation_final_mod'].copy()
    data = data[data['tissue'] != 'unassigned']
    
    return data
