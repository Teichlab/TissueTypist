"""
tissuetypist.features.gene_catalogue
=====================================
Step 1 of the TissueTypist pipeline.

Computes per-modality gene detection rates and derives shared gene pools
across Visium SD (3-prime), Visium SD (FFPE), and Visium HD (FFPE, segmented
cells). All operations are performed on raw counts.

Typical usage
-------------
>>> from tissuetypist.features.gene_catalogue import build_gene_pools, summarise_gene_pools
>>> pools = build_gene_pools(adata_sd3p, adata_sd_ffpe, adata_hd)
>>> summary = summarise_gene_pools(pools)
>>> genes = pools["shared_all"]   # cross-platform deployable feature set
"""

from __future__ import annotations

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODALITY_NAMES = ("SD_3prime", "SD_FFPE", "HD_FFPE")

# Minimum intersection size before we warn the user
MIN_SHARED_GENES_WARN = 500
MIN_SHARED_GENES_HARD = 200   # below this, cross-platform transfer is not recommended


# ── Core detection-rate functions ────────────────────────────────────────────

def compute_detection_rate(
    adata: ad.AnnData,
    min_counts: int = 1,
    layer: Optional[str] = None,
) -> pd.Series:
    """
    Compute the fraction of cells/spots in which each gene is detected.

    Detection is defined as raw count > ``min_counts``. Always run on raw
    counts (before normalisation). If your AnnData has been normalised,
    store raw counts in a layer and pass its name via ``layer``.

    Parameters
    ----------
    adata :
        AnnData for a single modality. Rows = cells/spots, columns = genes.
    min_counts :
        A gene is considered 'detected' in a cell if its count exceeds this
        value. Default 1 (i.e. count >= 1 means detected).
    layer :
        If provided, use ``adata.layers[layer]`` instead of ``adata.X``.
        Use this when ``adata.X`` has already been normalised.

    Returns
    -------
    pd.Series
        Detection rate per gene (values in [0, 1]), indexed by gene name.

    Examples
    --------
    >>> rates = compute_detection_rate(adata_sd3p)
    >>> rates.sort_values(ascending=False).head(10)
    """
    X = adata.layers[layer] if layer else adata.X

    if issparse(X):
        # Efficient sparse path: count non-zeros (or entries > min_counts)
        if min_counts == 0:
            detected = X.astype(bool)
        else:
            detected = X > min_counts
        rates = np.asarray(detected.mean(axis=0)).flatten()
    else:
        rates = (X > min_counts).mean(axis=0).flatten()

    return pd.Series(rates, index=adata.var_names, name="detection_rate")


def get_detected_genes(
    adata: ad.AnnData,
    min_detection_rate: float = 0.05,
    min_counts: int = 1,
    layer: Optional[str] = None,
) -> list[str]:
    """
    Return genes detected in at least ``min_detection_rate`` fraction of cells.

    Parameters
    ----------
    adata :
        AnnData for a single modality.
    min_detection_rate :
        Minimum fraction of cells that must express the gene. Default 0.05
        (5%). Lowering this retains more genes but increases noise.
    min_counts :
        Count threshold to call a gene detected in a single cell.
    layer :
        Layer containing raw counts (see ``compute_detection_rate``).

    Returns
    -------
    list[str]
        Sorted list of gene names passing the detection threshold.
    """
    rates = compute_detection_rate(adata, min_counts=min_counts, layer=layer)
    passing = rates[rates >= min_detection_rate].index.tolist()
    logger.info(
        "Detection filter (rate >= %.2f): %d / %d genes retained",
        min_detection_rate,
        len(passing),
        len(rates),
    )
    return sorted(passing)


# ── Gene pool construction ────────────────────────────────────────────────────

def _pools_from_sets(sets: dict) -> dict[str, list[str]]:
    """Internal helper: build pool dict from per-modality gene sets."""
    shared_all  = sets["SD_3prime"] & sets["SD_FFPE"] & sets["HD_FFPE"]
    shared_sd   = sets["SD_3prime"] & sets["SD_FFPE"]
    all_genes   = sets["SD_3prime"] | sets["SD_FFPE"] | sets["HD_FFPE"]
    shared_2of3 = {
        g for g in all_genes
        if sum(g in s for s in sets.values()) >= 2
    }

    pools = {
        "SD_3prime":   sorted(sets["SD_3prime"]),
        "SD_FFPE":     sorted(sets["SD_FFPE"]),
        "HD_FFPE":     sorted(sets["HD_FFPE"]),
        "shared_all":  sorted(shared_all),
        "shared_SD":   sorted(shared_sd),
        "shared_2of3": sorted(shared_2of3),
    }

    n_shared = len(pools["shared_all"])
    if n_shared < MIN_SHARED_GENES_HARD:
        logger.warning(
            "shared_all has only %d genes (< %d). Cross-platform transfer is "
            "not recommended. Consider using shared_2of3 (%d genes) instead, "
            "or lowering min_detection_rate.",
            n_shared, MIN_SHARED_GENES_HARD, len(pools["shared_2of3"]),
        )
    elif n_shared < MIN_SHARED_GENES_WARN:
        logger.warning(
            "shared_all has %d genes (< recommended %d). "
            "Consider using shared_2of3 (%d genes) or lowering "
            "min_detection_rate.",
            n_shared, MIN_SHARED_GENES_WARN, len(pools["shared_2of3"]),
        )
    else:
        logger.info(
            "shared_all: %d genes (sufficient for cross-platform use)", n_shared
        )

    return pools


def build_gene_pools(
    adata_sd3p: ad.AnnData,
    adata_sd_ffpe: ad.AnnData,
    adata_hd: ad.AnnData,
    min_detection_rate: float = 0.05,
    min_counts: int = 1,
    layer: Optional[str] = None,
) -> dict[str, list[str]]:
    """
    Compute per-modality gene detection pools and their intersections.

    All three AnnData objects must use gene symbols as ``var_names`` and
    contain raw counts in ``X`` (or in ``layer`` if specified).

    Parameters
    ----------
    adata_sd3p :
        Visium SD 3-prime AnnData (raw counts).
    adata_sd_ffpe :
        Visium SD FFPE AnnData (raw counts).
    adata_hd :
        Visium HD FFPE AnnData with segmented cells (raw counts).
    min_detection_rate :
        Minimum fraction of cells/spots that must express a gene for it to be
        included in that modality's pool. Default 0.05 (5%).
    min_counts :
        Count threshold to call a gene 'detected' in a single cell/spot.
    layer :
        Layer name containing raw counts, if ``X`` is not raw.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with the following keys:

        ``"SD_3prime"``
            Genes detected in Visium SD 3-prime.
        ``"SD_FFPE"``
            Genes detected in Visium SD FFPE.
        ``"HD_FFPE"``
            Genes detected in Visium HD (segmented cells).
        ``"shared_all"``
            Intersection of all three — the primary cross-platform deployable
            feature set.
        ``"shared_SD"``
            Intersection of SD-3prime and SD-FFPE only.
        ``"shared_2of3"``
            Union of genes present in at least 2 of 3 modalities — a relaxed
            alternative if ``shared_all`` is too small.

    Notes
    -----
    A warning is raised if ``shared_all`` contains fewer than
    ``MIN_SHARED_GENES_WARN`` (500) genes. If it contains fewer than
    ``MIN_SHARED_GENES_HARD`` (200), cross-platform transfer is not
    recommended and the function will suggest using ``shared_2of3`` instead.

    Examples
    --------
    >>> pools = build_gene_pools(adata_sd3p, adata_sd_ffpe, adata_hd)
    >>> print(len(pools["shared_all"]), "shared genes")
    """
    kwargs = dict(min_detection_rate=min_detection_rate,
                  min_counts=min_counts,
                  layer=layer)

    logger.info("Computing gene detection rates per modality...")
    sets = {
        "SD_3prime": set(get_detected_genes(adata_sd3p, **kwargs)),
        "SD_FFPE":   set(get_detected_genes(adata_sd_ffpe, **kwargs)),
        "HD_FFPE":   set(get_detected_genes(adata_hd, **kwargs)),
    }

    return _pools_from_sets(sets)


def build_gene_pools_from_paths(
    path_sd3p: str,
    path_sd_ffpe: str,
    path_hd: str,
    min_detection_rate: float = 0.05,
    min_counts: int = 1,
) -> dict[str, list[str]]:
    """
    Memory-efficient variant of ``build_gene_pools`` for machines with
    limited RAM (e.g. 16 GB).

    Loads each AnnData one at a time, computes detection rates, then
    immediately frees the matrix from memory before loading the next.
    Peak RAM usage is roughly the size of the largest single file rather
    than the sum of all three.

    Parameters
    ----------
    path_sd3p :
        Path to Visium SD 3-prime ``.h5ad`` file (raw counts).
    path_sd_ffpe :
        Path to Visium SD FFPE ``.h5ad`` file (raw counts).
    path_hd :
        Path to Visium HD FFPE ``.h5ad`` file (raw counts).
    min_detection_rate :
        Minimum fraction of cells/spots expressing a gene. Default 0.05.
    min_counts :
        Count threshold per cell. Default 1.

    Returns
    -------
    dict[str, list[str]]
        Same structure as ``build_gene_pools``.

    Examples
    --------
    >>> pools = build_gene_pools_from_paths(
    ...     "data/adata_sd_3p_raw.h5ad",
    ...     "data/adata_sd_ffpe_raw.h5ad",
    ...     "data/adata_hd_raw.h5ad",
    ... )
    """
    import gc

    kwargs = dict(min_detection_rate=min_detection_rate,
                  min_counts=min_counts)
    sets = {}

    for name, path in [
        ("SD_3prime", path_sd3p),
        ("SD_FFPE",   path_sd_ffpe),
        ("HD_FFPE",   path_hd),
    ]:
        logger.info("Loading %s from %s ...", name, path)
        adata = sc.read_h5ad(path)
        logger.info("  shape: %s", adata.shape)

        # Detection rate on raw X (files are raw counts)
        sets[name] = set(get_detected_genes(adata, **kwargs))

        # Free the full matrix immediately — only the gene name list is kept
        del adata
        gc.collect()
        logger.info("  %d detected genes retained; AnnData freed.", len(sets[name]))

    return _pools_from_sets(sets)


def compute_pseudobulk_detection_rate(
    adata: ad.AnnData,
    groupby: list[str],
    min_counts: int = 1,
    layer: Optional[str] = None,
    min_cells_per_group: int = 10,
) -> pd.Series:
    """
    Compute gene detection rate at the pseudobulk level.

    Rather than asking "in what fraction of individual cells is this gene
    expressed?" (which is very stringent for single-cell HD data), this asks
    "in what fraction of donor-niche-section groups is this gene expressed?"
    — a fairer comparison across modalities with different resolutions.

    A gene is considered 'expressed' in a pseudobulk group if its total
    summed count across cells in that group exceeds ``min_counts``.

    Parameters
    ----------
    adata :
        AnnData with raw counts. Rows = cells/spots, columns = genes.
    groupby :
        List of obs column names defining the pseudobulk grouping.
        e.g. ``["donor", "section_ID", "niche_coarse_Mar2026"]``
    min_counts :
        A gene is considered expressed in a group if its summed count
        across all cells in the group exceeds this value. Default 1.
    layer :
        Layer containing raw counts. If None, uses ``adata.X``.
    min_cells_per_group :
        Groups with fewer than this many cells are excluded. Prevents
        tiny spurious groups from inflating detection rates. Default 10.

    Returns
    -------
    pd.Series
        Detection rate per gene (values in [0, 1]), indexed by gene name.
        Values represent the fraction of pseudobulk groups in which the
        gene is expressed above ``min_counts``.

    Examples
    --------
    >>> rates = compute_pseudobulk_detection_rate(
    ...     adata_hd,
    ...     groupby=["donor", "section_ID", "niche_coarse_Mar2026"],
    ...     layer="raw_counts",
    ... )
    """
    import gc
    from scipy.sparse import issparse

    X = adata.layers[layer] if layer else adata.X

    # Build group labels
    group_keys = adata.obs[groupby].astype(str).agg("__".join, axis=1)
    unique_groups = group_keys.unique()

    # Filter out small groups
    group_sizes = group_keys.value_counts()
    valid_groups = group_sizes[group_sizes >= min_cells_per_group].index
    n_excluded = len(unique_groups) - len(valid_groups)
    if n_excluded > 0:
        logger.info(
            "Pseudobulk: excluded %d groups with < %d cells "
            "(%d groups remaining)",
            n_excluded, min_cells_per_group, len(valid_groups),
        )

    n_groups = len(valid_groups)
    n_genes  = adata.n_vars
    detected_count = np.zeros(n_genes, dtype=np.int32)

    for group in valid_groups:
        mask = (group_keys == group).values
        X_group = X[mask, :]
        if issparse(X_group):
            # Sum across cells in this group — result is (1, n_genes)
            group_sum = np.asarray(X_group.sum(axis=0)).flatten()
        else:
            group_sum = X_group.sum(axis=0).flatten()
        detected_count += (group_sum > min_counts).astype(np.int32)

    rates = detected_count / n_groups
    logger.info(
        "Pseudobulk detection: %d valid groups, %d genes evaluated",
        n_groups, n_genes,
    )
    return pd.Series(rates, index=adata.var_names, name="detection_rate_pseudobulk")


def get_detected_genes_pseudobulk(
    adata: ad.AnnData,
    groupby: list[str],
    min_detection_rate: float = 0.05,
    min_counts: int = 1,
    layer: Optional[str] = None,
    min_cells_per_group: int = 10,
) -> list[str]:
    """
    Return genes detected in >= ``min_detection_rate`` fraction of
    pseudobulk groups.

    Parameters
    ----------
    adata :
        AnnData with raw counts.
    groupby :
        obs columns defining pseudobulk groups.
        SD: ``["donor", "sample_for_cell2loc", "niche_coarse_Mar2026"]``
        HD: ``["donor", "section_ID", "niche_coarse_Mar2026"]``
    min_detection_rate :
        Minimum fraction of groups expressing the gene. Default 0.05.
    min_counts :
        Summed count threshold per group. Default 1.
    layer :
        Layer with raw counts.
    min_cells_per_group :
        Minimum cells per pseudobulk group to be included.

    Returns
    -------
    list[str]
        Sorted list of gene names passing the detection threshold.
    """
    rates = compute_pseudobulk_detection_rate(
        adata,
        groupby=groupby,
        min_counts=min_counts,
        layer=layer,
        min_cells_per_group=min_cells_per_group,
    )
    passing = rates[rates >= min_detection_rate].index.tolist()
    logger.info(
        "Pseudobulk detection filter (rate >= %.2f): %d / %d genes retained",
        min_detection_rate, len(passing), len(rates),
    )
    return sorted(passing)


def build_gene_pools_pseudobulk_from_paths(
    path_sd3p: str,
    path_sd_ffpe: str,
    path_hd: str,
    groupby_sd: Optional[list[str]] = None,
    groupby_hd: Optional[list[str]] = None,
    min_detection_rate: float = 0.05,
    min_counts: int = 1,
    min_cells_per_group: int = 10,
) -> dict[str, list[str]]:
    """
    Memory-efficient gene pool construction using pseudobulk detection rates.

    This is the recommended function for building gene pools when combining
    Visium SD (spot-level) and Visium HD (single-cell-level) data.
    Pseudobulk aggregation by donor x section x niche makes detection
    thresholds comparable across modalities regardless of resolution.

    Parameters
    ----------
    path_sd3p :
        Path to Visium SD 3-prime ``.h5ad`` (raw counts).
    path_sd_ffpe :
        Path to Visium SD FFPE ``.h5ad`` (raw counts).
    path_hd :
        Path to Visium HD FFPE ``.h5ad`` (raw counts, segmented cells).
    groupby_sd :
        obs columns for pseudobulk grouping in SD data.
        Default: ``["donor", "sample_for_cell2loc", "niche_coarse_Mar2026"]``
    groupby_hd :
        obs columns for pseudobulk grouping in HD data.
        Default: ``["donor", "section_ID", "niche_coarse_Mar2026"]``
    min_detection_rate :
        Minimum fraction of pseudobulk groups expressing a gene. Default 0.05.
    min_counts :
        Summed count threshold per pseudobulk group. Default 1.
    min_cells_per_group :
        Minimum cells per group; smaller groups excluded. Default 10.

    Returns
    -------
    dict[str, list[str]]
        Same structure as ``build_gene_pools``:
        ``shared_all``, ``shared_2of3``, ``shared_SD``,
        ``SD_3prime``, ``SD_FFPE``, ``HD_FFPE``.

    Examples
    --------
    >>> pools = build_gene_pools_pseudobulk_from_paths(
    ...     "data/adata_sd_3p_raw.h5ad",
    ...     "data/adata_sd_ffpe_raw.h5ad",
    ...     "data/adata_hd_raw.h5ad",
    ... )
    >>> print(len(pools["shared_all"]), "shared genes (pseudobulk)")
    """
    import gc

    # Defaults match the April 2026 cardiac data convention where all
    # modalities share `donor`, `section_ID`, `niche_coarse_Apr2026`.
    # Users with different column naming should pass explicit `groupby_sd`
    # and `groupby_hd` lists.
    if groupby_sd is None:
        groupby_sd = ["donor", "section_ID", "niche_coarse_Apr2026"]
    if groupby_hd is None:
        groupby_hd = ["donor", "section_ID", "niche_coarse_Apr2026"]

    sets = {}
    configs = [
        ("SD_3prime", path_sd3p,   groupby_sd),
        ("SD_FFPE",   path_sd_ffpe, groupby_sd),
        ("HD_FFPE",   path_hd,      groupby_hd),
    ]

    for name, path, groupby in configs:
        logger.info("Loading %s ...", name)
        adata = sc.read_h5ad(path)
        logger.info("  shape: %s | groupby: %s", adata.shape, groupby)

        # Report how many pseudobulk groups will be formed
        group_keys = adata.obs[groupby].astype(str).agg("__".join, axis=1)
        n_groups = group_keys.nunique()
        logger.info("  %d pseudobulk groups identified", n_groups)

        sets[name] = set(get_detected_genes_pseudobulk(
            adata,
            groupby=groupby,
            min_detection_rate=min_detection_rate,
            min_counts=min_counts,
            min_cells_per_group=min_cells_per_group,
        ))

        del adata
        gc.collect()
        logger.info("  %d genes retained; AnnData freed.", len(sets[name]))

    return _pools_from_sets(sets)


# ── Detection rate tables ─────────────────────────────────────────────────────

def compute_detection_table(
    adata_sd3p: ad.AnnData,
    adata_sd_ffpe: ad.AnnData,
    adata_hd: ad.AnnData,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a per-gene detection rate table across all three modalities.

    Useful for QC plots and deciding on detection thresholds.

    Parameters
    ----------
    adata_sd3p, adata_sd_ffpe, adata_hd :
        AnnData objects (raw counts).
    layer :
        Layer containing raw counts if X is not raw.

    Returns
    -------
    pd.DataFrame
        Rows = genes present in the union of all three modalities.
        Columns = ``["SD_3prime", "SD_FFPE", "HD_FFPE"]``.
        Missing genes (not in a modality's var_names) are filled with 0.

    Examples
    --------
    >>> table = compute_detection_table(adata_sd3p, adata_sd_ffpe, adata_hd)
    >>> table[table["shared_all"]].describe()
    """
    adatas = {
        "SD_3prime": adata_sd3p,
        "SD_FFPE":   adata_sd_ffpe,
        "HD_FFPE":   adata_hd,
    }
    rates = {
        name: compute_detection_rate(adata, layer=layer)
        for name, adata in adatas.items()
    }
    df = pd.DataFrame(rates).fillna(0.0)

    # Add a convenience column: number of modalities in which the gene is
    # detected above the default threshold (5%)
    df["n_modalities_5pct"] = (df >= 0.05).sum(axis=1)
    df["n_modalities_1pct"] = (df >= 0.01).sum(axis=1)

    return df.sort_values("n_modalities_5pct", ascending=False)


# ── Normalisation helpers ─────────────────────────────────────────────────────

def normalise_adata(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    store_raw_layer: str = "raw_counts",
    inplace: bool = True,
) -> ad.AnnData:
    """
    Apply library-size normalisation and log1p to an AnnData with raw counts.

    Stores the original raw counts in ``adata.layers[store_raw_layer]`` before
    overwriting ``adata.X``. Detection rate calculations should always use
    this layer (or the original raw ``X``), not the normalised values.

    Parameters
    ----------
    adata :
        AnnData with raw counts in ``X``.
    target_sum :
        Target total count per cell after normalisation. Default 1e4.
    store_raw_layer :
        Layer key to store raw counts. Default ``"raw_counts"``.
    inplace :
        If False, operate on a copy. Default True.

    Returns
    -------
    ad.AnnData
        AnnData with normalised log1p counts in ``X`` and raw counts in
        ``layers[store_raw_layer]``.

    Examples
    --------
    >>> adata_sd3p = normalise_adata(adata_sd3p)
    >>> # Detection rates should use the raw layer
    >>> rates = compute_detection_rate(adata_sd3p, layer="raw_counts")
    """
    if not inplace:
        adata = adata.copy()

    if store_raw_layer in adata.layers:
        logger.warning(
            "Layer '%s' already exists — skipping normalisation to avoid "
            "overwriting.", store_raw_layer
        )
        return adata

    logger.info("Storing raw counts in layer '%s'", store_raw_layer)
    adata.layers[store_raw_layer] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    logger.info(
        "Normalisation complete (target_sum=%.0f, log1p applied).", target_sum
    )
    return adata


# ── Platform intersection (for Xenium / MERFISH deployment) ──────────────────

def get_platform_intersection(
    genes_shared: list[str],
    platform_genes: list[str],
    platform_name: str = "external",
    min_genes: int = MIN_SHARED_GENES_HARD,
) -> dict:
    """
    Compute the intersection of the shared gene set with an external platform's
    gene panel. Used for Xenium / MERFISH deployment.

    Parameters
    ----------
    genes_shared :
        List of genes from ``pools["shared_all"]`` (or another pool).
    platform_genes :
        List of genes in the external platform's panel or detected gene set.
    platform_name :
        Label for logging/reporting.
    min_genes :
        Minimum intersection size for zero-shot transfer.

    Returns
    -------
    dict
        ``intersection`` : sorted list of overlapping genes.
        ``n_intersect``  : number of overlapping genes.
        ``n_shared``     : size of ``genes_shared``.
        ``n_platform``   : size of ``platform_genes``.
        ``coverage_pct`` : % of ``genes_shared`` covered by the platform.
        ``sufficient``   : whether zero-shot transfer is recommended.
        ``recommendation``: human-readable string.

    Examples
    --------
    >>> xenium_genes = list(adata_xenium.var_names)
    >>> result = get_platform_intersection(pools["shared_all"], xenium_genes,
    ...                                    platform_name="Xenium")
    >>> print(result["recommendation"])
    """
    shared_set   = set(genes_shared)
    platform_set = set(platform_genes)
    intersection = sorted(shared_set & platform_set)
    n = len(intersection)

    if n >= min_genes:
        rec = (
            f"Zero-shot transfer recommended. {n} / {len(shared_set)} "
            f"shared genes present in {platform_name} panel "
            f"({100 * n / len(shared_set):.1f}% coverage)."
        )
    else:
        rec = (
            f"Only {n} shared genes found in {platform_name} panel "
            f"(< threshold of {min_genes}). "
            f"Retraining on the intersection is recommended."
        )

    logger.info("[%s] %s", platform_name, rec)

    return {
        "intersection":   intersection,
        "n_intersect":    n,
        "n_shared":       len(shared_set),
        "n_platform":     len(platform_set),
        "coverage_pct":   round(100 * n / len(shared_set), 2),
        "sufficient":     n >= min_genes,
        "recommendation": rec,
    }


# ── Summary / reporting ───────────────────────────────────────────────────────

def summarise_gene_pools(
    pools: dict[str, list[str]],
    print_summary: bool = True,
) -> pd.DataFrame:
    """
    Return a summary DataFrame of gene pool sizes.

    Parameters
    ----------
    pools :
        Output of ``build_gene_pools``.
    print_summary :
        If True, print the table to stdout. Default True.

    Returns
    -------
    pd.DataFrame
        Columns: ``pool``, ``n_genes``, ``notes``.
    """
    notes = {
        "SD_3prime":   "Detected in Visium SD 3-prime",
        "SD_FFPE":     "Detected in Visium SD FFPE",
        "HD_FFPE":     "Detected in Visium HD (segmented cells)",
        "shared_all":  "Intersection of all 3 modalities — primary feature set",
        "shared_SD":   "Intersection of SD-3prime and SD-FFPE",
        "shared_2of3": "Detected in >= 2 of 3 modalities (relaxed)",
    }
    rows = [
        {"pool": k, "n_genes": len(v), "notes": notes.get(k, "")}
        for k, v in pools.items()
    ]
    df = pd.DataFrame(rows).set_index("pool")

    if print_summary:
        print("\n=== Gene Pool Summary ===")
        print(df.to_string())
        n_shared = len(pools.get("shared_all", []))
        if n_shared < MIN_SHARED_GENES_HARD:
            print(f"\n[WARNING] shared_all ({n_shared} genes) is below the "
                  f"recommended minimum of {MIN_SHARED_GENES_HARD}.")
            print("  → Consider using shared_2of3 or lowering "
                  "min_detection_rate in build_gene_pools().")
        print()

    return df


def save_gene_pools(
    pools: dict[str, list[str]],
    output_path: str,
) -> None:
    """
    Save gene pools to a CSV file for reproducibility.

    Each pool is saved as a column; shorter pools are padded with empty
    strings. The file can be loaded back with ``load_gene_pools``.

    Parameters
    ----------
    pools :
        Output of ``build_gene_pools``.
    output_path :
        Path to output CSV file (e.g. ``"results/gene_pools.csv"``).

    Examples
    --------
    >>> save_gene_pools(pools, "results/phase0_gene_pools.csv")
    """
    max_len = max(len(v) for v in pools.values())
    padded  = {k: v + [""] * (max_len - len(v)) for k, v in pools.items()}
    df = pd.DataFrame(padded)
    df.to_csv(output_path, index=False)
    logger.info("Gene pools saved to '%s'", output_path)


def load_gene_pools(input_path: str) -> dict[str, list[str]]:
    """
    Load gene pools saved by ``save_gene_pools``.

    Parameters
    ----------
    input_path :
        Path to CSV file produced by ``save_gene_pools``.

    Returns
    -------
    dict[str, list[str]]
        Same structure as output of ``build_gene_pools``.
    """
    df = pd.read_csv(input_path, dtype=str).fillna("")
    return {col: [g for g in df[col] if g != ""] for col in df.columns}
