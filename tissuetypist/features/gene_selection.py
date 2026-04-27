"""
tissuetypist.features.gene_selection
=====================================
Feature-gene selection (DEG + HVG) for TissueTypist.

Key design principles
---------------------
- **Union strategy**: DEGs are computed per niche using only the modalities
  that contain that niche, then the union is taken across all niches.
  This preserves signal from HD-exclusive and SD-exclusive niches.
- **Restricted to shared_all**: the final DEG set is intersected with
  ``shared_all`` to ensure deployability across platforms.
- **Modality-aware routing**: a niche modality map (derived from the data)
  determines which modalities contribute to each niche's DEG computation.

Typical usage
-------------
>>> from tissuetypist.features.gene_selection import compute_deg_features
>>> deg_genes = compute_deg_features(
...     adata_combined,
...     niche_col="niche_coarse_Mar2026",
...     genes_shared=pools["shared_all"],
...     niche_modality_map=niche_modality_map,
... )
"""

from __future__ import annotations

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


# ── Niche modality map ────────────────────────────────────────────────────────

def build_niche_modality_map(
    adata_sd3p: ad.AnnData,
    adata_sd_ffpe: ad.AnnData,
    adata_hd: ad.AnnData,
    niche_col: str,
) -> dict[str, list[str]]:
    """
    Derive which modalities contain each niche label, directly from the data.

    Rather than hardcoding modality coverage, this function inspects the
    obs annotations of each AnnData to build the map automatically. This
    handles edge cases (HD-exclusive niches, SD-only niches) without any
    manual curation.

    Parameters
    ----------
    adata_sd3p, adata_sd_ffpe, adata_hd :
        AnnData objects for each modality. Only ``obs[niche_col]`` is used.
    niche_col :
        obs column containing niche labels (e.g. ``"niche_coarse_Mar2026"``
        or ``"niche_fine_Mar2026"``).

    Returns
    -------
    dict[str, list[str]]
        Keys = niche labels. Values = sorted list of modality names that
        contain that niche.

    Examples
    --------
    >>> niche_map = build_niche_modality_map(
    ...     adata_sd3p, adata_sd_ffpe, adata_hd,
    ...     niche_col="niche_coarse_Mar2026"
    ... )
    >>> niche_map["Lymph node"]
    ['SD_FFPE']
    >>> niche_map["Ventricle"]
    ['HD_FFPE', 'SD_3prime', 'SD_FFPE']
    """
    modalities = {
        "SD_3prime": adata_sd3p,
        "SD_FFPE":   adata_sd_ffpe,
        "HD_FFPE":   adata_hd,
    }

    all_niches: set[str] = set()
    for adata in modalities.values():
        if niche_col not in adata.obs.columns:
            raise ValueError(
                f"Column '{niche_col}' not found in obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        all_niches |= set(adata.obs[niche_col].dropna().unique())

    niche_map: dict[str, list[str]] = {}
    for niche in sorted(all_niches):
        present_in = [
            name for name, adata in modalities.items()
            if niche in adata.obs[niche_col].values
        ]
        niche_map[niche] = sorted(present_in)

    # Log summary
    n_all3    = sum(1 for v in niche_map.values() if len(v) == 3)
    n_two     = sum(1 for v in niche_map.values() if len(v) == 2)
    n_one     = sum(1 for v in niche_map.values() if len(v) == 1)
    logger.info(
        "Niche modality map (%s): %d niches total — "
        "%d in all 3, %d in 2, %d in 1 modality",
        niche_col, len(niche_map), n_all3, n_two, n_one,
    )
    for niche, mods in niche_map.items():
        if len(mods) == 1:
            logger.info("  Single-modality niche: '%s' → %s", niche, mods)

    return niche_map


def summarise_niche_modality_map(
    niche_map: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Return a summary DataFrame of the niche modality map.

    Parameters
    ----------
    niche_map :
        Output of ``build_niche_modality_map``.

    Returns
    -------
    pd.DataFrame
        Columns: niche, n_modalities, modalities, SD_3prime, SD_FFPE, HD_FFPE.
    """
    rows = []
    for niche, mods in sorted(niche_map.items()):
        rows.append({
            "niche":        niche,
            "n_modalities": len(mods),
            "modalities":   ", ".join(mods),
            "SD_3prime":    "SD_3prime" in mods,
            "SD_FFPE":      "SD_FFPE"   in mods,
            "HD_FFPE":      "HD_FFPE"   in mods,
        })
    return pd.DataFrame(rows).set_index("niche")


# ── Combined AnnData construction ─────────────────────────────────────────────

def build_combined_adata(
    adata_sd3p: ad.AnnData,
    adata_sd_ffpe: ad.AnnData,
    adata_hd: ad.AnnData,
    genes: list[str],
    niche_col: str,
    modality_col: str = "modality",
) -> ad.AnnData:
    """
    Concatenate the three normalised AnnData objects, restricted to a gene set.

    Parameters
    ----------
    adata_sd3p, adata_sd_ffpe, adata_hd :
        Normalised (log1p) AnnData objects. Must contain ``niche_col`` in obs.
    genes :
        Gene list to subset to (e.g. ``pools["shared_all"]``).
    niche_col :
        obs column with niche labels.
    modality_col :
        Name of the new obs column recording which modality each
        observation came from. Default ``"modality"``.

    Returns
    -------
    ad.AnnData
        Concatenated AnnData with ``modality_col`` added to obs.
    """
    # Tag each modality
    adata_sd3p   = adata_sd3p.copy()
    adata_sd_ffpe = adata_sd_ffpe.copy()
    adata_hd     = adata_hd.copy()

    adata_sd3p.obs[modality_col]   = "SD_3prime"
    adata_sd_ffpe.obs[modality_col] = "SD_FFPE"
    adata_hd.obs[modality_col]     = "HD_FFPE"

    # Subset to shared genes
    for name, adata in [("SD_3prime", adata_sd3p),
                         ("SD_FFPE",   adata_sd_ffpe),
                         ("HD_FFPE",   adata_hd)]:
        missing = set(genes) - set(adata.var_names)
        if missing:
            logger.warning(
                "%s: %d requested genes not in var_names — will be excluded",
                name, len(missing),
            )

    # Only keep genes present in all three
    common_genes = sorted(
        set(genes)
        & set(adata_sd3p.var_names)
        & set(adata_sd_ffpe.var_names)
        & set(adata_hd.var_names)
    )
    logger.info(
        "Building combined AnnData: %d genes, modalities: SD-3prime %d | "
        "SD-FFPE %d | HD %d observations",
        len(common_genes),
        adata_sd3p.n_obs,
        adata_sd_ffpe.n_obs,
        adata_hd.n_obs,
    )

    combined = ad.concat(
        [
            adata_sd3p[:, common_genes],
            adata_sd_ffpe[:, common_genes],
            adata_hd[:, common_genes],
        ],
        join="inner",
        label=modality_col,
        keys=["SD_3prime", "SD_FFPE", "HD_FFPE"],
        merge="first",
    )
    # concat adds modality as index prefix — restore clean obs column
    combined.obs[modality_col] = combined.obs[modality_col].astype(str)

    logger.info("Combined AnnData shape: %s", combined.shape)
    return combined


# ── Sliding-window pseudobulk functions live in features.spatial ────────────
# Import here for backward compatibility (compute_degs_per_niche uses it).
from tissuetypist.features.spatial import (  # noqa: E402
    sliding_window_pseudobulk_hd,
    sliding_window_pseudobulk_cells,
    _calculate_window_corners_v2 as _calculate_window_corners,
    _validate_raw_counts,
)





# (Old definitions of _calculate_window_corners, _validate_raw_counts,
#  sliding_window_pseudobulk_hd, sliding_window_pseudobulk_cells removed.
#  Now imported from v1_spatial.py above.)


# ── DEG computation ───────────────────────────────────────────────────────────

def compute_degs_per_niche(
    adata_sd3p: ad.AnnData,
    adata_sd_ffpe: ad.AnnData,
    adata_hd: ad.AnnData,
    niche_col: str,
    genes_shared: list[str],
    niche_modality_map: dict[str, list[str]],
    method: str = "t-test_overestim_var",
    min_cells_per_niche: int = 20,   # kept for API compat, unused
    use_pseudobulk: bool = True,     # kept for API compat, unused
    microns_per_pixel_map: dict[str, float] = None,
    target_spot_um: float = 55.0,
    use_pseudobulk_hd: bool = True,
    min_groups: int = 3,
    n_top_markers: Optional[int] = None,
    deduplicate_shared: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Compute DEGs for each niche using a per-modality union strategy.

    Strategy
    --------
    - **SD modalities** (SD_3prime, SD_FFPE): spots are used directly —
      they are already ~55 µm, so no pseudobulking is needed.
    - **HD modality** (HD_FFPE): cells are aggregated into sliding-window
      pseudo-spots via :func:`sliding_window_pseudobulk_hd` before DEG
      testing. Window size is derived from ``microns_per_pixel_map`` so
      that each window approximates a standard Visium spot.
    - DEGs are computed independently per modality (one-vs-rest within
      that modality's data), ranking **all** shared genes (no top-N cap).
      Test statistics (scores) are not comparable across modalities with
      different sample sizes, so no cross-modality deduplication by score
      is performed. Instead, per-niche results are a simple set union of
      gene names across modalities, keeping all per-modality statistics
      for downstream filtering (p-adj, logFC, mean expression).
    - Mean expression of each gene in the target niche is computed and
      stored in a ``mean_expr`` column, enabling downstream filtering to
      remove lowly-expressed noise genes.
    - When ``deduplicate_shared=True``, genes appearing in multiple niches'
      DEG lists are resolved by assigning each gene exclusively to the niche
      where it has the highest score. This removes cross-niche noise and
      ensures each gene acts as a marker for exactly one niche. Recommended
      for small panels where the same gene frequently appears in multiple
      niches' top-N lists.

    Parameters
    ----------
    adata_sd3p, adata_sd_ffpe, adata_hd :
        Normalised (log1p) AnnData objects. HD must also have
        ``layers["raw_counts"]`` and ``obsm["spatial"]``.
    niche_col :
        obs column with niche labels.
    genes_shared :
        Gene pool to restrict all DEG computation to (``shared_all``).
    niche_modality_map :
        Output of :func:`build_niche_modality_map`.
    method :
        DEG test method for ``sc.tl.rank_genes_groups``.
        Default ``"t-test_overestim_var"``.
    min_cells_per_niche, use_pseudobulk :
        Kept for API compatibility; not used.
    microns_per_pixel_map :
        Dict mapping library ID → µm/px. Required for HD pseudobulk.
    target_spot_um :
        Target pseudo-spot size in µm. Default 55.0.
    use_pseudobulk_hd :
        If True (default), apply sliding-window pseudobulk to HD data.
        If False, use HD cells directly (slow, not recommended).
    min_groups :
        Minimum number of target observations (spots or windows) a niche
        must have in a modality for DEG testing to proceed. Default 3.
    n_top_markers : int or None
        When set, cap each modality's DEG list at this many genes (by score)
        **before** the cross-modality union. Ensures each modality contributes
        its own best markers independently of score differences between
        modalities. Default None — uses all genes passing significance filters.
        Recommended for small panels: 5, 10, 20.
    deduplicate_shared : bool
        When True, after all per-niche DEG lists are computed, remove any gene
        appearing in ≥2 niches' DEG lists entirely (Option A — strict).
        Only genes unique to a single niche are retained. No gene can serve
        as a marker for more than one niche. Default False.
        Recommended with ``n_top_markers`` for small panels where shared genes
        add noise to the LR decision boundary.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys = niche names. Values = DataFrame with columns
        ``["gene", "scores", "pvals", "pvals_adj", "logfoldchanges",
        "mean_expr", "niche", "modality"]``.
        When a gene appears in multiple modalities for the same niche,
        all rows are retained (one per modality).
    """
    shared_set = set(genes_shared)

    modality_adatas = {
        "SD_3prime": adata_sd3p,
        "SD_FFPE":   adata_sd_ffpe,
        "HD_FFPE":   adata_hd,
    }

    # ── Pre-compute HD sliding-window pseudobulk (once, before niche loop)
    hd_windows: ad.AnnData | None = None
    if "HD_FFPE" in {m for mods in niche_modality_map.values() for m in mods}:
        if use_pseudobulk_hd and microns_per_pixel_map:
            logger.info(
                "Pre-computing HD sliding-window pseudobulk "
                "(target_spot_um=%.1f µm)...", target_spot_um,
            )
            hd_windows = sliding_window_pseudobulk_hd(
                adata_hd,
                section_col="section_ID",
                niche_col=niche_col,
                library_col="library",
                microns_per_pixel_map=microns_per_pixel_map,
                target_spot_um=target_spot_um,
            )
        else:
            logger.info(
                "HD pseudobulk disabled or no microns_per_pixel_map — "
                "using HD cells directly."
            )

    results: dict[str, pd.DataFrame] = {}

    for niche, modalities in niche_modality_map.items():
        logger.info("Computing DEGs for niche '%s' | modalities: %s", niche, modalities)

        mod_dfs: list[pd.DataFrame] = []

        for mod in modalities:
            # Select the AnnData to use for this modality
            if mod == "HD_FFPE":
                adata_mod = hd_windows if hd_windows is not None else modality_adatas["HD_FFPE"]
            else:
                adata_mod = modality_adatas[mod]

            # Restrict to shared genes present in this modality
            genes_here = [g for g in genes_shared if g in adata_mod.var_names]
            sub = adata_mod[:, genes_here].copy()

            # Binary label: target niche vs rest
            sub.obs["_niche_binary"] = np.where(
                sub.obs[niche_col] == niche, niche, "rest"
            )

            n_target = int((sub.obs["_niche_binary"] == niche).sum())
            n_rest   = int((sub.obs["_niche_binary"] == "rest").sum())

            if n_target < min_groups:
                logger.warning(
                    "  %s / %s: only %d target obs (min_groups=%d) — skipping modality",
                    niche, mod, n_target, min_groups,
                )
                continue
            if n_rest < min_groups:
                logger.warning(
                    "  %s / %s: only %d rest obs — skipping modality",
                    niche, mod, n_rest,
                )
                continue

            logger.info(
                "  %s / %s: %d target vs %d rest (%d genes)",
                niche, mod, n_target, n_rest, sub.n_vars,
            )

            # Rank ALL shared genes (no top-N cap) so downstream filtering
            # by p-adj, logFC, and mean expression is the sole selection.
            n_rank = sub.n_vars
            try:
                sc.tl.rank_genes_groups(
                    sub,
                    groupby="_niche_binary",
                    groups=[niche],
                    reference="rest",
                    method=method,
                    n_genes=n_rank,
                    use_raw=False,
                )
            except Exception as e:
                logger.warning(
                    "  DEG computation failed for '%s' in %s: %s", niche, mod, e
                )
                continue

            deg_df = sc.get.rank_genes_groups_df(
                sub, group=niche, key="rank_genes_groups",
            )
            deg_df = deg_df.rename(columns={"names": "gene"})
            deg_df["niche"]    = niche
            deg_df["modality"] = mod

            # Compute mean expression in the TARGET niche (log1p space)
            # for downstream noise filtering.
            target_mask = sub.obs["_niche_binary"] == niche
            target_X = sub[target_mask, :].X
            if hasattr(target_X, "toarray"):
                target_X = target_X.toarray()
            gene_means = np.asarray(target_X, dtype=np.float64).mean(axis=0)
            mean_expr_map = dict(zip(sub.var_names, gene_means))
            deg_df["mean_expr"] = deg_df["gene"].map(mean_expr_map)

            # Optional cap per modality (for small-panel use cases only).
            if n_top_markers is not None:
                deg_df = deg_df.head(n_top_markers)

            logger.info(
                "  %s / %s: %d genes ranked", niche, mod, len(deg_df)
            )
            mod_dfs.append(deg_df)

        if not mod_dfs:
            logger.warning(
                "Niche '%s': no modality produced DEGs — skipping.", niche
            )
            continue

        # ── Union strategy: pool all modality DEG DataFrames
        # Restrict to shared_all genes. Keep all per-modality rows — no
        # dedup by score (scores are not comparable across modalities).
        # Downstream filtering (p-adj, logFC, mean_expr) is the sole
        # selection mechanism.
        combined = pd.concat(mod_dfs, ignore_index=True)
        combined = combined[combined["gene"].isin(shared_set)].reset_index(drop=True)

        results[niche] = combined
        n_unique = combined["gene"].nunique()
        logger.info(
            "Niche '%s': %d unique genes (%d rows across %d modalities)",
            niche, n_unique, len(combined), len(mod_dfs),
        )

    # ── Cross-niche deduplication (Option A) ──────────────────────────────
    # When deduplicate_shared=True, remove any gene appearing in ≥2 niches'
    # DEG lists entirely. Only genes unique to a single niche are kept.
    # This is the strictest approach — no gene can be a marker for more than
    # one niche. Recommended for small panels where shared genes add noise.
    if deduplicate_shared and len(results) > 1:
        # Count how many niches each gene appears in
        gene_niche_count: dict[str, int] = {}
        for niche, df in results.items():
            for gene in df["gene"]:
                gene_niche_count[gene] = gene_niche_count.get(gene, 0) + 1

        # Genes appearing in only one niche
        unique_genes: set[str] = {
            g for g, c in gene_niche_count.items() if c == 1
        }
        n_shared = len(gene_niche_count) - len(unique_genes)
        n_removed_total = 0

        # Keep only unique genes in each niche's DataFrame
        for niche in list(results.keys()):
            df = results[niche]
            before = len(df)
            df = df[df["gene"].isin(unique_genes)].reset_index(drop=True)
            n_removed = before - len(df)
            n_removed_total += n_removed
            results[niche] = df
            if n_removed > 0:
                logger.debug(
                    "Niche '%s': removed %d shared genes (kept %d unique markers)",
                    niche, n_removed, len(df),
                )

        logger.info(
            "Cross-niche deduplication (strict): %d shared genes removed entirely, "
            "%d gene-niche assignments removed across %d niches.",
            n_shared, n_removed_total, len(results),
        )

    return results


def get_deg_gene_set(
    deg_results: dict[str, pd.DataFrame],
    genes_shared: list[str],
    pval_adj_threshold: float = 0.05,
    min_logfc: float = 0.5,
    min_mean_expr: float = 0.5,
    max_degs_per_niche: int | None = None,
) -> list[str]:
    """
    Extract the union of DEGs across all niches, restricted to shared_all.

    For each niche, a gene passes if it meets the significance, effect-size,
    and expression thresholds in **any** modality (rows from
    ``compute_degs_per_niche`` may contain multiple modality entries per gene).
    Passed genes are unioned across all niches.

    Parameters
    ----------
    deg_results :
        Output of ``compute_degs_per_niche``. Each niche's DataFrame may
        contain multiple rows per gene (one per modality).
    genes_shared :
        Pool to restrict to. Genes not in this list are excluded.
    pval_adj_threshold :
        Adjusted p-value threshold. Default 0.05.
    min_logfc :
        Minimum log fold change (natural log, from log1p normalisation).
        Default 0.5 (≈1.65× fold change).
    min_mean_expr :
        Minimum mean expression (log1p space) in the target niche.
        Filters out lowly-expressed noise genes. Default 0.5.
    max_degs_per_niche : int or None
        If set, cap the number of DEGs per niche after filtering.
        Genes are ranked by their best (max) logFC across modalities,
        and the top-N are kept. This ensures balanced niche representation
        in the final gene set — critical for small panels where one niche
        can dominate with many more DEGs than others.
        Special values:

        - ``None`` (default): no cap, all passing DEGs are kept.
        - ``0``: adaptive cap — use the minimum per-niche count, so no
          niche has more markers than the least-represented niche.
        - Any positive int: fixed cap.

    Returns
    -------
    list[str]
        Sorted union of DEGs passing filters, restricted to ``genes_shared``.
    """
    shared_set = set(genes_shared)
    all_degs: set[str] = set()

    # First pass: filter DEGs per niche and collect per-niche gene sets
    niche_deg_info: dict[str, pd.DataFrame] = {}  # niche → filtered DataFrame
    niche_deg_sets: dict[str, set[str]] = {}       # niche → set of gene names

    for niche, df in deg_results.items():
        passing = df.copy()
        if "pvals_adj" in passing.columns:
            passing = passing[passing["pvals_adj"] < pval_adj_threshold]
        if "logfoldchanges" in passing.columns:
            passing = passing[passing["logfoldchanges"] >= min_logfc]
        if "mean_expr" in passing.columns:
            passing = passing[passing["mean_expr"] >= min_mean_expr]
        passing = passing[passing["gene"].isin(shared_set)]
        niche_deg_info[niche] = passing
        niche_deg_sets[niche] = set(passing["gene"])

    # Resolve adaptive cap (max_degs_per_niche=0 → min per-niche count)
    effective_cap = max_degs_per_niche
    if max_degs_per_niche == 0 and niche_deg_sets:
        per_niche_counts = [len(s) for s in niche_deg_sets.values() if len(s) > 0]
        effective_cap = min(per_niche_counts) if per_niche_counts else None
        logger.info(
            "Adaptive DEG cap: min per-niche count = %d (across %d niches with DEGs)",
            effective_cap, len(per_niche_counts),
        )

    # Second pass: apply cap and build union
    for niche in niche_deg_info:
        passing = niche_deg_info[niche]
        niche_degs = niche_deg_sets[niche]
        n_before_cap = len(niche_degs)

        if effective_cap is not None and len(niche_degs) > effective_cap:
            # Rank genes by best (max) logFC across modalities, take top-N
            best_logfc = passing.groupby("gene")["logfoldchanges"].max()
            top_genes = set(best_logfc.nlargest(effective_cap).index) & shared_set
            niche_degs = top_genes

        all_degs |= niche_degs

        cap_info = ""
        if effective_cap is not None and n_before_cap > effective_cap:
            cap_info = f" → capped to {len(niche_degs)}"
        logger.info(
            "Niche '%s': %d DEGs after filtering "
            "(p-adj<%.2f, logFC>=%.2f, mean_expr>=%.2f)%s",
            niche, n_before_cap,
            pval_adj_threshold, min_logfc, min_mean_expr,
            cap_info,
        )

    logger.info(
        "Union DEG set: %d genes across %d niches (restricted to shared_all%s)",
        len(all_degs), len(deg_results),
        f", max {effective_cap}/niche" if effective_cap is not None else "",
    )
    return sorted(all_degs)


# ── HVG computation ───────────────────────────────────────────────────────────

def compute_hvgs_per_modality(
    adata_sd3p: ad.AnnData,
    adata_sd_ffpe: ad.AnnData,
    adata_hd: ad.AnnData,
    genes_shared: list[str],
    n_top_genes: int = 4000,
    flavor: str = "seurat",
) -> dict[str, list[str]]:
    """
    Compute highly variable genes (HVGs) per modality within shared_all.

    Parameters
    ----------
    adata_sd3p, adata_sd_ffpe, adata_hd :
        AnnData objects (log-normalised counts).
    genes_shared :
        Restrict HVG computation to this gene pool.
    n_top_genes :
        Number of top HVGs to select per modality. Default 4000.
    flavor :
        HVG selection method. ``"seurat"`` (default) works on log-normalised
        data and requires no extra dependencies. Use ``"seurat_v3"`` for raw
        counts (requires ``scikit-misc``).

    Returns
    -------
    dict[str, list[str]]
        Keys: ``"SD_3prime"``, ``"SD_FFPE"``, ``"HD_FFPE"``.
        Values: HVG lists for each modality.
    """
    modality_adatas = {
        "SD_3prime": adata_sd3p,
        "SD_FFPE":   adata_sd_ffpe,
        "HD_FFPE":   adata_hd,
    }
    hvg_sets: dict[str, list[str]] = {}

    for name, adata in modality_adatas.items():
        # Skip empty or near-empty adatas — highly_variable_genes requires
        # at least 2 observations (Bessel's correction: n-1 > 0).
        # Empty adatas arise when a modality has no spots for a given niche
        # subset (e.g. SD modalities for HD-exclusive niches like Sinoatrial).
        if adata.n_obs < 2:
            logger.info(
                "%s: skipping HVG computation (%d obs — need ≥ 2).",
                name, adata.n_obs,
            )
            hvg_sets[name] = []
            continue

        # Subset to shared genes
        genes_here = [g for g in genes_shared if g in adata.var_names]
        sub = adata[:, genes_here].copy()

        # Guard: highly_variable_genes(flavor="seurat") requires log-normalised
        # input — it calls expm1 internally, which overflows on raw counts and
        # causes pd.cut to crash with "cannot specify integer bins when input
        # data contains infinity". Use scanpy's own check_nonnegative_integers
        # logic (dtype-first) to detect raw counts and normalise defensively.
        from numbers import Integral as _Integral
        # Mirrors is_log_normalised() in tissuetypist/utils/normalise.py.
        # Always convert to a plain numpy array first — sub.X may be a
        # memoryview (h5py backed), sparse matrix, or other array-like that
        # does not support dtype inspection or the % operator directly.
        _X = sub.X
        _raw = _X.data if hasattr(_X, "data") else _X
        _data = np.asarray(_raw, dtype=np.float64).ravel()
        if len(_data) == 0:
            _is_raw = False
        else:
            _orig_dtype = np.asarray(_raw).dtype
            _is_raw = (
                issubclass(_orig_dtype.type, _Integral)       # integer dtype → raw
                or (not np.signbit(_data).any()               # non-negative float
                    and not np.any((_data % 1) != 0))         # no fractional parts → raw
            )
        if _is_raw:
            logger.warning(
                "compute_hvgs_per_modality (%s): data appears to be raw counts "
                "(integer dtype or all-integer float values). "
                "Applying normalize_total + log1p before HVG selection. "
                "Pass log-normalised data to avoid this.",
                name,
            )
            sc.pp.normalize_total(sub, target_sum=1e4)
            sc.pp.log1p(sub)

        n_top = min(n_top_genes, len(genes_here))
        sc.pp.highly_variable_genes(
            sub,
            n_top_genes=n_top,
            flavor=flavor,
            subset=False,
        )
        hvgs = sub.var_names[sub.var["highly_variable"]].tolist()
        hvg_sets[name] = sorted(hvgs)
        logger.info(
            "%s: %d HVGs selected (from %d shared genes)",
            name, len(hvgs), len(genes_here),
        )

    return hvg_sets


def get_hvg_gene_set(
    hvg_sets: dict[str, list[str]],
    min_modalities: int = 2,
) -> list[str]:
    """
    Return genes that are HVG in at least ``min_modalities`` modalities.

    Parameters
    ----------
    hvg_sets :
        Output of ``compute_hvgs_per_modality``.
    min_modalities :
        Minimum number of modalities a gene must be HVG in. Default 2.
        Use 1 for the full union (most permissive).
        Use 3 for strict intersection across all modalities.

    Returns
    -------
    list[str]
        Sorted list of HVGs passing the threshold.
    """
    all_genes = set()
    for genes in hvg_sets.values():
        all_genes |= set(genes)

    passing = sorted([
        g for g in all_genes
        if sum(g in set(hvg_sets[m]) for m in hvg_sets) >= min_modalities
    ])
    logger.info(
        "HVG set (>= %d modalities): %d genes", min_modalities, len(passing)
    )
    return passing


# ── Combined DEG + HVG feature set (F2) ──────────────────────────────────────

def compute_f2_feature_set(
    adata_sd3p: ad.AnnData,
    adata_sd_ffpe: ad.AnnData,
    adata_hd: ad.AnnData,
    niche_col: str,
    genes_shared: list[str],
    niche_modality_map: dict[str, list[str]],
    n_top_degs: int = None,            # deprecated — ignored; kept for API compat
    n_top_hvgs: int = 4000,
    hvg_min_modalities: int = 2,
    pval_adj_threshold: float = 0.05,
    min_logfc: float = 0.5,
    min_mean_expr: float = 0.5,
    deg_method: str = "t-test_overestim_var",
    use_pseudobulk: bool = True,       # deprecated — has no effect; kept for API compat
    use_pseudobulk_hd: bool = True,
    microns_per_pixel_map: dict[str, float] = None,
    target_spot_um: float = 55.0,
    feature_strategy: str = "deg_hvg",
    n_top_markers: Optional[int] = None,
    deduplicate_shared: bool = False,
    max_degs_per_niche: int | None = None,
) -> dict:
    """
    Compute the F2 feature set: union of DEGs and HVGs, restricted to shared_all.

    DEG selection ranks all shared genes per niche per modality, then filters
    by significance (p-adj < 0.05), effect size (logFC >= 0.5), and expression
    level (mean_expr >= 0.5 in log1p space). No top-N cap is applied. DEGs are
    unioned across modalities per niche, then unioned across niches per model.

    Parameters
    ----------
    adata_sd3p, adata_sd_ffpe, adata_hd :
        Normalised AnnData objects (log1p counts in X, raw counts in
        ``layers["raw_counts"]``).
        When ``use_pseudobulk_hd=False``, ``adata_hd`` is expected to be a
        pre-computed sliding-window pseudobulk object (output of
        ``sliding_window_pseudobulk_hd`` or ``00_pseudobulk_hd.py``).
    niche_col :
        obs column with niche labels.
    genes_shared :
        Primary gene pool (``shared_all`` from Phase 0).
    niche_modality_map :
        Output of ``build_niche_modality_map`` for ``niche_col``.
    n_top_degs :
        **Deprecated** — ignored. Kept for backwards API compatibility.
        DEGs are now selected purely by significance, effect size, and
        expression filters. Pass any value; it will be silently ignored.
    n_top_hvgs :
        HVGs per modality. Default 4000.
    hvg_min_modalities :
        Minimum modalities for HVG inclusion. Default 2.
    pval_adj_threshold :
        Adjusted p-value cutoff for DEGs. Default 0.05.
    min_logfc :
        Minimum log fold-change for DEGs (natural log). Default 0.5
        (≈1.65× fold change).
    min_mean_expr :
        Minimum mean expression (log1p space) in the target niche for DEGs.
        Filters out lowly-expressed noise genes. Default 0.5.
    use_pseudobulk : bool
        **Deprecated** — has no effect. Kept for backwards API compatibility.
        Use ``use_pseudobulk_hd`` instead.
    use_pseudobulk_hd : bool
        If True (default), run sliding-window pseudobulk on ``adata_hd``
        before DEG computation. Set to False when ``adata_hd`` is already a
        pre-computed pseudobulk object (e.g. loaded from
        ``adata_hd_windows.h5ad``). Default True.
    microns_per_pixel_map :
        Dict mapping library ID → µm/px. Required when
        ``use_pseudobulk_hd=True``. Ignored when ``use_pseudobulk_hd=False``.
    target_spot_um :
        Target pseudo-spot size in µm. Default 55.0.
    feature_strategy : str
        Feature selection strategy. One of:

        ``"deg_hvg"`` (default)
            Union of filtered DEGs and HVGs. Standard F2 approach, optimal
            for large gene pools (≥500 genes) where HVGs add complementary
            information to DEGs.

        ``"deg_only"``
            Use only DEGs passing significance, effect-size, and expression
            filters. HVG computation is skipped entirely. Recommended for
            small gene panels (e.g. MERFISH 238 genes) where HVG selection
            tends to select all available genes, adding noise rather than
            signal.

        ``"hvg_only"``
            Use only HVGs. DEG computation is skipped. Useful as a baseline
            comparison.

    n_top_markers : int or None
        When set, take at most this many top-scoring DEGs per niche per
        modality (by score) before the cross-modality union. Default None
        (take all genes). Recommended for small panels: 5, 10, 20.
        Ignored when ``feature_strategy='hvg_only'``.

    Returns
    -------
    dict with keys:
        ``"genes"``        : sorted list of selected feature genes
        ``"deg_genes"``    : DEG component (empty list for hvg_only)
        ``"hvg_genes"``    : HVG component (empty list for deg_only)
        ``"deg_results"``  : raw DEG DataFrames per niche (empty dict for hvg_only)
        ``"hvg_sets"``     : HVG lists per modality (empty dict for deg_only)
        ``"n_total"``      : total feature set size
        ``"n_deg_only"``   : genes from DEGs not in HVG set
        ``"n_hvg_only"``   : genes from HVGs not in DEG set
        ``"n_overlap"``    : genes in both DEG and HVG sets
    """
    if feature_strategy not in ("deg_hvg", "deg_only", "hvg_only"):
        raise ValueError(
            f"feature_strategy must be 'deg_hvg', 'deg_only', or 'hvg_only', "
            f"got '{feature_strategy}'."
        )
    if n_top_degs is not None:
        logger.warning(
            "compute_f2_feature_set: 'n_top_degs' is deprecated and ignored. "
            "DEGs are selected by significance (p-adj<%.2f), effect size "
            "(logFC>=%.2f), and expression (mean_expr>=%.2f) filters only.",
            pval_adj_threshold, min_logfc, min_mean_expr,
        )
    if use_pseudobulk is not True:
        logger.warning(
            "compute_f2_feature_set: 'use_pseudobulk' is deprecated and has "
            "no effect. Use 'use_pseudobulk_hd' to control HD pseudobulking."
        )

    logger.info(
        "=== Computing F2 feature set (strategy='%s') ===", feature_strategy
    )
    logger.info(
        "DEG filters: p-adj < %.2f, logFC >= %.2f, mean_expr >= %.2f",
        pval_adj_threshold, min_logfc, min_mean_expr,
    )

    # ── DEG step (skipped for hvg_only) ───────────────────────────────────
    deg_genes:   list[str]              = []
    deg_results: dict[str, pd.DataFrame] = {}

    if feature_strategy in ("deg_hvg", "deg_only"):
        logger.info("Step 1/2: Computing DEGs per niche...")
        deg_results = compute_degs_per_niche(
            adata_sd3p, adata_sd_ffpe, adata_hd,
            niche_col=niche_col,
            genes_shared=genes_shared,
            niche_modality_map=niche_modality_map,
            method=deg_method,
            use_pseudobulk_hd=use_pseudobulk_hd,
            microns_per_pixel_map=microns_per_pixel_map,
            target_spot_um=target_spot_um,
            n_top_markers=n_top_markers,
            deduplicate_shared=deduplicate_shared,
        )
        deg_genes = get_deg_gene_set(
            deg_results, genes_shared,
            pval_adj_threshold=pval_adj_threshold,
            min_logfc=min_logfc,
            min_mean_expr=min_mean_expr,
            max_degs_per_niche=max_degs_per_niche,
        )
    else:
        logger.info("Step 1/2: DEG computation skipped (strategy='hvg_only').")

    # ── HVG step (skipped for deg_only) ───────────────────────────────────
    hvg_genes: list[str]          = []
    hvg_sets:  dict[str, list[str]] = {}

    if feature_strategy in ("deg_hvg", "hvg_only"):
        logger.info("Step 2/2: Computing HVGs per modality...")
        hvg_sets = compute_hvgs_per_modality(
            adata_sd3p, adata_sd_ffpe, adata_hd,
            genes_shared=genes_shared,
            n_top_genes=n_top_hvgs,
        )
        hvg_genes = get_hvg_gene_set(hvg_sets, min_modalities=hvg_min_modalities)
    else:
        logger.info("Step 2/2: HVG computation skipped (strategy='deg_only').")

    # ── Union / selection ──────────────────────────────────────────────────
    deg_set = set(deg_genes)
    hvg_set = set(hvg_genes)
    f2_genes = sorted(deg_set | hvg_set)

    n_deg_only = len(deg_set - hvg_set)
    n_hvg_only = len(hvg_set - deg_set)
    n_overlap  = len(deg_set & hvg_set)

    logger.info(
        "F2 feature set: %d genes total "
        "(DEG-only: %d | HVG-only: %d | overlap: %d)",
        len(f2_genes), n_deg_only, n_hvg_only, n_overlap,
    )

    return {
        "genes":       f2_genes,
        "deg_genes":   deg_genes,
        "hvg_genes":   hvg_genes,
        "deg_results": deg_results,
        "hvg_sets":    hvg_sets,
        "n_total":     len(f2_genes),
        "n_deg_only":  n_deg_only,
        "n_hvg_only":  n_hvg_only,
        "n_overlap":   n_overlap,
    }
