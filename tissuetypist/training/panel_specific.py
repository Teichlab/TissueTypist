"""
tissuetypist.training.panel_specific
=====================================
Panel-specific retraining for imaging-based ST (Xenium, MERFISH, CosMx).

When a query platform uses a targeted gene panel (rather than the full
transcriptome), the shipped reference models cannot be used directly:
the normalisation denominator of ``normalize_total(1e4)`` differs
between full-genome training data and a small panel's query data. The
fix is to retrain on ``panel ∩ shared_all`` with the reference data
normalised within that same gene space.

This module provides :func:`train_panel_specific` — a thin wrapper
around :func:`tissuetypist.training.hierarchical.train_all_models`
that applies the panel-specific gene subsetting + normalisation and
threads gene-list overrides into the chain walker.

Three gene-selection strategies
-------------------------------
1. ``"deg_hvg"`` (default) — fresh DEG+HVG computed on panel-normalised
   data for every sub-model stage.
2. ``"custom"`` — a user-supplied curated gene list (e.g. a manually
   curated marker set) is used for every stage, intersected with
   ``panel ∩ shared_all``. Useful when the panel is small and DEG+HVG
   on it is unstable.
3. ``"pre_computed"`` — per-stage gene lists from a prior full-genome
   training output directory (``*_gene_list.txt``) are intersected with
   ``panel ∩ shared_all``. Each stage keeps its own feature focus.

Example
-------
>>> from tissuetypist.training.panel_specific import train_panel_specific
>>> from tissuetypist.config import load_hierarchy
>>> train_panel_specific(
...     adata_sd3p_raw=adata_sd3p,
...     adata_sdffpe_raw=adata_sdffpe,
...     adata_hd_raw=adata_hd_windows,
...     gene_panel=list(adata_query.var_names),
...     genes_shared=shared_all,
...     outdir="results/panel_merfish",
...     config=TrainingConfig(hierarchy=load_hierarchy("cardiac")),
...     gene_strategy="pre_computed",
...     gene_lists_from="results/apr2026_default",
... )
"""
from __future__ import annotations

import logging
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .hierarchical import (
    TrainingConfig,
    _empty_adata,
    train_all_models,
)

if TYPE_CHECKING:
    from tissuetypist.config.hierarchy import HierarchySpec

logger = logging.getLogger("tissuetypist.training.panel_specific")


# ─────────────────────────────────────────────────────────────────────────────
# Gene-list helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_panel_shared(
    gene_panel: list[str],
    genes_shared: list[str],
    min_overlap_warn: int = 50,
) -> list[str]:
    """Return the sorted intersection ``gene_panel ∩ genes_shared``.

    Raises
    ------
    ValueError
        If the intersection is empty.
    """
    panel_set  = set(gene_panel)
    shared_set = set(genes_shared)
    panel_shared = sorted(panel_set & shared_set)

    n_panel   = len(panel_set)
    n_overlap = len(panel_shared)

    logger.info(
        "Gene panel overlap: %d panel × %d shared_all → %d overlap (%.1f%% of panel, "
        "%d panel-only genes ignored)",
        n_panel, len(shared_set), n_overlap,
        n_overlap / n_panel * 100 if n_panel > 0 else 0,
        n_panel - n_overlap,
    )
    if n_overlap == 0:
        raise ValueError(
            "Zero genes overlap between query panel and shared_all. "
            "Check that panel gene symbols match the shared-gene catalogue."
        )
    if n_overlap < min_overlap_warn:
        logger.warning(
            "Only %d genes in panel ∩ shared_all — retrained models may be "
            "unreliable (threshold for warning: %d).",
            n_overlap, min_overlap_warn,
        )
    return panel_shared


def _subset_and_normalise(
    adata: ad.AnnData,
    gene_list: list[str],
    name: str,
) -> ad.AnnData:
    """Subset to ``gene_list``, then ``normalize_total(1e4) + log1p``.

    Matches the published approach: normalise AFTER subsetting so the
    ``normalize_total`` denominator is over panel genes only, not the
    full genome. Critical for panel-specific retraining because the
    query data (e.g. MERFISH) is also normalised on its panel genes.

    Expects RAW counts in ``adata.X``. If ``adata.X`` already looks
    log-normalised, logs a warning and skips (defensive).
    """
    from tissuetypist.data.normalise import is_log_normalised

    genes_present = [g for g in gene_list if g in adata.var_names]
    if not genes_present:
        logger.warning(
            "  %s: 0 / %d panel genes found — returning as-is.",
            name, len(gene_list),
        )
        return adata
    sub = adata[:, genes_present].copy()

    if is_log_normalised(sub):
        logger.warning(
            "  %s: data appears already log-normalised after subsetting to "
            "%d genes — skipping normalize_total + log1p. This may indicate "
            "unexpected state if raw counts were expected.",
            name, len(genes_present),
        )
        return sub

    logger.info(
        "  %s: subsetting to %d / %d panel genes, then normalize_total + log1p",
        name, len(genes_present), len(gene_list),
    )
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# Strategy dispatch: build {stage.model_name: [genes]} overrides
# ─────────────────────────────────────────────────────────────────────────────

def _stage_model_names_from_spec(spec: "HierarchySpec") -> list[str]:
    """Return every stage model name across every sub-model in the spec."""
    out = ["coarse"]
    for sm in spec.sub_models.values():
        for stage in sm.stages:
            out.append(stage.model_name)
    return out


def _build_custom_overrides(
    spec: "HierarchySpec",
    custom_gene_list: list[str],
    panel_shared: list[str],
) -> dict[str, list[str]]:
    """Every stage uses ``custom_gene_list ∩ panel_shared``."""
    intersect = sorted(set(custom_gene_list) & set(panel_shared))
    logger.info(
        "Custom gene list: %d genes → %d after ∩ panel_shared",
        len(custom_gene_list), len(intersect),
    )
    return {model_name: intersect for model_name in _stage_model_names_from_spec(spec)}


def _build_precomputed_overrides(
    spec: "HierarchySpec",
    gene_lists_from: Path,
    panel_shared: list[str],
) -> dict[str, list[str]]:
    """For each stage, read ``{model_name}_gene_list.txt`` from ``gene_lists_from``
    and intersect with ``panel_shared``. Missing files are skipped with a warning
    (those stages fall through to DEG+HVG).
    """
    gene_lists_from = Path(gene_lists_from)
    panel_set = set(panel_shared)
    overrides: dict[str, list[str]] = {}
    for model_name in _stage_model_names_from_spec(spec):
        glist_path = gene_lists_from / f"{model_name}_gene_list.txt"
        if not glist_path.exists():
            logger.warning(
                "  stage %r: %s not found — will fall back to DEG+HVG.",
                model_name, glist_path,
            )
            continue
        with glist_path.open() as f:
            genes = [line.strip() for line in f if line.strip()]
        intersect = sorted(set(genes) & panel_set)
        logger.info(
            "  stage %r: pre-computed %d genes → %d after ∩ panel_shared",
            model_name, len(genes), len(intersect),
        )
        overrides[model_name] = intersect
    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Top-level API
# ─────────────────────────────────────────────────────────────────────────────

GeneStrategy = Literal["deg_hvg", "custom", "pre_computed"]


def train_panel_specific(
    adata_sd3p_raw: ad.AnnData,
    adata_sdffpe_raw: ad.AnnData,
    adata_hd_raw: ad.AnnData,
    gene_panel: list[str],
    genes_shared: list[str],
    outdir: str | Path,
    config: Optional[TrainingConfig] = None,
    gene_strategy: GeneStrategy = "deg_hvg",
    custom_gene_list: Optional[list[str]] = None,
    gene_lists_from: Optional[str | Path] = None,
    min_genes_submodel: int = 10,
    warn_genes_threshold: int = 200,
    qc_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain hierarchical models on ``panel ∩ shared_all``.

    Parameters
    ----------
    adata_sd3p_raw, adata_sdffpe_raw, adata_hd_raw :
        Reference training AnnDatas with RAW counts in ``X`` (not
        log-normalised). The function will subset + normalise them to
        the panel gene space internally. Pass empty AnnDatas (e.g. the
        output of :func:`tissuetypist.training.hierarchical._empty_adata`)
        for modalities you don't want to use.
    gene_panel :
        Gene names present on the query platform (e.g.
        ``list(query_adata.var_names)``).
    genes_shared :
        The shared-gene catalogue from Phase 0
        (``gene_pools.csv["shared_all"]``).
    outdir :
        Output directory for all per-stage pipelines, gene lists, the
        training summary, and ``hierarchy_config.json``.
    config :
        TrainingConfig carrying the :class:`HierarchySpec`. If ``None``,
        defaults to ``TrainingConfig()`` (which uses the shipped cardiac
        hierarchy).
    gene_strategy :
        ``"deg_hvg"`` (default) | ``"custom"`` | ``"pre_computed"``.
    custom_gene_list :
        Required when ``gene_strategy == "custom"``. A curated set of
        genes to use for every stage, intersected with ``panel ∩
        shared_all``.
    gene_lists_from :
        Required when ``gene_strategy == "pre_computed"``. Path to a
        prior training output directory (e.g. ``results/apr2026_default``)
        containing ``{model_name}_gene_list.txt`` files.
    min_genes_submodel :
        Skip any stage whose gene list (after overrides + intersection)
        falls below this count. Default 10.
    warn_genes_threshold :
        Emit a warning if the stage's gene list is below this count
        (still trains). Default 200.
    qc_dir :
        Optional path for edge-detection QC plots.

    Returns
    -------
    ``(summary_df, gene_count_df)`` — as from
    :func:`train_all_models`.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = TrainingConfig()
    spec = config.resolve_hierarchy()

    # 1. Intersect the panel with shared_all.
    panel_shared = compute_panel_shared(gene_panel, genes_shared)

    # 2. Subset + normalise reference AnnDatas to panel_shared.
    logger.info("Subsetting + normalising reference AnnDatas to panel space…")
    adata_sd3p_norm   = _subset_and_normalise(adata_sd3p_raw,   panel_shared, "SD_3prime") \
                        if adata_sd3p_raw.n_obs > 0 else _empty_adata(adata_sd3p_raw)
    adata_sdffpe_norm = _subset_and_normalise(adata_sdffpe_raw, panel_shared, "SD_FFPE") \
                        if adata_sdffpe_raw.n_obs > 0 else _empty_adata(adata_sdffpe_raw)
    adata_hd_norm     = _subset_and_normalise(adata_hd_raw,     panel_shared, "HD_windows") \
                        if adata_hd_raw.n_obs > 0 else _empty_adata(adata_hd_raw)

    # 3. Build stage gene overrides based on the requested strategy.
    stage_overrides: Optional[dict[str, list[str]]] = None
    if gene_strategy == "custom":
        if custom_gene_list is None:
            raise ValueError("gene_strategy='custom' requires custom_gene_list.")
        stage_overrides = _build_custom_overrides(spec, custom_gene_list, panel_shared)
    elif gene_strategy == "pre_computed":
        if gene_lists_from is None:
            raise ValueError("gene_strategy='pre_computed' requires gene_lists_from.")
        stage_overrides = _build_precomputed_overrides(
            spec, Path(gene_lists_from), panel_shared,
        )
    elif gene_strategy == "deg_hvg":
        stage_overrides = None  # chain walker computes DEG+HVG fresh on panel-normalised data
    else:
        raise ValueError(
            f"Unknown gene_strategy {gene_strategy!r}. "
            "Expected 'deg_hvg' | 'custom' | 'pre_computed'."
        )

    # 4. Emit per-stage size warnings for soft threshold.
    if stage_overrides is not None:
        for name, genes in stage_overrides.items():
            if 0 < len(genes) < warn_genes_threshold:
                logger.warning(
                    "Stage %r: only %d genes (< warn_genes_threshold=%d). "
                    "Training will proceed but accuracy may be reduced.",
                    name, len(genes), warn_genes_threshold,
                )

    # 5. Delegate to the generic chain-walker trainer. Pass panel_shared as
    #    genes_shared so feature-building (neighbour-max + edge) uses the
    #    panel gene space, not the full catalogue.
    logger.info(
        "Running train_all_models on panel-normalised data "
        "(strategy=%r, %d panel_shared genes)…",
        gene_strategy, len(panel_shared),
    )
    return train_all_models(
        adata_sd3p=adata_sd3p_norm,
        adata_sdffpe=adata_sdffpe_norm,
        adata_hd_windows=adata_hd_norm,
        genes_shared=panel_shared,
        outdir=outdir,
        config=config,
        qc_dir=qc_dir,
        stage_gene_overrides=stage_overrides,
        min_genes_submodel=min_genes_submodel,
    )


__all__ = [
    "train_panel_specific",
    "compute_panel_shared",
    "GeneStrategy",
]
