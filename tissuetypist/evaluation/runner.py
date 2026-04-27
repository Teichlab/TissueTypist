"""
tissuetypist/evaluation/runner.py
==================================
Top-level orchestration: predict once, then dispatch to metrics + plots.

Use this to run evaluation programmatically without touching the CLI
wrapper in ``scripts/03_predict_evaluate.py``.

Example
-------
>>> from tissuetypist.evaluation import evaluate
>>> evaluate(
...     adata=my_query_adata,
...     model_dir="results/apr2026_default",
...     outdir="results/eval_apr2026_sd3p",
...     modality="sd",
...     section_col="section_ID",
...     prefix="sd",
...     compute_eval=True,
... )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def evaluate(
    adata: "ad.AnnData",
    model_dir: str | Path,
    outdir: str | Path,
    modality: str,
    prefix: str,
    section_col: str = "section_ID",
    fine_col: Optional[str] = None,
    coarse_col: Optional[str] = None,
    theta: float = 0.5,
    compute_eval: bool = True,
    n_pcs: int = 30,
    n_sections: int = 3,
) -> "ad.AnnData":
    """Run prediction, save plots + metrics CSVs.

    Parameters
    ----------
    adata :
        Query AnnData. Must have ``obs[section_col]`` if section-based
        spatial plots are desired. If this is cell-level HD data, pre-
        pseudobulk it first (see ``tissuetypist.data.pseudobulk``).
    model_dir :
        Directory produced by ``tissuetypist train`` (contains
        ``hierarchy_config.json`` schema_version=2 + joblib pipelines).
    outdir :
        Where to write ``{prefix}_predicted.h5ad``,
        ``{prefix}_prediction_summary.csv``,
        ``{prefix}_classification_report.csv``,
        ``{prefix}_confusion_matrix.pdf``,
        ``{prefix}_spatial_<section>.pdf``,
        ``{prefix}_umap.pdf``,
        ``{prefix}_confidence_distributions.pdf``.
    modality :
        ``"sd"`` or ``"hd"``.
    prefix :
        Filename prefix (e.g. ``"sd"``, ``"hd"``, ``"merfish"``).
    section_col :
        obs column for section grouping.
    fine_col / coarse_col :
        Override the YAML's default columns, or pass ``None`` to use
        the spec's defaults.
    theta :
        Confidence threshold for Stage 2 routing. Default 0.5.
    compute_eval :
        If False, skip metrics + confusion matrix (useful when the
        query data has no ground-truth labels).
    n_pcs, n_sections :
        UMAP PCA dimension + number of sections to spatially plot.

    Returns
    -------
    AnnData
        With ``tt_*`` columns added. Also written to
        ``{outdir}/{prefix}_predicted.h5ad``.
    """
    from tissuetypist.prediction import predict_adata
    from tissuetypist.prediction.hierarchical import _load_hierarchy
    from .metrics import compute_metrics, save_prediction_summary
    from .plots import (
        plot_confidence_distributions,
        plot_confusion_matrix,
        plot_spatial,
        plot_umap,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load the spec from the model dir (already includes gt_label_remap + palette).
    hierarchy_bundle = _load_hierarchy(model_dir)
    spec = hierarchy_bundle["spec"]

    # Resolve column names (CLI override > spec default).
    fine_col   = fine_col   if fine_col   is not None else spec.fine_col
    coarse_col = coarse_col if coarse_col is not None else spec.coarse_col

    logger.info(
        "Evaluating with hierarchy %r (coarse_col=%r, fine_col=%r, theta=%.2f)",
        spec.name, coarse_col, fine_col, theta,
    )

    # ── Run prediction ────────────────────────────────────────────────────
    adata = predict_adata(
        adata,
        model_dir=str(model_dir),
        modality=modality,
        section_col=section_col,
        theta=theta,
    )

    # ── Persist predicted AnnData ────────────────────────────────────────
    predicted_path = outdir / f"{prefix}_predicted.h5ad"
    adata.write_h5ad(predicted_path)
    logger.info("Saved predicted AnnData → %s", predicted_path)

    # ── Summary CSV (always) ─────────────────────────────────────────────
    save_prediction_summary(adata, fine_col=fine_col, outdir=outdir,
                            prefix=prefix, spec=spec)

    # ── Plots ────────────────────────────────────────────────────────────
    if compute_eval and fine_col in adata.obs.columns:
        compute_metrics(adata, fine_col=fine_col, outdir=outdir,
                        prefix=prefix, spec=spec)
        plot_confusion_matrix(adata, fine_col=fine_col, outdir=outdir,
                              prefix=prefix, spec=spec)

    plot_spatial(adata, fine_col=fine_col, outdir=outdir,
                 prefix=prefix, spec=spec,
                 section_col=section_col, n_sections=n_sections)
    plot_umap(adata, fine_col=fine_col, outdir=outdir,
              prefix=prefix, spec=spec, n_pcs=n_pcs)
    plot_confidence_distributions(adata, outdir=outdir, prefix=prefix)

    return adata


__all__ = ["evaluate"]
