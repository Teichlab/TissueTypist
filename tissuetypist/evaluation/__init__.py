"""Evaluation + plotting for TissueTypist predictions.

Submodules:
    metrics  — precision/recall/F1, classification_report, prediction summary
    plots    — confusion matrix, spatial, UMAP, confidence distribution
    runner   — high-level ``evaluate()`` orchestration

The palette used by every plot is derived from
:attr:`tissuetypist.config.HierarchySpec.palette`, with a deterministic
tab20 fallback for any label not in the curated palette. Ground-truth
labels are remapped via :attr:`HierarchySpec.gt_label_remap` before
every metric / plot comparison.
"""
from .metrics import (
    compute_metrics,
    remap_ground_truth,
    save_prediction_summary,
)
from .plots import (
    build_palette_from_spec,
    plot_confidence_distributions,
    plot_confusion_matrix,
    plot_spatial,
    plot_umap,
)
from .runner import evaluate

__all__ = [
    # runner
    "evaluate",
    # metrics
    "compute_metrics",
    "remap_ground_truth",
    "save_prediction_summary",
    # plots
    "build_palette_from_spec",
    "plot_confidence_distributions",
    "plot_confusion_matrix",
    "plot_spatial",
    "plot_umap",
]
