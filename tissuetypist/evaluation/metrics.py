"""
tissuetypist/evaluation/metrics.py
===================================
Evaluation metrics for TissueTypist predictions.

All functions take a predicted AnnData (with ``tt_*`` columns in
``obs``) plus the ground-truth column name and a
:class:`~tissuetypist.config.HierarchySpec`. Ground-truth labels are
remapped via ``spec.gt_label_remap`` before comparison so pooled
terminal classes align with the prediction output.

Public API
----------
    remap_ground_truth(series, spec)     — apply spec.gt_label_remap
    compute_metrics(adata, spec, ...)    — weighted/macro F1 + per-class report
    save_prediction_summary(adata, ...)  — per-niche count + mean confidence CSV
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

if TYPE_CHECKING:
    import anndata as ad
    from tissuetypist.config.hierarchy import HierarchySpec

logger = logging.getLogger(__name__)


def remap_ground_truth(series: pd.Series, spec: "HierarchySpec") -> pd.Series:
    """Apply ``spec.gt_label_remap`` to a ground-truth label series.

    Labels not present in the remap are unchanged.
    """
    remap = dict(spec.gt_label_remap)
    if not remap:
        return series
    return series.map(lambda x: remap.get(x, x))


def compute_metrics(
    adata: "ad.AnnData",
    fine_col: str,
    outdir: Path,
    prefix: str,
    spec: "HierarchySpec",
) -> pd.DataFrame:
    """Compute per-niche precision, recall, F1 comparing ``tt_final_label``
    against ground-truth labels.

    Writes ``{outdir}/{prefix}_classification_report.csv`` and returns
    the report as a DataFrame. Ground-truth labels are remapped via
    ``spec.gt_label_remap`` before comparison.
    """
    obs = adata.obs.copy()

    # Exclude spots with no prediction or no ground truth.
    valid = obs["tt_final_label"].notna()
    if fine_col in obs.columns:
        valid &= obs[fine_col].notna()
    obs = obs[valid]

    if len(obs) == 0:
        logger.warning("No valid spots for evaluation.")
        return pd.DataFrame()

    y_true = remap_ground_truth(obs[fine_col].astype(str), spec)
    y_pred = obs["tt_final_label"].astype(str)

    # Overall metrics.
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    logger.info(
        "%s — weighted F1: %.4f | macro F1: %.4f | n_spots: %d",
        prefix, f1_weighted, f1_macro, len(obs),
    )

    # Per-class report.
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    report_path = Path(outdir) / f"{prefix}_classification_report.csv"
    report_df.to_csv(report_path)
    logger.info("Saved classification report → %s", report_path)

    return report_df


def save_prediction_summary(
    adata: "ad.AnnData",
    fine_col: str,
    outdir: Path,
    prefix: str,
    spec: "HierarchySpec",
) -> pd.DataFrame:
    """Save a summary CSV with prediction counts + mean confidence per niche.

    Columns: ``tt_final_label``, ``n_spots``, ``mean_coarse_score``,
    ``mean_joint_score``, ``n_low_conf``, ``n_ground_truth``. Ground-truth
    counts are ``spec.gt_label_remap``-remapped.
    """
    obs = adata.obs.copy()

    summary = obs.groupby("tt_final_label").agg(
        n_spots=("tt_final_label", "count"),
        mean_coarse_score=("tt_coarse_score", "mean"),
        mean_joint_score=(
            "tt_joint_score",
            lambda x: x.dropna().mean() if x.notna().any() else np.nan,
        ),
        n_low_conf=("tt_low_conf", "sum"),
    ).reset_index()

    if fine_col in obs.columns:
        gt_series = remap_ground_truth(obs[fine_col].astype(str), spec)
        gt_counts = (
            gt_series
            .value_counts()
            .rename("n_ground_truth")
            .reset_index()
            .rename(columns={"index": "tt_final_label",
                             fine_col: "tt_final_label"})
        )
        if "index" in gt_counts.columns:
            gt_counts = gt_counts.rename(columns={"index": "tt_final_label"})
        summary = summary.merge(gt_counts, on="tt_final_label", how="left")

    summary = summary.sort_values("n_spots", ascending=False)
    out_path = Path(outdir) / f"{prefix}_prediction_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Saved prediction summary → %s", out_path)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


__all__ = [
    "remap_ground_truth",
    "compute_metrics",
    "save_prediction_summary",
]
