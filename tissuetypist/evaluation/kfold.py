"""
tissuetypist/evaluation/kfold.py
================================
Stratified k-fold cross-validation for TissueTypist.

Companion to :mod:`tissuetypist.evaluation.loso`. Uses scikit-learn's
:class:`~sklearn.model_selection.StratifiedKFold` to split each modality's
spots into ``k`` train/test folds, stratifying by the fine niche label so
every fold sees a representative slice of every class.

**Important caveat — spatial leakage.** Standard k-fold places spots from
the *same section* into both train and test, which leaks neighbour-max
and distance-to-edge information across the split. This module is
therefore intended to be paired with the ``own_only`` weight preset
(``neighbour_weight=0.0`` and ``edge_weight=0.0``) so the LR sees only
gene expression and no spatial signal. ``run_kfold`` enforces that
constraint by default and emits a warning if the caller overrides it.

The orchestrator (``run_kfold``) mirrors ``loso.run_loso`` — same
per-fold parquet/JSON cache layout, same scoring helpers, same pooled
DataFrame shape — so the same notebook can read either set of outputs.

Public API
----------
    select_kfold_splits(...)           — per-modality stratified splits.
    run_single_kfold_fold(...)         — train + predict for one fold.
    run_kfold(...)                     — orchestrator, returns pooled DataFrame.
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace as dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from tissuetypist.config.hierarchy import load_hierarchy
from tissuetypist.data.normalise import normalise_if_needed
from tissuetypist.prediction.hierarchical import predict_adata
from tissuetypist.training.hierarchical import TrainingConfig, train_all_models

# Reuse the per-fold scorer + scoring-mode helper from the LOSO module.
from tissuetypist.evaluation.loso import (
    MOD_TO_PREDICT_ARG,
    SCORING_MODES,
    _score_fold,
)

if TYPE_CHECKING:
    import anndata as ad
    from tissuetypist.config.hierarchy import HierarchySpec

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fold record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KfoldSplit:
    """One k-fold's per-modality train/test indices."""
    fold_idx: int
    n_train_per_mod: dict[str, int]
    n_test_per_mod:  dict[str, int]


# ──────────────────────────────────────────────────────────────────────────────
# Stratified split construction
# ──────────────────────────────────────────────────────────────────────────────

def _stratify_labels(labels: np.ndarray, k: int, rare_label: str = "__rare__") -> np.ndarray:
    """Replace classes with fewer than ``k`` members with ``rare_label``.

    StratifiedKFold raises if any class has fewer members than ``n_splits``.
    Lumping rare classes into a single bin lets the splitter run; rare-class
    spots are still distributed across folds, just not stratified
    individually. This is fine for our purpose (we score per-niche after
    pooling all folds).
    """
    counts = Counter(labels.tolist())
    return np.array([l if counts[l] >= k else rare_label for l in labels])


def select_kfold_splits(
    adata_sd3p: "ad.AnnData",
    adata_sdffpe: "ad.AnnData",
    adata_hd: "ad.AnnData",
    k: int = 5,
    stratify_col: str = "niche_fine_Apr2026",
    seed: int = 42,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Per-modality stratified k-fold splits.

    Returns a dict keyed by modality (``"SD_3prime"`` / ``"SD_FFPE"`` /
    ``"HD"``) where each value is a list of ``k`` ``(train_idx, test_idx)``
    integer-array tuples (positional indices into that modality's
    AnnData). Modalities with no spots map to a list of empty splits.
    """
    from sklearn.model_selection import StratifiedKFold

    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, adata in [
        ("SD_3prime", adata_sd3p),
        ("SD_FFPE",   adata_sdffpe),
        ("HD",        adata_hd),
    ]:
        if adata is None or adata.n_obs == 0:
            empty = (np.array([], dtype=int), np.array([], dtype=int))
            splits[name] = [empty for _ in range(k)]
            continue
        if stratify_col not in adata.obs.columns:
            raise ValueError(
                f"{name}: stratify column {stratify_col!r} not in obs."
            )
        labels = adata.obs[stratify_col].astype(str).values
        labels_for_stratify = _stratify_labels(labels, k)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        splits[name] = [
            (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
            for tr, te in skf.split(np.zeros(len(labels_for_stratify)), labels_for_stratify)
        ]
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# Per-fold runner
# ──────────────────────────────────────────────────────────────────────────────

def _slice(adata: "ad.AnnData", idx: np.ndarray) -> "ad.AnnData":
    if adata.n_obs == 0 or len(idx) == 0:
        return adata[:0].copy()
    return adata[idx].copy()


def run_single_kfold_fold(
    fold_idx: int,
    adata_sd3p: "ad.AnnData",
    adata_sdffpe: "ad.AnnData",
    adata_hd: "ad.AnnData",
    splits_per_mod: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    gene_pool: list[str],
    spec: "HierarchySpec",
    config: TrainingConfig,
    workdir: Path,
    section_col: str = "section_ID",
    theta: float = 0.5,
) -> pd.DataFrame:
    """Train on ``train_idx`` per modality, predict on ``test_idx``.

    Returns a per-spot DataFrame matching the pooled-prediction shape from
    :func:`tissuetypist.evaluation.loso.run_single_fold`.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    train_sd3p   = _slice(adata_sd3p,   splits_per_mod["SD_3prime"][fold_idx][0])
    test_sd3p    = _slice(adata_sd3p,   splits_per_mod["SD_3prime"][fold_idx][1])
    train_sdffpe = _slice(adata_sdffpe, splits_per_mod["SD_FFPE"][fold_idx][0])
    test_sdffpe  = _slice(adata_sdffpe, splits_per_mod["SD_FFPE"][fold_idx][1])
    train_hd     = _slice(adata_hd,     splits_per_mod["HD"][fold_idx][0])
    test_hd      = _slice(adata_hd,     splits_per_mod["HD"][fold_idx][1])

    train_all_models(
        adata_sd3p=train_sd3p,
        adata_sdffpe=train_sdffpe,
        adata_hd_windows=train_hd,
        genes_shared=gene_pool,
        outdir=workdir,
        config=config,
        qc_dir=None,
    )

    rows: list[pd.DataFrame] = []
    for mod_name, test_adata in [
        ("SD_3prime", test_sd3p),
        ("SD_FFPE",   test_sdffpe),
        ("HD",        test_hd),
    ]:
        if test_adata.n_obs == 0:
            continue
        modality_arg = MOD_TO_PREDICT_ARG[mod_name]
        adata_pred = predict_adata(
            test_adata,
            model_dir=workdir,
            modality=modality_arg,
            section_col=section_col,
            theta=theta,
        )
        obs = adata_pred.obs.copy()
        gt_coarse = obs[spec.coarse_col].astype(str).values
        gt_fine   = obs[spec.fine_col].astype(str).values
        tt_cols   = [c for c in obs.columns if c.startswith("tt_")]
        df = pd.DataFrame({
            "spot_id":   obs.index.astype(str),
            "modality":  mod_name,
            "section_id": obs[section_col].astype(str).values
                          if section_col in obs.columns else "",
            "donor":     obs["donor"].astype(str).values
                          if "donor" in obs.columns else "",
            "fold_idx":  fold_idx,
            "ground_truth_coarse": gt_coarse,
            "ground_truth_fine":   gt_fine,
        })
        for col in tt_cols:
            df[col] = obs[col].values
        rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_kfold(
    adata_sd3p: "ad.AnnData",
    adata_sdffpe: "ad.AnnData",
    adata_hd: "ad.AnnData",
    gene_pool: list[str],
    outdir: Path,
    spec: Optional["HierarchySpec"] = None,
    training_config: Optional[TrainingConfig] = None,
    section_col: str = "section_ID",
    stratify_col: Optional[str] = None,
    k: int = 5,
    seed: int = 42,
    theta: float = 0.5,
    max_folds: Optional[int] = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full k-fold sweep and return pooled predictions + per-fold metrics.

    Mirrors :func:`tissuetypist.evaluation.loso.run_loso`. Per-fold parquet
    + JSON outputs land under ``{outdir}/per_fold/``. The pooled CSV and
    per-fold metrics CSV match the LOSO output schema (with an extra
    ``fold_idx`` column) so the same downstream notebook can compare
    LOSO-style and kfold-style runs side-by-side.

    By default, ``training_config`` is materialised with
    ``neighbour_weight=0.0`` and ``edge_weight=0.0`` (the ``own_only``
    preset) to avoid spatial leakage from same-section spots being split
    across train/test. A warning is emitted if the caller overrides this.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_fold_dir = outdir / "per_fold"
    per_fold_dir.mkdir(exist_ok=True)

    if spec is None:
        spec = load_hierarchy("cardiac")
    if stratify_col is None:
        stratify_col = spec.fine_col

    if training_config is None:
        training_config = TrainingConfig(
            hierarchy=spec, feature_set="deg_hvg",
            neighbour_weight=0.0, edge_weight=0.0,
            save_qc=False,
        )
    else:
        training_config = dataclass_replace(training_config, hierarchy=spec)

    if training_config.neighbour_weight != 0.0 or training_config.edge_weight != 0.0:
        logger.warning(
            "run_kfold: training_config has non-zero spatial weights "
            "(neighbour_weight=%.2f, edge_weight=%.2f). Standard k-fold "
            "leaks neighbourhood information across splits — use weight=0 "
            "(own_only) unless you know what you're doing.",
            training_config.neighbour_weight, training_config.edge_weight,
        )

    # Normalise once up front (same convention as run_loso).
    if adata_sd3p.n_obs:
        adata_sd3p = normalise_if_needed(adata_sd3p, "SD_3prime")
    if adata_sdffpe.n_obs:
        adata_sdffpe = normalise_if_needed(adata_sdffpe, "SD_FFPE")
    if adata_hd.n_obs:
        adata_hd = normalise_if_needed(adata_hd, "HD")

    splits = select_kfold_splits(
        adata_sd3p, adata_sdffpe, adata_hd,
        k=k, stratify_col=stratify_col, seed=seed,
    )
    roster_rows = [
        {
            "fold_idx": fold_idx,
            **{f"n_train_{m}": int(len(splits[m][fold_idx][0])) for m in splits},
            **{f"n_test_{m}":  int(len(splits[m][fold_idx][1])) for m in splits},
        }
        for fold_idx in range(k)
    ]
    roster_df = pd.DataFrame(roster_rows)
    roster_df.to_csv(outdir / "fold_roster.csv", index=False)
    logger.info("k-fold roster (k=%d):\n%s", k, roster_df.to_string(index=False))

    folds_to_run = list(range(k if max_folds is None else min(k, max_folds)))
    if max_folds is not None:
        logger.info("max_folds=%d — running first %d of %d folds.",
                    max_folds, len(folds_to_run), k)

    per_fold_rows: list[dict] = []
    for fold_idx in folds_to_run:
        tag = f"kfold_{fold_idx:02d}"
        pred_path   = per_fold_dir / f"{tag}__predictions.parquet"
        metric_path = per_fold_dir / f"{tag}__metrics.json"

        if resume and pred_path.exists() and metric_path.exists():
            logger.info("[fold %d/%d] %s — cached, skipping.",
                        fold_idx + 1, k, tag)
            per_fold_rows.append(json.loads(metric_path.read_text()))
            continue

        n_train = sum(len(splits[m][fold_idx][0]) for m in splits)
        n_test  = sum(len(splits[m][fold_idx][1]) for m in splits)
        logger.info("[fold %d/%d] %s — train=%d, test=%d, training + predicting…",
                    fold_idx + 1, k, tag, n_train, n_test)
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix=f"kfold_{tag}_") as td:
            fold_pred = run_single_kfold_fold(
                fold_idx=fold_idx,
                adata_sd3p=adata_sd3p,
                adata_sdffpe=adata_sdffpe,
                adata_hd=adata_hd,
                splits_per_mod=splits,
                gene_pool=gene_pool,
                spec=spec,
                config=training_config,
                workdir=Path(td),
                section_col=section_col,
                theta=theta,
            )
        dt = time.time() - t0
        fold_pred.to_parquet(pred_path)

        fold_metrics = _score_fold(fold_pred, spec)
        fold_metrics.update({
            "fold_idx": fold_idx,
            "n_train":  int(n_train),
            "n_test":   int(len(fold_pred)),
            "time_sec": float(dt),
        })
        metric_path.write_text(json.dumps(fold_metrics, indent=2, default=float))
        per_fold_rows.append(fold_metrics)

        logger.info(
            "[fold %d/%d] %s done in %.1fs — coarse F1(w)=%.4f | "
            "fine F1(w) strict=%.4f / fallback_excl=%.4f / res_aware=%.4f",
            fold_idx + 1, k, tag, dt,
            fold_metrics["coarse_f1_weighted"],
            fold_metrics["fine_f1_weighted_strict"],
            fold_metrics["fine_f1_weighted_fallback_excluded"],
            fold_metrics["fine_f1_weighted_resolution_aware"],
        )

    parquet_paths = sorted(per_fold_dir.glob("*__predictions.parquet"))
    if not parquet_paths:
        raise RuntimeError(
            f"No per-fold predictions under {per_fold_dir}. Run at least one fold."
        )
    pooled_df = pd.concat([pd.read_parquet(p) for p in parquet_paths],
                          ignore_index=True)
    pooled_path = outdir / "pooled_predictions.csv.gz"
    pooled_df.to_csv(pooled_path, index=False, compression="gzip")
    logger.info("Pooled k-fold predictions → %s  (%d rows)",
                pooled_path, len(pooled_df))

    per_fold_df = pd.DataFrame(per_fold_rows) if per_fold_rows else pd.DataFrame()
    if not per_fold_df.empty:
        per_fold_df.to_csv(outdir / "per_fold_metrics.csv", index=False)

    return pooled_df, per_fold_df


__all__ = [
    "KfoldSplit",
    "select_kfold_splits",
    "run_single_kfold_fold",
    "run_kfold",
]
