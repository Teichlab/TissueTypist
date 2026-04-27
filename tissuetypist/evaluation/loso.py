"""
tissuetypist/evaluation/loso.py
===============================
Leave-one-section-out cross-validation for manuscript accuracy metrics.

This module implements
``scripts/17_loso_accuracy.py`` on top of the April 2026 library layout
(YAML-driven chain walker, ``train_all_models``, ``predict_adata``).

Public API
----------
    select_loso_folds(...)           — build the held-out roster.
    run_single_fold(...)             — train on train-sections, predict held-out.
    run_loso(...)                    — orchestrator, returns pooled DataFrame.
    aggregate_overall_metrics(...)   — per-modality weighted/macro/micro F1.
    aggregate_per_niche_metrics(...) — classification report at the fine level.
    build_confusion_matrices(...)    — coarse / fine / per-sub-model CMs.

All scoring reuses ``tissuetypist.evaluation.metrics.remap_ground_truth``
so the hierarchy's ``gt_label_remap`` is respected.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, replace as dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from tissuetypist.config.hierarchy import load_hierarchy
from tissuetypist.data.normalise import normalise_if_needed
from tissuetypist.prediction.hierarchical import predict_adata
from tissuetypist.training.hierarchical import TrainingConfig, train_all_models
from tissuetypist.evaluation.metrics import remap_ground_truth

if TYPE_CHECKING:
    import anndata as ad
    from tissuetypist.config.hierarchy import HierarchySpec

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

# Donor pins for the cardiac reference dataset: for donors with more sections
# than ``max_sections_per_donor``, pin the specific sections to use instead of
# random selection. Useful when a particular donor has sections with unusually
# broad niche coverage.
DEFAULT_DONOR_PINS = {
    "C83": ["HCAHeartST10317184", "HCAHeartST10317186"],
}

MOD_TO_PREDICT_ARG = {"SD_3prime": "sd", "SD_FFPE": "sd", "HD": "hd"}


# ──────────────────────────────────────────────────────────────────────────────
# Fold roster
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LosoFold:
    """A single held-out section for LOSO CV."""
    modality: str          # "SD_3prime" | "SD_FFPE" | "HD"
    section_id: str
    donor: str
    n_spots: int


def _select_for_modality(
    adata: "ad.AnnData",
    mod_name: str,
    section_col: str,
    max_per_donor: int,
    donor_pins: dict[str, list[str]],
    rng: np.random.Generator,
) -> list[LosoFold]:
    """Select held-out sections for a single modality."""
    folds: list[LosoFold] = []
    if adata is None or adata.n_obs == 0:
        return folds
    if "donor" not in adata.obs.columns:
        raise ValueError(f"{mod_name}: 'donor' column required in obs.")
    if section_col not in adata.obs.columns:
        raise ValueError(f"{mod_name}: section column '{section_col}' not in obs.")

    for donor in sorted(adata.obs["donor"].astype(str).unique()):
        donor_mask = adata.obs["donor"].astype(str) == donor
        donor_sections = sorted(
            adata.obs.loc[donor_mask, section_col].astype(str).unique()
        )

        if donor in donor_pins:
            pinned = [s for s in donor_pins[donor] if s in donor_sections]
            if pinned:
                chosen = pinned[:max_per_donor]
            else:
                logger.warning(
                    "Donor %s pinned to %s but none present in %s — falling "
                    "back to random selection.",
                    donor, donor_pins[donor], mod_name,
                )
                chosen = donor_sections[:max_per_donor]
        elif len(donor_sections) <= max_per_donor:
            chosen = donor_sections
        else:
            chosen = sorted(rng.choice(
                donor_sections, size=max_per_donor, replace=False
            ).tolist())

        for s in chosen:
            n = int((donor_mask & (adata.obs[section_col].astype(str) == s)).sum())
            folds.append(LosoFold(
                modality=mod_name, section_id=s, donor=donor, n_spots=n,
            ))
    return folds


def select_loso_folds(
    adata_sd3p: "ad.AnnData",
    adata_sdffpe: "ad.AnnData",
    adata_hd: "ad.AnnData",
    section_col: str = "section_ID",
    max_sections_per_donor: int = 2,
    donor_pins: Optional[dict[str, list[str]]] = None,
    seed: int = 42,
) -> list[LosoFold]:
    """Build the LOSO fold roster across all three modalities.

    For donors with more than ``max_sections_per_donor`` sections in a
    modality, the roster defaults to a random subsample (deterministic
    given ``seed``). Entries in ``donor_pins`` override the random subsample
    for that donor with the listed section IDs.

    Parameters
    ----------
    adata_sd3p, adata_sdffpe, adata_hd :
        The three training reference AnnDatas. Pass an empty AnnData for
        modalities not in use.
    section_col :
        ``obs`` column used to identify sections. Defaults to
        ``"section_ID"``.
    max_sections_per_donor :
        Per-modality cap on sections per donor in the roster.
    donor_pins :
        ``{donor: [section, ...]}`` — when present, replaces the random
        subsample for that donor. Missing donors fall back to random.
    seed :
        RNG seed for reproducible selection.

    Returns
    -------
    list[LosoFold]
        Held-out sections, sorted by (modality, donor, section_id).
    """
    rng = np.random.default_rng(seed)
    pins = dict(donor_pins or {})
    folds: list[LosoFold] = []
    for mod_name, adata in [
        ("SD_3prime", adata_sd3p),
        ("SD_FFPE",   adata_sdffpe),
        ("HD",        adata_hd),
    ]:
        folds.extend(_select_for_modality(
            adata, mod_name, section_col, max_sections_per_donor, pins, rng,
        ))
    folds.sort(key=lambda f: (f.modality, f.donor, f.section_id))
    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Fold training + prediction
# ──────────────────────────────────────────────────────────────────────────────

def _drop_section(adata: "ad.AnnData", section_col: str, section_id: str) -> "ad.AnnData":
    if adata.n_obs == 0:
        return adata
    keep = adata.obs[section_col].astype(str) != section_id
    return adata[keep].copy()


def run_single_fold(
    fold: LosoFold,
    adata_sd3p: "ad.AnnData",
    adata_sdffpe: "ad.AnnData",
    adata_hd: "ad.AnnData",
    gene_pool: list[str],
    spec: "HierarchySpec",
    config: TrainingConfig,
    workdir: Path,
    section_col: str = "section_ID",
    theta: float = 0.5,
) -> pd.DataFrame:
    """Run one LOSO fold end-to-end.

    Trains a full hierarchy on every modality *except* ``fold.section_id``
    (removed only from ``fold.modality`` — other modalities are used in
    full). Predicts on the held-out section via ``predict_adata`` and
    returns a DataFrame with one row per held-out spot including ground
    truth, ``tt_*`` columns, and fold metadata.

    Expects the input adatas to already be log-normalised. See
    ``run_loso`` which does this once up front.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # ── Build per-fold training adatas ──────────────────────────────────────
    if fold.modality == "SD_3prime":
        train_sd3p   = _drop_section(adata_sd3p,   section_col, fold.section_id)
        train_sdffpe = adata_sdffpe
        train_hd     = adata_hd
    elif fold.modality == "SD_FFPE":
        train_sd3p   = adata_sd3p
        train_sdffpe = _drop_section(adata_sdffpe, section_col, fold.section_id)
        train_hd     = adata_hd
    elif fold.modality == "HD":
        train_sd3p   = adata_sd3p
        train_sdffpe = adata_sdffpe
        train_hd     = _drop_section(adata_hd,     section_col, fold.section_id)
    else:
        raise ValueError(f"Unknown modality: {fold.modality!r}")

    # ── Train ────────────────────────────────────────────────────────────────
    train_all_models(
        adata_sd3p=train_sd3p,
        adata_sdffpe=train_sdffpe,
        adata_hd_windows=train_hd,
        genes_shared=gene_pool,
        outdir=workdir,
        config=config,
        qc_dir=None,
    )

    # ── Build held-out query adata ──────────────────────────────────────────
    src = {
        "SD_3prime": adata_sd3p,
        "SD_FFPE":   adata_sdffpe,
        "HD":        adata_hd,
    }[fold.modality]
    held_mask = src.obs[section_col].astype(str) == fold.section_id
    adata_query = src[held_mask].copy()

    # ── Predict ──────────────────────────────────────────────────────────────
    modality_arg = MOD_TO_PREDICT_ARG[fold.modality]
    adata_pred = predict_adata(
        adata_query,
        model_dir=workdir,
        modality=modality_arg,
        section_col=section_col,
        theta=theta,
    )

    # ── Assemble row-level pooled DataFrame ─────────────────────────────────
    obs = adata_pred.obs.copy()
    # Ground truth columns — guaranteed to exist in the Apr 2026 data.
    gt_coarse = obs[spec.coarse_col].astype(str).values
    gt_fine   = obs[spec.fine_col].astype(str).values

    # Collect all tt_* columns the prediction wrote.
    tt_cols = [c for c in obs.columns if c.startswith("tt_")]

    result = pd.DataFrame({
        "spot_id":           obs.index.astype(str),
        "modality":          fold.modality,
        "section_id":        fold.section_id,
        "donor":             fold.donor,
        "ground_truth_coarse": gt_coarse,
        "ground_truth_fine":   gt_fine,
    })
    for col in tt_cols:
        result[col] = obs[col].values
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-fold scoring (lightweight — used for log output and the per-fold CSV)
# ──────────────────────────────────────────────────────────────────────────────

def _score_fold(pred_df: pd.DataFrame, spec: "HierarchySpec") -> dict:
    """Produce a flat dict of per-fold F1 metrics for logging + per_fold_metrics.csv.

    Emits coarse F1 (mode-independent) plus fine F1 under every mode in
    :data:`SCORING_MODES`, as columns
    ``fine_f1_{weighted,macro}_{mode}`` and ``fine_n_spots_{mode}``.
    """
    obs = pred_df.dropna(subset=["tt_coarse_label", "tt_final_label"])
    if obs.empty:
        out = {
            "coarse_f1_weighted": float("nan"),
            "coarse_f1_macro":    float("nan"),
        }
        for m in SCORING_MODES:
            out[f"fine_f1_weighted_{m}"] = float("nan")
            out[f"fine_f1_macro_{m}"]    = float("nan")
            out[f"fine_n_spots_{m}"]     = 0
        return out

    gt_coarse   = obs["ground_truth_coarse"].astype(str)
    pred_coarse = obs["tt_coarse_label"].astype(str)
    gt_fine     = remap_ground_truth(obs["ground_truth_fine"].astype(str), spec)
    pred_fine   = obs["tt_final_label"].astype(str)

    out = {
        "coarse_f1_weighted": float(f1_score(gt_coarse, pred_coarse, average="weighted", zero_division=0)),
        "coarse_f1_macro":    float(f1_score(gt_coarse, pred_coarse, average="macro",    zero_division=0)),
    }
    for mode in SCORING_MODES:
        gt_m, pred_m = _apply_scoring_mode(gt_fine, pred_fine, spec, mode=mode)
        out[f"fine_n_spots_{mode}"] = int(len(gt_m))
        if len(gt_m) == 0:
            out[f"fine_f1_weighted_{mode}"] = 0.0
            out[f"fine_f1_macro_{mode}"]    = 0.0
        else:
            out[f"fine_f1_weighted_{mode}"] = float(f1_score(gt_m, pred_m, average="weighted", zero_division=0))
            out[f"fine_f1_macro_{mode}"]    = float(f1_score(gt_m, pred_m, average="macro",    zero_division=0))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_loso(
    adata_sd3p: "ad.AnnData",
    adata_sdffpe: "ad.AnnData",
    adata_hd: "ad.AnnData",
    gene_pool: list[str],
    outdir: Path,
    spec: Optional["HierarchySpec"] = None,
    training_config: Optional[TrainingConfig] = None,
    section_col: str = "section_ID",
    max_sections_per_donor: int = 2,
    donor_pins: Optional[dict[str, list[str]]] = None,
    seed: int = 42,
    theta: float = 0.5,
    max_folds: Optional[int] = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[LosoFold]]:
    """Run the full LOSO CV sweep and return pooled predictions + per-fold metrics.

    Per-fold predictions (``.parquet``) and per-fold metrics (``.json``)
    are persisted to ``{outdir}/per_fold/`` after each fold. When ``resume``
    is true (default) those cached outputs are reused so the sweep can be
    interrupted and restarted.

    Parameters
    ----------
    adata_sd3p, adata_sdffpe, adata_hd :
        The three training AnnDatas. Pass an empty AnnData (e.g. via
        ``tissuetypist.training.hierarchical._empty_adata``) for modalities
        not in use. Input counts may be raw — ``normalise_if_needed`` is
        applied once up front.
    gene_pool :
        Shared-gene universe (from the Phase-0 catalogue or the
        intersection of var_names — same semantics as
        ``tissuetypist.training.resolve_gene_pool``).
    outdir :
        Output directory. Created if missing. Will contain:
          - ``fold_roster.csv``
          - ``per_fold/{fold_tag}__predictions.parquet``
          - ``per_fold/{fold_tag}__metrics.json``
          - ``per_fold_metrics.csv`` (concatenated from the JSONs)
          - ``pooled_predictions.csv.gz`` (concatenated from the parquets)
    spec :
        HierarchySpec to train against. Defaults to the shipped ``cardiac``
        spec via ``load_hierarchy("cardiac")``.
    training_config :
        Base TrainingConfig. If ``None``, uses defaults
        (``feature_set="deg_hvg"``, weights 0.3/5.0). The passed config
        is copied with ``hierarchy=spec`` applied.
    max_folds :
        If set, only run the first N folds of the sorted roster. Useful
        for dry-runs.

    Returns
    -------
    pooled_df, per_fold_df, folds
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_fold_dir = outdir / "per_fold"
    per_fold_dir.mkdir(exist_ok=True)

    if spec is None:
        spec = load_hierarchy("cardiac")
    if training_config is None:
        training_config = TrainingConfig(hierarchy=spec, feature_set="deg_hvg")
    else:
        training_config = dataclass_replace(training_config, hierarchy=spec)

    # ── Normalise all three references ONCE ─────────────────────────────────
    # ``train_all_models`` expects log-normalised inputs; normalising a
    # subset is equivalent to normalising the whole (normalize_total is
    # per-spot), so doing it once saves a lot of work across folds.
    if adata_sd3p.n_obs:
        adata_sd3p = normalise_if_needed(adata_sd3p, "SD_3prime")
    if adata_sdffpe.n_obs:
        adata_sdffpe = normalise_if_needed(adata_sdffpe, "SD_FFPE")
    if adata_hd.n_obs:
        adata_hd = normalise_if_needed(adata_hd, "HD")

    # ── Roster ──────────────────────────────────────────────────────────────
    folds = select_loso_folds(
        adata_sd3p, adata_sdffpe, adata_hd,
        section_col=section_col,
        max_sections_per_donor=max_sections_per_donor,
        donor_pins=donor_pins if donor_pins is not None else DEFAULT_DONOR_PINS,
        seed=seed,
    )
    roster_df = pd.DataFrame([asdict(f) for f in folds])
    roster_df.to_csv(outdir / "fold_roster.csv", index=False)
    logger.info("LOSO roster: %d folds", len(folds))
    logger.info("\n%s", roster_df.to_string(index=False))

    if max_folds is not None:
        folds_to_run = folds[:max_folds]
        logger.info(
            "max_folds=%d — running first %d of %d folds.",
            max_folds, len(folds_to_run), len(folds),
        )
    else:
        folds_to_run = folds

    # ── Fold loop ───────────────────────────────────────────────────────────
    per_fold_rows: list[dict] = []

    import tempfile  # local import: only needed when we actually run folds

    for i, fold in enumerate(folds_to_run, 1):
        tag = _fold_tag(fold)
        pred_path   = per_fold_dir / f"{tag}__predictions.parquet"
        metric_path = per_fold_dir / f"{tag}__metrics.json"

        if resume and pred_path.exists() and metric_path.exists():
            logger.info("[%d/%d] %s — cached, skipping.",
                        i, len(folds_to_run), tag)
            per_fold_rows.append(json.loads(metric_path.read_text()))
            continue

        logger.info(
            "[%d/%d] %s (n_spots=%d) — training + predicting …",
            i, len(folds_to_run), tag, fold.n_spots,
        )
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix=f"loso_{tag}_") as td:
            fold_pred = run_single_fold(
                fold=fold,
                adata_sd3p=adata_sd3p,
                adata_sdffpe=adata_sdffpe,
                adata_hd=adata_hd,
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
            "modality":   fold.modality,
            "section_id": fold.section_id,
            "donor":      fold.donor,
            "n_spots":    int(len(fold_pred)),
            "time_sec":   float(dt),
        })
        metric_path.write_text(json.dumps(fold_metrics, indent=2, default=float))
        per_fold_rows.append(fold_metrics)

        logger.info(
            "[%d/%d] %s done in %.1fs — coarse F1(w)=%.4f | "
            "fine F1(w) strict=%.4f / fallback_excl=%.4f / res_aware=%.4f",
            i, len(folds_to_run), tag, dt,
            fold_metrics["coarse_f1_weighted"],
            fold_metrics["fine_f1_weighted_strict"],
            fold_metrics["fine_f1_weighted_fallback_excluded"],
            fold_metrics["fine_f1_weighted_resolution_aware"],
        )

    # ── Concatenate pooled predictions ──────────────────────────────────────
    parquet_paths = sorted(per_fold_dir.glob("*__predictions.parquet"))
    if not parquet_paths:
        raise RuntimeError(
            f"No per-fold predictions found under {per_fold_dir}. "
            "Run at least one fold before aggregation."
        )
    pooled_df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    pooled_path = outdir / "pooled_predictions.csv.gz"
    pooled_df.to_csv(pooled_path, index=False, compression="gzip")
    logger.info("Pooled predictions → %s  (%d rows)", pooled_path, len(pooled_df))

    per_fold_df = pd.DataFrame(per_fold_rows) if per_fold_rows else pd.DataFrame()
    if not per_fold_df.empty:
        per_fold_df.to_csv(outdir / "per_fold_metrics.csv", index=False)

    return pooled_df, per_fold_df, folds


def _fold_tag(fold: LosoFold) -> str:
    """File-safe fold identifier."""
    safe_section = fold.section_id.replace(" ", "_").replace("/", "_")
    return f"{fold.modality}__{fold.donor}__{safe_section}"


# ──────────────────────────────────────────────────────────────────────────────
# Label taxonomy (leaves / intermediates / families / legacy-obs mapping)
# ──────────────────────────────────────────────────────────────────────────────
#
# The pooled predictions contain three kinds of labels in ``tt_final_label``:
#
#   (a) Leaf labels — a stage's class that doesn't route to a deeper stage
#       and isn't the key of a pool_from pooling (e.g. "Sinus horn",
#       "Atrium - Left", "Ventricle - Compact", plus the terminal coarse
#       labels "Epicardial region" and "Lymph node").
#   (b) Intermediate labels — appear when the chain-walker's confidence
#       drops below theta and it stops at a parent class (e.g. "Sinoatrial
#       region", "Atrial myocardium", "Great vessels", or the bare coarse
#       niche "Atrium" / "Ventricle" / etc. for Stage-1 fallbacks).
#   (c) Legacy intermediate-in-obs labels — only present in the GT column,
#       never predicted. Cardiac example: the SD 3' data carries
#       "Sinoatrial region - non-terminal category" as a bona-fide fine
#       label for spots we know are in the sinoatrial region but haven't
#       been resolved to a leaf sub-niche. These are declared via
#       ``pool_from`` entries in the YAML.
#
# Strict scoring treats (b) and (c) as class mismatches. For the
# manuscript we also report:
#
#   fallback_excluded   — drop spots with prediction of type (b).
#   resolution_aware    — evaluate GT-intermediate spots *at the
#                         intermediate level* (roll pred up if it falls in
#                         the intermediate's family); for GT-leaf spots
#                         still drop type-(b) predictions.
#
# ``_build_label_taxonomy`` derives everything needed for these modes
# from the :class:`HierarchySpec`. It uses only the public fields
# (``coarse_niches``, ``terminal_coarse``, ``sub_models.*.stages.*``) so
# it generalises to any user-supplied hierarchy.


SCORING_MODES = ("strict", "fallback_excluded", "resolution_aware")


def _build_label_taxonomy(spec: "HierarchySpec") -> dict:
    """Derive leaf / intermediate / family / legacy-obs maps from ``spec``.

    Returns a dict with keys:

    ``"leaves"`` — set of labels that are terminal in the hierarchy and
      can legitimately appear as ``tt_final_label`` for a confident
      prediction. Includes ``spec.terminal_coarse`` plus every stage class
      that isn't routed to a deeper stage and isn't a ``pool_from`` key,
      plus ``pool_from`` values that are themselves proper stage classes.

    ``"intermediates"`` — set of labels that represent a hierarchy level
      above a leaf. Includes every class in any stage's
      ``route_classes_to_next``, every ``pool_from`` key, and every
      non-terminal coarse niche.

    ``"families"`` — ``{intermediate: {labels_considered_equivalent}}``.
      A family is the intermediate itself + all labels at stages beneath
      it + all ``pool_from`` values at any stage under it. Used by
      resolution-aware scoring to decide whether a leaf prediction falls
      under a GT intermediate.

    ``"legacy_map"`` — ``{obs_label: canonical_intermediate}``. A
      ``pool_from`` value that isn't itself a stage class (e.g.
      ``"Sinoatrial region - non-terminal category"``) is treated as a
      legacy obs-only label that should be mapped to its ``pool_from`` key
      before evaluation.
    """
    all_stage_classes:   set[str] = set()
    routed_classes:      set[str] = set()
    pool_keys:           set[str] = set()
    pool_values_by_key:  dict[str, set[str]] = {}

    for parent_coarse, sm in spec.sub_models.items():
        for stage in sm.stages:
            all_stage_classes.update(stage.classes)
            routed_classes.update(stage.route_classes_to_next)
            if stage.pool_from:
                for pk, pv in stage.pool_from.items():
                    pool_keys.add(pk)
                    pool_values_by_key.setdefault(pk, set()).update(pv)

    # Leaves: terminal coarse + stage classes that are neither routed nor
    # pool-keys + pool_from values that ARE proper stage classes.
    leaves: set[str] = set(spec.terminal_coarse)
    for cls in all_stage_classes:
        if cls not in routed_classes and cls not in pool_keys:
            leaves.add(cls)
    for pvs in pool_values_by_key.values():
        for v in pvs:
            if v in all_stage_classes and v not in routed_classes and v not in pool_keys:
                leaves.add(v)

    # Intermediates: routed + pool keys + non-terminal coarse niches.
    intermediates: set[str] = set(routed_classes) | set(pool_keys)
    terminal_coarse_set = set(spec.terminal_coarse)
    for c in spec.coarse_niches:
        if c not in terminal_coarse_set:
            intermediates.add(c)

    # Families. Walk each sub-model to accumulate descendants.
    families: dict[str, set[str]] = {}
    for parent_coarse, sm in spec.sub_models.items():
        # Family of the non-terminal coarse niche = everything in its sub-model.
        fam_coarse: set[str] = {parent_coarse}
        for stage in sm.stages:
            fam_coarse.update(stage.classes)
            if stage.pool_from:
                for pv in stage.pool_from.values():
                    fam_coarse.update(pv)
        families[parent_coarse] = fam_coarse

        # Family of each intermediate within the chain.
        stages = sm.stages
        for i, stage in enumerate(stages):
            for cls in stage.classes:
                if cls not in routed_classes and cls not in pool_keys:
                    continue
                fam: set[str] = {cls}
                if stage.pool_from and cls in stage.pool_from:
                    fam.update(stage.pool_from[cls])
                if cls in routed_classes:
                    for j in range(i + 1, len(stages)):
                        fam.update(stages[j].classes)
                        if stages[j].pool_from:
                            for pv in stages[j].pool_from.values():
                                fam.update(pv)
                families.setdefault(cls, set()).update(fam)

    # Legacy obs-only labels: pool_from values that aren't stage classes.
    legacy_map: dict[str, str] = {}
    for pk, pvs in pool_values_by_key.items():
        for v in pvs:
            if v not in all_stage_classes and v not in leaves:
                legacy_map[v] = pk

    return {
        "leaves":        leaves,
        "intermediates": intermediates,
        "families":      families,
        "legacy_map":    legacy_map,
    }


def build_leaf_to_coarse_map(spec: "HierarchySpec") -> dict[str, str]:
    """Return a ``{label: parent_coarse_niche}`` map for every label in the hierarchy.

    Useful for colouring per-niche plots by parent coarse niche so the
    hierarchy structure is visible at a glance.

    - Each coarse niche maps to itself (including terminal coarse).
    - Every class in every sub-model stage (plus pool_from keys and
      pool_from values) maps to the sub-model's ``parent_coarse``.

    If a label appears under more than one coarse niche in the spec
    (which is structurally unusual), the first encountered parent wins.
    """
    result: dict[str, str] = {}
    for c in spec.coarse_niches:
        result[c] = c
    for parent_coarse, sm in spec.sub_models.items():
        for stage in sm.stages:
            for cls in stage.classes:
                result.setdefault(cls, parent_coarse)
            if stage.pool_from:
                for pk, pvs in stage.pool_from.items():
                    result.setdefault(pk, parent_coarse)
                    for v in pvs:
                        result.setdefault(v, parent_coarse)
    return result


def _apply_scoring_mode(
    y_true: pd.Series,
    y_pred: pd.Series,
    spec: "HierarchySpec",
    mode: str,
) -> tuple[pd.Series, pd.Series]:
    """Apply the scoring-mode transformations before computing metrics.

    See ``SCORING_MODES`` for the supported values and the module-level
    taxonomy docstring for what each mode does. For ``"strict"`` this is
    an identity function.
    """
    if mode not in SCORING_MODES:
        raise ValueError(f"Unknown scoring mode: {mode!r}. Expected one of {SCORING_MODES}.")
    if mode == "strict":
        return y_true, y_pred

    tax = _build_label_taxonomy(spec)

    if mode == "fallback_excluded":
        keep = y_pred.isin(tax["leaves"])
        return y_true[keep], y_pred[keep]

    # resolution_aware
    legacy_map    = tax["legacy_map"]
    intermediates = tax["intermediates"]
    families      = tax["families"]
    leaves        = tax["leaves"]

    y_true_canon = y_true.map(lambda x: legacy_map.get(x, x))
    is_inter_gt  = y_true_canon.isin(intermediates)

    y_pred_out = y_pred.copy()
    if is_inter_gt.any():
        for inter in y_true_canon[is_inter_gt].unique():
            fam = families.get(inter, {inter})
            mask = (y_true_canon == inter) & y_pred_out.isin(fam)
            y_pred_out[mask] = inter

    is_leaf_gt = ~is_inter_gt
    keep = is_inter_gt | (is_leaf_gt & y_pred_out.isin(leaves))
    return y_true_canon[keep], y_pred_out[keep]


# ──────────────────────────────────────────────────────────────────────────────
# Aggregates used by the plots + manuscript tables
# ──────────────────────────────────────────────────────────────────────────────

def _score_coarse_row(sub: pd.DataFrame) -> dict:
    """Coarse-level F1 row (mode-independent — no fallback/intermediate issue)."""
    gt   = sub["ground_truth_coarse"].astype(str)
    pred = sub["tt_coarse_label"].astype(str)
    return {
        "coarse_n_spots":      int(len(sub)),
        "coarse_f1_weighted":  float(f1_score(gt, pred, average="weighted", zero_division=0)),
        "coarse_f1_macro":     float(f1_score(gt, pred, average="macro",    zero_division=0)),
        "coarse_f1_micro":     float(f1_score(gt, pred, average="micro",    zero_division=0)),
    }


def _score_fine_row(
    sub: pd.DataFrame,
    spec: "HierarchySpec",
    mode: str,
) -> dict:
    """Fine-level F1 row for the given scoring mode."""
    gt   = remap_ground_truth(sub["ground_truth_fine"].astype(str), spec)
    pred = sub["tt_final_label"].astype(str)
    gt_m, pred_m = _apply_scoring_mode(gt, pred, spec, mode=mode)
    if len(gt_m) == 0:
        return {
            "fine_n_spots":     0,
            "fine_f1_weighted": 0.0,
            "fine_f1_macro":    0.0,
            "fine_f1_micro":    0.0,
        }
    return {
        "fine_n_spots":     int(len(gt_m)),
        "fine_f1_weighted": float(f1_score(gt_m, pred_m, average="weighted", zero_division=0)),
        "fine_f1_macro":    float(f1_score(gt_m, pred_m, average="macro",    zero_division=0)),
        "fine_f1_micro":    float(f1_score(gt_m, pred_m, average="micro",    zero_division=0)),
    }


def aggregate_overall_metrics(
    pooled_df: pd.DataFrame,
    spec: "HierarchySpec",
    modes: tuple[str, ...] = SCORING_MODES,
) -> pd.DataFrame:
    """Long-form per-modality × scoring_mode F1 table.

    One row per (modality, scoring_mode). Coarse scores are
    mode-independent and repeated across modes for convenience.
    Modalities: ``"SD_3prime"``, ``"SD_FFPE"``, ``"HD"``, ``"all"`` (the
    pooled total).
    """
    rows: list[dict] = []
    modalities = ["SD_3prime", "SD_FFPE", "HD", "all"]
    for mod in modalities:
        sub = pooled_df if mod == "all" else pooled_df[pooled_df["modality"] == mod]
        sub = sub.dropna(subset=["tt_coarse_label", "tt_final_label"])
        if sub.empty:
            continue
        coarse_row = _score_coarse_row(sub)
        for mode in modes:
            fine_row = _score_fine_row(sub, spec, mode)
            rows.append({
                "modality":       mod,
                "scoring_mode":   mode,
                "n_spots_pooled": int(len(sub)),
                **coarse_row,
                **fine_row,
            })
    return pd.DataFrame(rows)


def aggregate_per_niche_metrics(
    pooled_df: pd.DataFrame,
    spec: "HierarchySpec",
    mode: str = "resolution_aware",
) -> pd.DataFrame:
    """Per-niche precision / recall / F1 from the pooled fine predictions.

    Ground-truth labels are remapped via ``spec.gt_label_remap`` and
    through the scoring-mode transformations (see ``SCORING_MODES``).
    """
    sub = pooled_df.dropna(subset=["tt_final_label", "ground_truth_fine"])
    gt   = remap_ground_truth(sub["ground_truth_fine"].astype(str), spec)
    pred = sub["tt_final_label"].astype(str)
    gt_m, pred_m = _apply_scoring_mode(gt, pred, spec, mode=mode)
    if len(gt_m) == 0:
        return pd.DataFrame(columns=["niche", "precision", "recall", "f1-score", "support"])
    report = classification_report(gt_m, pred_m, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T
    df.index.name = "niche"
    return df.reset_index()


def build_confusion_matrices(
    pooled_df: pd.DataFrame,
    spec: "HierarchySpec",
    mode: str = "resolution_aware",
) -> dict[str, pd.DataFrame]:
    """Compute row-normalised confusion matrices.

    Returns a dict with keys:
      - ``"coarse"`` — coarse-level CM (mode-independent).
      - ``"fine"``   — fine-level CM after applying ``mode``.
      - ``"submodel__{parent_coarse}"`` — per parent-coarse-niche CM
        (rows filtered to spots with that ground-truth coarse label),
        also after applying ``mode``.

    Raw-count CMs can be recovered via ``sklearn.metrics.confusion_matrix``
    on the pooled DataFrame directly if needed.
    """
    out: dict[str, pd.DataFrame] = {}

    # Coarse (mode-independent).
    sub = pooled_df.dropna(subset=["tt_coarse_label"])
    if not sub.empty:
        gt     = sub["ground_truth_coarse"].astype(str)
        pred   = sub["tt_coarse_label"].astype(str)
        labels = sorted(set(gt) | set(pred))
        out["coarse"] = _cm_df(gt, pred, labels)

    # Fine (mode-aware).
    sub = pooled_df.dropna(subset=["tt_final_label", "ground_truth_fine"])
    if not sub.empty:
        gt   = remap_ground_truth(sub["ground_truth_fine"].astype(str), spec)
        pred = sub["tt_final_label"].astype(str)
        gt_m, pred_m = _apply_scoring_mode(gt, pred, spec, mode=mode)
        if len(gt_m) > 0:
            labels = sorted(set(gt_m) | set(pred_m))
            out["fine"] = _cm_df(gt_m, pred_m, labels)

    # Per-parent-coarse (sub-model scope), mode-aware.
    for parent_coarse, sub_model in spec.sub_models.items():
        if parent_coarse in spec.terminal_coarse:
            continue
        scoped = pooled_df[
            (pooled_df["ground_truth_coarse"].astype(str) == parent_coarse)
            & pooled_df["tt_final_label"].notna()
            & pooled_df["ground_truth_fine"].notna()
        ]
        if scoped.empty:
            continue
        gt   = remap_ground_truth(scoped["ground_truth_fine"].astype(str), spec)
        pred = scoped["tt_final_label"].astype(str)
        gt_m, pred_m = _apply_scoring_mode(gt, pred, spec, mode=mode)
        if len(gt_m) == 0:
            continue
        labels = sorted(set(gt_m) | set(pred_m))
        out[f"submodel__{parent_coarse}"] = _cm_df(gt_m, pred_m, labels)

    return out


def _cm_df(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    arr = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return pd.DataFrame(arr, index=labels, columns=labels)


__all__ = [
    "LosoFold",
    "DEFAULT_DONOR_PINS",
    "SCORING_MODES",
    "select_loso_folds",
    "run_single_fold",
    "run_loso",
    "aggregate_overall_metrics",
    "aggregate_per_niche_metrics",
    "build_confusion_matrices",
    "build_leaf_to_coarse_map",
]
