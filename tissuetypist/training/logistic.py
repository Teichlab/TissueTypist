"""
tissuetypist.training.logistic
================================
Logistic regression training, cross-validation, and evaluation
for TissueTypist.

Supports:
- Stratified k-fold CV (within combined data)
- Leave-one-modality-out CV (cross-modality transfer test)
- Coarse and fine-grained label levels
- L1, L2, and elastic net regularisation

Typical usage
-------------
>>> from tissuetypist.training.logistic import train_and_evaluate
>>> results = train_and_evaluate(
...     X, y,
...     modality_labels=adata.obs["modality"],
...     cv_strategy="stratified_kfold",
... )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Default LR hyperparameters.
# solver is chosen per-call based on penalty (see train_logistic):
#   lbfgs  — L2/none: fast convergence on large p, multinomial native
#   saga   — L1/elasticnet: only solver that supports these penalties
LR_DEFAULTS = dict(
    C=1.0,
    max_iter=1000,
)


# ── Core training functions ───────────────────────────────────────────────────

def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    penalty: str = "l2",
    C: float = 1.0,
    l1_ratio: Optional[float] = None,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a logistic regression classifier.

    Parameters
    ----------
    X_train :
        Feature matrix, shape (n_samples, n_features).
    y_train :
        Label array, shape (n_samples,). Can be strings or integers.
    penalty :
        Regularisation type: ``"l1"``, ``"l2"``, or ``"elasticnet"``.
        Default ``"l2"``.
    C :
        Inverse regularisation strength. Smaller = stronger regularisation.
        Default 1.0.
    l1_ratio :
        Elastic net mixing parameter (0 = L2, 1 = L1). Only used when
        ``penalty="elasticnet"``. Default None.
    class_weight :
        ``"balanced"`` adjusts for class imbalance. Default ``None``
        (no reweighting). 
    random_state :
        Random seed. Default 42.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline (StandardScaler → LogisticRegression).
    """
    # Pick solver based on penalty: lbfgs for L2/none, saga for L1/elasticnet
    solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"

    kwargs = dict(
        **LR_DEFAULTS,
        solver=solver,
        class_weight=class_weight,
        random_state=random_state,
    )
    kwargs["C"] = C
    # Translate penalty string → sklearn >= 1.8 l1_ratio API
    if penalty is None or penalty == "none":
        import math
        kwargs["C"] = math.inf
    elif penalty == "l2":
        kwargs["l1_ratio"] = 0.0
    elif penalty == "l1":
        kwargs["l1_ratio"] = 1.0
    elif penalty == "elasticnet":
        kwargs["l1_ratio"] = l1_ratio if l1_ratio is not None else 0.5
    else:
        raise ValueError(f"Unknown penalty: {penalty!r}")

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr",     LogisticRegression(**kwargs)),
    ])
    model.fit(X_train, y_train)
    return model


# ── Cross-validation ──────────────────────────────────────────────────────────

def stratified_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = 42,
    max_cells_cv: int = 100_000,
) -> dict:
    """
    Stratified k-fold cross-validation.

    Parameters
    ----------
    X :
        Feature matrix, shape (n_samples, n_features).
    y :
        Label array, shape (n_samples,).
    n_splits :
        Number of CV folds. Default 5.
    penalty, C, class_weight, random_state :
        Passed to ``train_logistic``.
    max_cells_cv :
        If > 0 and n_samples exceeds this value, subsample to this many
        cells (stratified) before running CV. Default 100,000.

    Returns
    -------
    dict with keys:
        ``"fold_results"``    : list of per-fold result dicts
        ``"mean_f1"``         : mean weighted F1 across folds
        ``"std_f1"``          : std of weighted F1
        ``"mean_accuracy"``   : mean accuracy across folds
        ``"per_class_f1"``    : DataFrame of per-class F1 per fold
        ``"confusion_matrix"``  : summed confusion matrix across all folds
        ``"classes"``         : label classes
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    if max_cells_cv > 0 and len(y) > max_cells_cv:
        n_orig = len(y)
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=max_cells_cv, random_state=random_state
        )
        sub_idx, _ = next(sss.split(X, y))
        X = X[sub_idx]
        y = y[sub_idx]
        logger.info(
            "stratified_kfold_cv: subsampled %d → %d cells (max_cells_cv=%d)",
            n_orig, max_cells_cv, max_cells_cv,
        )

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    fold_results = []
    all_preds = np.zeros_like(y_enc)
    all_true  = np.zeros_like(y_enc)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        model = train_logistic(
            X_train, y_train,
            penalty=penalty, C=C,
            class_weight=class_weight,
            random_state=random_state,
        )
        y_pred = model.predict(X_val)

        f1       = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        accuracy = accuracy_score(y_val, y_pred)

        per_class = f1_score(
            y_val, y_pred, average=None,
            labels=list(range(len(classes))),
            zero_division=0,
        )

        fold_results.append({
            "fold":          fold + 1,
            "f1_weighted":   f1,
            "accuracy":      accuracy,
            "per_class_f1":  dict(zip(classes, per_class)),
            "n_val":         len(val_idx),
        })
        all_preds[val_idx] = y_pred
        all_true[val_idx]  = y_enc[val_idx]

        logger.info(
            "Fold %d/%d — weighted F1: %.4f | accuracy: %.4f",
            fold + 1, n_splits, f1, accuracy,
        )

    f1_scores = [r["f1_weighted"] for r in fold_results]
    mean_f1   = np.mean(f1_scores)
    std_f1    = np.std(f1_scores)

    per_class_df = pd.DataFrame(
        [r["per_class_f1"] for r in fold_results],
        index=[f"fold_{r['fold']}" for r in fold_results],
    )

    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(classes))))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    logger.info(
        "Stratified %d-fold CV — mean weighted F1: %.4f ± %.4f",
        n_splits, mean_f1, std_f1,
    )

    return {
        "fold_results":     fold_results,
        "mean_f1":          mean_f1,
        "std_f1":           std_f1,
        "mean_accuracy":    np.mean([r["accuracy"] for r in fold_results]),
        "per_class_f1":     per_class_df,
        "confusion_matrix": cm_df,
        "classes":          classes,
    }


def leave_one_modality_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    modality_labels: np.ndarray,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> dict:
    """
    Leave-one-modality-out cross-validation (cross-modality transfer test).

    For each modality, trains on the other two and evaluates on the held-out
    one. This is the primary test of cross-modality generalisation.

    Parameters
    ----------
    X :
        Feature matrix, shape (n_samples, n_features).
    y :
        Label array.
    modality_labels :
        Array of modality names per sample (e.g. ``["SD_3prime", "HD_FFPE", ...]``).
    penalty, C, class_weight, random_state :
        Passed to ``train_logistic``.

    Returns
    -------
    dict
        Keys = modality names (held-out), values = result dicts with
        ``f1_weighted``, ``accuracy``, ``per_class_f1``,
        ``confusion_matrix``, ``classes``.
    """
    modalities = np.unique(modality_labels)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    results = {}

    for held_out in modalities:
        train_mask = modality_labels != held_out
        val_mask   = modality_labels == held_out

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y_enc[train_mask], y_enc[val_mask]

        if len(X_val) == 0:
            logger.warning("Modality '%s' has no samples — skipping.", held_out)
            continue

        # Check that all val classes are present in training
        val_classes  = set(np.unique(y_val))
        train_classes = set(np.unique(y_train))
        unseen = val_classes - train_classes
        if unseen:
            logger.warning(
                "Held-out modality '%s': %d classes not seen in training "
                "(HD-exclusive niches). These will be scored as 0 F1.",
                held_out, len(unseen),
            )

        model = train_logistic(
            X_train, y_train,
            penalty=penalty, C=C,
            class_weight=class_weight,
            random_state=random_state,
        )
        y_pred = model.predict(X_val)

        f1       = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        accuracy = accuracy_score(y_val, y_pred)
        per_class = f1_score(
            y_val, y_pred, average=None,
            labels=list(range(len(classes))),
            zero_division=0,
        )
        cm = confusion_matrix(
            y_val, y_pred, labels=list(range(len(classes)))
        )

        logger.info(
            "Leave-%s-out — weighted F1: %.4f | accuracy: %.4f "
            "(train n=%d, val n=%d)",
            held_out, f1, accuracy, len(X_train), len(X_val),
        )

        results[held_out] = {
            "f1_weighted":    f1,
            "accuracy":       accuracy,
            "per_class_f1":   dict(zip(classes, per_class)),
            "confusion_matrix": pd.DataFrame(cm, index=classes, columns=classes),
            "classes":        classes,
            "n_train":        len(X_train),
            "n_val":          len(X_val),
        }

    return results


# ── Convenience wrapper ───────────────────────────────────────────────────────

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    modality_labels: Optional[np.ndarray] = None,
    cv_strategy: str = "both",
    n_splits: int = 5,
    penalty: str = "l2",
    C: float = 1.0,
    experiment_id: str = "",
) -> dict:
    """
    Train and evaluate a logistic regression model.

    Parameters
    ----------
    X :
        Feature matrix, shape (n_samples, n_features).
    y :
        Label array.
    modality_labels :
        Modality per sample — required for ``"lomo"`` and ``"both"``
        strategies.
    cv_strategy :
        ``"stratified_kfold"`` — within-data CV only.
        ``"lomo"`` — leave-one-modality-out only.
        ``"both"`` — run both (default).
    n_splits :
        Folds for stratified k-fold. Default 5.
    penalty, C :
        Regularisation settings.
    experiment_id :
        Optional label for logging (e.g. ``"F1_coarse"``).

    Returns
    -------
    dict with keys:
        ``"experiment_id"``
        ``"cv_results"``   (if stratified_kfold or both)
        ``"lomo_results"`` (if lomo or both)
        ``"feature_dim"``
        ``"n_samples"``
        ``"n_classes"``
    """
    label = f"[{experiment_id}] " if experiment_id else ""
    logger.info(
        "%sTraining LR: %d samples, %d features, %d classes, "
        "penalty=%s, C=%.3f, strategy=%s",
        label, len(y), X.shape[1],
        len(np.unique(y)), penalty, C, cv_strategy,
    )

    output = {
        "experiment_id": experiment_id,
        "feature_dim":   X.shape[1],
        "n_samples":     len(y),
        "n_classes":     len(np.unique(y)),
    }

    if cv_strategy in ("stratified_kfold", "both"):
        logger.info("%sRunning stratified %d-fold CV...", label, n_splits)
        output["cv_results"] = stratified_kfold_cv(
            X, y, n_splits=n_splits, penalty=penalty, C=C,
        )

    if cv_strategy in ("lomo", "both"):
        if modality_labels is None:
            raise ValueError(
                "modality_labels is required for leave-one-modality-out CV."
            )
        logger.info("%sRunning leave-one-modality-out CV...", label)
        output["lomo_results"] = leave_one_modality_out_cv(
            X, y, modality_labels=modality_labels,
            penalty=penalty, C=C,
        )

    return output


# ── Results formatting ────────────────────────────────────────────────────────

def format_cv_summary(results: dict) -> pd.DataFrame:
    """
    Format train_and_evaluate output into a tidy summary DataFrame.

    Parameters
    ----------
    results :
        Output of ``train_and_evaluate``.

    Returns
    -------
    pd.DataFrame
        One row per evaluation split with columns:
        experiment_id, cv_type, split, f1_weighted, accuracy.
    """
    rows = []
    exp_id = results.get("experiment_id", "")

    if "cv_results" in results:
        cv = results["cv_results"]
        for fold_r in cv["fold_results"]:
            rows.append({
                "experiment_id": exp_id,
                "cv_type":       "stratified_kfold",
                "split":         f"fold_{fold_r['fold']}",
                "f1_weighted":   fold_r["f1_weighted"],
                "accuracy":      fold_r["accuracy"],
                "n_val":         fold_r["n_val"],
            })
        rows.append({
            "experiment_id": exp_id,
            "cv_type":       "stratified_kfold",
            "split":         "mean",
            "f1_weighted":   cv["mean_f1"],
            "accuracy":      cv["mean_accuracy"],
            "n_val":         None,
        })

    if "lomo_results" in results:
        for modality, lomo_r in results["lomo_results"].items():
            rows.append({
                "experiment_id": exp_id,
                "cv_type":       "leave_one_modality_out",
                "split":         f"held_out_{modality}",
                "f1_weighted":   lomo_r["f1_weighted"],
                "accuracy":      lomo_r["accuracy"],
                "n_val":         lomo_r["n_val"],
            })

    return pd.DataFrame(rows)


# ── Phase 2: weighted neighbourhood pipeline ──────────────────────────────────

class AmplifyTransformer(BaseEstimator, TransformerMixin):
    """
    Multiply all feature values by a scalar factor.

    Used as the final step in the neighbour-feature sub-pipeline to apply
    the neighbourhood weight, and directly on ``distance_to_edge`` to apply
    the edge weight.  A factor of 0.0 effectively zeroes out the feature
    group (equivalent to not using it).

    Parameters
    ----------
    factor :
        Multiplicative scale factor. Default 1.0.
    """

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def fit(self, X, y=None):
        self.is_fitted_ = True    # required by sklearn >= 1.8 check_is_fitted
        return self

    def transform(self, X):
        return X * self.factor


def build_weighted_pipeline(
    own_features: list[str],
    neighbour_features: list[str],
    neighbour_weight: float,
    edge_weight: float,
) -> Pipeline:
    """
    Build a weighted ColumnTransformer → LogisticRegression Pipeline.

    Feature preprocessing:

    - **own gene features**:       ``StandardScaler()``
    - **neighbour-max features**:  ``StandardScaler()`` →
                                   ``AmplifyTransformer(neighbour_weight)``
    - **distance_to_edge**:        ``AmplifyTransformer(edge_weight)``
                                   (no scaling — already normalised + log1p)

    Classifier: ``LR(lbfgs, C=0.01, max_iter=500, class_weight=None)``.
    C=0.01 provides stronger regularisation appropriate for the high-dimensional
    spatial feature space.  class_weight=None and max_iter=500 are the
    verified baseline for the cardiac training set.

    Parameters
    ----------
    own_features :
        Column names of ``<gene>_own`` features.
    neighbour_features :
        Column names of ``<gene>_neighbour-max`` features.
    neighbour_weight :
        Amplification factor for neighbour features (0.0 = no neighbourhood).
    edge_weight :
        Amplification factor for distance_to_edge (0 = ignore edge).

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    preprocessor = ColumnTransformer([
        ("own_scale",
         StandardScaler(),
         own_features),
        ("neigh_scale_weight",
         Pipeline([
             ("scale",  StandardScaler()),
             ("weight", AmplifyTransformer(factor=neighbour_weight)),
         ]),
         neighbour_features),
        ("edge_amp",
         AmplifyTransformer(factor=edge_weight),
         ["distance_to_edge"]),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(
            solver="lbfgs",
            C=0.01,
            max_iter=500,
        )),
    ])


def run_weighted_cv(
    data: pd.DataFrame,
    own_features: list[str],
    neighbour_features: list[str],
    neighbour_weight: float,
    edge_weight: float,
    niche_col: str,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified k-fold CV with a weighted neighbourhood pipeline.

    Both overall metrics and per-class F1 are computed inside a single
    ``cross_validate`` call using ``make_scorer`` — one scorer per niche
    class.  This yields one row per fold (not one averaged row), enabling
    Kruskal–Wallis statistical testing per niche in the analysis notebook.

    Mirrors the v1 ``cross_validation()`` function in train.py exactly.

    Parameters
    ----------
    data :
        DataFrame containing ``own_features``, ``neighbour_features``,
        ``"distance_to_edge"``, and ``niche_col`` columns.
    own_features :
        Column names of ``<gene>_own`` features.
    neighbour_features :
        Column names of ``<gene>_neighbour-max`` features.
    neighbour_weight :
        Amplification factor for neighbour features.
    edge_weight :
        Amplification factor for distance_to_edge.
    niche_col :
        Column in ``data`` containing niche/class labels.
    n_splits :
        Number of CV folds. Default 5.

    Returns
    -------
    df_folds : pd.DataFrame
        Per-fold overall metrics. Shape: (n_splits, 6).
        Columns: ``fold``, ``neighbour_weight``, ``edge_weight``,
        ``accuracy``, ``balanced_accuracy``, ``f1_weighted``.

    df_f1_per_class : pd.DataFrame
        Per-fold per-class F1. Shape: (n_splits, n_classes + 3).
        Columns: ``f1_class_<niche>`` × n_classes, ``fold``,
        ``neighbour_weight``, ``edge_weight``.
        One row per fold — suitable for Kruskal–Wallis testing.
    """
    from sklearn.metrics import make_scorer

    pipeline = build_weighted_pipeline(
        own_features, neighbour_features, neighbour_weight, edge_weight
    )

    feature_cols = own_features + neighbour_features + ["distance_to_edge"]
    X_df    = data[feature_cols]
    y       = data[niche_col].astype(str).values
    classes = sorted(np.unique(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ── Build scoring dict: overall metrics + one make_scorer per class ───────
    # Identical approach to v1 cross_validation() in train.py:
    #   f1_class_{cls}: make_scorer(f1_score, labels=[cls], average="macro")
    scoring = {
        "accuracy":          "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_weighted":       "f1_weighted",
        **{
            f"f1_class_{cls}": make_scorer(
                f1_score, labels=[cls], average="macro", zero_division=0
            )
            for cls in classes
        },
    }

    # ── Single cross_validate call — returns per-fold scores for everything ───
    cv = cross_validate(
        pipeline, X_df, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
    )

    # ── df_folds: overall metrics, one row per fold ───────────────────────────
    df_folds = pd.DataFrame({
        "fold":              np.arange(1, n_splits + 1),
        "neighbour_weight":  neighbour_weight,
        "edge_weight":       edge_weight,
        "accuracy":          cv["test_accuracy"],
        "balanced_accuracy": cv["test_balanced_accuracy"],
        "f1_weighted":       cv["test_f1_weighted"],
    })

    # ── df_f1_per_class: per-class F1, one row per fold ───────────────────────
    class_data = {
        f"f1_class_{cls}": cv[f"test_f1_class_{cls}"]
        for cls in classes
    }
    df_f1_per_class = pd.DataFrame(class_data)
    df_f1_per_class.insert(0, "fold",             np.arange(1, n_splits + 1))
    df_f1_per_class["neighbour_weight"] = neighbour_weight
    df_f1_per_class["edge_weight"]      = edge_weight

    logger.info(
        "Weighted CV (nw=%.1f, ew=%d): mean weighted F1 = %.4f ± %.4f",
        neighbour_weight, edge_weight,
        cv["test_f1_weighted"].mean(),
        cv["test_f1_weighted"].std(),
    )

    return df_folds, df_f1_per_class
