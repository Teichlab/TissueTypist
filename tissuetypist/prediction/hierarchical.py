"""
tissuetypist/prediction/hierarchical.py
========================================
TissueTypist — hierarchical two-stage prediction.

Prediction proceeds in two stages:

  Stage 1: coarse LR model → 7-class prediction + coarse_score
  Stage 2: fine-grained sub-classifier per coarse niche → fine_label + fine_score

Confidence gating
-----------------
  - Terminal niches (Epicardium, Lymph node) bypass Stage 2 entirely (hard rule).
  - Non-terminal spots with coarse_score < theta are returned with coarse label
    only and tt_low_conf=True (soft rule, tunable via theta parameter).
  - Passing spots are routed to the appropriate sub-classifier.

Multi-level hierarchies
-----------------------
  Three coarse niches have three-level sub-model hierarchies:

  - Atrium:  stage 2a (Atrial myocardium vs Endocardium - Atrial)
             stage 2b (Atrium - Left / Right / Transitional)
  - Pacemaker conduction system:  stage 2a (Sinoatrial region vs AV node)
                                  stage 2b (Sinus horn / SA node - Head / Tail)
  - Vasculature:  stage 2a (Great vessels vs Coronary vessel vs Connective tissue)
                  stage 2b (Great vessel vs Ductus arteriosus)

Output columns added to adata.obs (or returned in query_df)
------------------------------------------------------------
  tt_coarse_label  — Stage 1 prediction
  tt_coarse_score  — max(predict_proba) from coarse model
  tt_fine_label    — Stage 2 prediction (NaN if terminal or low_conf)
  tt_fine_score    — max(predict_proba) from sub-classifier (NaN if not run)
  tt_joint_score   — coarse_score × fine_score (NaN if fine not run)
  tt_final_label   — recommended column: fine_label where available,
                     else coarse_label
  tt_low_conf      — bool: True when coarse_score < theta
  tt_stage2a_score — stage 2a confidence (Atrium and PCS three-level only)

Typical usage
-------------
>>> from tissuetypist.prediction import predict, predict_adata
>>>
>>> # Low-level: operate on a pre-built feature DataFrame
>>> query_df = build_neighbourhood_features_sd(adata, genes=..., ...)
>>> result_df = predict(query_df, model_dir="results/hierarchical")
>>>
>>> # High-level: operate directly on an AnnData
>>> adata = predict_adata(
...     adata, model_dir="results/hierarchical",
...     modality="sd", section_col="section_ID",
... )
>>> adata.obs[["tt_final_label", "tt_coarse_score"]]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

logger = logging.getLogger(__name__)

# Shared normalisation helpers (also used by 02_hierarchical_train.py)
from tissuetypist.data.normalise import normalise_if_needed, is_log_normalised


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (spec-driven, Phase 3b)
# ─────────────────────────────────────────────────────────────────────────────

def _read_gene_list(path: Path) -> list[str]:
    """Read a plain-text gene list (one gene per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_hierarchy(model_dir: str | Path) -> dict:
    """Load hierarchy_config.json (schema_version=2) + all pipelines.

    Returns a runtime bundle:

        {
          "coarse_pipeline": fitted sklearn Pipeline,
          "coarse_genes":    list[str],
          "spec":            HierarchySpec,               # parsed hierarchy
          "stage_runtime":   { model_name: {"pipeline":..., "genes":...} },
                              # loaded for every stage whose artifact_present is True
          "schema_version":  int,
        }

    Stage artifacts missing from disk are logged as warnings and omitted
    from ``stage_runtime``; the chain walker then treats that stage as a
    no-op (spots stop at the previous confident label).

    Schema-version-1 files (pre-Apr-2026) raise a clear error instructing
    the user to retrain with Phase-3b code.
    """
    from tissuetypist.config.hierarchy import (
        HierarchySpec, SubModel, SubModelStage,
    )

    model_dir = Path(model_dir)
    config_path = model_dir / "hierarchy_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"hierarchy_config.json not found in {model_dir}. "
            "Have you run `tissuetypist train` (or scripts/02_hierarchical_train.py)?"
        )

    with open(config_path) as f:
        cfg = json.load(f)

    schema_version = cfg.get("schema_version")
    if schema_version is None:
        # Legacy format — surfacing a clear error is better than a silent
        # KeyError deep inside the dispatcher.
        raise ValueError(
            f"hierarchy_config.json in {model_dir} is in the pre-Apr-2026 "
            "format (no schema_version field). TissueTypist's prediction "
            "code was refactored; please retrain with Phase-3b code. If you "
            "need to read the old format, check out the git tag "
            "`pre-restructure`."
        )
    if schema_version != 2:
        raise ValueError(
            f"Unsupported hierarchy_config.json schema_version={schema_version} "
            f"in {model_dir}. This prediction code expects schema_version=2."
        )

    # ── Coarse stage ──────────────────────────────────────────────────────
    coarse_pipeline_path  = model_dir / cfg["coarse_pipeline"]
    coarse_gene_list_path = model_dir / cfg["coarse_gene_list"]
    if not coarse_pipeline_path.exists():
        raise FileNotFoundError(f"Coarse pipeline not found: {coarse_pipeline_path}")

    # ── Parse HierarchySpec from the embedded payload ─────────────────────
    h_payload = cfg["hierarchy"]

    def _parse_stage_from_cfg(d: dict) -> "SubModelStage":
        pool_from = d.get("pool_from")
        if pool_from is not None:
            pool_from = {k: list(v) for k, v in pool_from.items()}
        return SubModelStage(
            model_name=d["model_name"],
            classes=list(d["classes"]),
            modalities=list(d.get("modalities", [])),
            pool_from=pool_from,
            intermediate_label_in_data=d.get("intermediate_label_in_data"),
            route_classes_to_next=list(d.get("route_classes_to_next", [])),
            low_confidence_route=d.get("low_confidence_route"),
            fallback_label=d.get("fallback_label"),
        )

    sub_models: dict[str, SubModel] = {}
    for parent, sm_dict in h_payload["sub_models"].items():
        stages = [_parse_stage_from_cfg(st) for st in sm_dict["stages"]]
        sub_models[parent] = SubModel(parent_coarse=parent, stages=stages)

    spec = HierarchySpec(
        name=h_payload["name"],
        coarse_col=h_payload["coarse_col"],
        fine_col=h_payload.get("fine_col"),
        coarse_niches=list(h_payload["coarse_niches"]),
        terminal_coarse=list(h_payload.get("terminal_coarse", [])),
        sub_models=sub_models,
        description=h_payload.get("description", ""),
    )

    # ── Load each stage's pipeline + gene list (skip if absent on disk) ───
    stage_runtime: dict[str, dict] = {}
    missing: list[str] = []
    for sm in spec.sub_models.values():
        # Walk the serialised sub_model stages in parallel to get the per-stage
        # filename metadata (pipeline/gene_list paths).
        sm_dict = h_payload["sub_models"][sm.parent_coarse]
        for stage, stage_dict in zip(sm.stages, sm_dict["stages"]):
            pp = model_dir / stage_dict["pipeline"]
            gp = model_dir / stage_dict["gene_list"]
            if not pp.exists() or not gp.exists():
                missing.append(stage.model_name)
                continue
            stage_runtime[stage.model_name] = {
                "pipeline": joblib.load(pp),
                "genes": _read_gene_list(gp),
            }
    if missing:
        logger.warning(
            "Hierarchy load: stage pipelines missing (chain walker will "
            "stop before these stages): %s", ", ".join(missing),
        )

    logger.info(
        "Hierarchy loaded: %r  (%d coarse, %d sub-models, %d stage pipelines)",
        spec.name, len(spec.coarse_niches), len(spec.sub_models), len(stage_runtime),
    )

    return {
        "coarse_pipeline": joblib.load(coarse_pipeline_path),
        "coarse_genes":    _read_gene_list(coarse_gene_list_path),
        "spec":            spec,
        "stage_runtime":   stage_runtime,
        "schema_version":  schema_version,
    }


def _union_of_all_gene_lists(hierarchy: dict) -> list[str]:
    """Return sorted union of coarse gene list and every loaded stage gene list."""
    all_genes: set[str] = set(hierarchy["coarse_genes"])
    for rt in hierarchy["stage_runtime"].values():
        all_genes.update(rt["genes"])
    return sorted(all_genes)


def _build_feature_matrix(
    query_df: pd.DataFrame,
    gene_list: list[str],
) -> pd.DataFrame:
    """Select own + neighbour-max features + distance_to_edge for gene_list.

    Missing genes (present in gene_list but absent from query_df columns)
    are filled with 0, consistent with v1 behaviour: a gene absent from
    the query platform is treated as unexpressed.
    """
    own_cols   = [f"{g}_own"           for g in gene_list]
    neigh_cols = [f"{g}_neighbour-max" for g in gene_list]
    all_cols   = own_cols + neigh_cols + ["distance_to_edge"]

    missing = [c for c in all_cols if c not in query_df.columns]
    if missing:
        n_missing_genes = len([c for c in missing if c.endswith("_own")])
        logger.debug(
            "%d genes missing from query features — filling with 0.",
            n_missing_genes,
        )

    return query_df.reindex(columns=all_cols, fill_value=0.0)


def _run_pipeline(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run a fitted sklearn Pipeline and return (labels, max_proba scores).

    The fitted ColumnTransformer records feature_names_in_. We re-index
    X to that exact order so prediction is robust to column ordering.
    """
    preprocessor = pipeline.named_steps.get(
        "preprocessor",
        pipeline.named_steps.get("scaler", None),
    )
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        feature_names = list(preprocessor.feature_names_in_)
        X = X.reindex(columns=feature_names, fill_value=0.0)

    labels = pipeline.predict(X)
    probas = pipeline.predict_proba(X)
    scores = probas.max(axis=1)
    return labels, scores


# ─────────────────────────────────────────────────────────────────────────────
# Generic sub-model chain walker (replaces hardcoded Atrium / PCS / Vasc)
# ─────────────────────────────────────────────────────────────────────────────

def _predict_sub_model_chain(
    query_df: pd.DataFrame,
    hierarchy: dict,
    sub_model,           # SubModel
    theta: float,
) -> None:
    """Walk ``sub_model.stages`` sequentially, writing predictions in-place.

    For every stage:

    1. Select spots whose current predicted label at this point in the
       walk is ``parent_coarse`` (for stage 0) or whose previous stage
       routed them here (``tt_<prev>_label in stage.route_classes_to_next``
       for stage 1+).
    2. Build a feature matrix restricted to ``stage.genes`` (from
       ``hierarchy["stage_runtime"]``).
    3. Run the pipeline → (label, score).
    4. Write ``tt_<stage.model_name>_label`` / ``tt_<stage.model_name>_score``.
    5. Handle confidence gating:
       - score >= theta → apply routing: if label is in
         ``stage.route_classes_to_next`` the spot continues to the next
         stage; otherwise the label is terminal and written to
         ``tt_fine_label``.
       - score <  theta → mark ``tt_low_conf = True``, then:
           * if ``stage.low_confidence_route`` is set, treat as that class
             and continue to next stage (permissive routing);
           * else write ``stage.fallback_label`` (or the class predicted
             at the previous stage, defaulting to ``parent_coarse`` for
             stage 0) to ``tt_fine_label`` and stop for this spot.
    6. Update ``tt_joint_score`` multiplicatively by this stage's score.

    Legacy aliases (kept for one release):
      - ``tt_stage2a_score``  = first-stage score for Atrium + PCS sub-models
      - ``tt_vasc2a_score``   = first-stage score for Vasculature sub-model
    """
    spec: "HierarchySpec" = hierarchy["spec"]
    parent_coarse = sub_model.parent_coarse
    stage_runtime = hierarchy["stage_runtime"]

    coarse_mask = query_df["tt_coarse_label"].astype(str) == parent_coarse
    if not coarse_mask.any():
        return

    # Spots below theta at the coarse gate don't enter this chain.
    coarse_conf = query_df["tt_coarse_score"] >= theta
    enter_mask = coarse_mask & coarse_conf
    low_at_coarse = coarse_mask & ~coarse_conf
    if low_at_coarse.any():
        query_df.loc[low_at_coarse, "tt_low_conf"] = True

    n_enter = int(enter_mask.sum())
    if n_enter == 0:
        return

    logger.info(
        "  [%s] chain entry: %d/%d spots above theta=%.2f (depth=%d)",
        parent_coarse, n_enter, int(coarse_mask.sum()), theta, sub_model.depth,
    )

    # Track per-spot "currently active" mask and the running joint score.
    # We start by copying the coarse score as the accumulator.
    active = enter_mask.copy()
    # Ensure per-stage columns exist.
    for stage in sub_model.stages:
        label_col = f"tt_{stage.model_name}_label"
        score_col = f"tt_{stage.model_name}_score"
        if label_col not in query_df.columns:
            query_df[label_col] = pd.NA
        if score_col not in query_df.columns:
            query_df[score_col] = np.nan

    # Previous-stage predicted label per spot, for fallback resolution.
    # Starts as the coarse label (so stage 0's fallback points up to coarse).
    prev_label = query_df["tt_coarse_label"].copy()

    for stage_idx, stage in enumerate(sub_model.stages):
        if not active.any():
            break

        rt = stage_runtime.get(stage.model_name)
        if rt is None:
            # Pipeline missing — stop the chain here; leave prev_label as
            # each spot's final fine label.
            logger.warning(
                "  [%s] stage %r pipeline not available — stopping chain. "
                "Spots will retain %r as fine label.",
                parent_coarse, stage.model_name,
                "<previous stage>" if stage_idx > 0 else parent_coarse,
            )
            query_df.loc[active, "tt_fine_label"] = prev_label[active]
            break

        n_active = int(active.sum())
        logger.info(
            "  [%s] stage %d/%d (%s): predicting on %d spots",
            parent_coarse, stage_idx + 1, sub_model.depth, stage.model_name, n_active,
        )

        X = _build_feature_matrix(query_df.loc[active], rt["genes"])
        labels, scores = _run_pipeline(rt["pipeline"], X)

        # Write per-stage columns.
        label_col = f"tt_{stage.model_name}_label"
        score_col = f"tt_{stage.model_name}_score"
        active_idx = query_df.index[active]
        query_df.loc[active_idx, label_col] = labels
        query_df.loc[active_idx, score_col] = scores

        # Legacy aliases (stage 0 of three-level sub-models only).
        if stage_idx == 0:
            if parent_coarse in {"Atrium", "Pacemaker conduction system"}:
                query_df.loc[active_idx, "tt_stage2a_score"] = scores
            elif parent_coarse == "Vasculature":
                query_df.loc[active_idx, "tt_vasc2a_score"] = scores

        # Joint score update: multiply running product.
        # On stage 0, running product is coarse_score; on later stages, it is
        # already updated from prior iterations. Use tt_joint_score as the
        # accumulator, initialised below to the coarse score for these spots.
        # We keep it simple by recomputing based on the current product:
        running = query_df.loc[active_idx, "tt_joint_score"]
        # On stage 0, running is still NaN — seed with coarse score.
        if stage_idx == 0:
            running = query_df.loc[active_idx, "tt_coarse_score"].astype(float)
        query_df.loc[active_idx, "tt_joint_score"] = running.values * np.asarray(scores, dtype=float)

        # Now decide per-spot: route / terminal / fallback / permissive.
        # Work with boolean Series indexed by active_idx.
        labels_s = pd.Series(labels, index=active_idx)
        scores_s = pd.Series(scores, index=active_idx, dtype=float)

        confident = scores_s >= theta
        low = ~confident

        # Confident spots: route-to-next or terminal at this stage.
        route_set = set(stage.route_classes_to_next)
        confident_routed = confident & labels_s.isin(route_set)
        confident_terminal = confident & ~labels_s.isin(route_set)

        if confident_terminal.any():
            idx_term = labels_s.index[confident_terminal]
            query_df.loc[idx_term, "tt_fine_label"] = labels_s.loc[idx_term]
            query_df.loc[idx_term, "tt_fine_score"] = scores_s.loc[idx_term]

        # Low-confidence: either permissive route or fallback terminal.
        if low.any():
            # Mark low_conf for these spots.
            query_df.loc[labels_s.index[low], "tt_low_conf"] = True

            if stage.low_confidence_route is not None:
                # Permissive: treat as the configured class and continue.
                idx_low = labels_s.index[low]
                # Overwrite the predicted label to the permissive target so
                # downstream gating treats them as routed.
                labels_s.loc[idx_low] = stage.low_confidence_route
                # Now they count as confident-routed for next-stage purposes.
                confident_routed = confident_routed | low
            else:
                # Terminal fallback: fallback_label, or previous stage label.
                fb_label = stage.fallback_label
                if fb_label is None:
                    # Default: whatever label we had before this stage.
                    fallback_vals = prev_label.loc[labels_s.index[low]]
                else:
                    fallback_vals = pd.Series(fb_label, index=labels_s.index[low])
                idx_low = labels_s.index[low]
                query_df.loc[idx_low, "tt_fine_label"] = fallback_vals.values
                # Keep tt_fine_score as the stage's (low) score — this is
                # informative even though we chose to fall back.
                query_df.loc[idx_low, "tt_fine_score"] = scores_s.loc[idx_low]

        # Prepare for next iteration.
        # Active = only spots that should continue to the next stage =
        # those whose (possibly overridden) label is in route_classes_to_next.
        # We rebuild the boolean Series over the whole query_df:
        next_active_in_this_subset = labels_s.isin(route_set)

        # Translate back into the full-DataFrame mask.
        new_active = pd.Series(False, index=query_df.index)
        new_active.loc[labels_s.index[next_active_in_this_subset]] = True
        active = new_active

        # Record prev_label for fallback wiring of the NEXT stage.
        prev_label = pd.Series(prev_label)
        prev_label.loc[labels_s.index] = labels_s.values

    # ── End of chain: any spots still "active" fell off the last stage's
    #    route_classes_to_next — set their fine label to whatever label
    #    they carry at this point (prev_label).
    if active.any():
        # This happens only if the final stage listed route_classes_to_next
        # (which is unusual — the final stage should have an empty list).
        query_df.loc[active, "tt_fine_label"] = prev_label[active]


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    query_df: pd.DataFrame,
    model_dir: str | Path,
    theta: float = 0.5,
) -> pd.DataFrame:
    """
    Run hierarchical two-stage prediction on a pre-built feature DataFrame.

    Parameters
    ----------
    query_df : pd.DataFrame
        Output of build_neighbourhood_features_sd() or _hd(), indexed by
        spot/window ID. Must contain <gene>_own, <gene>_neighbour-max,
        and distance_to_edge columns.
    model_dir : str or Path
        Directory containing hierarchy_config.json and all .joblib files.
    theta : float
        Confidence threshold on coarse_score below which Stage 2 is skipped
        and tt_low_conf is set True. Default 0.5.

    Returns
    -------
    query_df with new columns:
        tt_coarse_label, tt_coarse_score,
        tt_fine_label, tt_fine_score, tt_joint_score,
        tt_final_label, tt_low_conf
    """
    hierarchy  = _load_hierarchy(model_dir)
    spec       = hierarchy["spec"]
    n_spots    = len(query_df)
    query_df   = query_df.copy()

    # ── Stage 1: coarse prediction ─────────────────────────────────────────
    logger.info("Stage 1: coarse prediction (%d spots)...", n_spots)

    X_coarse = _build_feature_matrix(query_df, hierarchy["coarse_genes"])
    coarse_labels, coarse_scores = _run_pipeline(
        hierarchy["coarse_pipeline"], X_coarse
    )

    query_df["tt_coarse_label"] = coarse_labels
    query_df["tt_coarse_score"] = coarse_scores

    # Initialise Stage 2+ output columns.
    query_df["tt_fine_label"]    = pd.NA
    query_df["tt_fine_score"]    = np.nan
    query_df["tt_joint_score"]   = np.nan
    query_df["tt_low_conf"]      = False
    # Legacy aliases (one-release deprecation; filled per-spot by chain walker):
    query_df["tt_stage2a_score"] = np.nan    # Atrium + PCS first-stage score
    query_df["tt_vasc2a_score"]  = np.nan    # Vasculature first-stage score

    terminal_niches = set(spec.terminal_coarse)

    # ── Stage 2+: walk each sub-model chain ───────────────────────────────
    logger.info(
        "Stage 2+: walking %d sub-model chain(s) (%s)...",
        len(spec.sub_models), ", ".join(spec.sub_models),
    )

    for parent_coarse, sub_model in spec.sub_models.items():
        if parent_coarse in terminal_niches:
            continue
        _predict_sub_model_chain(
            query_df=query_df,
            hierarchy=hierarchy,
            sub_model=sub_model,
            theta=theta,
        )

    # ── Terminal coarse niches: coarse label is final, no Stage 2 ─────────
    for tn in terminal_niches:
        term_mask = query_df["tt_coarse_label"].astype(str) == tn
        if term_mask.any():
            # Confidence gate still applies — low coarse score → tt_low_conf.
            low = term_mask & (query_df["tt_coarse_score"] < theta)
            if low.any():
                query_df.loc[low, "tt_low_conf"] = True

    # ── tt_final_label: fine where assigned, else coarse ──────────────────
    query_df["tt_final_label"] = np.where(
        query_df["tt_fine_label"].isna(),
        query_df["tt_coarse_label"],
        query_df["tt_fine_label"],
    )

    # Ensure tt_fine_label is object dtype with np.nan (not pd.NA) so that
    # h5py can serialise it. pd.NA is a pandas NAType that h5py cannot handle.
    query_df["tt_fine_label"] = (
        query_df["tt_fine_label"]
        .astype(object)
        .where(query_df["tt_fine_label"].notna(), other=np.nan)
    )

    # ── Summary ────────────────────────────────────────────────────────────
    n_fine_predicted = int(query_df["tt_fine_label"].notna().sum())
    n_terminal       = int(query_df["tt_coarse_label"].isin(terminal_niches).sum())
    n_low_conf_total = int(query_df["tt_low_conf"].sum())
    n_final_filled   = int(query_df["tt_final_label"].notna().sum())

    logger.info(
        "\nPrediction summary (%d spots):\n"
        "  Fine-grained label assigned  : %d\n"
        "  Terminal niche (coarse only) : %d\n"
        "  Low-confidence spots         : %d\n"
        "  tt_final_label filled        : %d",
        n_spots, n_fine_predicted, n_terminal, n_low_conf_total, n_final_filled,
    )

    return query_df


def predict_adata(
    adata: ad.AnnData,
    model_dir: str | Path,
    modality: str,
    section_col: str = "section_ID",
    theta: float = 0.5,
    hd_windows: Optional[ad.AnnData] = None,
    sliding_window_col: Optional[str] = None,
) -> ad.AnnData:
    """
    High-level wrapper: build neighbourhood features, run predict(),
    and write results back into adata.obs.

    Parameters
    ----------
    adata : AnnData
        Query AnnData. For HD data, this can be:
          (a) Pre-computed pseudobulk windows (pass hd_windows=None and
              ensure adata already has window_col/window_row in obs), OR
          (b) Raw HD cells — in which case pass hd_windows=<windows AnnData>
              and sliding_window_col to map predictions back to cells.
    model_dir : str or Path
        Path to trained model directory.
    modality : str
        'sd' or 'hd'.
    section_col : str
        obs column for section/library grouping. Default: 'section_ID'.
    theta : float
        Coarse confidence threshold. Default 0.5.
    hd_windows : AnnData, optional
        Pre-computed HD pseudobulk windows (output of 00_pseudobulk_hd.py
        or sliding_window_pseudobulk_hd). Required when modality='hd' and
        adata is raw HD cells. If modality='hd' and adata already IS the
        windows object, leave this as None.
    sliding_window_col : str, optional
        obs column in raw adata mapping each cell to its window ID.
        Required only when hd_windows is provided (HD raw → window mapping).

    Returns
    -------
    adata with tt_* columns added to adata.obs.
    """
    from tissuetypist.features.spatial import (
        build_neighbourhood_features_sd,
        build_neighbourhood_features_hd,
    )

    hierarchy  = _load_hierarchy(model_dir)
    spec       = hierarchy["spec"]
    all_genes  = _union_of_all_gene_lists(hierarchy)

    logger.info(
        "Building neighbourhood features for %d genes (union of all models)...",
        len(all_genes),
    )

    if modality == "sd":
        # Normalise SD query data if raw counts — neighbourhood features
        # must be built from log-normalised expression, not raw integers.
        adata = normalise_if_needed(adata, name="SD query")
        coarse_col = spec.coarse_col
        query_df = build_neighbourhood_features_sd(
            adata,
            genes=all_genes,
            niche_col=coarse_col if coarse_col in adata.obs.columns else None,
            section_col=section_col,
        )

    elif modality == "hd":
        # HD windows now store raw summed counts (since 00_pseudobulk_hd.py fix).
        # normalise_if_needed handles them exactly like SD query data.
        windows_adata = hd_windows if hd_windows is not None else adata
        windows_adata = normalise_if_needed(windows_adata, name="HD windows")
        coarse_col = spec.coarse_col
        query_df = build_neighbourhood_features_hd(
            windows_adata,
            genes=all_genes,
            niche_col=coarse_col if coarse_col in windows_adata.obs.columns else None,
            section_col=section_col,
        )

    else:
        raise ValueError(f"modality must be 'sd' or 'hd', got '{modality!r}'.")

    # ── Run hierarchical prediction ─────────────────────────────────────────
    query_df = predict(query_df, model_dir=model_dir, theta=theta)

    # Base result columns (always present).
    result_cols = [
        "tt_coarse_label", "tt_coarse_score",
        "tt_fine_label",   "tt_fine_score",
        "tt_joint_score",  "tt_final_label",
        "tt_low_conf",
        # Legacy aliases (deprecated; kept for one release):
        "tt_stage2a_score",
        "tt_vasc2a_score",
    ]
    # Plus per-stage named columns (tt_<stage.model_name>_label / _score)
    # for every sub-model stage. These are generated by the chain walker.
    for sub_model in spec.sub_models.values():
        for stage in sub_model.stages:
            label_col = f"tt_{stage.model_name}_label"
            score_col = f"tt_{stage.model_name}_score"
            if label_col in query_df.columns:
                result_cols.append(label_col)
            if score_col in query_df.columns:
                result_cols.append(score_col)

    # ── Map predictions back to adata.obs ──────────────────────────────────
    if modality == "hd" and hd_windows is not None and sliding_window_col is not None:
        # HD raw cells: map window-level predictions back to individual cells
        if sliding_window_col not in adata.obs.columns:
            raise ValueError(
                f"sliding_window_col='{sliding_window_col}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        for col in result_cols:
            mapping = query_df[col].to_dict()
            adata.obs[col] = (
                adata.obs[sliding_window_col]
                .astype(str)
                .map(mapping)
            )
    else:
        # SD (or HD windows directly): 1-to-1 mapping by obs index
        for col in result_cols:
            adata.obs[col] = query_df[col].reindex(adata.obs_names)

    # Report unmapped spots
    n_nan = adata.obs["tt_final_label"].isna().sum()
    if n_nan > 0:
        logger.warning(
            "%d cells have no prediction "
            "(likely outside sliding windows or unmatched index).",
            n_nan,
        )

    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: predict from saved h5ad (CLI-friendly entry point)
# ─────────────────────────────────────────────────────────────────────────────

def predict_from_file(
    h5ad_path: str,
    model_dir: str,
    modality: str,
    section_col: str = "section_ID",
    out_path: str = "predicted.h5ad",
    theta: float = 0.5,
    hd_windows_path: Optional[str] = None,
) -> None:
    """
    Load an AnnData, run predict_adata(), and save with tt_* columns.

    Parameters
    ----------
    h5ad_path : str
        Path to query AnnData.
    model_dir : str
        Path to model directory.
    modality : str
        'sd' or 'hd'.
    section_col : str
        Section obs column. Default: 'section_ID'.
    out_path : str
        Where to save the annotated AnnData.
    theta : float
        Confidence threshold. Default 0.5.
    hd_windows_path : str, optional
        Path to pre-computed HD windows h5ad. Required for raw HD input.
    """
    import scanpy as sc

    logger.info("Loading query AnnData: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)

    hd_windows = None
    if hd_windows_path is not None:
        logger.info("Loading HD windows: %s", hd_windows_path)
        hd_windows = sc.read_h5ad(hd_windows_path)

    adata = predict_adata(
        adata,
        model_dir=model_dir,
        modality=modality,
        section_col=section_col,
        theta=theta,
        hd_windows=hd_windows,
    )

    logger.info("Saving annotated AnnData: %s", out_path)
    adata.write_h5ad(out_path)

    # Print label distribution summary
    if "tt_final_label" in adata.obs.columns:
        logger.info("\ntt_final_label distribution:")
        counts = adata.obs["tt_final_label"].value_counts()
        for label, n in counts.items():
            pct = 100 * n / len(adata.obs)
            logger.info("  %-40s  %5d  (%.1f%%)", label, n, pct)
