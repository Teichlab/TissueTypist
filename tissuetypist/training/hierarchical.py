"""
tissuetypist.training.hierarchical
===================================
Core library for TissueTypist hierarchical training.

The niche hierarchy is declarative. :class:`TrainingConfig` carries a
:class:`~tissuetypist.config.HierarchySpec` loaded from YAML (shipped:
``tissuetypist/config/hierarchies/cardiac.yaml``; or a user-supplied path).
Training walks every non-terminal coarse niche's ordered chain of sub-model
stages, of any depth. The cardiac hierarchy has chains of depth 1 (Ventricle,
AV junction), 2 (Pacemaker conduction system, Vasculature), and 3 (Atrium).

This module contains all training logic as importable functions.
``scripts/02_hierarchical_train.py`` is the thin CLI wrapper.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc

if TYPE_CHECKING:
    from tissuetypist.config.hierarchy import (
        HierarchySpec,
        SubModel,
        SubModelStage,
    )

logger = logging.getLogger("tissuetypist.training.hierarchical")


# Active modality tag → obs column / sub-model modality name.
# Keep in sync with the "modalities" field in HierarchySpec stages.
_MOD_TAG_TO_LOG_NAME = {"sd3p": "SD_3prime", "sd_ffpe": "SD_FFPE", "hd": "HD_FFPE"}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Configuration for hierarchical training.

    This replaces the argparse.Namespace that was previously threaded
    through all training functions. Users can construct this directly
    for programmatic access.

    Attributes
    ----------
    hierarchy :
        :class:`~tissuetypist.config.HierarchySpec` describing the niche
        hierarchy. If ``None`` (default), the shipped cardiac hierarchy
        is loaded at training time.
    coarse_col / fine_col :
        obs-column overrides. If ``None`` (default), the values from
        ``hierarchy.coarse_col`` / ``hierarchy.fine_col`` are used. Set
        these to override the YAML's defaults from Python or from the
        CLI (useful for ad-hoc relabeling without editing the YAML).
    """
    hierarchy: Optional["HierarchySpec"] = None
    coarse_col: Optional[str] = None
    fine_col: Optional[str] = None
    neighbour_weight: float = 0.3
    edge_weight: float = 5.0
    n_top_hvgs: int = 4000
    feature_set: str = "deg_hvg"       # "deg_hvg" or "deg_only"
    cv: bool = False
    cv_folds: int = 5
    coarse_only: bool = False
    save_qc: bool = True

    def resolve_hierarchy(self) -> "HierarchySpec":
        """Return the effective HierarchySpec for this training run.

        Loads the shipped ``cardiac`` hierarchy if :attr:`hierarchy` is
        ``None``. Applies :attr:`coarse_col` / :attr:`fine_col`
        overrides on top of the spec if they are set.
        """
        from tissuetypist.config.hierarchy import load_hierarchy

        spec = self.hierarchy if self.hierarchy is not None else load_hierarchy("cardiac")
        if self.coarse_col is None and self.fine_col is None:
            return spec
        return dataclass_replace(
            spec,
            coarse_col=self.coarse_col if self.coarse_col is not None else spec.coarse_col,
            fine_col=self.fine_col if self.fine_col is not None else spec.fine_col,
        )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _empty_adata(reference: ad.AnnData) -> ad.AnnData:
    """Return an empty AnnData with the same var (gene) space."""
    import scipy.sparse as sp
    empty = ad.AnnData(
        X=sp.csr_matrix((0, reference.n_vars)),
        var=reference.var.copy(),
        obs=pd.DataFrame(columns=reference.obs.columns),
    )
    return empty


def resolve_gene_pool(
    path: Optional[str | Path],
    reference_adatas: list[ad.AnnData],
) -> list[str]:
    """Resolve the ``shared_all`` gene pool from a file or from reference adatas.

    Three accepted inputs:

    1. ``path`` is a ``.csv`` file → read the ``shared_all`` column.
       This is the canonical format produced by
       :func:`tissuetypist.features.gene_catalogue.build_gene_pools_from_paths`
       (i.e. the output of ``tissuetypist build-catalogue``).
    2. ``path`` is any other extension (``.txt``, ``.tsv``, ``.list``, etc.)
       → read one gene per line. Useful when the user has a curated
       marker list they want to constrain feature selection to.
    3. ``path`` is ``None`` → compute the intersection of
       ``var_names`` across every provided non-empty AnnData in
       ``reference_adatas``. Requires at least one non-empty input.

    Parameters
    ----------
    path :
        Path to ``gene_pools.csv``, a plain-text gene list, or ``None``.
    reference_adatas :
        AnnDatas that would be used for training. Only used when
        ``path is None`` (to compute the intersection).

    Returns
    -------
    list[str]
        The sorted gene list to use as the ``shared_all`` universe.
    """
    if path is not None:
        p = Path(path)
        if p.suffix.lower() == ".csv":
            logger.info("Loading gene pools from CSV: %s", p)
            pools = pd.read_csv(p)
            if "shared_all" not in pools.columns:
                raise ValueError(
                    f"{p}: expected a 'shared_all' column. "
                    f"Found: {list(pools.columns)}"
                )
            genes_shared = pools["shared_all"].dropna().tolist()
        else:
            logger.info("Loading gene pool (one per line): %s", p)
            with p.open() as f:
                genes_shared = [line.strip() for line in f if line.strip()]
        logger.info("  gene pool: %d genes", len(genes_shared))
        return genes_shared

    # path=None → intersection across non-empty reference adatas.
    non_empty = [a for a in reference_adatas if a is not None and a.n_obs > 0]
    if not non_empty:
        raise ValueError(
            "No gene-pool file provided and no non-empty reference AnnDatas — "
            "cannot determine the training gene universe."
        )
    sets = [set(a.var_names) for a in non_empty]
    genes_shared = sorted(set.intersection(*sets))
    logger.info(
        "No --gene_pools supplied — using var_names intersection across "
        "%d reference AnnData%s: %d genes",
        len(non_empty), "" if len(non_empty) == 1 else "s", len(genes_shared),
    )
    return genes_shared


def load_data(
    sd3p_path: str,
    gene_pools_path: Optional[str] = None,
    sd_ffpe_path: Optional[str] = None,
    hd_windows_path: Optional[str] = None,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, list[str]]:
    """
    Load available modality AnnDatas and resolve the gene-pool universe.

    ``sd_ffpe_path`` and ``hd_windows_path`` are optional — if not
    provided, empty AnnDatas with the same gene space as the primary
    input are returned.

    ``gene_pools_path`` is optional — see :func:`resolve_gene_pool` for
    the three accepted inputs. When omitted, uses the intersection of
    ``var_names`` across the provided non-empty AnnDatas.

    Returns
    -------
    adata_sd3p, adata_sdffpe, adata_hd_windows, genes_shared
    """
    from tissuetypist.data.normalise import normalise_if_needed

    logger.info("Loading primary reference: %s", sd3p_path)
    adata_sd3p = sc.read_h5ad(sd3p_path)
    logger.info("  %d spots × %d genes", adata_sd3p.n_obs, adata_sd3p.n_vars)
    adata_sd3p = normalise_if_needed(adata_sd3p, "SD_3prime")

    if sd_ffpe_path:
        logger.info("Loading secondary reference: %s", sd_ffpe_path)
        adata_sdffpe = sc.read_h5ad(sd_ffpe_path)
        logger.info("  %d spots × %d genes", adata_sdffpe.n_obs, adata_sdffpe.n_vars)
        adata_sdffpe = normalise_if_needed(adata_sdffpe, "SD_FFPE")
    else:
        logger.info("No secondary reference provided — using empty placeholder.")
        adata_sdffpe = _empty_adata(adata_sd3p)

    if hd_windows_path:
        logger.info("Loading tertiary reference: %s", hd_windows_path)
        adata_hd_windows = sc.read_h5ad(hd_windows_path)
        logger.info(
            "  %d obs × %d genes", adata_hd_windows.n_obs, adata_hd_windows.n_vars
        )
        adata_hd_windows = normalise_if_needed(adata_hd_windows, "HD_windows")
    else:
        logger.info("No tertiary reference provided — using empty placeholder.")
        adata_hd_windows = _empty_adata(adata_sd3p)

    genes_shared = resolve_gene_pool(
        gene_pools_path,
        reference_adatas=[adata_sd3p, adata_sdffpe, adata_hd_windows],
    )

    return adata_sd3p, adata_sdffpe, adata_hd_windows, genes_shared


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_f2_genes_coarse(
    adata_sd3p: ad.AnnData,
    adata_sdffpe: ad.AnnData,
    adata_hd_windows: ad.AnnData,
    coarse_col: str,
    genes_shared: list[str],
    n_top_degs: int = None,
    n_top_hvgs: int = 4000,
    feature_strategy: str = "deg_hvg",
) -> list[str]:
    """
    Compute DEG ∪ HVG gene set for the coarse model on the full dataset.
    """
    from tissuetypist.features.gene_selection import (
        build_niche_modality_map,
        compute_f2_feature_set,
    )

    strategy_label = "DEG ∪ HVG" if feature_strategy == "deg_hvg" else "DEG-only"
    logger.info("Computing coarse %s feature set...", strategy_label)

    niche_modality_map = build_niche_modality_map(
        adata_sd3p, adata_sdffpe, adata_hd_windows,
        niche_col=coarse_col,
    )

    # Auto-relax hvg_min_modalities based on how many modalities actually
    # carry any data. Default is 2 (genes must be HVG in ≥2 modalities — a
    # cross-modality consistency filter for the canonical cardiac 3-modality
    # setup). For single-modality runs (e.g. flat-mode training on just SD 3'
    # or a non-cardiac user's one dataset) that threshold would drop all
    # HVGs. Clamp to the number of non-empty modalities.
    n_active = sum(
        1 for a in (adata_sd3p, adata_sdffpe, adata_hd_windows) if a.n_obs > 0
    )
    hvg_min = min(2, n_active) if n_active >= 1 else 1
    if hvg_min < 2:
        logger.info(
            "Only %d modality with data — relaxing hvg_min_modalities from 2 to %d.",
            n_active, hvg_min,
        )

    f2 = compute_f2_feature_set(
        adata_sd3p=adata_sd3p,
        adata_sd_ffpe=adata_sdffpe,
        adata_hd=adata_hd_windows,
        niche_col=coarse_col,
        genes_shared=genes_shared,
        niche_modality_map=niche_modality_map,
        n_top_hvgs=n_top_hvgs,
        hvg_min_modalities=hvg_min,
        use_pseudobulk_hd=False,
        microns_per_pixel_map=None,
        feature_strategy=feature_strategy,
    )
    logger.info(
        "Coarse feature set: %d genes "
        "(DEG: %d | HVG: %d | DEG-only: %d | HVG-only: %d | overlap: %d)",
        f2["n_total"], len(f2["deg_genes"]), len(f2["hvg_genes"]),
        f2["n_deg_only"], f2["n_hvg_only"], f2["n_overlap"],
    )
    return f2["genes"]


def compute_f2_genes_fine(
    adata_sd3p_sub: ad.AnnData,
    adata_sdffpe_sub: ad.AnnData,
    adata_hd_sub: ad.AnnData,
    fine_col: str,
    genes_shared: list[str],
    coarse_niche: str,
    n_top_degs: int = None,
    n_top_hvgs: int = 4000,
    feature_strategy: str = "deg_hvg",
) -> list[str]:
    """
    Compute DEG ∪ HVG gene set for a fine-grained sub-model.

    Each input adata has already been subset to spots belonging to one
    coarse niche. Empty adatas are handled gracefully.
    hvg_min_modalities=1 (relaxed from coarse model's 2).
    """
    from tissuetypist.features.gene_selection import (
        build_niche_modality_map,
        compute_f2_feature_set,
    )

    non_empty: dict[str, ad.AnnData] = {}
    for name, adata in [
        ("SD_3prime", adata_sd3p_sub),
        ("SD_FFPE",   adata_sdffpe_sub),
        ("HD_FFPE",   adata_hd_sub),
    ]:
        if adata.n_obs == 0:
            logger.warning(
                "  Sub-model '%s': %s has 0 spots — skipping this modality.",
                coarse_niche, name,
            )
        else:
            non_empty[name] = adata

    if not non_empty:
        logger.warning("  Sub-model '%s': all modalities empty.", coarse_niche)
        return []

    if len(non_empty) == 1:
        logger.warning(
            "  Sub-model '%s': only 1 modality has data (%s).",
            coarse_niche, list(non_empty.keys())[0],
        )

    sd3p_sub   = non_empty.get("SD_3prime", _empty_adata(adata_sd3p_sub))
    sdffpe_sub = non_empty.get("SD_FFPE",   _empty_adata(adata_sdffpe_sub))
    hd_sub     = non_empty.get("HD_FFPE",   _empty_adata(adata_hd_sub))

    niche_modality_map = build_niche_modality_map(
        sd3p_sub, sdffpe_sub, hd_sub,
        niche_col=fine_col,
    )

    if not niche_modality_map:
        logger.warning(
            "  Sub-model '%s': no niches found — cannot compute features.",
            coarse_niche,
        )
        return []

    f2 = compute_f2_feature_set(
        adata_sd3p=sd3p_sub,
        adata_sd_ffpe=sdffpe_sub,
        adata_hd=hd_sub,
        niche_col=fine_col,
        genes_shared=genes_shared,
        niche_modality_map=niche_modality_map,
        n_top_hvgs=n_top_hvgs,
        hvg_min_modalities=1,
        use_pseudobulk_hd=False,
        microns_per_pixel_map=None,
        feature_strategy=feature_strategy,
    )
    logger.info(
        "  '%s' feature set: %d genes (DEG: %d | HVG: %d)",
        coarse_niche, f2["n_total"], len(f2["deg_genes"]), len(f2["hvg_genes"]),
    )
    return f2["genes"]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SUBSETTING
# ══════════════════════════════════════════════════════════════════════════════

def subset_features(
    full_data: pd.DataFrame,
    gene_list: list[str],
    coarse_col: str,
    fine_col: str,
    row_mask: Optional[pd.Series] = None,
    niche_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Subset pre-built feature DataFrame by rows and gene columns.

    Parameters
    ----------
    full_data : Pre-built features with "{gene}_own", "{gene}_neighbour-max" cols.
    gene_list : Genes to keep.
    row_mask : Boolean mask to select rows. If None, all rows are kept.
    """
    df = full_data if row_mask is None else full_data.loc[row_mask].copy()

    own_cols = [f"{g}_own" for g in gene_list if f"{g}_own" in full_data.columns]
    neigh_cols = [
        f"{g}_neighbour-max" for g in gene_list
        if f"{g}_neighbour-max" in full_data.columns
    ]

    meta_cols = [
        "distance_to_edge", "n_neighbours", "is_edge",
        "section", "x", "y", "modality",
        coarse_col, fine_col,
    ]
    keep_cols = own_cols + neigh_cols + [c for c in meta_cols if c in df.columns]

    return df[keep_cols]


# ══════════════════════════════════════════════════════════════════════════════
# NEIGHBOURHOOD FEATURE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_features(
    adata_sd3p: ad.AnnData,
    adata_sdffpe: ad.AnnData,
    adata_hd_windows: ad.AnnData,
    gene_list: list[str],
    coarse_col: str,
    fine_col: str,
    niche_col_for_features: str,
    plot: bool = False,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build neighbourhood feature DataFrames for all available modalities
    and concatenate them.

    All modalities use section_col='section_ID'.
    """
    from tissuetypist.features.spatial import (
        build_neighbourhood_features_sd,
        build_neighbourhood_features_hd,
    )

    frames = []

    if adata_sd3p.n_obs > 0:
        logger.info(
            "  Building SD 3-prime neighbourhood features (%d spots, %d genes)...",
            adata_sd3p.n_obs, len(gene_list),
        )
        df_sd3p = build_neighbourhood_features_sd(
            adata_sd3p, genes=gene_list,
            niche_col=niche_col_for_features, section_col="section_ID",
            plot=plot, save_dir=save_dir, data_tag="SD_3prime",
        )
        df_sd3p["modality"] = "SD_3prime"
        frames.append(df_sd3p)

    if adata_sdffpe.n_obs > 0:
        logger.info(
            "  Building SD FFPE neighbourhood features (%d spots, %d genes)...",
            adata_sdffpe.n_obs, len(gene_list),
        )
        df_sdffpe = build_neighbourhood_features_sd(
            adata_sdffpe, genes=gene_list,
            niche_col=niche_col_for_features, section_col="section_ID",
            plot=plot, save_dir=save_dir, data_tag="SD_FFPE",
        )
        df_sdffpe["modality"] = "SD_FFPE"
        frames.append(df_sdffpe)

    if adata_hd_windows.n_obs > 0:
        logger.info(
            "  Building HD neighbourhood features (%d windows, %d genes)...",
            adata_hd_windows.n_obs, len(gene_list),
        )
        df_hd = build_neighbourhood_features_hd(
            adata_hd_windows, genes=gene_list,
            niche_col=niche_col_for_features, section_col="section_ID",
            plot=plot, save_dir=save_dir,
        )
        df_hd["modality"] = "HD_FFPE"
        frames.append(df_hd)

    if not frames:
        raise ValueError("No modalities had data — cannot build features.")

    data = pd.concat(frames, ignore_index=False)
    data = data.rename(columns={"niche": niche_col_for_features})

    # Attach fine labels. In flat mode (no sub-models), ``fine_col`` is
    # ``None`` — no fine column exists in obs and sub-model training is
    # skipped altogether, so we don't need this column in the feature frame.
    if fine_col is not None:
        fine_labels = pd.concat([
            adata_sd3p.obs[fine_col],
            adata_sdffpe.obs[fine_col],
            adata_hd_windows.obs[fine_col],
        ])
        data[fine_col] = fine_labels.loc[data.index].values

    logger.info(
        "  Combined feature matrix: %d rows × %d columns",
        len(data), len(data.columns),
    )
    return data


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FIT + SAVE
# ══════════════════════════════════════════════════════════════════════════════

def fit_and_save_pipeline(
    data: pd.DataFrame,
    niche_col: str,
    gene_list: list[str],
    outdir: Path,
    filekey: str,
    neighbour_weight: float,
    edge_weight: float,
    run_cv: bool = False,
    cv_folds: int = 5,
) -> None:
    """
    Build, optionally cross-validate, and fit a weighted LR pipeline.
    Save the pipeline and gene list to outdir.
    """
    from tissuetypist.training.logistic import (
        build_weighted_pipeline,
        run_weighted_cv,
    )

    own_features       = [c for c in data.columns if c.endswith("_own")]
    neighbour_features = [c for c in data.columns if c.endswith("_neighbour-max")]
    feature_cols       = own_features + neighbour_features + ["distance_to_edge"]

    y = data[niche_col].astype(str).values
    n_classes = len(np.unique(y))
    n_spots   = len(data)

    logger.info(
        "  Fitting pipeline '%s': %d spots, %d classes, %d features",
        filekey, n_spots, n_classes, len(feature_cols),
    )

    if run_cv:
        logger.info("  Running %d-fold stratified CV...", cv_folds)
        df_folds, df_per_class = run_weighted_cv(
            data=data,
            own_features=own_features,
            neighbour_features=neighbour_features,
            neighbour_weight=neighbour_weight,
            edge_weight=edge_weight,
            niche_col=niche_col,
            n_splits=cv_folds,
        )
        cv_path = outdir / f"cv_{filekey}.csv"
        df_folds.to_csv(cv_path, index=False)
        cv_pc_path = outdir / f"cv_{filekey}_per_class.csv"
        df_per_class.to_csv(cv_pc_path, index=False)
        logger.info(
            "  CV mean weighted F1: %.4f ± %.4f (saved to %s)",
            df_folds["f1_weighted"].mean(),
            df_folds["f1_weighted"].std(),
            cv_path,
        )

    pipeline = build_weighted_pipeline(
        own_features=own_features,
        neighbour_features=neighbour_features,
        neighbour_weight=neighbour_weight,
        edge_weight=edge_weight,
    )
    pipeline.fit(data[feature_cols], y)

    pipeline_path  = outdir / f"{filekey}_pipeline.joblib"
    gene_list_path = outdir / f"{filekey}_gene_list.txt"

    joblib.dump(pipeline, pipeline_path)
    gene_list_path.write_text("\n".join(gene_list))

    logger.info("  Saved pipeline → %s", pipeline_path)
    logger.info("  Saved gene list (%d genes) → %s", len(gene_list), gene_list_path)


# ══════════════════════════════════════════════════════════════════════════════
# (Legacy three-level training helpers removed Apr 2026 — see git tag
#  pre-restructure for the hardcoded _train_three_level /
#  _train_vasculature_three_level. Their behaviour is now subsumed by
#  _train_sub_model_chain below, which handles chains of any depth.)

# ══════════════════════════════════════════════════════════════════════════════
# SPEC-DRIVEN CHAIN TRAINING (Phase 3b)
# ══════════════════════════════════════════════════════════════════════════════
#
# These helpers generalise training over a :class:`SubModel` chain of any
# depth. They supersede ``_train_three_level`` and
# ``_train_vasculature_three_level`` (which remain above as dead code pending
# removal in a follow-up cleanup) and handle all five cardiac sub-models plus
# any user-supplied hierarchy via a single code path.


def _build_stage_remap(stage: "SubModelStage") -> dict[str, str]:
    """Build a ``data_label → stage_class`` remap table for one stage.

    The remap unifies three sources of stage-class evidence:

    1. **Direct labels** — every class in ``stage.classes`` that is NOT
       a pool target maps to itself (identity). Direct labels appear
       verbatim in ``adata.obs[fine_col]``.
    2. **pool_from** — each ``synth_cls: [data_labels]`` entry contributes
       ``data_label → synth_cls`` entries (many-to-one).
    3. **intermediate_label_in_data** — legacy mechanism: a single data
       label that maps onto the intermediate class of this stage. The
       target is inferred from ``route_classes_to_next`` (must be
       unambiguous — exactly one non-pooled routed class).

    Returns
    -------
    dict
        ``data_label → class_in_stage.classes``.

    Raises
    ------
    ValueError
        If ``intermediate_label_in_data`` is set but the target class
        cannot be disambiguated.
    """
    pool_targets = set(stage.pool_from.keys()) if stage.pool_from else set()
    remap: dict[str, str] = {}

    # 1. Direct (non-pooled) classes keep their identity mapping.
    for cls in stage.classes:
        if cls not in pool_targets:
            remap[cls] = cls

    # 2. pool_from entries.
    if stage.pool_from:
        for synth_cls, data_labels in stage.pool_from.items():
            for dl in data_labels:
                if dl in remap and remap[dl] != synth_cls:
                    raise ValueError(
                        f"Stage {stage.model_name!r}: data label {dl!r} is "
                        f"mapped to both {remap[dl]!r} and {synth_cls!r}. "
                        "pool_from entries must be disjoint."
                    )
                remap[dl] = synth_cls

    # 3. intermediate_label_in_data (legacy path, rarely used alongside pool_from).
    if stage.intermediate_label_in_data:
        candidates = [c for c in stage.route_classes_to_next if c not in pool_targets]
        if len(candidates) == 1:
            target = candidates[0]
        elif len(candidates) == 0:
            # Fall back to route_classes_to_next ∩ pool_targets (intermediate is
            # pooled; intermediate_label_in_data is just one more data label
            # that routes to that same pooled class).
            if len(stage.route_classes_to_next) == 1:
                target = stage.route_classes_to_next[0]
            else:
                raise ValueError(
                    f"Stage {stage.model_name!r}: intermediate_label_in_data "
                    f"is set but the target class is ambiguous "
                    f"(route_classes_to_next={stage.route_classes_to_next!r})."
                )
        else:
            raise ValueError(
                f"Stage {stage.model_name!r}: intermediate_label_in_data "
                f"target is ambiguous — multiple non-pooled routed classes "
                f"{candidates!r}."
            )
        remap[stage.intermediate_label_in_data] = target

    return remap


def _remap_for_stage(
    adata: ad.AnnData,
    fine_col: str,
    stage: "SubModelStage",
    modality_tag: str,
    remap: Optional[dict[str, str]] = None,
) -> Optional[ad.AnnData]:
    """Filter ``adata`` to spots usable by this stage and remap their labels.

    Parameters
    ----------
    adata :
        AnnData for one modality, already subset to a single coarse niche.
    fine_col :
        Fine-label obs column.
    stage :
        The stage whose classes / pool_from / modalities govern filtering.
    modality_tag :
        ``"sd3p"``, ``"sd_ffpe"`` or ``"hd"``. If not in
        ``stage.modalities``, this modality is silently dropped.
    remap :
        Optional pre-computed remap (from :func:`_build_stage_remap`). If
        omitted it is built on the fly.

    Returns
    -------
    Optional[AnnData]
        Filtered+remapped AnnData, or ``None`` if no rows survive or the
        modality is not used by this stage.
    """
    if modality_tag not in stage.modalities:
        return None
    if adata.n_obs == 0:
        return None
    if remap is None:
        remap = _build_stage_remap(stage)

    fv = adata.obs[fine_col].astype(str)
    mask = fv.isin(remap.keys())
    if not mask.any():
        return None

    filtered = adata[mask].copy()
    filtered.obs[fine_col] = filtered.obs[fine_col].astype(str).map(remap).values

    # Post-condition: every remapped label is a member of stage.classes.
    bad = set(filtered.obs[fine_col].astype(str)) - set(stage.classes)
    if bad:
        raise RuntimeError(
            f"Stage {stage.model_name!r}: remapped labels {sorted(bad)!r} "
            f"are not in stage.classes {stage.classes!r}."
        )
    return filtered


def _prepare_stage_frame(
    full_data: pd.DataFrame,
    coarse_col: str,
    fine_col: str,
    coarse_niche: str,
    stage: "SubModelStage",
    remap: dict[str, str],
) -> pd.DataFrame:
    """Equivalent of :func:`_remap_for_stage` at the feature-frame level.

    ``full_data`` is the pre-built neighbourhood feature frame for all
    spots. We filter to rows under ``coarse_niche`` whose fine label
    maps into this stage and rewrite the fine column.
    """
    row_mask = full_data[coarse_col] == coarse_niche
    frame = full_data.loc[row_mask].copy()
    fv = frame[fine_col].astype(str)
    mask = fv.isin(remap.keys())
    frame = frame.loc[mask].copy()
    frame[fine_col] = frame[fine_col].astype(str).map(remap).values
    return frame


def _train_stage(
    stage: "SubModelStage",
    stage_label: str,
    sub_sd3p: ad.AnnData,
    sub_sdffpe: ad.AnnData,
    sub_hd: ad.AnnData,
    full_data: pd.DataFrame,
    coarse_col: str,
    fine_col: str,
    coarse_niche: str,
    genes_shared: list[str],
    outdir: Path,
    config: "TrainingConfig",
    summary_rows: list[dict],
    gene_count_rows: list[dict],
    gene_override: Optional[list[str]] = None,
    min_genes_submodel: int = 1,
) -> bool:
    """Train a single stage of a sub-model chain.

    Returns ``True`` on success (pipeline saved), ``False`` if the stage
    was skipped (no data, too few classes, feature selection returned
    nothing, too few genes, etc.).

    ``gene_override`` lets panel-specific training bypass DEG+HVG selection
    and supply the stage's gene list directly (e.g. the intersection of a
    user-provided curated list with the panel). When ``None``, DEG+HVG runs.

    ``min_genes_submodel`` is the floor; any resulting gene list below this
    count triggers a skip with a warning. Default 1 (back-compat with the
    full-genome path). Panel-specific training sets this to e.g. 10.
    """
    remap = _build_stage_remap(stage)
    logger.info(
        "  [%s] stage %r classes=%s modalities=%s",
        coarse_niche, stage.model_name, stage.classes, stage.modalities,
    )
    if stage.pool_from:
        for synth_cls, data_labels in stage.pool_from.items():
            logger.info(
                "  [%s]   pool_from: %r <- %s",
                coarse_niche, synth_cls, list(data_labels),
            )

    # ── Filter + remap each modality's AnnData for this stage ──────────────
    staged = {}
    for mod_tag, adata in [("sd3p", sub_sd3p), ("sd_ffpe", sub_sdffpe), ("hd", sub_hd)]:
        out = _remap_for_stage(adata, fine_col, stage, mod_tag, remap=remap)
        staged[mod_tag] = out  # may be None

    mod_tags_with_data = [t for t, a in staged.items() if a is not None and a.n_obs > 0]
    if not mod_tags_with_data:
        logger.warning(
            "  [%s] stage %r: no spots after filtering — SKIPPING.",
            coarse_niche, stage.model_name,
        )
        return False

    # ── Stage gene list: override OR fresh DEG+HVG ─────────────────────────
    if gene_override is not None:
        stage_genes = [g for g in gene_override if g in genes_shared]
        logger.info(
            "  [%s] stage %r: using override gene list (%d genes, %d ∩ shared_all)",
            coarse_niche, stage.model_name,
            len(gene_override), len(stage_genes),
        )
    else:
        # For DEG+HVG we need the three-arg (sd3p, sdffpe, hd) tuple; pass empty
        # AnnDatas where the stage doesn't use that modality so
        # compute_f2_genes_fine silently skips them.
        a_sd3p = staged["sd3p"] or _empty_adata(sub_sd3p)
        a_ffpe = staged["sd_ffpe"] or _empty_adata(sub_sdffpe)
        a_hd = staged["hd"] or _empty_adata(sub_hd)
        stage_genes = compute_f2_genes_fine(
            a_sd3p, a_ffpe, a_hd,
            fine_col=fine_col,
            genes_shared=genes_shared,
            coarse_niche=stage.model_name,
            n_top_hvgs=config.n_top_hvgs,
            feature_strategy=config.feature_set,
        )

    if not stage_genes:
        logger.warning(
            "  [%s] stage %r: 0 genes selected — SKIPPING.",
            coarse_niche, stage.model_name,
        )
        return False
    if len(stage_genes) < min_genes_submodel:
        logger.warning(
            "  [%s] stage %r: only %d genes (< min_genes_submodel=%d) — SKIPPING.",
            coarse_niche, stage.model_name, len(stage_genes), min_genes_submodel,
        )
        return False

    # ── Feature frame for this stage (filtered + remapped rows from full_data) ──
    stage_frame_init = _prepare_stage_frame(
        full_data, coarse_col, fine_col, coarse_niche, stage, remap,
    )
    if stage_frame_init.empty:
        logger.warning(
            "  [%s] stage %r: feature frame empty after filter — SKIPPING.",
            coarse_niche, stage.model_name,
        )
        return False

    try:
        stage_frame = subset_features(
            stage_frame_init,
            gene_list=stage_genes,
            coarse_col=coarse_col, fine_col=fine_col,
            niche_col=fine_col,
        )
    except Exception as exc:
        logger.error(
            "  [%s] stage %r: feature subsetting failed: %s — SKIPPING.",
            coarse_niche, stage.model_name, exc,
        )
        return False

    n_classes = stage_frame[fine_col].nunique()
    if n_classes < 2:
        logger.warning(
            "  [%s] stage %r: only %d class present — cannot train. SKIPPING.",
            coarse_niche, stage.model_name, n_classes,
        )
        return False

    logger.info(
        "  [%s] stage %r training data: %d rows, %d classes: %s",
        coarse_niche, stage.model_name, len(stage_frame), n_classes,
        sorted(stage_frame[fine_col].unique()),
    )

    fit_and_save_pipeline(
        data=stage_frame,
        niche_col=fine_col,
        gene_list=stage_genes,
        outdir=outdir,
        filekey=stage.model_name,
        neighbour_weight=config.neighbour_weight,
        edge_weight=config.edge_weight,
        run_cv=config.cv,
        cv_folds=config.cv_folds,
    )

    used_mod_names = [_MOD_TAG_TO_LOG_NAME[t] for t in mod_tags_with_data]
    summary_rows.append({
        "model": stage.model_name,
        "stage": stage_label,
        "n_spots": len(stage_frame),
        "n_classes": n_classes,
        "n_genes": len(stage_genes),
        "modalities": ", ".join(used_mod_names),
    })
    gene_count_rows.append({"model": stage.model_name, "n_genes": len(stage_genes)})
    return True


def _train_sub_model_chain(
    sub_model: "SubModel",
    sub_sd3p: ad.AnnData,
    sub_sdffpe: ad.AnnData,
    sub_hd: ad.AnnData,
    full_data: pd.DataFrame,
    coarse_col: str,
    fine_col: str,
    genes_shared: list[str],
    outdir: Path,
    config: "TrainingConfig",
    summary_rows: list[dict],
    gene_count_rows: list[dict],
    stage_gene_overrides: Optional[dict[str, list[str]]] = None,
    min_genes_submodel: int = 1,
) -> None:
    """Train every stage in ``sub_model.stages`` in order.

    If an early stage in the chain fails (no spots, too few classes),
    later stages are still attempted — each is independent at training
    time, depending only on the subset of ``full_data`` that matches its
    pool_from / class definitions.

    ``stage_gene_overrides`` is an optional ``{model_name: [genes]}``
    map used by panel-specific training to substitute a pre-selected
    gene list for one or more stages in lieu of DEG+HVG.
    """
    logger.info(
        "\n=== Sub-model chain for %r  (depth=%d) ===",
        sub_model.parent_coarse, sub_model.depth,
    )
    stage_labels = _stage_labels_for_depth(sub_model.depth)
    for stage, stage_label in zip(sub_model.stages, stage_labels):
        override = (
            stage_gene_overrides.get(stage.model_name)
            if stage_gene_overrides is not None else None
        )
        _train_stage(
            stage=stage,
            stage_label=stage_label,
            sub_sd3p=sub_sd3p, sub_sdffpe=sub_sdffpe, sub_hd=sub_hd,
            full_data=full_data,
            coarse_col=coarse_col, fine_col=fine_col,
            coarse_niche=sub_model.parent_coarse,
            genes_shared=genes_shared,
            outdir=outdir, config=config,
            summary_rows=summary_rows, gene_count_rows=gene_count_rows,
            gene_override=override,
            min_genes_submodel=min_genes_submodel,
        )


def _stage_labels_for_depth(depth: int) -> list[str]:
    """Human-readable stage labels by chain depth (for logs + summary CSV).

    depth=1 → ['2']                   (flat sub-model)
    depth=2 → ['2a', '2b']            (legacy "three-level")
    depth=3 → ['2a', '2b', '2c']      (Atrium Apr 2026)
    depth≥4 → ['2', '2b', '2c', '2d', ...] (future-proof)
    """
    if depth == 1:
        return ["2"]
    letters = ["a", "b", "c", "d", "e", "f"]
    return [f"2{letters[i]}" for i in range(depth)]


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHY CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def _serialise_stage(stage: "SubModelStage") -> dict:
    """Serialise a :class:`SubModelStage` to a JSON-safe dict."""
    out: dict = {
        "model_name": stage.model_name,
        "classes": list(stage.classes),
        "modalities": list(stage.modalities),
        "route_classes_to_next": list(stage.route_classes_to_next),
        "pipeline":  f"{stage.model_name}_pipeline.joblib",
        "gene_list": f"{stage.model_name}_gene_list.txt",
    }
    if stage.pool_from:
        out["pool_from"] = {k: list(v) for k, v in stage.pool_from.items()}
    if stage.intermediate_label_in_data:
        out["intermediate_label_in_data"] = stage.intermediate_label_in_data
    if stage.low_confidence_route:
        out["low_confidence_route"] = stage.low_confidence_route
    if stage.fallback_label:
        out["fallback_label"] = stage.fallback_label
    return out


def _serialise_hierarchy(spec: "HierarchySpec") -> dict:
    """Serialise a :class:`HierarchySpec` to a JSON-safe dict.

    This is the source-of-truth for the on-disk hierarchy_config.json
    format (schema_version 2). :func:`prediction.hierarchical._load_hierarchy`
    parses it back into a spec for chain-walking at predict time.
    """
    return {
        "name": spec.name,
        "coarse_col": spec.coarse_col,
        "fine_col": spec.fine_col,
        "coarse_niches": list(spec.coarse_niches),
        "terminal_coarse": list(spec.terminal_coarse),
        "description": spec.description,
        "sub_models": {
            parent: {
                "parent_coarse": sm.parent_coarse,
                "stages": [_serialise_stage(st) for st in sm.stages],
            }
            for parent, sm in spec.sub_models.items()
        },
    }


def save_hierarchy_config(
    outdir: Path,
    config: TrainingConfig,
    spec: Optional["HierarchySpec"] = None,
) -> Path:
    """Build and save hierarchy_config.json. Returns the written path.

    Schema version 2 (Apr 2026 restructure):
      - ``schema_version: 2``
      - ``hierarchy: <serialised HierarchySpec>`` — the full spec,
        including each stage's pipeline/gene_list filenames.
      - ``coarse_pipeline`` + ``coarse_gene_list`` — convenience pointers
        to the Stage-1 artifact (always ``"coarse_pipeline.joblib"``).
      - ``training_args`` — weights, feature_set, HVG count, DEG filters.

    For forward compatibility the writer also emits the schema-v1 keys
    (``coarse_label_col``, ``fine_label_col``, ``terminal_niches``) so
    older prediction code can partially read the file; but the new
    prediction code reads ``hierarchy`` directly and ignores the v1 keys.
    """
    if spec is None:
        spec = config.resolve_hierarchy()

    hierarchy_payload = _serialise_hierarchy(spec)

    # Decorate each stage dict with a liveness flag so downstream readers
    # can warn about missing joblibs without inspecting the filesystem.
    missing: list[str] = []
    for sm_dict in hierarchy_payload["sub_models"].values():
        for stage_dict in sm_dict["stages"]:
            pp = outdir / stage_dict["pipeline"]
            gp = outdir / stage_dict["gene_list"]
            stage_dict["artifact_present"] = pp.exists() and gp.exists()
            if not stage_dict["artifact_present"]:
                missing.append(stage_dict["model_name"])

    if missing:
        logger.warning(
            "Hierarchy config: stage pipelines not found on disk (likely skipped): %s",
            ", ".join(missing),
        )

    hierarchy_config = {
        "schema_version": 2,
        # v2 canonical content
        "hierarchy": hierarchy_payload,
        "coarse_pipeline":  "coarse_pipeline.joblib",
        "coarse_gene_list": "coarse_gene_list.txt",
        "training_args": {
            "neighbour_weight": config.neighbour_weight,
            "edge_weight":      config.edge_weight,
            "feature_set":      config.feature_set,
            "n_top_hvgs":       config.n_top_hvgs,
            "deg_filters": {
                "pval_adj": 0.05,
                "min_logfc": 0.5,
                "min_mean_expr": 0.5,
            },
        },
        # v1 shim fields (deprecated — v2 readers ignore these)
        "coarse_label_col": spec.coarse_col,
        "fine_label_col":   spec.fine_col,
        "terminal_niches":  list(spec.terminal_coarse),
    }

    config_path = outdir / "hierarchy_config.json"
    with open(config_path, "w") as f:
        json.dump(hierarchy_config, f, indent=2)
    logger.info("Saved hierarchy_config.json (schema_version=2) → %s", config_path)
    return config_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    adata_sd3p: ad.AnnData,
    adata_sdffpe: ad.AnnData,
    adata_hd_windows: ad.AnnData,
    genes_shared: list[str],
    outdir: str | Path,
    config: Optional[TrainingConfig] = None,
    qc_dir: Optional[str] = None,
    stage_gene_overrides: Optional[dict[str, list[str]]] = None,
    min_genes_submodel: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train the full hierarchical model suite.

    This is the main entry point for programmatic use.

    Parameters
    ----------
    adata_sd3p : SD 3-prime reference (log-normalised).
    adata_sdffpe : SD FFPE reference (log-normalised or empty).
    adata_hd_windows : HD pseudobulk windows (log-normalised or empty).
    genes_shared : Shared gene list across modalities.
    outdir : Output directory for models and config.
    config : Training configuration. If None, uses defaults.
    qc_dir : Directory for QC plots. None = no plots.

    Returns
    -------
    summary_df : Training summary (model, stage, n_spots, n_classes, n_genes, modalities).
    gene_count_df : Gene counts per model.
    """
    if config is None:
        config = TrainingConfig()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Resolve the effective HierarchySpec (load cardiac if None) ─────────
    spec = config.resolve_hierarchy()
    coarse_col = spec.coarse_col
    fine_col = spec.fine_col

    logger.info("Hierarchy: %r  (coarse=%r, fine=%r)", spec.name, coarse_col, fine_col)
    logger.info("Coarse niches (%d): %s", len(spec.coarse_niches), spec.coarse_niches)
    logger.info("Terminal coarse (no Stage 2): %s", spec.terminal_coarse)

    # Validate required obs columns
    for name, adata in [
        ("SD_3prime",  adata_sd3p),
        ("SD_FFPE",    adata_sdffpe),
        ("HD_windows", adata_hd_windows),
    ]:
        if adata.n_obs == 0:
            continue
        for col in [c for c in (coarse_col, fine_col) if c is not None]:
            if col not in adata.obs.columns:
                raise ValueError(
                    f"{name}: obs column '{col}' not found. "
                    f"Available: {list(adata.obs.columns)}"
                )

    active_mods = []
    for name, adata in [
        ("SD_3prime", adata_sd3p), ("SD_FFPE", adata_sdffpe),
        ("HD_FFPE", adata_hd_windows),
    ]:
        if adata.n_obs > 0:
            active_mods.append(name)
    logger.info("Active modalities: %s", ", ".join(active_mods))
    logger.info("Feature set: %s", config.feature_set)

    summary_rows: list[dict] = []
    gene_count_rows: list[dict] = []

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD FEATURES ONCE — on full dataset with ALL shared genes
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE BUILDING: Computing neighbourhood features (all %d shared genes)",
                len(genes_shared))
    logger.info("=" * 70)

    plot_qc = qc_dir is not None
    full_data = build_features(
        adata_sd3p, adata_sdffpe, adata_hd_windows,
        gene_list=genes_shared,
        coarse_col=coarse_col,
        fine_col=fine_col,
        niche_col_for_features=coarse_col,
        plot=plot_qc,
        save_dir=qc_dir,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Coarse model
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: Coarse model (%d-class)", len(spec.coarse_niches))
    logger.info("=" * 70)

    coarse_override = (
        stage_gene_overrides.get("coarse")
        if stage_gene_overrides is not None else None
    )
    if coarse_override is not None:
        coarse_genes = [g for g in coarse_override if g in genes_shared]
        logger.info(
            "Coarse: using override gene list (%d genes, %d ∩ shared_all)",
            len(coarse_override), len(coarse_genes),
        )
        if not coarse_genes:
            raise ValueError(
                "Coarse gene override produced an empty list after intersecting "
                "with shared_all. Check that your override genes are in the "
                "shared-gene catalogue."
            )
    else:
        coarse_genes = compute_f2_genes_coarse(
            adata_sd3p, adata_sdffpe, adata_hd_windows,
            coarse_col=coarse_col,
            genes_shared=genes_shared,
            n_top_hvgs=config.n_top_hvgs,
            feature_strategy=config.feature_set,
        )

    logger.info("Subsetting coarse neighbourhood features...")
    coarse_data = subset_features(
        full_data, gene_list=coarse_genes,
        coarse_col=coarse_col, fine_col=fine_col, niche_col=coarse_col,
    )

    n_coarse_classes = coarse_data[coarse_col].nunique()
    logger.info(
        "Coarse training data: %d spots, %d classes: %s",
        len(coarse_data), n_coarse_classes,
        sorted(coarse_data[coarse_col].unique()),
    )

    fit_and_save_pipeline(
        data=coarse_data, niche_col=coarse_col, gene_list=coarse_genes,
        outdir=outdir, filekey="coarse",
        neighbour_weight=config.neighbour_weight, edge_weight=config.edge_weight,
        run_cv=config.cv, cv_folds=config.cv_folds,
    )

    summary_rows.append({
        "model": "coarse", "stage": 1,
        "n_spots": len(coarse_data), "n_classes": n_coarse_classes,
        "n_genes": len(coarse_genes), "modalities": ", ".join(active_mods),
    })
    gene_count_rows.append({"model": "coarse", "n_genes": len(coarse_genes)})

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2+ — Sub-model chains (any depth) driven by HierarchySpec
    # ═══════════════════════════════════════════════════════════════════════
    if config.coarse_only:
        logger.info("\ncoarse_only=True — skipping Stage 2+ sub-models.")
    else:
        logger.info("\n" + "=" * 70)
        logger.info(
            "STAGE 2+: Spec-driven sub-model chains (%d parent niches)",
            len(spec.sub_models),
        )
        logger.info("=" * 70)

        for parent_coarse, sub_model in spec.sub_models.items():
            if parent_coarse in spec.terminal_coarse:
                # Shouldn't happen for a well-formed spec, but be defensive.
                logger.info(
                    "Skipping sub-model chain for terminal coarse niche %r.",
                    parent_coarse,
                )
                continue

            logger.info("\n--- Sub-model chain: %r ---", parent_coarse)

            # Subset each modality to spots under this coarse niche.
            def _sub(adata, tag):
                if adata.n_obs == 0:
                    return _empty_adata(adata)
                mask = adata.obs[coarse_col].astype(str) == parent_coarse
                return adata[mask].copy() if mask.any() else _empty_adata(adata)

            sub_sd3p = _sub(adata_sd3p, "sd3p")
            sub_sdffpe = _sub(adata_sdffpe, "sd_ffpe")
            sub_hd = _sub(adata_hd_windows, "hd")

            n_total = sub_sd3p.n_obs + sub_sdffpe.n_obs + sub_hd.n_obs
            logger.info(
                "  Spots per modality: SD_3prime=%d | SD_FFPE=%d | HD=%d | total=%d",
                sub_sd3p.n_obs, sub_sdffpe.n_obs, sub_hd.n_obs, n_total,
            )
            if n_total == 0:
                logger.warning(
                    "  %r: no spots found for this coarse niche — SKIPPING chain.",
                    parent_coarse,
                )
                continue

            _train_sub_model_chain(
                sub_model=sub_model,
                sub_sd3p=sub_sd3p, sub_sdffpe=sub_sdffpe, sub_hd=sub_hd,
                full_data=full_data,
                coarse_col=coarse_col, fine_col=fine_col,
                genes_shared=genes_shared,
                outdir=outdir, config=config,
                summary_rows=summary_rows, gene_count_rows=gene_count_rows,
                stage_gene_overrides=stage_gene_overrides,
                min_genes_submodel=min_genes_submodel,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Save config and summaries
    # ═══════════════════════════════════════════════════════════════════════
    save_hierarchy_config(outdir, config, spec=spec)

    summary_df    = pd.DataFrame(summary_rows)
    gene_count_df = pd.DataFrame(gene_count_rows)

    summary_df.to_csv(outdir / "training_summary.csv", index=False)
    gene_count_df.to_csv(outdir / "gene_counts.csv", index=False)

    logger.info("Saved training_summary.csv")
    logger.info("Saved gene_counts.csv")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    if len(summary_df) > 0:
        logger.info("\n%s", summary_df.to_string(index=False))
    logger.info("\nOutput directory: %s", outdir)
    logger.info(
        "Models saved:\n%s",
        "\n".join(f"  {f.name}" for f in sorted(outdir.glob("*.joblib"))),
    )

    return summary_df, gene_count_df
