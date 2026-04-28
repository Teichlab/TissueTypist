"""
tissuetypist/cli/main.py
=========================
The ``tissuetypist`` command-line dispatcher.

Registered in ``pyproject.toml`` as::

    [project.scripts]
    tissuetypist = "tissuetypist.cli.main:entry"

After ``pip install -e .``, the ``tissuetypist`` command becomes
available with eight subcommands (see :mod:`tissuetypist.cli`). Each
subcommand's argparse + handler is defined in this file.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("tissuetypist.cli")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(log_file: str | None, outdir: Path, cmd_name: str) -> None:
    """Configure root + file logging. File path defaults to
    ``{outdir}/{cmd_name}_YYYY-MM-DD_HHMMSS.log`` unless the caller
    passed an explicit path or ``"none"``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if log_file is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_file = str(outdir / f"{cmd_name}_{ts}.log")
    if log_file.lower() == "none":
        return
    outdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to file: %s", log_file)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: `tissuetypist train`
# ─────────────────────────────────────────────────────────────────────────────

def _add_train_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "train",
        help="Hierarchical training on full-genome reference data",
        description="Train a hierarchical (coarse + sub-model chain) "
                    "classifier. Supports shipped YAML, user YAML, "
                    "--flat (no sub-models), or --auto_infer "
                    "(2-level hierarchy from obs columns).",
    )
    p.add_argument(
        "--reference", "--sd3p", dest="reference", required=True,
        help="Primary reference AnnData (raw counts). Treated as modality "
             "tag 'sd3p' by the chain walker — your hierarchy YAML's "
             "`modalities:` declarations use this tag. "
             "Alias --sd3p kept for backward compatibility with the cardiac pipeline.",
    )
    p.add_argument(
        "--reference_secondary", "--sd_ffpe", dest="reference_secondary", default=None,
        help="Secondary reference AnnData (optional; tag 'sd_ffpe'). "
             "Alias --sd_ffpe kept for cardiac compatibility.",
    )
    p.add_argument(
        "--reference_tertiary", "--hd_windows", dest="reference_tertiary", default=None,
        help="Tertiary reference AnnData (optional; tag 'hd'). "
             "For cardiac, this is pre-computed HD pseudobulk windows from "
             "`tissuetypist pseudobulk-hd`. Alias --hd_windows kept.",
    )
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument(
        "--gene_pools", default=None,
        help="Optional. (a) Path to gene_pools.csv from `build-catalogue` "
             "(uses the 'shared_all' column). (b) Path to a plain-text "
             "one-gene-per-line file (uses all listed genes). "
             "(c) Omit entirely → uses the intersection of var_names "
             "across all provided reference AnnDatas.",
    )
    p.add_argument("--neighbour_weight", type=float, default=0.3)
    p.add_argument("--edge_weight",      type=float, default=5.0)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--hierarchy", default=None,
        help="YAML path or shipped name ('cardiac'). Default: 'cardiac'.",
    )
    mode.add_argument(
        "--flat", action="store_true",
        help="Flat mode: single column, no sub-models. Requires --coarse_col.",
    )
    mode.add_argument(
        "--auto_infer", action="store_true",
        help="Auto-infer 2-level hierarchy from obs. Requires --coarse_col + --fine_col.",
    )

    p.add_argument("--coarse_col", default=None)
    p.add_argument("--fine_col",   default=None)
    p.add_argument("--strict_infer",    action="store_true",  default=True)
    p.add_argument("--no_strict_infer", action="store_false", dest="strict_infer")
    p.add_argument("--cv",           action="store_true")
    p.add_argument("--cv_folds",     type=int, default=5)
    p.add_argument("--coarse_only",  action="store_true")
    p.add_argument("--n_top_hvgs",   type=int, default=4000)
    p.add_argument("--feature_set",  default="deg_hvg", choices=["deg_hvg", "deg_only"])
    p.add_argument("--save_qc",      action="store_true",  default=True)
    p.add_argument("--no_save_qc",   action="store_false", dest="save_qc")
    p.add_argument("--log_file",     default=None)

    p.set_defaults(func=_run_train)
    return p


def _run_train(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from tissuetypist.config.hierarchy import (
        flat_hierarchy,
        infer_hierarchy_from_data,
        load_hierarchy,
    )
    from tissuetypist.training.hierarchical import (
        TrainingConfig, load_data, train_all_models,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_file, outdir, cmd_name="training")

    # Resolve hierarchy via the three modes.
    if args.flat:
        if not args.coarse_col:
            raise SystemExit("--flat requires --coarse_col.")
        import scanpy as sc
        peek = sc.read_h5ad(args.reference)
        if args.coarse_col not in peek.obs.columns:
            raise SystemExit(
                f"--flat: --coarse_col {args.coarse_col!r} not in adata.obs."
            )
        classes = sorted(peek.obs[args.coarse_col].dropna().astype(str).unique())
        spec = flat_hierarchy(args.coarse_col, classes, name="flat_from_cli")
        logger.info("Flat hierarchy: %d classes", len(classes))
    elif args.auto_infer:
        if not args.coarse_col or not args.fine_col:
            raise SystemExit("--auto_infer requires --coarse_col and --fine_col.")
        import scanpy as sc
        peek = sc.read_h5ad(args.reference)
        spec = infer_hierarchy_from_data(
            peek, coarse_col=args.coarse_col, fine_col=args.fine_col,
            name="auto_infer", strict=args.strict_infer,
        )
        logger.info(
            "Auto-inferred hierarchy: %d coarse, %d sub-models (strict=%s)",
            len(spec.coarse_niches), len(spec.sub_models), args.strict_infer,
        )
    else:
        spec = load_hierarchy(args.hierarchy) if args.hierarchy else load_hierarchy("cardiac")
        logger.info("YAML hierarchy: %r (%d sub-models)", spec.name, len(spec.sub_models))

    qc_dir = str(outdir / "qc_plots") if args.save_qc else None
    if qc_dir:
        Path(qc_dir).mkdir(parents=True, exist_ok=True)

    adata_sd3p, adata_sdffpe, adata_hd, genes_shared = load_data(
        sd3p_path=args.reference,
        gene_pools_path=args.gene_pools,
        sd_ffpe_path=args.reference_secondary,
        hd_windows_path=args.reference_tertiary,
    )

    config = TrainingConfig(
        hierarchy=spec,
        coarse_col=args.coarse_col,
        fine_col=args.fine_col,
        neighbour_weight=args.neighbour_weight,
        edge_weight=args.edge_weight,
        n_top_hvgs=args.n_top_hvgs,
        feature_set=args.feature_set,
        cv=args.cv, cv_folds=args.cv_folds,
        coarse_only=args.coarse_only,
        save_qc=args.save_qc,
    )

    train_all_models(
        adata_sd3p=adata_sd3p, adata_sdffpe=adata_sdffpe,
        adata_hd_windows=adata_hd, genes_shared=genes_shared,
        outdir=outdir, config=config, qc_dir=qc_dir,
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: `tissuetypist train-panel`
# ─────────────────────────────────────────────────────────────────────────────

def _add_train_panel_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "train-panel",
        help="Panel-specific retraining for imaging-based ST",
        description="Retrain all hierarchical models on panel ∩ shared_all. "
                    "Three gene-selection strategies: --custom_gene_list, "
                    "--gene_lists_from, or default (fresh DEG+HVG on panel).",
    )
    panel = p.add_mutually_exclusive_group(required=True)
    panel.add_argument("--query",         help="Query AnnData; panel read from var_names.")
    panel.add_argument("--gene_panel_txt", help="Plain text file, one gene per line.")

    p.add_argument(
        "--reference", "--sd3p", dest="reference", required=True,
        help="Primary reference AnnData (raw counts; modality tag 'sd3p'). "
             "Alias --sd3p kept.",
    )
    p.add_argument(
        "--reference_secondary", "--sd_ffpe", dest="reference_secondary", default=None,
        help="Secondary reference AnnData (tag 'sd_ffpe'). Alias --sd_ffpe kept.",
    )
    p.add_argument(
        "--reference_tertiary", "--hd_windows", dest="reference_tertiary", default=None,
        help="Tertiary reference AnnData (tag 'hd'). Alias --hd_windows kept.",
    )
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--gene_pools", default=None,
        help="Optional gene-pool file (CSV with 'shared_all' column or one-per-line "
             "text). If omitted, uses the var_names intersection of provided references.",
    )

    p.add_argument("--hierarchy",  default=None)
    p.add_argument("--coarse_col", default=None)
    p.add_argument("--fine_col",   default=None)

    p.add_argument("--neighbour_weight", type=float, default=0.3)
    p.add_argument("--edge_weight",      type=float, default=5.0)
    p.add_argument("--n_top_hvgs",       type=int,   default=4000)
    p.add_argument("--feature_set",      default="deg_hvg", choices=["deg_hvg", "deg_only"])
    p.add_argument("--cv",               action="store_true")
    p.add_argument("--cv_folds",         type=int, default=5)
    p.add_argument("--coarse_only",      action="store_true")

    p.add_argument("--min_genes_submodel",  type=int, default=10)
    p.add_argument("--warn_genes_threshold", type=int, default=200)

    gene_grp = p.add_mutually_exclusive_group()
    gene_grp.add_argument("--custom_gene_list", default=None,
                          help="File: one gene per line; used for all stages.")
    gene_grp.add_argument("--gene_lists_from", default=None,
                          help="Directory with per-stage {model_name}_gene_list.txt files.")

    p.add_argument("--query_section_col",    default="sample_id")
    p.add_argument("--min_cells_per_window", type=int, default=1)
    p.add_argument("--sd3p_only",            action="store_true")
    p.add_argument("--log_file",             default=None)

    p.set_defaults(func=_run_train_panel)
    return p


def _run_train_panel(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import scanpy as sc
    import pandas as pd
    from tissuetypist.config.hierarchy import load_hierarchy
    from tissuetypist.training import TrainingConfig, train_panel_specific
    from tissuetypist.training.hierarchical import _empty_adata

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_file, outdir, cmd_name="training")

    # 1. Gene panel
    if args.query:
        logger.info("Gene panel from: %s", args.query)
        adata_q = sc.read_h5ad(args.query)
        gene_panel = list(adata_q.var_names)
        logger.info("  panel: %d genes", len(gene_panel))
        # Cell-level query? Pseudobulk once for downstream.
        if "spatial" in adata_q.obsm and adata_q.n_obs > 50_000:
            from tissuetypist.data.pseudobulk import sliding_window_pseudobulk
            sc_col = args.query_section_col
            if sc_col not in adata_q.obs.columns:
                raise SystemExit(
                    f"--query_section_col {sc_col!r} not in query obs."
                )
            logger.info(
                "  cell-level query detected (%d cells × %d sections) — pseudobulking…",
                adata_q.n_obs, adata_q.obs[sc_col].nunique(),
            )
            windows = sliding_window_pseudobulk(
                adata_q, section_col=sc_col, window_size=50,
                coord_columns=None, log_normalise=True,
            )
            windows.write_h5ad(outdir / "query_windows.h5ad")
            logger.info("  query windows → %s", outdir / "query_windows.h5ad")

            # Save the per-cell window assignment so callers can project
            # window-level predictions back onto individual cells:
            #     mapping = pd.read_csv(outdir / "query_cell_window_assignment.csv.gz",
            #                           index_col=0)
            #     adata.obs["sliding_window_assignment"] = mapping["sliding_window_assignment"]
            #     adata = predict_adata(adata, hd_windows=adata_windows,
            #                           sliding_window_col="sliding_window_assignment")
            if "sliding_window_assignment" in adata_q.obs.columns:
                mapping_path = outdir / "query_cell_window_assignment.csv.gz"
                adata_q.obs[[sc_col, "sliding_window_assignment"]].to_csv(
                    mapping_path, compression="gzip"
                )
                logger.info("  query cell→window mapping → %s", mapping_path)
    else:
        with open(args.gene_panel_txt) as f:
            gene_panel = [line.strip() for line in f if line.strip()]
        logger.info("Gene panel from %s: %d genes", args.gene_panel_txt, len(gene_panel))

    # 2. Reference AnnDatas (raw counts — panel_specific normalises internally)
    adata_sd3p = sc.read_h5ad(args.reference)
    use_ffpe = args.reference_secondary and not args.sd3p_only
    adata_sdffpe = (
        sc.read_h5ad(args.reference_secondary) if use_ffpe
        else _empty_adata(adata_sd3p)
    )
    use_hd = args.reference_tertiary and not args.sd3p_only
    adata_hd = (
        sc.read_h5ad(args.reference_tertiary) if use_hd
        else _empty_adata(adata_sd3p)
    )

    from tissuetypist.training.hierarchical import resolve_gene_pool
    genes_shared = resolve_gene_pool(
        args.gene_pools,
        reference_adatas=[adata_sd3p, adata_sdffpe, adata_hd],
    )

    # 3. Custom list
    custom_gene_list = None
    if args.custom_gene_list:
        with open(args.custom_gene_list) as f:
            custom_gene_list = [line.strip() for line in f if line.strip()]
        logger.info("Custom gene list: %d genes", len(custom_gene_list))

    # 4. Strategy
    if custom_gene_list is not None:
        strategy = "custom"
    elif args.gene_lists_from:
        strategy = "pre_computed"
    else:
        strategy = "deg_hvg"

    spec = load_hierarchy(args.hierarchy) if args.hierarchy else load_hierarchy("cardiac")
    logger.info("Hierarchy: %r (strategy=%r)", spec.name, strategy)

    config = TrainingConfig(
        hierarchy=spec,
        coarse_col=args.coarse_col, fine_col=args.fine_col,
        neighbour_weight=args.neighbour_weight,
        edge_weight=args.edge_weight,
        n_top_hvgs=args.n_top_hvgs,
        feature_set=args.feature_set,
        cv=args.cv, cv_folds=args.cv_folds,
        coarse_only=args.coarse_only,
    )

    train_panel_specific(
        adata_sd3p_raw=adata_sd3p,
        adata_sdffpe_raw=adata_sdffpe,
        adata_hd_raw=adata_hd,
        gene_panel=gene_panel, genes_shared=genes_shared,
        outdir=outdir, config=config,
        gene_strategy=strategy,
        custom_gene_list=custom_gene_list,
        gene_lists_from=args.gene_lists_from,
        min_genes_submodel=args.min_genes_submodel,
        warn_genes_threshold=args.warn_genes_threshold,
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: `tissuetypist predict`  (prediction only, no plots)
# ─────────────────────────────────────────────────────────────────────────────

def _add_predict_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "predict",
        help="Hierarchical prediction (writes h5ad + summary CSV; no plots)",
        description="Run hierarchical prediction on query data. Output: "
                    "{prefix}_predicted.h5ad + {prefix}_prediction_summary.csv. "
                    "For plots + metrics, use `tissuetypist evaluate`.",
    )
    p.add_argument("--query",      required=True, help="Query AnnData (.h5ad)")
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--outdir",     required=True)
    p.add_argument("--modality",   choices=["sd", "hd"], default="sd")
    p.add_argument("--section_col", default="section_ID")
    p.add_argument("--theta",       type=float, default=0.5)
    p.add_argument("--prefix",      default="predict")
    p.add_argument("--log_file",    default=None)
    p.set_defaults(func=_run_predict)
    return p


def _run_predict(args: argparse.Namespace) -> int:
    import scanpy as sc
    from tissuetypist.prediction import predict_adata
    from tissuetypist.prediction.hierarchical import _load_hierarchy
    from tissuetypist.evaluation.metrics import save_prediction_summary

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_file, outdir, cmd_name="predict")

    logger.info("Loading query: %s", args.query)
    adata = sc.read_h5ad(args.query)
    logger.info("  %d × %d", adata.n_obs, adata.n_vars)

    adata = predict_adata(
        adata, model_dir=args.model_dir,
        modality=args.modality, section_col=args.section_col,
        theta=args.theta,
    )

    out_h5ad = outdir / f"{args.prefix}_predicted.h5ad"
    adata.write_h5ad(out_h5ad)
    logger.info("Saved: %s", out_h5ad)

    bundle = _load_hierarchy(args.model_dir)
    spec = bundle["spec"]
    save_prediction_summary(
        adata, fine_col=spec.fine_col or "", outdir=outdir,
        prefix=args.prefix, spec=spec,
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: `tissuetypist evaluate`  (predict + plots + metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _add_evaluate_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "evaluate",
        help="Predict + plots + metrics",
        description="Run hierarchical prediction and produce evaluation "
                    "plots (confusion matrix, spatial, UMAP, confidence "
                    "distributions) + classification-report CSVs.",
    )
    p.add_argument("--query_sd",    help="SD query AnnData (.h5ad)")
    p.add_argument("--query_hd",    help="HD query AnnData (windows).")
    p.add_argument("--model_dir",   required=True)
    p.add_argument("--outdir",      required=True)
    p.add_argument("--modality",    choices=["sd", "hd", "both"], default="both")
    p.add_argument("--section_col", default="section_ID")
    p.add_argument("--fine_col",    default=None)
    p.add_argument("--coarse_col",  default=None)
    p.add_argument("--theta",       type=float, default=0.5)
    p.add_argument("--no_eval",     action="store_true")
    p.add_argument("--n_pcs",       type=int, default=30)
    p.add_argument("--plot_sections", type=int, default=3)
    p.add_argument("--log_file",    default=None)
    p.set_defaults(func=_run_evaluate)
    return p


def _run_evaluate(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import scanpy as sc
    from tissuetypist.evaluation import evaluate

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_file, outdir, cmd_name="evaluate")

    run_sd = args.query_sd is not None and args.modality in ("sd", "both")
    run_hd = args.query_hd is not None and args.modality in ("hd", "both")
    if not run_sd and not run_hd:
        raise SystemExit("No query provided. Supply --query_sd or --query_hd.")

    for modality, query_path, prefix in [
        ("sd", args.query_sd, "sd"),
        ("hd", args.query_hd, "hd"),
    ]:
        if (modality == "sd" and not run_sd) or (modality == "hd" and not run_hd):
            continue
        logger.info("Loading %s query: %s", modality.upper(), query_path)
        adata = sc.read_h5ad(query_path)
        evaluate(
            adata=adata, model_dir=args.model_dir, outdir=outdir,
            modality=modality, prefix=prefix,
            section_col=args.section_col,
            fine_col=args.fine_col, coarse_col=args.coarse_col,
            theta=args.theta, compute_eval=not args.no_eval,
            n_pcs=args.n_pcs, n_sections=args.plot_sections,
        )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: `tissuetypist pseudobulk-hd`
# ─────────────────────────────────────────────────────────────────────────────

def _add_pseudobulk_hd_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "pseudobulk-hd",
        help="Aggregate raw Visium HD cells into sliding-window pseudo-spots",
        description="Pre-compute HD pseudobulk windows so training doesn't "
                    "re-run the expensive sliding-window step. Output: "
                    "{outdir}/adata_hd_windows.h5ad + pseudobulk_summary.csv.",
    )
    p.add_argument("--hd",           required=True)
    p.add_argument("--scalefactors", required=True)
    p.add_argument("--outdir",       required=True)
    p.add_argument("--fine_col",     default="niche_fine_Apr2026")
    p.add_argument("--coarse_col",   default="niche_coarse_Apr2026")
    p.add_argument("--target_spot_um", type=float, default=55.0)
    p.add_argument("--min_cells",      type=int,   default=1)
    p.add_argument("--log_file",     default=None)
    p.set_defaults(func=_run_pseudobulk_hd)
    return p


def _run_pseudobulk_hd(args: argparse.Namespace) -> int:
    # Defer to the existing script's core logic, which already uses the new
    # tissuetypist.data.pseudobulk path internally after the Phase 3b sweep.
    import json
    import scanpy as sc
    from tissuetypist.data.pseudobulk import sliding_window_pseudobulk_hd

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_file, outdir, cmd_name="pseudobulk")

    logger.info("Loading scale factors: %s", args.scalefactors)
    with open(args.scalefactors) as f:
        sfs = json.load(f)

    logger.info("Loading HD AnnData: %s", args.hd)
    adata_hd = sc.read_h5ad(args.hd)
    logger.info("  %d cells × %d genes", adata_hd.n_obs, adata_hd.n_vars)

    for col in (args.fine_col, args.coarse_col):
        if col not in adata_hd.obs.columns:
            raise SystemExit(
                f"obs column {col!r} not found. "
                f"Pass --fine_col / --coarse_col to override."
            )

    adata_windows = sliding_window_pseudobulk_hd(
        adata_hd,
        microns_per_pixel_map=sfs.get("microns_per_pixel_map", sfs),
        target_spot_um=args.target_spot_um,
        min_cells=args.min_cells,
        niche_fine_col=args.fine_col,
        niche_coarse_col=args.coarse_col,
    )
    out_path = outdir / "adata_hd_windows.h5ad"
    adata_windows.write_h5ad(out_path)
    logger.info("Saved: %s  (%d windows)", out_path, adata_windows.n_obs)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: `tissuetypist build-catalogue`
# ─────────────────────────────────────────────────────────────────────────────

def _add_build_catalogue_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "build-catalogue",
        help="Phase 0: per-modality gene detection rates + shared_all pool",
        description="Computes the shared gene catalogue across one or more "
                    "reference datasets. Writes {outdir}/gene_pools.csv used "
                    "by `train` / `train-panel` as --gene_pools.\n\n"
                    "Accepts variable inputs: pass --reference once per "
                    "dataset (minimum 1). For the cardiac reference pipeline, "
                    "use --sd3p / --sd_ffpe / --hd aliases.",
    )
    p.add_argument(
        "--reference", action="append", default=[],
        help="Reference AnnData (.h5ad). Repeat this flag once per dataset. "
             "When only one is provided, shared_all equals that dataset's "
             "detected genes.",
    )
    # Cardiac-legacy aliases (kept for backward compatibility).
    p.add_argument("--sd3p",    default=None,
                   help="Cardiac-legacy alias: adds this file to --reference.")
    p.add_argument("--sd_ffpe", default=None,
                   help="Cardiac-legacy alias: adds this file to --reference.")
    p.add_argument("--hd",      default=None,
                   help="Cardiac-legacy alias: adds this file to --reference.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--pseudobulk", action="store_true",
                   help="Use pseudobulked detection rates (recommended — and "
                        "required to reproduce the cardiac-paper Phase 0 "
                        "shared_all pool).")
    p.add_argument(
        "--groupby_sd", nargs="+", default=None,
        help="Pseudobulk groupby columns for SD modalities (repeatable). "
             "Default: ['donor', 'section_ID', 'niche_coarse_Apr2026']. "
             "Used only with --pseudobulk.",
    )
    p.add_argument(
        "--groupby_hd", nargs="+", default=None,
        help="Pseudobulk groupby columns for HD modality (repeatable). "
             "Default: ['donor', 'section_ID', 'niche_coarse_Apr2026']. "
             "Used only with --pseudobulk.",
    )
    p.add_argument("--log_file", default=None)
    p.set_defaults(func=_run_build_catalogue)
    return p


def _run_build_catalogue(args: argparse.Namespace) -> int:
    # Validate arguments before importing heavy deps so usage errors surface
    # cleanly in minimal environments (no scanpy required).
    paths: list[str] = list(args.reference)
    for alias in (args.sd3p, args.sd_ffpe, args.hd):
        if alias:
            paths.append(alias)
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    if not paths:
        raise SystemExit(
            "build-catalogue requires at least one reference dataset. "
            "Pass --reference (repeatable) or --sd3p/--sd_ffpe/--hd."
        )

    import scanpy as sc
    from tissuetypist.features.gene_catalogue import (
        build_gene_pools_from_paths,
        build_gene_pools_pseudobulk_from_paths,
        save_gene_pools,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_file, outdir, cmd_name="catalogue")

    logger.info("Reference datasets (%d):", len(paths))
    for p in paths:
        logger.info("  %s", p)

    # Single input: shared_all = detected genes in that dataset.
    if len(paths) == 1:
        logger.info("Single-reference mode: shared_all = var_names of the input.")
        adata = sc.read_h5ad(paths[0])
        # `save_gene_pools` takes a dict {pool_name: [genes]}.
        pools = {"shared_all": list(adata.var_names)}

    elif len(paths) == 3:
        # 3 inputs: delegate to the canonical library wrappers (cardiac
        # pipeline's detection-rate-based intersection, with the option of
        # pseudobulk detection rates).
        logger.info("Using canonical %s gene-pool library function for 3 references.",
                    "pseudobulk" if args.pseudobulk else "per-cell")
        if args.pseudobulk:
            kwargs = {"path_sd3p": paths[0], "path_sd_ffpe": paths[1], "path_hd": paths[2]}
            if args.groupby_sd:
                kwargs["groupby_sd"] = list(args.groupby_sd)
            if args.groupby_hd:
                kwargs["groupby_hd"] = list(args.groupby_hd)
            pools = build_gene_pools_pseudobulk_from_paths(**kwargs)
        else:
            pools = build_gene_pools_from_paths(
                path_sd3p=paths[0], path_sd_ffpe=paths[1], path_hd=paths[2],
            )

    else:
        # 2 or 4+ inputs: hand-roll detection + intersection with
        # `get_detected_genes`, which is what the 3-arg library function does
        # internally anyway. Pseudobulk detection is 3-arg-only, so we fall
        # back to per-cell detection with a warning.
        import gc
        from tissuetypist.features.gene_catalogue import get_detected_genes
        if args.pseudobulk:
            logger.warning(
                "--pseudobulk currently requires exactly 3 references "
                "(library limitation). Falling back to per-cell detection "
                "rates for %d inputs.", len(paths),
            )
        if len(paths) > 3:
            logger.warning(
                "Dropping extras beyond 3 references (detection-rate loop "
                "keeps the first 3 slot names): %s", paths[3:],
            )

        default_names = ["SD_3prime", "SD_FFPE", "HD_FFPE"]
        sets_by_modality: dict[str, set[str]] = {}
        for i, path in enumerate(paths):
            name = default_names[i] if i < len(default_names) else f"modality_{i}"
            logger.info("Detecting expressed genes: %s  (slot %s)", path, name)
            adata = sc.read_h5ad(path)
            sets_by_modality[name] = set(get_detected_genes(adata))
            del adata
            gc.collect()
            logger.info("  %d detected genes retained.", len(sets_by_modality[name]))
        shared = sorted(set.intersection(*sets_by_modality.values()))
        pools = {"shared_all": shared}
        for name, genes in sets_by_modality.items():
            pools[name] = sorted(genes)
        logger.info("shared_all = %d genes across %d modalities", len(shared), len(paths))

    out_path = outdir / "gene_pools.csv"
    save_gene_pools(pools, out_path)
    logger.info("Saved gene pools → %s", out_path)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: `tissuetypist validate-hierarchy`
# ─────────────────────────────────────────────────────────────────────────────

def _add_validate_hierarchy_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "validate-hierarchy",
        help="Check a hierarchy YAML against an AnnData",
        description="Loads a YAML and, optionally, an AnnData, reports any "
                    "missing obs columns / unreachable labels / modality "
                    "mismatches. No training is run.",
    )
    p.add_argument("--hierarchy", required=True, help="Shipped name or path to YAML.")
    p.add_argument("--adata",     default=None, help="AnnData to validate against (optional).")
    p.set_defaults(func=_run_validate_hierarchy)
    return p


def _run_validate_hierarchy(args: argparse.Namespace) -> int:
    from tissuetypist.config.hierarchy import load_hierarchy
    # Console-only logging (no file) — validate is read-only.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    spec = load_hierarchy(args.hierarchy)
    logger.info("Loaded %r  (coarse=%r, fine=%r)",
                spec.name, spec.coarse_col, spec.fine_col)
    logger.info("  coarse niches (%d): %s", len(spec.coarse_niches), spec.coarse_niches)
    logger.info("  terminal coarse: %s", spec.terminal_coarse)
    logger.info("  sub-models (%d):", len(spec.sub_models))
    for parent, sm in spec.sub_models.items():
        logger.info("    %r (depth=%d)", parent, sm.depth)
        for i, st in enumerate(sm.stages, 1):
            logger.info("      stage %d: %r  classes=%s  modalities=%s",
                        i, st.model_name, st.classes, st.modalities)
    if spec.palette:
        logger.info("  palette entries: %d", len(spec.palette))
    if spec.gt_label_remap:
        logger.info("  gt_label_remap: %s", spec.gt_label_remap)

    if args.adata is None:
        logger.info("No --adata provided — schema-only validation complete.")
        return 0

    import scanpy as sc
    logger.info("Loading: %s", args.adata)
    adata = sc.read_h5ad(args.adata)

    problems = []
    if spec.coarse_col not in adata.obs.columns:
        problems.append(f"coarse_col {spec.coarse_col!r} not in obs")
    if spec.fine_col and spec.fine_col not in adata.obs.columns:
        problems.append(f"fine_col {spec.fine_col!r} not in obs")
    if not problems:
        coarse_in_obs = set(adata.obs[spec.coarse_col].dropna().astype(str).unique())
        missing_coarse = set(spec.coarse_niches) - coarse_in_obs
        extra_coarse   = coarse_in_obs - set(spec.coarse_niches)
        if missing_coarse:
            logger.warning("  coarse niches not in obs: %s", sorted(missing_coarse))
        if extra_coarse:
            logger.warning("  coarse labels in obs but not in spec: %s", sorted(extra_coarse))
    for p in problems:
        logger.error("  %s", p)
    return 1 if problems else 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: `tissuetypist info`
# ─────────────────────────────────────────────────────────────────────────────

def _add_info_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "info",
        help="List shipped weight presets and hierarchies",
        description="Prints the weight presets and hierarchy YAMLs shipped "
                    "with the installed tissuetypist package.",
    )
    p.set_defaults(func=_run_info)
    return p


def _run_info(args: argparse.Namespace) -> int:
    from tissuetypist import __version__
    from tissuetypist.config.hierarchy import list_shipped_hierarchies, load_hierarchy
    from tissuetypist.config.presets import WEIGHT_PRESETS
    from tissuetypist.models import list_shipped_presets

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"TissueTypist v{__version__}\n")

    installed = set(list_shipped_presets())
    print("Weight presets:")
    for name, preset in sorted(WEIGHT_PRESETS.items()):
        marker = "[installed]" if name in installed else "[NOT installed]"
        print(f"  {name:16s}  neigh={preset.neighbour_weight:<4}  "
              f"edge={preset.edge_weight:<4}  {marker}")
        print(f"      {preset.description}")
    missing = sorted(p for p in WEIGHT_PRESETS if p not in installed)
    if missing:
        print()
        print(f"  Missing presets: {missing}.")
        print(f"  After training, populate with "
              f"`bash scripts/07_populate_preset_models.sh`.")
    print()

    print("Shipped hierarchies:")
    for name in list_shipped_hierarchies():
        spec = load_hierarchy(name)
        print(f"  {name:16s}  ({len(spec.coarse_niches)} coarse, "
              f"{len(spec.sub_models)} sub-models)")
        print(f"      {spec.description.split(chr(10))[0][:100] if spec.description else ''}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Root dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tissuetypist",
        description="TissueTypist — tissue-niche classifier for spatial "
                    "transcriptomics. See `tissuetypist <subcommand> --help` "
                    "for each subcommand's options.",
    )
    p.add_argument(
        "--version", action="store_true",
        help="Print the tissuetypist version and exit.",
    )
    sub = p.add_subparsers(dest="cmd", metavar="<subcommand>")
    _add_train_parser(sub)
    _add_train_panel_parser(sub)
    _add_predict_parser(sub)
    _add_evaluate_parser(sub)
    _add_pseudobulk_hd_parser(sub)
    _add_build_catalogue_parser(sub)
    _add_validate_hierarchy_parser(sub)
    _add_info_parser(sub)
    return p


def entry(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from tissuetypist import __version__
        print(f"tissuetypist {__version__}")
        return 0

    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(entry())
