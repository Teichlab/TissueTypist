"""
scripts/18_loso_accuracy.py
===========================
Leave-one-section-out accuracy metrics for the TissueTypist manuscript.

Each fold retrains the full hierarchy on all sections *except* the
held-out one (per-sub-model DEG+HVG recomputed from the fold's training
set) and predicts the held-out section.

Outputs (under ``--outdir``):

    fold_roster.csv                             — the 14-fold roster
    per_fold/<fold_tag>__predictions.parquet    — per-spot predictions
    per_fold/<fold_tag>__metrics.json           — per-fold F1 summary
    per_fold_metrics.csv                        — concatenated per-fold metrics
    pooled_predictions.csv.gz                   — concatenated per-spot table
    overall_f1_by_modality.csv                  — headline table
    per_niche_f1.csv                            — per-class precision/recall/F1
    confusion_{coarse,fine}.csv                 — row-normalised CMs
    confusion_submodel_<parent>.csv             — per sub-model CMs
    plots/overall_f1_by_modality.pdf
    plots/per_niche_f1.pdf
    plots/confusion_matrices.pdf
    plots/per_submodel_confusion_matrices.pdf

Usage
-----
    cd ~/GitHub/TissueTypist
    conda activate tissuetypist

    # Dry-run: only the first fold (validates end-to-end plumbing).
    python scripts/18_loso_accuracy.py \\
        --sd3p        data/adata_sd_3p_raw.h5ad \\
        --sd_ffpe     data/adata_sd_ffpe_raw.h5ad \\
        --hd_windows  data/adata_hd_windows.h5ad \\
        --gene_pools  results/phase0_pseudobulk/gene_pools.csv \\
        --outdir      results/loso_accuracy_apr2026 \\
        --folds       1

    # Full 14-fold sweep (run in tmux + caffeinate -i).
    python scripts/18_loso_accuracy.py \\
        --sd3p        data/adata_sd_3p_raw.h5ad \\
        --sd_ffpe     data/adata_sd_ffpe_raw.h5ad \\
        --hd_windows  data/adata_hd_windows.h5ad \\
        --gene_pools  results/phase0_pseudobulk/gene_pools.csv \\
        --outdir      results/loso_accuracy_apr2026

    # Skip training entirely and regenerate plots from cached per-fold parquets.
    python scripts/18_loso_accuracy.py \\
        --outdir      results/loso_accuracy_apr2026 \\
        --plots_only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

from tissuetypist.config.hierarchy import load_hierarchy
from tissuetypist.evaluation.loso import (
    DEFAULT_DONOR_PINS,
    SCORING_MODES,
    aggregate_overall_metrics,
    aggregate_per_niche_metrics,
    build_confusion_matrices,
    run_loso,
)
from tissuetypist.evaluation.loso_plots import (
    plot_confusion_matrices,
    plot_overall_f1_by_modality,
    plot_per_niche_f1,
    plot_per_submodel_confusion_matrices,
)
from tissuetypist.training.hierarchical import (
    TrainingConfig,
    _empty_adata,
    resolve_gene_pool,
)


LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s  %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("18_loso_accuracy")


# Cardiac-specific per-niche footnotes for the paper figure. The keys must
# match the label strings used in ``niche_fine_Apr2026``. Only niches listed
# here get a marker on the plot; everything else is unannotated. Users of a
# different tissue / hierarchy can override via ``--annotations`` JSON.
CARDIAC_DEFAULT_ANNOTATIONS: dict[str, str] = {
    "Lymph node": (
        "LOSO artefact: single-donor class (Hst45, SD FFPE) — no training "
        "examples remain in the other section's fold."
    ),
    "AV ring": (
        "Systematic confusion with Endocardial cushion - Valve within the "
        "AV-junction sub-model."
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOSO cross-validation for manuscript accuracy metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sd3p",       help="SD 3-prime reference h5ad.")
    p.add_argument("--sd_ffpe",    help="SD FFPE reference h5ad.")
    p.add_argument("--hd_windows", help="HD pseudobulk windows h5ad.")
    p.add_argument(
        "--gene_pools",
        help="Optional path to gene_pools.csv or a plain gene list. "
             "If omitted, uses the intersection of var_names across the "
             "three references (same fallback as `tissuetypist train`).",
    )
    p.add_argument(
        "--outdir",
        default="results/loso_accuracy_apr2026",
        help="Output directory (created if missing).",
    )
    p.add_argument("--section_col", default="section_ID")
    p.add_argument(
        "--max_sections_per_donor", type=int, default=2,
        help="Maximum number of sections per donor per modality.",
    )
    p.add_argument(
        "--donor_pins", default=None,
        help="Optional JSON mapping {donor: [section_id, ...]} to pin "
             "specific sections for a donor (bypasses random selection). "
             "If omitted, the built-in default pin {'C83': [ST10317184, "
             "ST10317186]} from loso.DEFAULT_DONOR_PINS is used.",
    )
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--theta", type=float, default=0.5,
                   help="Coarse confidence threshold for chain-walker fallbacks.")
    p.add_argument(
        "--feature_set", default="deg_hvg", choices=["deg_hvg", "deg_only"],
    )
    p.add_argument(
        "--neighbour_weight", type=float, default=0.3,
        help="Weight for the neighbour-max spatial features (default: 0.3, "
             "matches the shipped 'default' preset). Set to 0.0 to use the "
             "'own_only' preset (gene-expression only, no spatial features).",
    )
    p.add_argument(
        "--edge_weight", type=float, default=5.0,
        help="Weight for the distance-to-edge feature (default: 5.0, matches "
             "the shipped 'default' preset). Set to 0.0 alongside "
             "--neighbour_weight=0 for the 'own_only' preset.",
    )
    p.add_argument(
        "--folds", type=int, default=None,
        help="If set, run only the first N folds (useful for dry-runs).",
    )
    p.add_argument(
        "--no_resume", action="store_true",
        help="Ignore cached per-fold outputs and rerun every fold.",
    )
    p.add_argument(
        "--plots_only", action="store_true",
        help="Skip training entirely. Expects per_fold/*.parquet to "
             "already exist under --outdir; regenerates tables + plots.",
    )
    p.add_argument(
        "--hierarchy", default="cardiac",
        help="Hierarchy name (shipped) or path to a YAML spec.",
    )
    p.add_argument(
        "--annotations", default=None,
        help="Optional JSON {niche: footnote_text} for per-niche plot markers. "
             "If omitted and --hierarchy=cardiac, uses "
             "CARDIAC_DEFAULT_ANNOTATIONS. Pass '{}' to disable all footnotes.",
    )
    return p.parse_args()


def _parse_donor_pins(raw: str | None) -> dict[str, list[str]] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--donor_pins is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise SystemExit("--donor_pins must be a JSON object.")
    for donor, sections in parsed.items():
        if not isinstance(sections, list) or not all(isinstance(s, str) for s in sections):
            raise SystemExit(f"--donor_pins[{donor!r}] must be a list of strings.")
    return parsed


def _load_reference(path: str | None, name: str, template) -> "sc.AnnData":
    if path:
        logger.info("Loading %s reference: %s", name, path)
        return sc.read_h5ad(path)
    logger.info("No %s reference provided — using empty placeholder.", name)
    return _empty_adata(template)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Hierarchy spec ─────────────────────────────────────────────────────
    logger.info("Loading hierarchy: %s", args.hierarchy)
    spec = load_hierarchy(args.hierarchy)

    # ── Resolve annotations (JSON flag > cardiac default > empty) ─────────
    if args.annotations is not None:
        try:
            annotations = json.loads(args.annotations)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--annotations is not valid JSON: {e}") from e
        if not isinstance(annotations, dict):
            raise SystemExit("--annotations must be a JSON object.")
    elif args.hierarchy == "cardiac":
        annotations = dict(CARDIAC_DEFAULT_ANNOTATIONS)
    else:
        annotations = {}

    # ── Plots-only short-circuit ───────────────────────────────────────────
    if args.plots_only:
        parquets = sorted((outdir / "per_fold").glob("*__predictions.parquet"))
        if not parquets:
            raise SystemExit(
                f"--plots_only but no per_fold/*.parquet under {outdir}."
            )
        logger.info("plots_only: concatenating %d per-fold parquets.", len(parquets))
        pooled_df = pd.concat([pd.read_parquet(p) for p in parquets],
                              ignore_index=True)
        pooled_df.to_csv(outdir / "pooled_predictions.csv.gz",
                         index=False, compression="gzip")
        _write_tables_and_plots(pooled_df, spec, outdir, annotations=annotations)
        return 0

    # ── Load references ────────────────────────────────────────────────────
    if args.sd3p is None:
        raise SystemExit("--sd3p is required unless --plots_only is set.")
    logger.info("Loading SD 3-prime reference: %s", args.sd3p)
    adata_sd3p = sc.read_h5ad(args.sd3p)
    adata_sdffpe = _load_reference(args.sd_ffpe,    "SD FFPE", adata_sd3p)
    adata_hd     = _load_reference(args.hd_windows, "HD windows", adata_sd3p)

    # ── Gene pool ──────────────────────────────────────────────────────────
    gene_pool = resolve_gene_pool(
        args.gene_pools,
        reference_adatas=[adata_sd3p, adata_sdffpe, adata_hd],
    )
    logger.info("Gene pool size: %d", len(gene_pool))

    # ── Training config ────────────────────────────────────────────────────
    training_config = TrainingConfig(
        hierarchy=spec,
        feature_set=args.feature_set,
        neighbour_weight=args.neighbour_weight,
        edge_weight=args.edge_weight,
        save_qc=False,
    )
    logger.info(
        "TrainingConfig: feature_set=%s, neighbour_weight=%.2f, edge_weight=%.2f",
        args.feature_set, args.neighbour_weight, args.edge_weight,
    )

    # ── Donor pins ─────────────────────────────────────────────────────────
    donor_pins = _parse_donor_pins(args.donor_pins) or dict(DEFAULT_DONOR_PINS)
    logger.info("Donor pins: %s", donor_pins)

    # ── Run LOSO ────────────────────────────────────────────────────────────
    pooled_df, per_fold_df, folds = run_loso(
        adata_sd3p=adata_sd3p,
        adata_sdffpe=adata_sdffpe,
        adata_hd=adata_hd,
        gene_pool=gene_pool,
        outdir=outdir,
        spec=spec,
        training_config=training_config,
        section_col=args.section_col,
        max_sections_per_donor=args.max_sections_per_donor,
        donor_pins=donor_pins,
        seed=args.seed,
        theta=args.theta,
        max_folds=args.folds,
        resume=not args.no_resume,
    )
    logger.info("Completed %d folds; pooled %d spots.", len(folds), len(pooled_df))

    _write_tables_and_plots(pooled_df, spec, outdir, annotations=annotations)
    return 0


def _write_tables_and_plots(
    pooled_df, spec, outdir: Path,
    annotations: dict[str, str] | None = None,
) -> None:
    """Aggregate metrics, write CSVs, produce plots (one set per scoring mode).

    Scoring modes in :data:`tissuetypist.evaluation.loso.SCORING_MODES`:
      - ``strict``             — everything counts (original behaviour).
      - ``fallback_excluded``  — drop spots with a non-leaf prediction.
      - ``resolution_aware``   — legacy GT intermediate mapping + intermediate-
        level rollup for GT-intermediate spots + fallback-excluded for GT-leaf.

    The overall-F1 table is long-form (one row per modality × mode). The
    per-niche tables, confusion matrices, and per-sub-model CMs are
    written separately for each mode so you can pick which one to cite
    in the paper.
    """
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── Overall F1 (long-form, all modes in one CSV + one PDF) ─────────────
    overall_df = aggregate_overall_metrics(pooled_df, spec)
    overall_df.to_csv(outdir / "overall_f1_by_modality.csv", index=False)
    logger.info("\n%s", overall_df.to_string(index=False))

    plot_overall_f1_by_modality(
        overall_df,
        outpath=plots_dir / "overall_f1_by_modality.pdf",
    )

    # ── Coarse CM is mode-independent — write once. ────────────────────────
    cms_ref = build_confusion_matrices(pooled_df, spec, mode="strict")
    if "coarse" in cms_ref:
        cms_ref["coarse"].to_csv(outdir / "confusion_coarse.csv")

    # ── Per-mode: per-niche + fine + per-sub-model CMs + plots ────────────
    for mode in SCORING_MODES:
        per_niche_df = aggregate_per_niche_metrics(pooled_df, spec, mode=mode)
        per_niche_df.to_csv(outdir / f"per_niche_f1_{mode}.csv", index=False)

        cms_mode = build_confusion_matrices(pooled_df, spec, mode=mode)
        for key, cm in cms_mode.items():
            if key == "coarse":
                continue  # already written above
            safe = key.replace("submodel__", "submodel_").replace(" ", "_")
            cm.to_csv(outdir / f"confusion_{safe}_{mode}.csv")

        plot_per_niche_f1(
            per_niche_df,
            spec=spec,
            outpath=plots_dir / f"per_niche_f1_{mode}.pdf",
            title=f"Per-niche F1 (pooled across LOSO folds, scoring_mode={mode})",
            annotations=annotations,
        )
        plot_confusion_matrices(
            cms_mode,
            outpath=plots_dir / f"confusion_matrices_{mode}.pdf",
            title=f"LOSO confusion matrices (row-normalised, scoring_mode={mode})",
        )
        plot_per_submodel_confusion_matrices(
            cms_mode,
            outpath=plots_dir / f"per_submodel_confusion_matrices_{mode}.pdf",
            title=f"Per sub-model confusion matrices (scoring_mode={mode})",
        )

    logger.info("All outputs written under %s", outdir)


if __name__ == "__main__":
    raise SystemExit(main())
