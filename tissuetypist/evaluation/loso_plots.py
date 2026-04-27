"""
tissuetypist/evaluation/loso_plots.py
=====================================
Manuscript-ready plots driven by the pooled DataFrame that
``tissuetypist.evaluation.loso.run_loso`` produces.

Four plot types (one PDF each):

1. ``plot_overall_f1_by_modality``  — grouped bar chart of weighted +
   macro F1 per modality × {coarse, fine}.
2. ``plot_per_niche_f1``            — horizontal bar of per-niche F1
   sorted descending, with per-class support on the y-ticks.
3. ``plot_confusion_matrices``      — coarse + fine row-normalised
   heatmaps as a 2-panel PDF.
4. ``plot_per_submodel_confusion_matrices`` — one small heatmap per
   parent-coarse sub-model chain, tiled into a single PDF.

All palettes flow through
``tissuetypist.evaluation.plots.build_palette_from_spec`` so the
cardiac ``spec.palette`` is respected.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tissuetypist.evaluation.plots import build_palette_from_spec

if TYPE_CHECKING:
    from tissuetypist.config.hierarchy import HierarchySpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Overall F1 bar chart
# ─────────────────────────────────────────────────────────────────────────────

_MODALITY_ORDER  = ["SD_3prime", "SD_FFPE", "HD", "all"]
_MODALITY_LABELS = {
    "SD_3prime": "Visium SD 3'",
    "SD_FFPE":   "Visium SD FFPE",
    "HD":        "Visium HD",
    "all":       "All (pooled)",
}


def plot_overall_f1_by_modality(
    overall_df: pd.DataFrame,
    outpath: Path,
    title: str = "LOSO F1 by modality",
) -> None:
    """Grouped bar chart: modality groups × (coarse weighted, coarse macro,
    fine weighted, fine macro), one subplot row per scoring mode.

    Accepts both the long-form DataFrame produced by
    ``loso.aggregate_overall_metrics`` (rows keyed on
    ``(modality, scoring_mode)`` with columns ``coarse_f1_weighted``,
    ``coarse_f1_macro``, ``fine_f1_weighted``, ``fine_f1_macro``, and an
    ``n_spots_pooled`` / ``fine_n_spots`` column) AND the legacy wide form
    (one row per modality, no ``scoring_mode`` column).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    has_mode = "scoring_mode" in overall_df.columns
    modes    = list(overall_df["scoring_mode"].unique()) if has_mode else [None]

    mods_present = [m for m in _MODALITY_ORDER if m in overall_df["modality"].values]
    if not mods_present:
        logger.warning("plot_overall_f1_by_modality: overall_df is empty.")
        return

    metric_cols = [
        ("coarse_f1_weighted", "Coarse F1 (weighted)", "#4c72b0"),
        ("coarse_f1_macro",    "Coarse F1 (macro)",    "#8da0cb"),
        ("fine_f1_weighted",   "Fine F1 (weighted)",   "#dd8452"),
        ("fine_f1_macro",      "Fine F1 (macro)",      "#e3a77e"),
    ]

    x = np.arange(len(mods_present))
    w = 0.2
    nrows = len(modes)
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(max(6, 1.2 * len(mods_present) + 3), 3.6 * nrows + 0.8),
        sharey=True,
        squeeze=False,
    )

    for row, mode in enumerate(modes):
        ax = axes[row][0]
        if has_mode:
            df_mode = overall_df[overall_df["scoring_mode"] == mode]
        else:
            df_mode = overall_df

        for i, (col, label, colour) in enumerate(metric_cols):
            vals = []
            for m in mods_present:
                sub = df_mode[df_mode["modality"] == m]
                vals.append(float(sub[col].iloc[0]) if not sub.empty else 0.0)
            ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=colour)
            for xpos, v in zip(x + (i - 1.5) * w, vals):
                ax.text(xpos, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels([_MODALITY_LABELS.get(m, m) for m in mods_present])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 score")

        subtitle = title if mode is None else f"{title}  —  scoring_mode = {mode}"
        ax.set_title(subtitle, fontsize=10)
        if row == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)

        # Annotate sample sizes. For mode-aware rows, show fine_n_spots
        # (which may be smaller than the pooled total when the mode drops
        # spots); otherwise fall back to n_spots / n_spots_pooled.
        n_col = "fine_n_spots" if "fine_n_spots" in df_mode.columns else \
                ("n_spots_pooled" if "n_spots_pooled" in df_mode.columns else "n_spots")
        for xi, m in zip(x, mods_present):
            sub = df_mode[df_mode["modality"] == m]
            if sub.empty:
                continue
            n = int(sub[n_col].iloc[0])
            ax.text(xi, -0.06, f"n = {n:,}", ha="center", va="top",
                    fontsize=7, color="#555555",
                    transform=ax.get_xaxis_transform())

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved overall-F1 plot → %s", outpath)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-niche F1 bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_niche_f1(
    per_niche_df: pd.DataFrame,
    spec: "HierarchySpec",
    outpath: Path,
    title: str = "Per-niche F1 (pooled across LOSO folds)",
    exclude_summary_rows: tuple[str, ...] = ("accuracy", "macro avg", "weighted avg"),
    drop_zero_support: bool = True,
    group_by_coarse: bool = True,
    colour_by_coarse: bool = True,
    annotations: Optional[dict[str, str]] = None,
) -> None:
    """Horizontal bar chart of F1-score per fine niche.

    Parameters
    ----------
    per_niche_df :
        Output of ``loso.aggregate_per_niche_metrics``. Must contain
        ``niche``, ``f1-score``, ``support``.
    spec :
        HierarchySpec (used for palette + coarse grouping).
    exclude_summary_rows :
        Names to drop from the report (``accuracy``, ``macro avg``, ...).
    drop_zero_support :
        Drop rows with ``support == 0`` (default). These are residual
        false-positive classes that add no information to the figure.
    group_by_coarse :
        Sort bars by parent coarse niche first, then by F1 within each
        group. Keeps related sub-niches adjacent. Default ``True``.
    colour_by_coarse :
        Colour every leaf bar with its parent coarse niche's palette
        entry, so the hierarchy structure is visible at a glance.
        Default ``True``. When ``False``, uses the leaf's own palette
        entry (legacy behaviour).
    annotations :
        Optional ``{niche: footnote_text}`` mapping. Niches present here
        get a marker († / ‡ / § / ¶) appended to their tick label; the
        figure gets a footnote block at the bottom listing the markers.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = per_niche_df[~per_niche_df["niche"].isin(exclude_summary_rows)].copy()
    if drop_zero_support and "support" in df.columns:
        df = df[df["support"].astype(float) > 0]
    if df.empty:
        logger.warning("plot_per_niche_f1: no per-niche rows after filtering.")
        return

    # ── Ordering ──────────────────────────────────────────────────────────
    if group_by_coarse:
        from tissuetypist.evaluation.loso import build_leaf_to_coarse_map
        leaf_to_coarse = build_leaf_to_coarse_map(spec)
        # Keep the YAML coarse-niche order for group ordering.
        coarse_order = {c: i for i, c in enumerate(spec.coarse_niches)}
        df["_coarse"]     = df["niche"].map(lambda n: leaf_to_coarse.get(n, "Other"))
        df["_coarse_ord"] = df["_coarse"].map(lambda c: coarse_order.get(c, 99))
        # Within group: ascending F1 so the best-performing sub-niche is on top.
        df = df.sort_values(["_coarse_ord", "f1-score"], ascending=[True, True])
        df = df.reset_index(drop=True)
    else:
        df = df.sort_values("f1-score", ascending=True).reset_index(drop=True)
        df["_coarse"] = df["niche"]

    # ── Colours ───────────────────────────────────────────────────────────
    if colour_by_coarse:
        palette = build_palette_from_spec(spec, labels=list(df["_coarse"].unique()))
        colours = [palette.get(c, "#cccccc") for c in df["_coarse"]]
    else:
        palette = build_palette_from_spec(spec, labels=df["niche"].tolist())
        colours = [palette.get(n, "#cccccc") for n in df["niche"]]

    # ── Annotation markers ────────────────────────────────────────────────
    annotations = dict(annotations or {})
    marker_order = ("†", "‡", "§", "¶", "#", "*")
    label_to_marker: dict[str, str] = {}
    if annotations:
        for i, lbl in enumerate(sorted(annotations)):
            if i < len(marker_order):
                label_to_marker[lbl] = marker_order[i]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(df) + 1.8))
    y    = np.arange(len(df))
    bars = ax.barh(y, df["f1-score"].astype(float), color=colours,
                   edgecolor="#333333", linewidth=0.3)

    tick_labels = []
    for n, s in zip(df["niche"], df["support"].astype(float)):
        marker = label_to_marker.get(n, "")
        suffix = f" {marker}" if marker else ""
        tick_labels.append(f"{n}  (n={int(s):,}){suffix}")
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("F1 score")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#bbbbbb", zorder=0)

    for b, v in zip(bars, df["f1-score"].astype(float)):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", ha="left", fontsize=7)

    # ── Legend (coarse-niche colour key) ──────────────────────────────────
    if colour_by_coarse:
        import matplotlib.patches as mpatches
        unique_coarse = list(dict.fromkeys(df["_coarse"].tolist()))  # preserve order
        handles = [
            mpatches.Patch(color=palette.get(c, "#cccccc"), label=c)
            for c in unique_coarse
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=7,
            frameon=False,
            title="Coarse niche",
            title_fontsize=8,
        )

    # ── Footnotes ─────────────────────────────────────────────────────────
    if label_to_marker:
        footnote_lines = [
            f"{label_to_marker[lbl]}  {annotations[lbl]}"
            for lbl in sorted(label_to_marker, key=lambda x: marker_order.index(label_to_marker[x]))
        ]
        fig.text(0.01, 0.0, "\n".join(footnote_lines),
                 fontsize=7, va="bottom", ha="left", color="#333333")

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-niche F1 plot → %s", outpath)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Coarse + fine confusion matrices (single PDF)
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap(ax, cm: pd.DataFrame, title: str, small: bool = False) -> None:
    im = ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    labels = cm.index.tolist()
    fs = 6 if small else 7
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, fontsize=10)


def plot_confusion_matrices(
    cms: dict[str, pd.DataFrame],
    outpath: Path,
    title: str = "LOSO confusion matrices (row-normalised)",
) -> None:
    """Two-panel PDF: coarse CM (left) + fine CM (right).

    ``cms`` is the output of ``loso.build_confusion_matrices``; this
    function only looks at the ``"coarse"`` and ``"fine"`` keys.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    coarse_cm = cms.get("coarse")
    fine_cm   = cms.get("fine")
    if coarse_cm is None and fine_cm is None:
        logger.warning("plot_confusion_matrices: nothing to plot.")
        return

    width = 10 + (0 if fine_cm is None else max(0, len(fine_cm) - 8) * 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(width, 8))
    if coarse_cm is not None:
        _heatmap(axes[0], coarse_cm, "Coarse")
    else:
        axes[0].axis("off")
    if fine_cm is not None:
        _heatmap(axes[1], fine_cm, "Fine (tt_final_label vs GT)")
    else:
        axes[1].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved coarse+fine confusion matrices → %s", outpath)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-sub-model confusion matrices (tiled PDF)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_submodel_confusion_matrices(
    cms: dict[str, pd.DataFrame],
    outpath: Path,
    title: str = "Per sub-model confusion matrices (row-normalised)",
) -> None:
    """Tiled PDF: one mini-heatmap per ``submodel__<parent>`` entry in ``cms``.

    Tiles scale to at most 3 per row; useful as a supplementary figure
    so reviewers can see where each sub-model's errors sit.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(k for k in cms if k.startswith("submodel__"))
    if not keys:
        logger.warning("plot_per_submodel_confusion_matrices: no sub-model CMs.")
        return

    ncols = min(3, len(keys))
    nrows = int(math.ceil(len(keys) / ncols))

    # Size tiles by the largest sub-model CM so everything fits.
    max_dim = max(len(cms[k]) for k in keys)
    tile_w  = max(3.5, 0.4 * max_dim + 2)
    tile_h  = max(3.0, 0.35 * max_dim + 1.6)

    fig, axes = plt.subplots(nrows, ncols, figsize=(tile_w * ncols, tile_h * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, key in enumerate(keys):
        r, c = divmod(i, ncols)
        parent = key[len("submodel__"):]
        _heatmap(axes[r][c], cms[key], parent, small=True)

    # Hide unused axes.
    for j in range(len(keys), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-sub-model CM plot → %s", outpath)


__all__ = [
    "plot_overall_f1_by_modality",
    "plot_per_niche_f1",
    "plot_confusion_matrices",
    "plot_per_submodel_confusion_matrices",
]
