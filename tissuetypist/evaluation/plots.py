"""
tissuetypist/evaluation/plots.py
=================================
Plotting helpers for TissueTypist prediction evaluation.

All plots accept a :class:`~tissuetypist.config.HierarchySpec` and
derive their colour palette via :func:`build_palette_from_spec`, which
uses ``spec.palette`` as the curated source and falls back to tab20
iteration colours for any label absent from the spec. This means the
same plotting code works for cardiac, non-cardiac, Apr2026 labels,
and any user-supplied YAML.

Public API
----------
    build_palette_from_spec(spec, labels=None)
    plot_confusion_matrix(adata, fine_col, outdir, prefix, spec)
    plot_spatial(adata, fine_col, outdir, prefix, spec, ...)
    plot_umap(adata, fine_col, outdir, prefix, spec, n_pcs=30)
    plot_confidence_distributions(adata, outdir, prefix)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad
    from tissuetypist.config.hierarchy import HierarchySpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Palette builder
# ─────────────────────────────────────────────────────────────────────────────

def build_palette_from_spec(
    spec: "HierarchySpec",
    labels: Optional[Iterable[str]] = None,
    fallback_cmap: str = "tab20",
) -> dict[str, str]:
    """Construct a ``{label: hex_colour}`` mapping for plotting.

    Labels drawn from ``spec.palette`` take precedence. Labels present in
    ``labels`` but absent from ``spec.palette`` are filled with colours
    from ``fallback_cmap`` in alphabetical order (deterministic across
    runs).

    Parameters
    ----------
    spec :
        The loaded :class:`HierarchySpec`.
    labels :
        If given, ensures every label here has an entry in the returned
        dict (filling from the fallback cmap). If ``None``, the returned
        dict equals ``spec.palette``.
    fallback_cmap :
        Matplotlib colormap name used for fill-ins.

    Returns
    -------
    dict[str, str]
        Keyed by label; values are hex colours (or colour strings that
        matplotlib accepts).
    """
    import matplotlib.cm as cm

    palette: dict[str, str] = dict(spec.palette)

    if labels is None:
        return palette

    missing = [lbl for lbl in labels if lbl not in palette]
    if not missing:
        return palette

    cmap = cm.get_cmap(fallback_cmap)
    # tab20 → 20 colours; iterate with wrap for long label lists.
    n_colours = cmap.N if hasattr(cmap, "N") else 20
    for i, lbl in enumerate(sorted(missing)):
        rgba = cmap(i % n_colours)
        palette[lbl] = _rgba_to_hex(rgba)

    return palette


def _rgba_to_hex(rgba) -> str:
    r, g, b, _ = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    adata: "ad.AnnData",
    fine_col: str,
    outdir: Path,
    prefix: str,
    spec: "HierarchySpec",
) -> None:
    """Plot normalised confusion matrix (rows=GT, cols=predicted).

    GT labels are remapped through ``spec.gt_label_remap`` before
    computing the matrix.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from .metrics import remap_ground_truth

    obs = adata.obs.copy()
    valid = obs["tt_final_label"].notna()
    if fine_col in obs.columns:
        valid &= obs[fine_col].notna()
    obs = obs[valid]
    if len(obs) == 0:
        logger.warning("plot_confusion_matrix: no spots with prediction + GT.")
        return

    y_true  = remap_ground_truth(obs[fine_col].astype(str), spec)
    y_pred  = obs["tt_final_label"].astype(str)
    classes = sorted(set(y_true) | set(y_pred))

    cm_arr = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

    fig, ax = plt.subplots(
        figsize=(max(10, len(classes) * 0.6), max(8, len(classes) * 0.5))
    )
    im = ax.imshow(cm_arr, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall (row-normalised)")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted (tt_final_label)")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{prefix} — Confusion matrix (row-normalised)")

    plt.tight_layout()
    out_path = Path(outdir) / f"{prefix}_confusion_matrix.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial(
    adata: "ad.AnnData",
    fine_col: str,
    outdir: Path,
    prefix: str,
    spec: "HierarchySpec",
    section_col: str = "section_ID",
    n_sections: int = 3,
) -> None:
    """For the first ``n_sections`` sections, plot GT | prediction | confidence."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if "spatial" not in adata.obsm:
        logger.warning("obsm['spatial'] not found — skipping spatial plots.")
        return

    has_gt = fine_col in adata.obs.columns
    sections = (
        adata.obs[section_col].unique()
        if section_col in adata.obs.columns
        else adata.obs.get("section_ID", pd.Series()).unique()
    )
    sections = list(sections)[:n_sections]

    all_labels = sorted(
        (set(adata.obs[fine_col].dropna().astype(str).unique()) if has_gt else set())
        | set(adata.obs["tt_final_label"].dropna().astype(str).unique())
    )
    palette = build_palette_from_spec(spec, labels=all_labels)

    for section in sections:
        mask = adata.obs[section_col] == section
        sub  = adata[mask]
        xy   = sub.obsm["spatial"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1 — ground truth.
        if has_gt:
            labels  = sub.obs[fine_col].astype(str).values
            colours = [palette.get(l, "#cccccc") for l in labels]
            axes[0].scatter(xy[:, 0], xy[:, 1], c=colours, s=4, linewidths=0)
            axes[0].set_title(f"Ground truth\n{section}", fontsize=9)
        else:
            axes[0].set_title(f"Ground truth (not available)\n{section}", fontsize=9)
            axes[0].text(0.5, 0.5, "No ground truth",
                         transform=axes[0].transAxes,
                         ha="center", va="center", fontsize=9, color="grey")
        axes[0].set_aspect("equal")
        axes[0].axis("off")

        # Panel 2 — prediction.
        labels  = sub.obs["tt_final_label"].astype(str).values
        colours = [palette.get(l, "#cccccc") for l in labels]
        axes[1].scatter(xy[:, 0], xy[:, 1], c=colours, s=4, linewidths=0)
        axes[1].set_title(f"TissueTypist prediction\n{section}", fontsize=9)
        axes[1].set_aspect("equal")
        axes[1].axis("off")

        # Panel 3 — coarse confidence heatmap.
        scores = sub.obs["tt_coarse_score"].values.astype(float)
        sc_plot = axes[2].scatter(
            xy[:, 0], xy[:, 1], c=scores,
            cmap="RdYlGn", vmin=0, vmax=1, s=4, linewidths=0,
        )
        plt.colorbar(sc_plot, ax=axes[2], label="tt_coarse_score")
        axes[2].set_title(f"Coarse confidence\n{section}", fontsize=9)
        axes[2].set_aspect("equal")
        axes[2].axis("off")

        legend_labels = sorted(set(
            sub.obs[fine_col].dropna().astype(str).unique()
            if has_gt else
            sub.obs["tt_final_label"].dropna().astype(str).unique()
        ))
        handles = [
            mpatches.Patch(color=palette.get(l, "#cccccc"), label=l)
            for l in legend_labels
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=min(6, len(legend_labels)),
            fontsize=7, bbox_to_anchor=(0.5, -0.02),
        )

        plt.tight_layout()
        safe_section = str(section).replace("/", "_").replace(" ", "_")
        out_path = Path(outdir) / f"{prefix}_spatial_{safe_section}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved spatial plot → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap(
    adata: "ad.AnnData",
    fine_col: str,
    outdir: Path,
    prefix: str,
    spec: "HierarchySpec",
    n_pcs: int = 30,
) -> None:
    """PCA → neighbours → UMAP, then plot GT | prediction | confidence | low_conf."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import scanpy as sc
    from tissuetypist.data.normalise import normalise_if_needed

    logger.info("Computing PCA + UMAP (%d PCs)...", n_pcs)

    adata_umap = adata.copy()
    adata_umap = normalise_if_needed(adata_umap, name="UMAP input")

    n_genes = adata_umap.n_vars
    n_hvg   = min(3000, n_genes)
    flavor  = "seurat" if n_genes >= 500 else "cell_ranger"
    if n_genes < 3000:
        logger.info(
            "  Small gene panel (%d genes) — using n_top_genes=%d, flavor='%s'.",
            n_genes, n_hvg, flavor,
        )
    sc.pp.highly_variable_genes(adata_umap, n_top_genes=n_hvg, flavor=flavor)
    n_pcs_actual = min(n_pcs, n_genes - 1, adata_umap.n_obs - 1)
    if n_pcs_actual < n_pcs:
        logger.info(
            "  Reducing n_pcs from %d to %d (limited by n_genes=%d, n_obs=%d).",
            n_pcs, n_pcs_actual, n_genes, adata_umap.n_obs,
        )
    sc.pp.pca(adata_umap, n_comps=n_pcs_actual, use_highly_variable=True)
    sc.pp.neighbors(adata_umap, n_pcs=n_pcs_actual)
    sc.tl.umap(adata_umap)

    for col in ["tt_final_label", "tt_coarse_label", "tt_coarse_score",
                "tt_low_conf", fine_col]:
        if col in adata.obs.columns:
            adata_umap.obs[col] = adata.obs[col].values

    has_gt = fine_col in adata_umap.obs.columns
    umap_xy = adata_umap.obsm["X_umap"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    # Panels 1 + 2 — categorical (GT, prediction).
    for ax, col, title in [
        (axes[0], fine_col,         "Ground truth"),
        (axes[1], "tt_final_label", "Prediction"),
    ]:
        if col not in adata_umap.obs.columns:
            ax.set_title(f"{title} (not available)", fontsize=9)
            ax.text(0.5, 0.5, "No ground truth", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="grey")
            ax.axis("off")
            continue
        labels   = adata_umap.obs[col].astype(str).fillna("unknown")
        present  = sorted(labels.unique())
        palette  = build_palette_from_spec(spec, labels=present)
        colours  = [palette.get(l, "#cccccc") for l in labels]
        ax.scatter(umap_xy[:, 0], umap_xy[:, 1],
                   c=colours, s=1, linewidths=0, alpha=0.6)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        handles = [mpatches.Patch(color=palette.get(l, "#cccccc"), label=l)
                   for l in present if l != "unknown"]
        ax.legend(handles=handles, fontsize=5, loc="lower left",
                  framealpha=0.7, ncol=2)

    # Panel 3 — coarse confidence.
    scores = adata_umap.obs["tt_coarse_score"].values.astype(float)
    sc3 = axes[2].scatter(umap_xy[:, 0], umap_xy[:, 1],
                          c=scores, cmap="RdYlGn",
                          vmin=0, vmax=1, s=1, linewidths=0, alpha=0.6)
    plt.colorbar(sc3, ax=axes[2], label="tt_coarse_score")
    axes[2].set_title("Coarse confidence", fontsize=9)
    axes[2].axis("off")

    # Panel 4 — low confidence flag.
    low_conf = adata_umap.obs["tt_low_conf"].fillna(False).astype(bool)
    colours4 = np.where(low_conf, "red", "lightgrey")
    axes[3].scatter(umap_xy[:, 0], umap_xy[:, 1],
                    c=colours4, s=1, linewidths=0, alpha=0.6)
    axes[3].set_title(
        f"Low confidence (red)\n{low_conf.sum()} / {len(low_conf)} spots",
        fontsize=9,
    )
    axes[3].axis("off")

    plt.suptitle(f"{prefix} — UMAP", fontsize=11)
    plt.tight_layout()
    out_path = Path(outdir) / f"{prefix}_umap.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved UMAP → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_distributions(
    adata: "ad.AnnData",
    outdir: Path,
    prefix: str,
) -> None:
    """Violin plots of ``tt_coarse_score`` and ``tt_joint_score`` per predicted coarse niche.

    Helps identify niches where the model is uncertain.
    """
    import matplotlib.pyplot as plt

    obs = adata.obs.copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, score_col, title in [
        (axes[0], "tt_coarse_score", "Coarse confidence (tt_coarse_score)"),
        (axes[1], "tt_joint_score",  "Joint confidence (tt_joint_score)"),
    ]:
        plot_data = obs[["tt_coarse_label", score_col]].dropna()
        if plot_data.empty:
            continue

        niches  = sorted(plot_data["tt_coarse_label"].unique())
        data_by_niche = [
            plot_data.loc[plot_data["tt_coarse_label"] == n, score_col].values
            for n in niches
        ]

        parts = ax.violinplot(data_by_niche, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(niches) + 1))
        ax.set_xticklabels(niches, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8,
                   label="theta=0.5")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle(f"{prefix} — Confidence distributions", fontsize=11)
    plt.tight_layout()
    out_path = Path(outdir) / f"{prefix}_confidence_distributions.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confidence distributions → %s", out_path)


__all__ = [
    "build_palette_from_spec",
    "plot_confusion_matrix",
    "plot_spatial",
    "plot_umap",
    "plot_confidence_distributions",
]
