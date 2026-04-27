"""Shipped pre-trained model directories for TissueTypist.

Three cardiac hierarchies are shipped here, one per weight preset:

    default/            — neighbour_weight=0.3,  edge_weight=5.0
    own_only/           — neighbour_weight=0.0,  edge_weight=0.0
    neighbour_heavy/    — neighbour_weight=1.0,  edge_weight=5.0

Each directory is a complete Phase 3b training output for the cardiac
Apr 2026 hierarchy:

    coarse_pipeline.joblib + coarse_gene_list.txt
    ventricle_pipeline.joblib + ventricle_gene_list.txt
    avjunction_pipeline.joblib + avjunction_gene_list.txt
    atrium_split_pipeline.joblib + atrium_split_gene_list.txt
    atrium_transitional_pipeline.joblib + atrium_transitional_gene_list.txt
    atrium_lr_pipeline.joblib + atrium_lr_gene_list.txt
    pcs_split_pipeline.joblib + pcs_split_gene_list.txt
    pcs_sinoatrial_pipeline.joblib + pcs_sinoatrial_gene_list.txt
    vasc_split_pipeline.joblib + vasc_split_gene_list.txt
    vasc_fine_pipeline.joblib + vasc_fine_gene_list.txt
    hierarchy_config.json   (schema_version=2)
    training_summary.csv
    gene_counts.csv

The ``load_preset("<name>")`` helper in ``tissuetypist`` returns the path
to the selected preset; pass that to ``predict_adata(..., model_dir=...)``.

Repository developers populate these directories by running the helper
script ``scripts/07_populate_preset_models.sh`` after running all three
training configurations. The populator copies from ``results/apr2026_<name>/``
into this directory, excluding training logs and QC plots.
"""
from pathlib import Path


def preset_dir(name: str) -> Path:
    """Return the path to the shipped preset directory (may not exist).

    Existence is NOT checked — use ``tissuetypist.load_preset(name)`` for
    that. This helper is purely for path arithmetic.
    """
    return Path(__file__).parent / name


def list_shipped_presets() -> list[str]:
    """Return the names of preset directories that exist on disk AND
    contain a ``hierarchy_config.json`` (i.e. are complete).
    """
    root = Path(__file__).parent
    names = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "hierarchy_config.json").exists():
            names.append(child.name)
    return names


__all__ = ["preset_dir", "list_shipped_presets"]
