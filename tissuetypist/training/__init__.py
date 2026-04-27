"""Training modules for TissueTypist."""
from .logistic import (
    format_cv_summary,
    leave_one_modality_out_cv,
    stratified_kfold_cv,
    train_and_evaluate,
    train_logistic,
)
from .hierarchical import (
    TrainingConfig,
    train_all_models,
    load_data,
    resolve_gene_pool,
    build_features,
    compute_f2_genes_coarse,
    compute_f2_genes_fine,
    subset_features,
    fit_and_save_pipeline,
)
from .panel_specific import (
    GeneStrategy,
    compute_panel_shared,
    train_panel_specific,
)

__all__ = [
    # logistic
    "format_cv_summary",
    "leave_one_modality_out_cv",
    "stratified_kfold_cv",
    "train_and_evaluate",
    "train_logistic",
    # hierarchical
    "TrainingConfig",
    "train_all_models",
    "load_data",
    "resolve_gene_pool",
    "build_features",
    "compute_f2_genes_coarse",
    "compute_f2_genes_fine",
    "subset_features",
    "fit_and_save_pipeline",
    # panel_specific
    "GeneStrategy",
    "compute_panel_shared",
    "train_panel_specific",
]
