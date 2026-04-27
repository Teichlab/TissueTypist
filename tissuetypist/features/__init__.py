"""Feature engineering modules for TissueTypist.

Submodules:
    gene_catalogue   — per-modality detection rates + shared_all gene pool
    gene_selection   — DEG + HVG + F2 feature-gene selection
    spatial          — neighbour-max + edge-detection features
"""
from .gene_catalogue import (
    build_gene_pools,
    build_gene_pools_from_paths,
    build_gene_pools_pseudobulk_from_paths,
    compute_detection_rate,
    compute_detection_table,
    compute_pseudobulk_detection_rate,
    get_detected_genes,
    get_detected_genes_pseudobulk,
    get_platform_intersection,
    load_gene_pools,
    normalise_adata,
    save_gene_pools,
    summarise_gene_pools,
)
from .gene_selection import (
    build_niche_modality_map,
    compute_degs_per_niche,
    compute_f2_feature_set,
    compute_hvgs_per_modality,
    get_deg_gene_set,
    get_hvg_gene_set,
    summarise_niche_modality_map,
)
from .spatial import (
    include_neighbours,
    annotate_edge,
    build_neighbourhood_features_sd,
    build_neighbourhood_features_hd,
)

__all__ = [
    # gene_catalogue
    "build_gene_pools",
    "build_gene_pools_from_paths",
    "build_gene_pools_pseudobulk_from_paths",
    "compute_detection_rate",
    "compute_detection_table",
    "compute_pseudobulk_detection_rate",
    "get_detected_genes",
    "get_detected_genes_pseudobulk",
    "get_platform_intersection",
    "load_gene_pools",
    "normalise_adata",
    "save_gene_pools",
    "summarise_gene_pools",
    # gene_selection
    "build_niche_modality_map",
    "compute_degs_per_niche",
    "compute_f2_feature_set",
    "compute_hvgs_per_modality",
    "get_deg_gene_set",
    "get_hvg_gene_set",
    "summarise_niche_modality_map",
    # spatial
    "include_neighbours",
    "annotate_edge",
    "build_neighbourhood_features_sd",
    "build_neighbourhood_features_hd",
]
