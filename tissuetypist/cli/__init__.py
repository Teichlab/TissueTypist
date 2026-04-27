"""TissueTypist command-line interface.

The ``tissuetypist`` console script (registered in ``pyproject.toml``
under ``[project.scripts]``) dispatches to the subcommands below via
:func:`tissuetypist.cli.main.entry`:

    tissuetypist train              — hierarchical training (full genome)
    tissuetypist train-panel        — panel-specific retraining (Xenium / MERFISH / CosMx)
    tissuetypist predict            — prediction only (writes h5ad + summary CSV)
    tissuetypist evaluate           — prediction + plots + metrics
    tissuetypist pseudobulk-hd      — HD sliding-window pseudobulk
    tissuetypist build-catalogue    — Phase 0 gene-detection catalogue
    tissuetypist validate-hierarchy — check a hierarchy YAML against an AnnData
    tissuetypist info               — list shipped presets + hierarchies

The subcommands delegate to the library modules:

    tissuetypist.training.hierarchical        (train, predict deps)
    tissuetypist.training.panel_specific      (train-panel)
    tissuetypist.evaluation.runner            (evaluate)
    tissuetypist.data.pseudobulk              (pseudobulk-hd)
    tissuetypist.features.gene_catalogue      (build-catalogue)
    tissuetypist.config.hierarchy             (validate-hierarchy, info)
    tissuetypist.config.presets               (info)
"""
from .main import entry

__all__ = ["entry"]
