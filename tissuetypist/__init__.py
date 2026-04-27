"""TissueTypist — cardiac tissue-niche classifier for spatial transcriptomics.

High-level public API
---------------------
Prediction (use a shipped or user-trained model):

    >>> from tissuetypist import predict_adata, load_preset
    >>> adata = predict_adata(
    ...     adata,
    ...     model_dir=load_preset("default"),
    ...     modality="sd",
    ...     section_col="section_ID",
    ... )

Training (on cardiac data with the shipped hierarchy):

    >>> from tissuetypist import TrainingConfig, load_hierarchy
    >>> from tissuetypist.training import train_all_models, load_data
    >>> spec = load_hierarchy("cardiac")        # or path to your YAML
    >>> # ... build TrainingConfig and call train_all_models(...)

Inspecting presets:

    >>> from tissuetypist import WEIGHT_PRESETS, list_presets
    >>> list_presets()
    ['default', 'neighbour_heavy', 'own_only']

Module layout
-------------
- ``tissuetypist.config``      — hierarchy spec + weight presets
- ``tissuetypist.data``        — normalisation + pseudobulk
- ``tissuetypist.features``    — gene-selection + spatial features
- ``tissuetypist.training``    — hierarchical + logistic trainers
- ``tissuetypist.prediction``  — hierarchical prediction
- ``tissuetypist.models``      — shipped pre-trained models (three presets)
"""
from __future__ import annotations

__version__ = "0.1.0"

# ─── Public API re-exports ───────────────────────────────────────────────────
# Keep this list short and curated. Users should reach for subpackages for
# finer-grained access.

# Lightweight imports: config is dataclasses + YAML + (lazy) pandas. Safe to
# import eagerly even when scanpy/sklearn/matplotlib aren't installed — this
# is what makes `tissuetypist info` and `tissuetypist --version` usable in
# minimal environments.
from .config import (
    DEFAULT_PRESET_NAME,
    HierarchySpec,
    WEIGHT_PRESETS,
    WeightPreset,
    flat_hierarchy,
    get_preset,
    infer_hierarchy_from_data,
    list_presets,
    list_shipped_hierarchies,
    load_hierarchy,
)

# Heavy imports (``predict``, ``predict_adata``, ``TrainingConfig``,
# ``evaluate``) are resolved lazily via ``__getattr__`` so that `import
# tissuetypist` doesn't force loading of scanpy / sklearn / matplotlib.
# Users continue to access them as ``tissuetypist.predict_adata`` etc.;
# the first access triggers the submodule import.

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "predict":        ("tissuetypist.prediction",          "predict"),
    "predict_adata":  ("tissuetypist.prediction",          "predict_adata"),
    "TrainingConfig": ("tissuetypist.training",            "TrainingConfig"),
    "evaluate":       ("tissuetypist.evaluation",          "evaluate"),
}


def __getattr__(name):  # PEP 562 lazy attribute access
    if name in _LAZY_ATTRS:
        import importlib
        mod_path, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val   # memoise
        return val
    raise AttributeError(f"module 'tissuetypist' has no attribute {name!r}")


def load_preset(name: str = DEFAULT_PRESET_NAME):
    """Return the filesystem path of a shipped pre-trained model directory.

    Parameters
    ----------
    name :
        Preset name. One of :func:`list_presets` (e.g. ``"default"``,
        ``"own_only"``, ``"neighbour_heavy"``).

    Returns
    -------
    pathlib.Path
        Path to the directory containing the 9 pipeline ``joblib`` files,
        ``gene_list.txt`` files, and ``hierarchy_config.json`` for the
        selected preset.

    Notes
    -----
    If you have a locally-trained model, pass its directory path
    directly to ``predict_adata(..., model_dir=...)`` rather than using
    ``load_preset``.
    """
    from pathlib import Path

    get_preset(name)  # validates name; raises KeyError on unknown
    path = Path(__file__).parent / "models" / name
    if not path.exists():
        raise FileNotFoundError(
            f"Shipped preset {name!r} not found at {path}. "
            "The package may have been installed without models, or the "
            "preset has not been trained yet. Train it locally with "
            f"`tissuetypist train --preset {name} ...` and pass its output "
            "directory directly to `predict_adata`."
        )
    return path


__all__ = [
    "__version__",
    # High-level API
    "predict",
    "predict_adata",
    "load_preset",
    # Config
    "TrainingConfig",
    "HierarchySpec",
    "load_hierarchy",
    "list_shipped_hierarchies",
    "infer_hierarchy_from_data",
    "WEIGHT_PRESETS",
    "WeightPreset",
    "DEFAULT_PRESET_NAME",
    "get_preset",
    "list_presets",
]
