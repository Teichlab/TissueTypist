"""Configuration modules for TissueTypist.

Public API:
    HierarchySpec               — declarative niche hierarchy
    load_hierarchy              — load a shipped name or YAML file
    list_shipped_hierarchies    — list shipped YAML names
    infer_hierarchy_from_data   — auto-infer 2-level hierarchy from obs

    WeightPreset                — spatial-feature weight preset
    WEIGHT_PRESETS              — dict of shipped presets
    DEFAULT_PRESET_NAME         — "default"
    get_preset                  — look up a preset by name
    list_presets                — list known preset names
"""
from .hierarchy import (
    HierarchySpec,
    SubModel,
    SubModelStage,
    flat_hierarchy,
    infer_hierarchy_from_data,
    list_shipped_hierarchies,
    load_hierarchy,
)
from .presets import (
    DEFAULT_PRESET_NAME,
    WEIGHT_PRESETS,
    WeightPreset,
    get_preset,
    list_presets,
)

__all__ = [
    # hierarchy
    "HierarchySpec",
    "SubModel",
    "SubModelStage",
    "flat_hierarchy",
    "infer_hierarchy_from_data",
    "list_shipped_hierarchies",
    "load_hierarchy",
    # presets
    "DEFAULT_PRESET_NAME",
    "WEIGHT_PRESETS",
    "WeightPreset",
    "get_preset",
    "list_presets",
]
