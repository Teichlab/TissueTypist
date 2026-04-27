"""
tissuetypist/config/presets.py
===============================
Default spatial-feature weight presets for TissueTypist.

Each preset corresponds to a complete trained hierarchy shipped under
``tissuetypist/models/<preset_name>/``. Users can either:

  1. Predict with a preset directly:
         tissuetypist predict --preset default --query my.h5ad

  2. Retrain with the same weights on their own data:
         tissuetypist train --preset default --sd3p my.h5ad ...

  3. Retrain with custom weights (override the preset):
         tissuetypist train --neighbour_weight 0.5 --edge_weight 2 ...

Semantics
---------
``neighbour_weight`` is the amplification factor applied to the
StandardScaler-scaled neighbour-max features before they enter the
logistic regression model. ``edge_weight`` is the same for the
``distance_to_edge`` feature. See
:func:`tissuetypist.training.logistic.build_weighted_pipeline` for how
these are applied inside the pipeline.

A preset whose ``neighbour_weight`` and ``edge_weight`` are both zero is
an "own-only" model: spatial features are effectively disabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class WeightPreset:
    """A named combination of spatial-feature weights."""
    name: str
    neighbour_weight: float
    edge_weight: float
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# Shipped presets
# ─────────────────────────────────────────────────────────────────────────────
# Keep in sync with ``tissuetypist/models/<preset_name>/`` directories.

WEIGHT_PRESETS: Dict[str, WeightPreset] = {
    "own_only": WeightPreset(
        name="own_only",
        neighbour_weight=0.0,
        edge_weight=0.0,
        description=(
            "Own expression only. Disables neighbour-max and distance-to-edge "
            "features. Useful when query spatial coordinates are unreliable "
            "(e.g. dissociated reference data, poorly aligned registrations, "
            "or sanity checks to isolate spatial signal contribution)."
        ),
    ),
    "default": WeightPreset(
        name="default",
        neighbour_weight=0.3,
        edge_weight=5.0,
        description=(
            "Recommended default. Default TissueTypist weights. Modest "
            "amplification of neighbour-max features and strong weight on "
            "distance-to-edge. Balances gene signal and spatial context."
        ),
    ),
    "neighbour_heavy": WeightPreset(
        name="neighbour_heavy",
        neighbour_weight=1.0,
        edge_weight=5.0,
        description=(
            "Stronger neighbourhood weighting. Use when tissue architecture "
            "is expected to be highly locally-organised and query coordinates "
            "are reliable."
        ),
    ),
}


DEFAULT_PRESET_NAME = "default"


def get_preset(name: str) -> WeightPreset:
    """Return the :class:`WeightPreset` registered under ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not one of the known preset names.
    """
    if name not in WEIGHT_PRESETS:
        known = ", ".join(sorted(WEIGHT_PRESETS))
        raise KeyError(
            f"Unknown weight preset '{name}'. Known presets: {known}"
        )
    return WEIGHT_PRESETS[name]


def list_presets() -> list[str]:
    """Return the list of known preset names."""
    return sorted(WEIGHT_PRESETS)


__all__ = [
    "WeightPreset",
    "WEIGHT_PRESETS",
    "DEFAULT_PRESET_NAME",
    "get_preset",
    "list_presets",
]
