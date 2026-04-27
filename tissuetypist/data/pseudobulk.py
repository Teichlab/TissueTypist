"""
tissuetypist/data/pseudobulk.py
================================
Sliding-window pseudobulk functions for TissueTypist.

These currently live physically in :mod:`tissuetypist.features.spatial`
(re-exported from features.spatial)
so that the public API reflects their conceptual home — ``data.pseudobulk``
— independently of where they are implemented. A follow-up phase will
move the bodies here; until then, updating your imports to use
``tissuetypist.data.pseudobulk`` is forward-compatible.

Public API
----------
    sliding_window_pseudobulk        — general (Visium SD-style) pseudobulk
    sliding_window_pseudobulk_hd     — Visium HD cells → 55 µm windows
    sliding_window_pseudobulk_cells  — imaging-based ST (µm coords) → windows
"""
from __future__ import annotations

from tissuetypist.features.spatial import (
    sliding_window_pseudobulk,
    sliding_window_pseudobulk_hd,
    sliding_window_pseudobulk_cells,
)

__all__ = [
    "sliding_window_pseudobulk",
    "sliding_window_pseudobulk_hd",
    "sliding_window_pseudobulk_cells",
]
