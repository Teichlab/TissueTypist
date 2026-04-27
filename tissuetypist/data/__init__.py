"""Data handling modules for TissueTypist.

Submodules:
    normalise   — raw-count detection and log-normalisation
    pseudobulk  — sliding-window pseudobulk for Visium HD and
                  imaging-based ST (Xenium / MERFISH / CosMx)
"""
from .normalise import (
    is_log_normalised,
    normalise_if_needed,
)
from .pseudobulk import (
    sliding_window_pseudobulk,
    sliding_window_pseudobulk_hd,
    sliding_window_pseudobulk_cells,
)

__all__ = [
    # normalise
    "is_log_normalised",
    "normalise_if_needed",
    # pseudobulk
    "sliding_window_pseudobulk",
    "sliding_window_pseudobulk_hd",
    "sliding_window_pseudobulk_cells",
]
