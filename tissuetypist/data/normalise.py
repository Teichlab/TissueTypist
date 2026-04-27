"""
tissuetypist/data/normalise.py
===============================
Shared normalisation helpers for TissueTypist.

Used by the training and prediction entry points to load raw AnnData
from disk and ensure log-normalised data reaches the feature-selection
functions in ``tissuetypist.features.gene_selection``, which always
assume log-normalised input.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


def is_log_normalised(adata: ad.AnnData) -> bool:
    """
    Check whether adata.X contains log-normalised data or raw counts.

    Mirrors the logic of scanpy's ``check_nonnegative_integers`` (in
    ``scanpy._utils``), inverted to return True for log-normalised data:

    1. **Integer dtype** (e.g. int32, int64) → definitively raw counts.
       No value inspection needed.
    2. **Float dtype with negative values** → log-normalised (or centred).
    3. **Float dtype, all non-negative** → check modulo: if any value has
       a non-zero fractional part, data is log-normalised.
    4. **All zeros** → cannot determine; conservatively returns False (raw).

    This dtype-first approach is more reliable than sampling-based heuristics
    and avoids the false-positive failure mode where sparse data with
    median non-zero = 1 was misidentified as log-normalised.

    Parameters
    ----------
    adata :
        AnnData whose X matrix is to be inspected.

    Returns
    -------
    bool
        True if data appears already log-normalised; False if raw counts.
    """
    from numbers import Integral
    from scipy.sparse import issparse

    X = adata.X
    # Always convert to a plain numpy array first — X may be a memoryview
    # (h5py backed), sparse matrix, or other array-like that does not support
    # dtype inspection or the % operator directly.
    raw = X.data if issparse(X) else X
    orig_dtype = np.asarray(raw).dtype   # preserve original dtype before float cast
    data = np.asarray(raw, dtype=np.float64).ravel()

    if len(data) == 0 or np.all(data == 0):
        logger.warning("is_log_normalised: matrix is all zeros — assuming raw.")
        return False

    # Step 1: integer dtype → definitively raw counts (same as scanpy)
    if issubclass(orig_dtype.type, Integral):
        return False

    # Step 2: any negatives → log-normalised (or centred)
    if np.signbit(data).any():
        return True

    # Step 3: float dtype — check for fractional parts
    # Raw counts stored as float still have data % 1 == 0 for all values;
    # log-normalised data (normalize_total + log1p) does not.
    return bool(np.any((data % 1) != 0))


def normalise_if_needed(
    adata: ad.AnnData,
    name: str,
    target_sum: float = 1e4,
) -> ad.AnnData:
    """
    Apply normalize_total + log1p to an AnnData if it appears to contain
    raw counts.

    Raw counts are preserved in ``layers["raw_counts"]`` before normalisation
    so they remain accessible downstream (e.g. for future pseudobulk steps).
    If the data already appears log-normalised, it is returned unchanged.

    Parameters
    ----------
    adata :
        AnnData to (potentially) normalise. Modified in place.
    name : str
        Human-readable label used in log messages (e.g. ``"SD_3prime"``).
    target_sum : float
        Library size for normalize_total. Default 1e4.

    Returns
    -------
    ad.AnnData
        The same adata object, normalised if needed.
    """
    if is_log_normalised(adata):
        logger.info("  %s: already log-normalised — skipping.", name)
        return adata

    logger.info(
        "  %s: raw counts detected — applying normalize_total "
        "(target_sum=%.0f) + log1p.",
        name, target_sum,
    )

    # Preserve raw counts before overwriting X
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    logger.info("  %s: normalisation complete.", name)
    return adata
