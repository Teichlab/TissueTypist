"""Prediction modules for TissueTypist.

Public entry points:
    predict         — DataFrame-level hierarchical prediction (low-level)
    predict_adata   — AnnData-level hierarchical prediction (high-level, recommended)

Example
-------
>>> from tissuetypist.prediction import predict_adata
>>> adata = predict_adata(
...     adata,
...     model_dir="results/hierarchical",
...     modality="sd",
...     section_col="section_ID",
... )
"""
from .hierarchical import (
    predict,
    predict_adata,
)

__all__ = [
    "predict",
    "predict_adata",
]
