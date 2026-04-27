"""Shared pytest fixtures and skip-markers for TissueTypist tests."""
from __future__ import annotations

import pytest


def _has_module(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


# Skip markers for tests that need the heavy scientific stack.
needs_scanpy = pytest.mark.skipif(
    not _has_module("scanpy"),
    reason="scanpy not installed in this environment",
)
needs_anndata = pytest.mark.skipif(
    not _has_module("anndata"),
    reason="anndata not installed",
)
needs_sklearn = pytest.mark.skipif(
    not _has_module("sklearn"),
    reason="scikit-learn not installed",
)
