"""Lightweight smoke tests for the `tissuetypist` CLI.

Only exercises the zero-dependency subcommands (``info``,
``validate-hierarchy``, ``--version``, ``--help``). Tests for heavy
subcommands (``train``, ``predict``, ``evaluate``, ``train-panel``,
``pseudobulk-hd``, ``build-catalogue``) require scanpy / anndata / sklearn
/ matplotlib and are covered by the full end-to-end runs documented in
``RESTRUCTURE_NOTES.md``.
"""
from __future__ import annotations

import io
import sys

import pytest


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Invoke tissuetypist.cli.main.entry(argv) and capture stdout+stderr.

    Argparse writes usage/error messages to stderr, and
    ``raise SystemExit("message")`` also prints the message to stderr, so
    we merge both streams here for simpler string-assertion tests.
    """
    from tissuetypist.cli.main import entry

    old_stdout, old_stderr = sys.stdout, sys.stderr
    buf = io.StringIO()
    sys.stdout = buf
    sys.stderr = buf
    try:
        try:
            rc = entry(argv) or 0
        except SystemExit as e:
            code = e.code
            if isinstance(code, str):
                # SystemExit("message") — print message (mirrors Python's
                # default behaviour) and report exit code 1.
                print(code)
                rc = 1
            elif code is None:
                rc = 0
            else:
                rc = int(code)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return rc, buf.getvalue()


def test_cli_version():
    rc, out = _run_cli(["--version"])
    assert rc == 0
    assert "tissuetypist" in out.lower()


def test_cli_help_lists_all_subcommands():
    # --help triggers argparse to print and SystemExit(0); capture both.
    rc, out = _run_cli(["--help"])
    assert rc == 0
    for sub in [
        "train", "train-panel", "predict", "evaluate",
        "pseudobulk-hd", "build-catalogue", "validate-hierarchy", "info",
    ]:
        assert sub in out, f"Subcommand {sub!r} missing from top-level --help"


def test_cli_info_lists_presets_and_cardiac():
    rc, out = _run_cli(["info"])
    assert rc == 0
    # All three presets should be referenced (installed or not).
    for name in ("default", "own_only", "neighbour_heavy"):
        assert name in out
    assert "cardiac" in out


def test_cli_validate_hierarchy_cardiac_schema_only():
    rc, out = _run_cli(["validate-hierarchy", "--hierarchy", "cardiac"])
    assert rc == 0


def test_cli_build_catalogue_errors_with_no_refs():
    """Before any heavy imports, build-catalogue must surface a clear
    error when called with no reference datasets."""
    rc, out = _run_cli(["build-catalogue", "--outdir", "/tmp/should_not_be_created"])
    # Expect a non-zero exit and a clear message mentioning --reference.
    assert rc != 0
    assert "--reference" in out or "reference" in out.lower()


def test_cli_train_and_auto_infer_are_mutex():
    """argparse must reject --flat alongside --auto_infer."""
    rc, out = _run_cli([
        "train",
        "--reference", "/tmp/does_not_exist.h5ad",
        "--outdir", "/tmp/x",
        "--flat", "--auto_infer",
        "--coarse_col", "x", "--fine_col", "y",
    ])
    # argparse mutex violation exits non-zero and prints usage.
    assert rc != 0


def test_cli_train_flat_and_hierarchy_are_mutex():
    rc, out = _run_cli([
        "train",
        "--reference", "/tmp/does_not_exist.h5ad",
        "--outdir", "/tmp/x",
        "--hierarchy", "cardiac",
        "--flat",
        "--coarse_col", "x",
    ])
    assert rc != 0
