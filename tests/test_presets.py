"""Tests for weight presets and shipped models."""
from __future__ import annotations

import pytest

from tissuetypist.config.presets import (
    DEFAULT_PRESET_NAME,
    WEIGHT_PRESETS,
    get_preset,
    list_presets,
)
from tissuetypist.models import list_shipped_presets, preset_dir


EXPECTED_PRESETS = {"default", "own_only", "neighbour_heavy"}


def test_three_weight_presets_registered():
    assert set(WEIGHT_PRESETS) == EXPECTED_PRESETS
    assert DEFAULT_PRESET_NAME in WEIGHT_PRESETS


def test_preset_weights_match_names():
    """The named presets must have the weights documented in the YAML/README."""
    own = get_preset("own_only")
    assert own.neighbour_weight == 0.0 and own.edge_weight == 0.0

    default = get_preset("default")
    assert default.neighbour_weight == 0.3 and default.edge_weight == 5.0

    heavy = get_preset("neighbour_heavy")
    assert heavy.neighbour_weight == 1.0 and heavy.edge_weight == 5.0


def test_list_presets_sorted_and_complete():
    assert list_presets() == sorted(EXPECTED_PRESETS)


def test_get_preset_raises_on_unknown():
    with pytest.raises(KeyError):
        get_preset("gibberish")


# ─────────────────────────────────────────────────────────────────────────────
# Shipped model directories (only present after `scripts/07_populate_preset_models.sh`)
# ─────────────────────────────────────────────────────────────────────────────

def test_preset_dir_returns_path_per_preset():
    for name in EXPECTED_PRESETS:
        p = preset_dir(name)
        assert p.name == name
        # preset_dir doesn't require existence; that's load_preset's job.


@pytest.mark.parametrize("name", sorted(EXPECTED_PRESETS))
def test_shipped_preset_has_full_hierarchy(name):
    """If a preset is installed (artifacts present on disk), it must be complete:
    hierarchy_config.json + at least the coarse pipeline + gene list."""
    if name not in list_shipped_presets():
        pytest.skip(f"{name} preset not installed — "
                    f"run scripts/07_populate_preset_models.sh to ship it.")
    d = preset_dir(name)
    assert (d / "hierarchy_config.json").exists(), \
        f"{name} missing hierarchy_config.json"
    assert (d / "coarse_pipeline.joblib").exists(), \
        f"{name} missing coarse_pipeline.joblib"
    assert (d / "coarse_gene_list.txt").exists(), \
        f"{name} missing coarse_gene_list.txt"


def test_shipped_hierarchy_config_is_schema_v2():
    """Every installed preset must emit the current schema."""
    import json
    for name in list_shipped_presets():
        cfg_path = preset_dir(name) / "hierarchy_config.json"
        with cfg_path.open() as f:
            cfg = json.load(f)
        assert cfg["schema_version"] == 2, \
            f"{name}: hierarchy_config.json has unexpected schema_version"
        assert "hierarchy" in cfg, f"{name}: missing embedded HierarchySpec"


def test_load_preset_returns_existing_path():
    """tissuetypist.load_preset resolves to a directory that exists for every
    installed preset, and raises clearly for uninstalled ones."""
    # Lazy import: load_preset is on the top-level tissuetypist module.
    from tissuetypist import load_preset

    installed = list_shipped_presets()
    for name in installed:
        path = load_preset(name)
        assert path.exists() and path.is_dir()
        assert (path / "hierarchy_config.json").exists()

    # If any preset is NOT installed, load_preset should raise clearly.
    missing = sorted(set(EXPECTED_PRESETS) - set(installed))
    for name in missing:
        with pytest.raises(FileNotFoundError):
            load_preset(name)
