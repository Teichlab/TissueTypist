"""Round-trip and structural tests for :class:`HierarchySpec` + cardiac YAML.

These tests don't require scanpy / anndata / sklearn — they only exercise
the hierarchy parser and dataclass contract.
"""
from __future__ import annotations

import pytest

from tissuetypist.config.hierarchy import (
    HierarchySpec,
    SubModel,
    SubModelStage,
    flat_hierarchy,
    infer_hierarchy_from_data,
    list_shipped_hierarchies,
    load_hierarchy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shipped cardiac spec
# ─────────────────────────────────────────────────────────────────────────────

def test_cardiac_ships_by_default():
    """The cardiac hierarchy must always be shipped with the package."""
    assert "cardiac" in list_shipped_hierarchies()


def test_cardiac_loads_successfully():
    spec = load_hierarchy("cardiac")
    assert spec.name.startswith("cardiac")
    assert spec.coarse_col.startswith("niche_coarse_")
    assert spec.fine_col is not None


def test_cardiac_has_seven_coarse_with_two_terminal():
    spec = load_hierarchy("cardiac")
    assert len(spec.coarse_niches) == 7
    assert len(spec.terminal_coarse) == 2
    # Every terminal_coarse must also appear in coarse_niches.
    for t in spec.terminal_coarse:
        assert t in spec.coarse_niches
    # Coverage: terminal + sub-model parents should equal the coarse set.
    accounted = set(spec.terminal_coarse) | set(spec.sub_models.keys())
    assert accounted == set(spec.coarse_niches)


def test_cardiac_atrium_is_three_stage_chain():
    """Atrium chain is the new depth=3 structure introduced in Apr 2026."""
    spec = load_hierarchy("cardiac")
    atrium = spec.sub_models["Atrium"]
    assert atrium.depth == 3
    assert [s.model_name for s in atrium.stages] == [
        "atrium_split", "atrium_transitional", "atrium_lr",
    ]


def test_cardiac_atrium_transitional_has_low_confidence_route():
    """Atrium stage 2 uses low_confidence_route=Atrium - LR to continue to stage 3."""
    spec = load_hierarchy("cardiac")
    stage2b = spec.sub_models["Atrium"].stages[1]
    assert stage2b.low_confidence_route == "Atrium - LR"
    assert stage2b.low_confidence_route in stage2b.route_classes_to_next


def test_cardiac_pcs_is_two_stage_chain():
    spec = load_hierarchy("cardiac")
    pcs = spec.sub_models["Pacemaker conduction system"]
    assert pcs.depth == 2
    assert pcs.stages[0].model_name == "pcs_split"
    assert pcs.stages[1].model_name == "pcs_sinoatrial"


def test_cardiac_vasculature_pools_great_vessels():
    """Vasculature stage 1 pools Great vessel + Ductus arteriosus into the
    synthetic intermediate 'Great vessels'."""
    spec = load_hierarchy("cardiac")
    stage2a = spec.sub_models["Vasculature"].stages[0]
    assert "Great vessels" in stage2a.pool_from
    assert set(stage2a.pool_from["Great vessels"]) == {
        "Great vessel", "Ductus arteriosus",
    }


def test_cardiac_has_palette_and_gt_label_remap():
    spec = load_hierarchy("cardiac")
    assert spec.palette, "cardiac.yaml should ship a `palette:` section"
    # Apr2026 rename targets must be in the palette.
    assert "Atrium - LR" in spec.palette
    assert "Epicardial region" in spec.palette
    assert "AV nodal region" in spec.palette
    assert spec.gt_label_remap, "cardiac.yaml should ship a `gt_label_remap:` section"
    assert "Adventitia" in spec.gt_label_remap
    assert spec.gt_label_remap["Adventitia"] == "Connective tissue"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass contract + derived properties
# ─────────────────────────────────────────────────────────────────────────────

def test_submodel_depth_property():
    sm = SubModel(
        parent_coarse="Whatever",
        stages=[SubModelStage(
            model_name="only", classes=["A", "B"], modalities=["sd3p"],
        )],
    )
    assert sm.depth == 1
    assert sm.three_level is False
    assert sm.stage2b is None


def test_submodel_three_level_property_true_for_depth_ge_2():
    sm = SubModel(
        parent_coarse="X",
        stages=[
            SubModelStage(model_name="a", classes=["P", "Q"], modalities=["sd3p"]),
            SubModelStage(model_name="b", classes=["P1", "P2"], modalities=["sd3p"]),
        ],
    )
    assert sm.depth == 2
    assert sm.three_level is True
    assert sm.stage2b is not None
    assert sm.stage2b.model_name == "b"


# ─────────────────────────────────────────────────────────────────────────────
# Flat hierarchy builder
# ─────────────────────────────────────────────────────────────────────────────

def test_flat_hierarchy_has_no_sub_models():
    spec = flat_hierarchy("my_niche", ["A", "B", "C"])
    assert spec.fine_col is None
    assert spec.coarse_niches == ["A", "B", "C"]
    assert spec.terminal_coarse == ["A", "B", "C"]
    assert spec.sub_models == {}


def test_flat_hierarchy_sorts_classes():
    spec = flat_hierarchy("my", ["Z", "A", "M"])
    assert spec.coarse_niches == ["A", "M", "Z"]


# ─────────────────────────────────────────────────────────────────────────────
# Auto-infer builder (uses pandas; skip if pandas absent)
# ─────────────────────────────────────────────────────────────────────────────

def test_infer_hierarchy_from_obs_strict_ok():
    pd = pytest.importorskip("pandas")
    obs = pd.DataFrame({
        "coarse": ["Heart"] * 4 + ["Lung"] * 4 + ["Liver"] * 3,
        "fine":   ["Atrium", "Atrium", "Ventricle", "Ventricle",
                   "Alveolus", "Alveolus", "Bronchus", "Bronchus",
                   "Hepatocyte", "Hepatocyte", "Hepatocyte"],
    })
    spec = infer_hierarchy_from_data(obs, "coarse", "fine")
    assert set(spec.coarse_niches) == {"Heart", "Lung", "Liver"}
    # Liver has only one fine label → terminal, no sub-model.
    assert "Liver" in spec.terminal_coarse
    assert "Liver" not in spec.sub_models
    # Heart and Lung have ≥2 fine labels → get sub-models.
    assert "Heart" in spec.sub_models
    assert sorted(spec.sub_models["Heart"].stages[0].classes) == ["Atrium", "Ventricle"]


def test_infer_hierarchy_strict_raises_on_ambiguous_fine():
    pd = pytest.importorskip("pandas")
    obs = pd.DataFrame({
        "coarse": ["A", "A", "B", "B"],
        "fine":   ["X", "Y", "X", "Z"],    # 'X' spans A and B
    })
    with pytest.raises(ValueError, match="more than one coarse niche"):
        infer_hierarchy_from_data(obs, "coarse", "fine", strict=True)


def test_infer_hierarchy_non_strict_assigns_to_majority():
    pd = pytest.importorskip("pandas")
    obs = pd.DataFrame({
        "coarse": ["A"] * 10 + ["B"] * 3,
        "fine":   ["X"] * 7 + ["Y"] * 3 + ["X"] * 3,   # X: 7 in A, 3 in B
    })
    spec = infer_hierarchy_from_data(obs, "coarse", "fine", strict=False)
    # X should be under A (majority); B loses X.
    heart_children = spec.sub_models.get("A", None)
    assert heart_children is not None
    assert "X" in heart_children.stages[0].classes
    # B had only X (after majority-remap → none), so it's terminal.
    assert "B" in spec.terminal_coarse or "B" not in spec.sub_models


def test_infer_hierarchy_fine_none_returns_flat():
    pd = pytest.importorskip("pandas")
    obs = pd.DataFrame({"coarse": ["A", "B", "A", "C"]})
    spec = infer_hierarchy_from_data(obs, "coarse", fine_col=None)
    assert spec.fine_col is None
    assert spec.sub_models == {}
    assert set(spec.terminal_coarse) == {"A", "B", "C"}


# ─────────────────────────────────────────────────────────────────────────────
# Invalid spec guards
# ─────────────────────────────────────────────────────────────────────────────

def test_low_confidence_route_must_be_routable(tmp_path):
    """low_confidence_route must also appear in route_classes_to_next."""
    pytest.importorskip("yaml")
    import yaml
    bad = {
        "name": "bad", "coarse_col": "c", "fine_col": "f",
        "coarse_niches": ["X"], "terminal_coarse": [],
        "sub_models": [{
            "parent": "X",
            "stages": [{
                "model_name": "s1", "classes": ["A", "B"],
                "modalities": ["hd"],
                "route_classes_to_next": [],           # nothing routes anywhere
                "low_confidence_route": "A",           # illegal
            }],
        }],
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="low_confidence_route"):
        load_hierarchy(str(p))
