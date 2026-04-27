"""
tissuetypist/config/hierarchy.py
=================================
Declarative niche-hierarchy specification for TissueTypist.

A :class:`HierarchySpec` describes, for a given tissue or experiment:
  - which ``obs`` columns hold the coarse and (optional) fine labels,
  - the set of coarse niches,
  - which coarse niches are "terminal" (no Stage 2 sub-model),
  - for each non-terminal coarse niche, either:
      * a flat list of fine children (standard two-stage sub-model), or
      * a three-level spec (Stage 2a → intermediate node → Stage 2b).

The spec is read from YAML at training time and embedded into the saved
model directory (``hierarchy_config.json``) at training time so the
prediction code can reconstruct the routing.

This module currently defines the dataclasses and loader. The training
and prediction code still uses hardcoded cardiac definitions; phase 3b
of the restructure will switch them over to read from this module.

Shipped hierarchies
-------------------
    cardiac  — 7 coarse / 3 intermediate / 21 terminal (Mar 2026)
               YAML: ``tissuetypist/config/hierarchies/cardiac.yaml``

User-supplied YAML files can be loaded with :func:`load_hierarchy`.

Example
-------
>>> from tissuetypist.config.hierarchy import load_hierarchy
>>> spec = load_hierarchy("cardiac")                # shipped
>>> spec = load_hierarchy("/path/to/my.yaml")       # user-provided
>>> spec.coarse_col
'niche_coarse_Apr2026'
>>> spec.coarse_niches
['Ventricle', 'Atrium', 'Pacemaker conduction system', ...]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

# Supported modality tags. Kept permissive so non-cardiac users can adapt.
_KNOWN_MODALITIES = {"sd3p", "sd_ffpe", "hd"}


@dataclass(frozen=True)
class SubModelStage:
    """A single LR stage within a sub-model chain.

    A sub-model attached to a coarse niche is a chain of one or more stages.
    At each stage, an LR classifier discriminates between the classes in
    ``classes``. Output classes that are listed in ``route_classes_to_next``
    trigger routing to the next stage; the rest are terminal outputs at
    this stage (no further classification).
    """

    #: The name used to save/load the pipeline (e.g. ``"ventricle"``,
    #: ``"atrium_split"``, ``"atrium_transitional"``).
    model_name: str

    #: The labels this stage discriminates between. Labels may be either
    #: "direct" (present in ``obs`` as-is) or "synthetic" (pooled from
    #: multiple fine labels via ``pool_from``).
    classes: List[str]

    #: Which input modalities contribute training data for this stage.
    #: Subset of ``{"sd3p", "sd_ffpe", "hd"}``.
    modalities: List[str]

    #: Optional — map from a class name in ``classes`` to the list of
    #: fine labels that should be pooled to form that class at training
    #: time. Used for synthetic intermediate nodes (e.g. Vasculature's
    #: "Great vessels" = "Great vessel" ∪ "Ductus arteriosus", or Atrium's
    #: "Atrial myocardium" = "Atrium - Left" ∪ "Atrium - Right" ∪
    #: "Atrium - Transitional"). Classes not in this dict are taken
    #: directly from ``obs``.
    pool_from: Optional[Dict[str, List[str]]] = None

    #: Optional — explicit intermediate-node label already present in
    #: ``obs`` (alternative to ``pool_from``). Legacy path: used by PCS
    #: where the Mar/Apr 2026 SD 3' data carries "Sinoatrial region -
    #: non-terminal category" as an explicit fine label.
    intermediate_label_in_data: Optional[str] = None

    #: Which of the classes at this stage route to the next stage in the
    #: chain. Classes not listed here are terminal at this stage. Must be
    #: a subset of ``classes``; empty for the final stage.
    route_classes_to_next: List[str] = field(default_factory=list)

    #: Optional — **permissive routing on low confidence.** When set, a
    #: spot whose stage score is below theta is routed as if this class
    #: had been predicted, continuing to the next stage of the chain
    #: instead of stopping at ``fallback_label``. The value must be a
    #: member of ``route_classes_to_next`` (otherwise there is no next
    #: stage to continue into). ``tt_low_conf`` still flips to True so
    #: downstream users can filter these spots out if desired.
    #:
    #: Useful when this stage's classifier is known to be underpowered
    #: (e.g. HD-only training data on a minority class) but a downstream
    #: stage is well-trained on the larger class — you prefer the deeper
    #: prediction-with-uncertainty to stopping at the parent intermediate.
    low_confidence_route: Optional[str] = None

    #: Fallback label when this stage's score is below theta **and**
    #: ``low_confidence_route`` is not set (the "confident predecessor"
    #: — typically the class at the previous stage whose expansion
    #: produced this stage, or the parent coarse niche for stage 2a).
    fallback_label: Optional[str] = None


@dataclass(frozen=True)
class SubModel:
    """Sub-model chain attached to a single non-terminal coarse niche.

    ``stages`` is an ordered list of one or more :class:`SubModelStage`
    instances. A chain of length 1 is a "flat" sub-model (e.g. Ventricle,
    AV junction). A chain of length 2 is "three-level" (e.g. Vasculature's
    2a + 2b, PCS's 2a + 2b). A chain of length 3 is "four-level" (Atrium
    in the April 2026 hierarchy: 2a splits Atrial myocardium vs
    Endocardium - Atrial, 2b splits Atrium - LR vs Atrium - Transitional,
    2c splits Atrium - Left vs Atrium - Right).
    """

    #: The coarse-niche label this sub-model is attached to.
    parent_coarse: str

    #: Ordered chain of classifier stages. Stage k's output classes that
    #: appear in ``stage[k].route_classes_to_next`` are expanded by stage
    #: k+1; other outputs are terminal at stage k.
    stages: List[SubModelStage] = field(default_factory=list)

    @property
    def stage2(self) -> SubModelStage:
        """Convenience accessor for the first stage (stage 2a)."""
        return self.stages[0]

    @property
    def stage2b(self) -> Optional[SubModelStage]:
        """Convenience accessor for stage 2b (None for flat sub-models)."""
        return self.stages[1] if len(self.stages) >= 2 else None

    @property
    def three_level(self) -> bool:
        """True when the chain has 2 or more stages (legacy shorthand)."""
        return len(self.stages) >= 2

    @property
    def depth(self) -> int:
        """Number of classifier stages in this sub-model chain."""
        return len(self.stages)


@dataclass(frozen=True)
class HierarchySpec:
    """Full niche hierarchy for one tissue / experiment."""

    #: Unique name (used in logs and in shipped model directories).
    name: str

    #: ``obs`` column containing the coarse (Stage-1) label.
    coarse_col: str

    #: ``obs`` column containing the fine label. May be ``None`` for
    #: flat-only hierarchies (no sub-models).
    fine_col: Optional[str]

    #: All coarse-niche class labels (the Stage-1 output classes).
    coarse_niches: List[str]

    #: Coarse niches that have no sub-model — predictions stop at Stage 1.
    terminal_coarse: List[str]

    #: Sub-models, keyed by parent coarse niche.
    sub_models: Dict[str, SubModel] = field(default_factory=dict)

    #: Free-text provenance / citation.
    description: str = ""

    #: Optional — map from ground-truth label in ``obs[fine_col]`` to the
    #: label emitted by TissueTypist for the same class. Used for evaluation
    #: metrics to align pooled terminal labels (e.g. cardiac:
    #: ``Adventitia`` and ``Cardiac mesenchyme`` → ``Connective tissue``).
    #: Keys are original ``obs`` labels; values are the prediction output
    #: labels TissueTypist is expected to emit.
    gt_label_remap: Dict[str, str] = field(default_factory=dict)

    #: Optional — curated colour palette keyed by label (coarse, intermediate,
    #: or leaf). Values are hex codes (``"#1f77b4"``) or matplotlib-parseable
    #: colour names. Used by ``tissuetypist.evaluation.plots`` for confusion
    #: matrix / spatial / UMAP plots. Labels not present here fall back to
    #: tab20 iteration colours.
    palette: Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# YAML loader
# ─────────────────────────────────────────────────────────────────────────────

def _parse_pool_from(raw: Any, context: str) -> Optional[Dict[str, List[str]]]:
    """Parse the ``pool_from`` field.

    Accepts three shapes for forward/backward compatibility:

    * ``None`` — no pooling at this stage.
    * ``dict[str, list[str]]`` — new canonical form; keys are synthetic
      class names, values are the fine labels to pool into that class.
    * ``list[str]`` — legacy form, used in early cardiac YAML for
      Vasculature: a flat list of labels to pool into the single
      synthetic class. Caller must pass ``legacy_pool_target`` via
      ``_parse_stage`` when using this form.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {k: list(v) for k, v in raw.items()}
    if isinstance(raw, list):
        raise ValueError(
            f"pool_from at {context!r} is a list; the new canonical form "
            "is a dict mapping synthetic-class-name → list-of-fine-labels. "
            "Example: `pool_from: {Great vessels: [Great vessel, "
            "Ductus arteriosus]}`."
        )
    raise TypeError(
        f"pool_from at {context!r} must be a dict (or None); got {type(raw).__name__}"
    )


def _parse_stage(d: Dict[str, Any], default_model_name: str) -> SubModelStage:
    modalities = list(d.get("modalities", []))
    model_name = d.get("model_name", default_model_name)
    _validate_modalities(modalities, context=model_name)
    pool_from = _parse_pool_from(d.get("pool_from"), context=model_name)
    if pool_from is not None:
        unknown_targets = [k for k in pool_from if k not in d["classes"]]
        if unknown_targets:
            raise ValueError(
                f"pool_from keys {unknown_targets!r} at {model_name!r} are not "
                f"in this stage's classes {list(d['classes'])!r}."
            )
    route_next = list(d.get("route_classes_to_next", []))
    unknown_routes = [r for r in route_next if r not in d["classes"]]
    if unknown_routes:
        raise ValueError(
            f"route_classes_to_next values {unknown_routes!r} at "
            f"{model_name!r} are not in this stage's classes "
            f"{list(d['classes'])!r}."
        )
    low_conf_route = d.get("low_confidence_route")
    if low_conf_route is not None:
        if low_conf_route not in d["classes"]:
            raise ValueError(
                f"low_confidence_route {low_conf_route!r} at {model_name!r} "
                f"is not in this stage's classes {list(d['classes'])!r}."
            )
        if low_conf_route not in route_next:
            raise ValueError(
                f"low_confidence_route {low_conf_route!r} at {model_name!r} "
                f"must also appear in route_classes_to_next={route_next!r} "
                "(otherwise there is no next stage to route low-confidence "
                "spots into). Consider using `fallback_label` instead."
            )
    return SubModelStage(
        model_name=model_name,
        classes=list(d["classes"]),
        modalities=modalities,
        pool_from=pool_from,
        intermediate_label_in_data=d.get("intermediate_label_in_data"),
        route_classes_to_next=route_next,
        low_confidence_route=low_conf_route,
        fallback_label=d.get("fallback_label"),
    )


def _validate_modalities(modalities: List[str], context: str) -> None:
    unknown = [m for m in modalities if m not in _KNOWN_MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown modalities {unknown} in hierarchy spec at {context!r}. "
            f"Known: {sorted(_KNOWN_MODALITIES)}"
        )


def _parse_sub_model(d: Dict[str, Any]) -> SubModel:
    """Parse a sub-model block from YAML.

    Accepts three forms:

    * ``stages: [<stage>, <stage>, ...]`` — new canonical form; chain of
      any length. This is what every YAML shipped after April 2026 uses.
    * ``three_level: {stage_2a: <stage>, stage_2b: <stage>}`` — legacy
      two-stage form from the Mar 2026 YAML. Silently upcast to a
      two-element ``stages`` chain.
    * Flat stage fields at the top level — legacy one-stage form.
    """
    parent = d["parent"]
    if "stages" in d:
        stages = [
            _parse_stage(st, f"{parent}_stage{i + 1}")
            for i, st in enumerate(d["stages"])
        ]
        if not stages:
            raise ValueError(f"Sub-model for {parent!r} has empty `stages` list.")
        return SubModel(parent_coarse=parent, stages=stages)
    if "three_level" in d:
        tl = d["three_level"]
        stage2a = _parse_stage(tl["stage_2a"], f"{parent}_2a")
        stage2b = _parse_stage(tl["stage_2b"], f"{parent}_2b")
        return SubModel(parent_coarse=parent, stages=[stage2a, stage2b])
    # Flat (single-stage) sub-model.
    stage2 = _parse_stage(d, parent)
    return SubModel(parent_coarse=parent, stages=[stage2])


def _hierarchies_dir() -> Path:
    return Path(__file__).parent / "hierarchies"


def load_hierarchy(source: Union[str, Path]) -> HierarchySpec:
    """Load a :class:`HierarchySpec` from a shipped name or a YAML file.

    Parameters
    ----------
    source :
        Either the name of a shipped hierarchy (e.g. ``"cardiac"``), or a
        path to a YAML file.

    Returns
    -------
    HierarchySpec

    Raises
    ------
    FileNotFoundError
        If the shipped name is unknown and the path does not exist.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for load_hierarchy(); "
            "install with `pip install pyyaml`."
        ) from exc

    src_path: Optional[Path] = None
    # 1) shipped name
    shipped = _hierarchies_dir() / f"{source}.yaml"
    if shipped.exists():
        src_path = shipped
    else:
        p = Path(source)
        if p.exists():
            src_path = p
    if src_path is None:
        known = sorted(p.stem for p in _hierarchies_dir().glob("*.yaml"))
        raise FileNotFoundError(
            f"Could not resolve hierarchy source {source!r}. "
            f"Not a shipped name ({known}) nor an existing path."
        )

    with src_path.open("r") as f:
        raw = yaml.safe_load(f)

    sub_models: Dict[str, SubModel] = {}
    for entry in raw.get("sub_models", []):
        sm = _parse_sub_model(entry)
        sub_models[sm.parent_coarse] = sm

    return HierarchySpec(
        name=raw["name"],
        coarse_col=raw["coarse_col"],
        fine_col=raw.get("fine_col"),
        coarse_niches=list(raw["coarse_niches"]),
        terminal_coarse=list(raw.get("terminal_coarse", [])),
        sub_models=sub_models,
        description=raw.get("description", ""),
        gt_label_remap=dict(raw.get("gt_label_remap", {})),
        palette=dict(raw.get("palette", {})),
    )


def list_shipped_hierarchies() -> List[str]:
    """Return the names of YAMLs shipped under ``config/hierarchies/``."""
    return sorted(p.stem for p in _hierarchies_dir().glob("*.yaml"))


# ─────────────────────────────────────────────────────────────────────────────
# Auto-inference from data (helper for flat-coarse / 2-level users)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODALITIES = ["sd3p", "sd_ffpe", "hd"]


def flat_hierarchy(
    coarse_col: str,
    coarse_classes: List[str],
    name: str = "flat",
    description: Optional[str] = None,
) -> HierarchySpec:
    """Build a :class:`HierarchySpec` with no sub-models (coarse-only).

    Use this when the user's dataset has a single label column and they
    simply want TissueTypist to train one LR classifier over those
    classes — no hierarchy. Prediction will emit ``tt_coarse_label`` /
    ``tt_coarse_score`` / ``tt_final_label`` (= coarse) only.

    Parameters
    ----------
    coarse_col :
        The ``obs`` column holding the class labels.
    coarse_classes :
        List of class labels. Usually ``sorted(adata.obs[coarse_col].unique())``.
    name :
        Identifier for logs / saved ``hierarchy_config.json``.
    description :
        Optional free-text.

    Returns
    -------
    HierarchySpec
        With ``fine_col=None``, every class in ``terminal_coarse``,
        and an empty ``sub_models`` dict.
    """
    classes = sorted(coarse_classes)
    return HierarchySpec(
        name=name,
        coarse_col=coarse_col,
        fine_col=None,
        coarse_niches=classes,
        terminal_coarse=classes,
        sub_models={},
        description=description or f"Flat (single-stage) hierarchy over obs[{coarse_col!r}].",
    )


def infer_hierarchy_from_data(
    adata_or_obs: Any,
    coarse_col: str,
    fine_col: Optional[str] = None,
    name: str = "auto",
    strict: bool = True,
    modalities: Optional[List[str]] = None,
) -> HierarchySpec:
    """Infer a :class:`HierarchySpec` from an AnnData (or its ``obs`` frame).

    This covers the common case of a user who has ``coarse`` and optionally
    ``fine`` obs columns, and wants a default two-level hierarchy where
    each coarse niche automatically gets a sub-model over the fine labels
    that appear beneath it in the data.

    Parameters
    ----------
    adata_or_obs :
        Either an AnnData or a pandas DataFrame with the relevant columns.
    coarse_col :
        Name of the coarse-label column.
    fine_col :
        Name of the fine-label column. If omitted, a flat-only spec with
        no sub-models is returned (equivalent to :func:`flat_hierarchy`).
    name :
        Name to assign to the resulting spec.
    strict :
        If True (default), raise ``ValueError`` when any fine label maps
        to more than one coarse label (ambiguous parent assignment). If
        False, warn and assign to the majority parent.
    modalities :
        Training modalities to declare on each inferred sub-model stage.
        Default ``["sd3p", "sd_ffpe", "hd"]`` — the chain walker silently
        skips any modality whose AnnData has 0 spots in a given niche, so
        declaring all three here is safe and data-driven.

    Returns
    -------
    HierarchySpec
    """
    import pandas as pd

    obs = adata_or_obs.obs if hasattr(adata_or_obs, "obs") else adata_or_obs
    if coarse_col not in obs.columns:
        raise KeyError(f"coarse column {coarse_col!r} not in obs")

    coarse_values = sorted(obs[coarse_col].dropna().astype(str).unique())
    if not coarse_values:
        raise ValueError(f"No non-null values found in obs[{coarse_col!r}]")

    if fine_col is None:
        # Pure flat hierarchy.
        return flat_hierarchy(
            coarse_col=coarse_col,
            coarse_classes=coarse_values,
            name=name,
            description=f"Auto-inferred flat hierarchy from obs[{coarse_col!r}].",
        )

    if fine_col not in obs.columns:
        raise KeyError(f"fine column {fine_col!r} not in obs")

    mods = list(modalities) if modalities else list(_DEFAULT_MODALITIES)
    _validate_modalities(mods, context="infer_hierarchy_from_data")

    crosstab = pd.crosstab(
        obs[fine_col].astype(str), obs[coarse_col].astype(str)
    )

    sub_models: Dict[str, SubModel] = {}
    terminal_coarse: List[str] = []
    import logging as _logging
    _log = _logging.getLogger(__name__)

    for c in coarse_values:
        if c not in crosstab.columns:
            terminal_coarse.append(c)
            continue
        col = crosstab[c]
        fine_here = col[col > 0].index.tolist()
        # Enforce unambiguous parent if strict.
        if strict:
            overlap = [f for f in fine_here if (crosstab.loc[f] > 0).sum() > 1]
            if overlap:
                raise ValueError(
                    f"Fine label(s) {overlap!r} appear under more than one "
                    f"coarse niche. Pass strict=False to permit majority "
                    f"assignment, or clean up the labels first."
                )
        else:
            # Non-strict mode: warn about ambiguous fine labels; assign to the
            # coarse column with the highest count for that fine label (majority).
            fine_here_majority = []
            for f in fine_here:
                row = crosstab.loc[f]
                if (row > 0).sum() > 1:
                    majority = row.idxmax()
                    if majority == c:
                        fine_here_majority.append(f)
                        _log.warning(
                            "infer_hierarchy_from_data: fine label %r appears "
                            "under %d coarse niches; assigning to majority %r.",
                            f, int((row > 0).sum()), majority,
                        )
                    # else: skip — will be assigned under its majority's entry.
                else:
                    fine_here_majority.append(f)
            fine_here = fine_here_majority
        if len(fine_here) < 2:
            # Only one fine label under this coarse — no sub-model needed.
            terminal_coarse.append(c)
            continue
        model_name = c.lower().replace(" ", "_").replace("-", "_")
        sub_models[c] = SubModel(
            parent_coarse=c,
            stages=[SubModelStage(
                model_name=model_name,
                classes=sorted(fine_here),
                modalities=mods,
                # No pool_from / intermediate / routing — single flat sub-model.
                route_classes_to_next=[],
                fallback_label=c,   # low-conf at this stage → report coarse niche
            )],
        )

    return HierarchySpec(
        name=name,
        coarse_col=coarse_col,
        fine_col=fine_col,
        coarse_niches=coarse_values,
        terminal_coarse=terminal_coarse,
        sub_models=sub_models,
        description=(
            f"Auto-inferred 2-level hierarchy from "
            f"obs[{coarse_col!r}] × obs[{fine_col!r}]"
        ),
    )


__all__ = [
    "HierarchySpec",
    "SubModel",
    "SubModelStage",
    "load_hierarchy",
    "list_shipped_hierarchies",
    "infer_hierarchy_from_data",
    "flat_hierarchy",
]
