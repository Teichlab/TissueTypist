# The niche hierarchy

TissueTypist's niche structure is declared in a **YAML file**, not
hardcoded. This means you can adapt TissueTypist to any tissue by
writing one YAML, and the training / prediction / evaluation code
adapts with no Python changes.

- [The shipped cardiac hierarchy](#the-shipped-cardiac-hierarchy)
- [Multi-stage sub-model chains](#multi-stage-sub-model-chains)
- [The YAML spec](#the-yaml-spec)
- [Writing your own hierarchy](#writing-your-own-hierarchy)
- [Auto-inferring a 2-level hierarchy from data](#auto-inferring-a-2-level-hierarchy-from-data)

---

## The shipped cardiac hierarchy

File: `tissuetypist/config/hierarchies/cardiac.yaml` (April 2026 release).

- **7 coarse niches**: Ventricle, Atrium, Pacemaker conduction system,
  Epicardial region, Vasculature, AV junction, Lymph node.
- **3 intermediate nodes** (synthetic pooled classes, not present in
  `obs` directly — see "pool_from" in the spec below):
  Atrial myocardium (under Atrium), Sinoatrial region (under Pacemaker
  conduction system), Great vessels (under Vasculature).
- The Atrium chain additionally introduces **Atrium - LR** as a
  second-level intermediate (= Atrium - Left ∪ Atrium - Right).
- **21 terminal leaves**: AV nodal region, AV ring, Atrium - Left,
  Atrium - Right, Atrium - Transitional, Connective tissue, Coronary
  vessel, Ductus arteriosus, Endocardial cushion - Valve,
  Endocardium - Atrial, Endocardium - Ventricular, Epicardial region,
  Great vessel, Lymph node, SA node - Head, SA node - Tail, Sinus horn,
  VCS - Distal, VCS - Proximal, Ventricle - Compact,
  Ventricle - Trabeculated.

Terminal coarse niches with no Stage 2 sub-model: **Epicardial region**,
**Lymph node**.

See `niche_tree.pdf` in the repo root for the diagram.

### Inspect the spec programmatically

```python
from tissuetypist import load_hierarchy, list_shipped_hierarchies

list_shipped_hierarchies()            # ['cardiac']
spec = load_hierarchy("cardiac")
spec.coarse_col                       # 'niche_coarse_Apr2026'
spec.fine_col                         # 'niche_fine_Apr2026'
spec.sub_models["Atrium"].depth       # 3
```

---

## Multi-stage sub-model chains

Each non-terminal coarse niche has a **chain** of one or more classifier
stages. A chain of length 1 is a flat sub-model; longer chains split the
decision into successively finer steps so each stage can be trained using
only the modalities that carry the relevant labels.

| Coarse niche | Depth | Stages |
|---|---|---|
| Ventricle | 1 | Compact / Trabeculated / VCS-P / VCS-D / Endocardium-V |
| AV junction | 1 | AV ring / Endocardial cushion - Valve |
| **Atrium** | **3** | 1: Atrial myocardium vs Endocardium-Atrial · 2: Atrium-LR vs Transitional · 3: Left vs Right |
| PCS | 2 | 1: Sinoatrial region vs AV nodal region · 2: Sinus horn / SA-Head / SA-Tail |
| Vasculature | 2 | 1: Great vessels / Coronary / Connective tissue · 2: Great vessel vs Ductus arteriosus |

At prediction time the chain walker runs each stage, gates by confidence
θ (default 0.5), and either continues to the next stage or falls back
to the `fallback_label`. The Atrium chain uses
**`low_confidence_route: Atrium - LR`** on stage 2 so that atrium spots
with low confidence at the HD-trained Transitional-split stage are
routed through the Left-vs-Right stage anyway (with `tt_low_conf=True`
flagged).

---

## The YAML spec

Top-level fields of a hierarchy YAML:

| Field | Required | Purpose |
|---|---|---|
| `name` | yes | Identifier used in logs + saved `hierarchy_config.json`. |
| `coarse_col` | yes | `obs` column holding the coarse label. |
| `fine_col` | no | `obs` column holding the fine label. Omit for flat-only hierarchies. |
| `coarse_niches` | yes | List of all coarse class labels. |
| `terminal_coarse` | yes | Coarse niches that have no Stage 2 model. |
| `sub_models` | yes | List of sub-model definitions (see below). |
| `palette` | no | `{label: "#hex"}` for consistent plot colours. |
| `gt_label_remap` | no | `{data_label: output_label}` for aligning pooled terminal labels in evaluation metrics. |

Per sub-model, inside `sub_models:`:

```yaml
- parent: <coarse_niche_name>
  stages:
    - model_name: <string>              # filename stem for the joblib
      classes: [<class_1>, <class_2>, ...]
      modalities: [sd3p, sd_ffpe, hd]   # which of these contribute training data
      pool_from:                        # optional: synthesise a class by pooling data labels
        <synthetic_class_name>:
          - <data_label_1>
          - <data_label_2>
      intermediate_label_in_data: <string>  # optional: for non-synthetic intermediates already present in obs
      route_classes_to_next: [<class>]  # which classes continue to the next stage
      low_confidence_route: <class>     # optional: permissive routing on low θ
      fallback_label: <string>          # label to emit if this stage's confidence < θ
    # ... more stages …
```

### Cardiac Atrium chain excerpt (3 stages, 2 synthetic intermediates)

```yaml
- parent: Atrium
  stages:
    - model_name: atrium_split
      classes: [Atrial myocardium, Endocardium - Atrial]
      modalities: [sd3p, sd_ffpe, hd]
      pool_from:
        Atrial myocardium: [Atrium - Left, Atrium - Right, Atrium - Transitional]
      route_classes_to_next: [Atrial myocardium]
      fallback_label: Atrium

    - model_name: atrium_transitional
      classes: [Atrium - LR, Atrium - Transitional]
      modalities: [hd]
      pool_from:
        Atrium - LR: [Atrium - Left, Atrium - Right]
      route_classes_to_next: [Atrium - LR]
      low_confidence_route: Atrium - LR
      fallback_label: Atrial myocardium

    - model_name: atrium_lr
      classes: [Atrium - Left, Atrium - Right]
      modalities: [sd3p, sd_ffpe, hd]
      fallback_label: Atrium - LR
```

### Cardiac PCS chain excerpt (2 stages, 1 synthetic intermediate pooling mixed data labels)

```yaml
- parent: Pacemaker conduction system
  stages:
    - model_name: pcs_split
      classes: [Sinoatrial region, AV nodal region]
      modalities: [sd3p, hd]
      pool_from:
        Sinoatrial region:
          - "Sinoatrial region - non-terminal category"  # SD 3' explicit intermediate
          - Sinus horn                                   # HD sub-regions
          - SA node - Head
          - SA node - Tail
      route_classes_to_next: [Sinoatrial region]
      fallback_label: Pacemaker conduction system

    - model_name: pcs_sinoatrial
      classes: [Sinus horn, SA node - Head, SA node - Tail]
      modalities: [hd]
      fallback_label: Sinoatrial region
```

---

## Writing your own hierarchy

1. Copy the shipped cardiac YAML as a starting point.
2. Replace the coarse / fine / intermediate / terminal labels with yours.
3. Update `coarse_col` / `fine_col` to your AnnData obs columns.
4. For each non-terminal coarse niche, decide its stage depth:
    - If you only have flat labels → 1 stage listing all children.
    - If some children exist only in a subset of modalities → multi-stage
      chain that routes from "intermediate" classes (pooled from the
      restricted-modality labels) down to terminal leaves.
5. Add a `palette:` section (label → hex colour) so your plots render
   consistently. Labels absent from the palette fall back to `tab20`.
6. Add `gt_label_remap:` if you pool terminal labels (e.g. if your
   training labels include synonyms you want to collapse at metric time).
7. Validate:

```bash
tissuetypist validate-hierarchy my_tissue.yaml --adata my_data.h5ad
```

### Available modality tags

The built-in modality tags are `sd3p`, `sd_ffpe`, `hd` — they map to the
three `--reference*` CLI slots (primary / secondary / tertiary) at
training time. For non-cardiac tissues these are just slot identifiers;
use them in the order of your training data.

---

## Auto-inferring a 2-level hierarchy from data

If you have clean coarse + fine columns where each fine label maps to
exactly one coarse niche, skip the YAML entirely:

```bash
tissuetypist train \
    --reference my_data.h5ad \
    --outdir    results/my_auto \
    --auto_infer --coarse_col my_coarse --fine_col my_fine
```

Builds a `HierarchySpec` on the fly with one flat sub-model per coarse
niche. Pass `--no_strict_infer` to permit fine labels that appear under
multiple coarse niches (assigned to their majority parent).

From Python:

```python
from tissuetypist import infer_hierarchy_from_data
spec = infer_hierarchy_from_data(adata, "my_coarse", "my_fine", strict=True)
```
