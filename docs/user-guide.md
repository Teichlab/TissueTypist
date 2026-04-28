# User guide

Detailed walkthroughs for the three common TissueTypist workflows:

- [Predicting on Visium data](#predicting-on-visium-data)
- [Training on the cardiac reference pipeline](#training-on-the-cardiac-reference-pipeline)
- [Training on your own data](#training-on-your-own-data)
- [Retraining for imaging-based ST (Xenium / MERFISH / CosMx)](#retraining-for-imaging-based-st)
- [Evaluating predictions (plots + metrics)](#evaluating-predictions)

See also:
- [docs/hierarchy.md](hierarchy.md) — the cardiac niche hierarchy + the YAML spec.
- [docs/output-columns.md](output-columns.md) — the `tt_*` columns TissueTypist adds.

---

## Predicting on Visium data

TissueTypist ships with three pre-trained cardiac hierarchies (see
[Shipped weight presets](#shipped-weight-presets) below). No retraining is
needed unless you want to use different weights or a different tissue.

```bash
tissuetypist predict \
    --query       data/my_visium.h5ad \
    --model_dir   $(python -c "import tissuetypist; print(tissuetypist.load_preset('default'))") \
    --modality    sd \
    --section_col section_ID \
    --outdir      results/pred_my_visium
```

Or programmatically:

```python
from tissuetypist import predict_adata, load_preset
adata = predict_adata(
    adata,
    model_dir=load_preset("default"),
    modality="sd",
    section_col="section_ID",
)
```

Output: `{prefix}_predicted.h5ad` plus `{prefix}_prediction_summary.csv`.
Column meanings are described in [docs/output-columns.md](output-columns.md).

### Shipped weight presets

| Preset | `neighbour_weight` | `edge_weight` | When to use |
|---|---|---|---|
| `default` | 0.3 | 5.0 | Recommended. Matches v1 TissueTypist weights. |
| `own_only` | 0.0 | 0.0 | Disables spatial context — use when query coordinates are unreliable. |
| `neighbour_heavy` | 1.0 | 5.0 | Strong neighbourhood weighting. Use when tissue architecture is highly locally-organised. |

`tissuetypist info` lists which presets are currently installed.

---

## Training on the cardiac reference pipeline

Reproduces the manuscript's pipeline from scratch on all three modalities.

### Step 0 — HD sliding-window pseudobulk (run once)

```bash
tissuetypist pseudobulk-hd \
    --hd           data/adata_hd_raw.h5ad \
    --scalefactors configs/hd_scalefactors.json \
    --outdir       data/
```

Output: `data/adata_hd_windows.h5ad`.

### Step 1 — Shared-gene catalogue (recommended)

```bash
tissuetypist build-catalogue \
    --sd3p    data/adata_sd_3p_raw.h5ad \
    --sd_ffpe data/adata_sd_ffpe_raw.h5ad \
    --hd      data/adata_hd_raw.h5ad \
    --outdir  results/phase0_pseudobulk \
    --pseudobulk
```

Output: `results/phase0_pseudobulk/gene_pools.csv` with a `shared_all`
column. You can skip this step and pass `--gene_pools my_genes.txt` later
if you'd rather use a curated marker list, or omit `--gene_pools` entirely
to use the intersection of the reference AnnDatas' var_names.

### Step 2 — Hierarchical training

```bash
tissuetypist train \
    --reference           data/adata_sd_3p_raw.h5ad \
    --reference_secondary data/adata_sd_ffpe_raw.h5ad \
    --reference_tertiary  data/adata_hd_windows.h5ad \
    --outdir              results/apr2026_default \
    --gene_pools          results/phase0_pseudobulk/gene_pools.csv
```

Produces 10 sklearn pipelines (1 coarse + 9 sub-model stages), their gene
lists, and a `hierarchy_config.json`. See
[docs/hierarchy.md](hierarchy.md) for the stage structure.

### Training the alternative weight presets

The shipped `default` preset has `neighbour_weight=0.3, edge_weight=5.0`.
To also train `own_only` (0, 0) and `neighbour_heavy` (1, 5):

```bash
tissuetypist train --outdir results/apr2026_own_only ... \
    --neighbour_weight 0.0 --edge_weight 0.0

tissuetypist train --outdir results/apr2026_neighbour_heavy ... \
    --neighbour_weight 1.0 --edge_weight 5.0

# Then populate the shipped models directory:
bash scripts/07_populate_preset_models.sh
```

---

## Training on your own data

TissueTypist is tissue-agnostic. Three paths in increasing order of effort:

### 1. Flat — single label column, no sub-models

```bash
tissuetypist train \
    --reference my_data.h5ad \
    --outdir    results/my_flat \
    --flat --coarse_col my_niche
```

Produces a single LR classifier over the classes in `obs["my_niche"]`.
Prediction emits `tt_coarse_label` / `tt_coarse_score` / `tt_final_label`
(= coarse); `tt_fine_label` is unused.

### 2. Auto-infer — coarse + fine columns, 2-level hierarchy

```bash
tissuetypist train \
    --reference my_data.h5ad \
    --outdir    results/my_auto \
    --auto_infer --coarse_col my_coarse --fine_col my_fine
```

Builds one flat sub-model per coarse niche, whose children are the fine
labels observed beneath it. Use `--no_strict_infer` to permit ambiguous
fine labels (assigned to their majority-count coarse niche).

### 3. Custom hierarchy — your own YAML

Copy `tissuetypist/config/hierarchies/cardiac.yaml`, edit the niche names
/ modalities / stages / palette / remap, then:

```bash
tissuetypist train \
    --reference my_data.h5ad \
    --hierarchy my_tissue.yaml \
    --outdir    results/my_hier
```

Validate before a long run:

```bash
tissuetypist validate-hierarchy my_tissue.yaml --adata my_data.h5ad
```

See [docs/hierarchy.md](hierarchy.md) for the full YAML schema.

### Using your own gene universe

`--gene_pools` accepts three forms:

| Input | Behaviour |
|---|---|
| `path/to/gene_pools.csv` | Reads the `shared_all` column. Canonical `build-catalogue` output. |
| `path/to/genes.txt` | One gene per line. Use this for a curated marker list. |
| *(omit)* | Uses the intersection of `var_names` across the provided reference AnnDatas. |

---

## Retraining for imaging-based ST

Targeted gene panels (Xenium, MERFISH, CosMx) cover a small fraction of
the transcriptome. Using the shipped full-genome models directly is
wrong because `normalize_total(1e4)` produces different expression
scales between full-genome training and a panel query. Fix: retrain on
`panel ∩ shared_all`.

### Three gene-selection strategies

**1. Pre-computed gene lists from a prior full-genome training** (recommended first try):

```bash
tissuetypist train-panel \
    --query               data/merfish.h5ad \
    --reference           data/adata_sd_3p_raw.h5ad \
    --reference_secondary data/adata_sd_ffpe_raw.h5ad \
    --reference_tertiary  data/adata_hd_windows.h5ad \
    --gene_pools          results/phase0_pseudobulk/gene_pools.csv \
    --gene_lists_from     results/apr2026_default \
    --outdir              results/panel_merfish
```

Each stage intersects its Apr2026 gene list with the query panel. Tends
to produce ~200 genes per stage on a 238-gene MERFISH panel.

**2. Fresh DEG+HVG on panel-normalised data** (default, no flag):

```bash
tissuetypist train-panel \
    --query data/merfish.h5ad \
    --reference ... --reference_secondary ... --reference_tertiary ... \
    --gene_pools results/phase0_pseudobulk/gene_pools.csv \
    --outdir results/panel_merfish_deghvg \
    --feature_set deg_hvg --min_logfc 0.5
```

Runs fresh DEG+HVG selection per stage on data normalised within the
panel gene space. Controllable via `--feature_set deg_only`,
`--min_logfc`, `--max_degs_per_niche`.

**3. Custom curated gene list**:

```bash
tissuetypist train-panel \
    --query data/merfish.h5ad --reference ... \
    --custom_gene_list path/to/markers.txt \
    --outdir results/panel_merfish_curated
```

### Cell-level query data

For raw MERFISH / Xenium cell-level h5ads (>50 k cells with
`obsm['spatial']`), `train-panel` auto-detects and pseudobulks into
windows. The windowed query is saved to `<outdir>/query_windows.h5ad`.
You can also pre-window yourself with
`tissuetypist.data.pseudobulk.sliding_window_pseudobulk_cells`.

### Gene panel overlap (measured)

| Platform | Panel genes | Overlap with cardiac `shared_all` |
|---|---|---|
| Xenium 5K | 5,001 | 4,191 (83.8%) |
| MERFISH | 238 | 232 (97.5%) |

---

## Evaluating predictions

`tissuetypist evaluate` runs prediction and writes a full set of plots +
metrics, not just the h5ad:

```bash
tissuetypist evaluate \
    --query_sd   data/my_visium.h5ad \
    --model_dir  results/apr2026_default \
    --outdir     results/eval_my_visium \
    --modality   sd --section_col section_ID
```

Per-modality outputs:

- `{prefix}_predicted.h5ad`
- `{prefix}_prediction_summary.csv` — per-niche counts + mean confidence
- `{prefix}_classification_report.csv` — if ground-truth column present
- `{prefix}_confusion_matrix.pdf` — if ground-truth column present
- `{prefix}_spatial_<section>.pdf` — GT / prediction / confidence panels
- `{prefix}_umap.pdf` — GT / prediction / confidence / low-conf panels
- `{prefix}_confidence_distributions.pdf` — violin per coarse niche

Colours come from the YAML's `palette:` section (see
[docs/hierarchy.md](hierarchy.md)). Pass `--no_eval` to skip the
metrics + confusion matrix on external data without ground truth.

## CLI reference (full)

| Command | Purpose |
|---|---|
| `tissuetypist info` | List shipped presets + hierarchies; report install status. |
| `tissuetypist predict` | Run prediction; writes `{prefix}_predicted.h5ad` + summary. |
| `tissuetypist evaluate` | Predict + confusion matrix + spatial / UMAP / confidence plots. |
| `tissuetypist train` | Train on your own reference data — any tissue. Supports `--flat`, `--auto_infer`, or a custom YAML hierarchy. |
| `tissuetypist train-panel` | Retrain for an imaging-based ST panel (Xenium / MERFISH / CosMx). |
| `tissuetypist build-catalogue` | Build the shared-gene pool across 1–3 reference datasets. |
| `tissuetypist pseudobulk-hd` | HD sliding-window pseudobulk (run once before HD training). |
| `tissuetypist validate-hierarchy` | Schema + optional `obs` check for a hierarchy YAML. |

Every subcommand has its own `--help`.
