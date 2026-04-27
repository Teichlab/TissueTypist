# Prediction output columns

Columns that TissueTypist writes to `adata.obs` (after `predict_adata`)
or to the prediction DataFrame (after the lower-level `predict`).

## Core columns (always present)

| Column | Type | Meaning |
|---|---|---|
| `tt_coarse_label` | str | Stage 1 prediction (one of the coarse niches). |
| `tt_coarse_score` | float | `max(predict_proba)` from the coarse classifier. |
| `tt_fine_label` | str / NaN | Most-specific sub-model prediction. `NaN` if the coarse was terminal (e.g. Epicardial region / Lymph node) or every downstream stage was low-confidence. |
| `tt_fine_score` | float / NaN | `max(predict_proba)` of the most-specific stage that ran. |
| `tt_joint_score` | float / NaN | Product of all stage confidences traversed. For a fully-resolved cardiac Atrium Left / Right spot this is `coarse × stage2a × stage2b × stage2c`. |
| `tt_final_label` | str | **Recommended.** `tt_fine_label` where present, else `tt_coarse_label`. |
| `tt_low_conf` | bool | `True` when any traversed stage's score was below θ (default 0.5). |

## Per-stage columns

One pair per sub-model stage defined in the hierarchy YAML:

| Column | Meaning |
|---|---|
| `tt_<stage.model_name>_label` | That stage's predicted label (whether the spot ultimately routed further or stopped). |
| `tt_<stage.model_name>_score` | That stage's `max(predict_proba)`. |

For the shipped cardiac hierarchy, these expand to:

- `tt_ventricle_label` / `tt_ventricle_score`
- `tt_avjunction_label` / `tt_avjunction_score`
- `tt_atrium_split_*`, `tt_atrium_transitional_*`, `tt_atrium_lr_*`
- `tt_pcs_split_*`, `tt_pcs_sinoatrial_*`
- `tt_vasc_split_*`, `tt_vasc_fine_*`

Only the stages traversed for a given spot are populated (others remain
`NaN`).

## Legacy backward-compat columns

Retained for one release so analyses written against older cardiac
outputs keep working:

| Column | Meaning |
|---|---|
| `tt_stage2a_score` | Alias of the first sub-model stage's score for Atrium / PCS spots. |
| `tt_vasc2a_score` | Alias of Vasculature's stage 1 score. |

These duplicate the corresponding `tt_<model_name>_score` columns. New
analyses should prefer the per-stage columns.

## Interpretation tips

- **Filter by confidence:** `adata.obs[adata.obs["tt_low_conf"] == False]`
  keeps only spots whose full chain passed θ.
- **Rank by most confident prediction:**
  `adata.obs.sort_values("tt_joint_score", ascending=False)`.
- **Per-coarse-niche diagnostic:** group by `tt_coarse_label` and look
  at the distribution of `tt_joint_score` vs `tt_coarse_score` — a large
  gap indicates a stage deeper in the chain is where confidence drops.
- **Terminal coarse niches** (e.g. Epicardial region, Lymph node in
  cardiac) have `tt_fine_label = NaN` and `tt_joint_score = NaN` —
  there's no Stage 2 to compute a joint with. Use `tt_final_label` to
  include these in downstream analyses.

## Example

```python
import anndata as ad
from tissuetypist import predict_adata, load_preset

adata = ad.read_h5ad("my_visium.h5ad")
adata = predict_adata(adata, model_dir=load_preset("default"),
                      modality="sd", section_col="section_ID")

# Core per-spot summary
adata.obs[["tt_final_label", "tt_coarse_score",
           "tt_joint_score",  "tt_low_conf"]].head()

# Per-stage confidence for atrium spots
atrium = adata.obs[adata.obs["tt_coarse_label"] == "Atrium"]
atrium[["tt_atrium_split_score",
        "tt_atrium_transitional_score",
        "tt_atrium_lr_score",
        "tt_final_label"]].describe()
```
