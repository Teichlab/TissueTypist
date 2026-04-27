# Reference training data

The cardiac reference datasets used to train the shipped TissueTypist
presets are not stored in this repository — they are too large for GitHub.

## Files

| File | Modality | Spots / cells |
|------|----------|---------------|
| `adata_sd_3p_raw.h5ad` | Visium SD 3-prime | 19,590 |
| `adata_sd_ffpe_raw.h5ad` | Visium SD FFPE | 19,633 |
| `adata_hd_raw.h5ad` | Visium HD FFPE | 215,913 (cell-level) |
| `adata_hd_windows.h5ad` | HD pseudobulk | 9,583 (window-level) |

Each `obs` table contains: `donor`, `section_ID`, `library`,
`niche_fine_Mar2026`, `niche_coarse_Mar2026`. HD windows additionally:
`window_col`, `window_row`, `window_col_idx`, `window_row_idx`, `_n_cells`.

`adata.X` holds raw counts in every file (float32 with integer values for
SD; raw summed counts for HD windows). Normalisation is applied
automatically at load time by the package.

## Download

The download links for the cardiac reference datasets will be made
available at the time of manuscript publication.

In the meantime, you can already use TissueTypist in two ways without
needing these specific files:

- **Predict on full-transcriptomics queries** (Visium SD / Visium HD)
  — the shipped model presets in
  [`tissuetypist/models/`](../tissuetypist/models) cover the cardiac
  hierarchy out of the box. See
  [`notebooks/demo_predict_only.ipynb`](../notebooks/demo_predict_only.ipynb).
- **Train on your own (non-cardiac) reference data** — TissueTypist is
  tissue-agnostic. Provide your own annotated AnnData and a hierarchy
  YAML (or use `--flat` for a single-stage classifier) and run
  `tissuetypist train`. See
  [`docs/user-guide.md`](../docs/user-guide.md) and
  [`notebooks/demo_lung_loso.ipynb`](../notebooks/demo_lung_loso.ipynb)
  for a worked non-cardiac example.

After downloading, place the files directly in this `data/` directory:

```
data/
├── adata_sd_3p_raw.h5ad
├── adata_sd_ffpe_raw.h5ad
├── adata_hd_raw.h5ad
└── adata_hd_windows.h5ad
```

`adata_hd_windows.h5ad` can also be regenerated from `adata_hd_raw.h5ad`
with:

```bash
tissuetypist pseudobulk-hd \
    --hd           data/adata_hd_raw.h5ad \
    --scalefactors configs/hd_scalefactors.json \
    --outdir       data/
```

## Citation

If you use these datasets, please cite the preprint listed in the root
[`CITATION.cff`](../CITATION.cff).
