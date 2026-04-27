#!/usr/bin/env bash
# scripts/07_populate_preset_models.sh
# ===============================================================
# Populate tissuetypist/models/{default,own_only,neighbour_heavy}/
# from results/apr2026_{default,own_only,neighbour_heavy}/.
#
# Run this ONCE after training all three weight presets:
#
#   tissuetypist train --outdir results/apr2026_default ...
#   tissuetypist train --outdir results/apr2026_own_only ... \
#       --neighbour_weight 0.0 --edge_weight 0.0
#   tissuetypist train --outdir results/apr2026_neighbour_heavy ... \
#       --neighbour_weight 1.0 --edge_weight 5.0
#
# Excluded from the copy (to keep the shipped package small):
#   - training_*.log      (per-run training logs)
#   - qc_plots/           (edge-detection QC plots)
#
# The copy includes:
#   - *_pipeline.joblib   (9 sub-models + 1 coarse)
#   - *_gene_list.txt     (one per pipeline)
#   - hierarchy_config.json
#   - training_summary.csv
#   - gene_counts.csv
#
# After running, `pip install -e .` picks up the new files via the
# [tool.setuptools.package-data] stanza in pyproject.toml, and
# `tissuetypist info` reports all three presets as "installed".
# ===============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
MODELS_DIR="${REPO_ROOT}/tissuetypist/models"

PRESETS=(default own_only neighbour_heavy)

echo "Populating ${MODELS_DIR} from ${RESULTS_DIR}/apr2026_<preset>/ ..."
echo ""

for preset in "${PRESETS[@]}"; do
    src="${RESULTS_DIR}/apr2026_${preset}"
    dst="${MODELS_DIR}/${preset}"

    if [[ ! -d "${src}" ]]; then
        echo "  [skip] ${preset}: source ${src} not found — train it first."
        continue
    fi

    if [[ ! -f "${src}/hierarchy_config.json" ]]; then
        echo "  [skip] ${preset}: ${src}/hierarchy_config.json missing — incomplete training."
        continue
    fi

    mkdir -p "${dst}"
    echo "  [copy] ${preset}: ${src}  →  ${dst}"

    # Copy pipelines + gene lists + config + summaries, skipping logs and QC.
    # rsync honours `--exclude` for precise filtering.
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete \
              --exclude='training_*.log' \
              --exclude='qc_plots/' \
              --exclude='qc_plots' \
              "${src}/" "${dst}/"
    else
        # Fallback: cp + manual cleanup.
        rm -rf "${dst:?}"/*
        cp -R "${src}/." "${dst}/"
        rm -f "${dst}"/training_*.log
        rm -rf "${dst}/qc_plots"
    fi

    n_pipelines="$(ls "${dst}"/*_pipeline.joblib 2>/dev/null | wc -l | tr -d ' ')"
    n_genelists="$(ls "${dst}"/*_gene_list.txt   2>/dev/null | wc -l | tr -d ' ')"
    size="$(du -sh "${dst}" | awk '{print $1}')"
    echo "         ${n_pipelines} joblib + ${n_genelists} gene_list.txt + config.json  (${size})"
done

echo ""
echo "Done. Quick sanity check:"
echo ""
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
from tissuetypist.models import list_shipped_presets
presets = list_shipped_presets()
print(f"  tissuetypist.models.list_shipped_presets() → {presets}")
if len(presets) == 3:
    print("  All three presets installed ✓")
else:
    print(f"  Note: only {len(presets)} / 3 presets installed.")
PY
fi

echo ""
echo "Next: commit the changes (git add tissuetypist/models/) and"
echo "      run 'tissuetypist info' to verify."
