#!/bin/bash
# Local smoke sweep: run every registry model through the base + reports variants,
# crossed with the baseline + clinical_context ablation tiers (--context, see
# src/dataset/context_ablation.py), for 4 inference combinations per model.
#
# Deliberately NO `set -e`: we want the sweep to keep going even when one
# model/variant/context combo fails (OOM, corrupt merge, unsupported arch, ...). Each step is
# run via run_step(), which logs the failure and continues; a summary of every
# failed step is printed at the end so nothing fails silently.

# currently running on full test dataset
# LIMIT=5000

# Tee all stdout+stderr to a timestamped log file (under logs/, matching the
# SLURM log convention) while still printing to the console.
mkdir -p logs
LOG_FILE="logs/run_local_context_ablation_test_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "logging to ${LOG_FILE}"
echo

failures=()

run_step() {
    # run_step <label> <command...>
    local label="$1"; shift
    echo "=== [START] ${label} ==="
    if "$@"; then
        echo "=== [OK]    ${label} ==="
    else
        local rc=$?
        echo "=== [FAIL]  ${label} (exit ${rc}) — continuing ===" >&2
        failures+=("${label} (exit ${rc})")
    fi
    echo
}

# --model <hf-id>: restrict the sweep to a single registry model (default: all).
MODEL_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -n "$MODEL_FILTER" ]; then
    models=("$MODEL_FILTER")
else
    # Derive model list by grepping "name" entries from the registry — no Python import needed.
    mapfile -t models < <(grep -oP '"name":\s*"\K[^"]+' src/model_registry.py)
fi

contexts=(baseline clinical_context)

for model in "${models[@]}"; do
    slug=$(echo "$model" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')  # match helpers.sanitize_slug

    for context in "${contexts[@]}"; do
        # base (untrained baseline) — omit --variant entirely, same as run_local_test.sh
        run_step "${model} :: base :: ${context} :: $(date '+%F %T')" \
            python -m src.inference --model "$model" --context "$context"
        base_parquet="data/inference_${slug}__ablation_${context}.parquet"

        # reports
        run_step "${model} :: reports :: ${context} :: $(date '+%F %T')" \
            python -m src.inference --model "$model" --variant reports --context "$context"
        reports_parquet="data/inference_${slug}__reports__ablation_${context}.parquet"
    done
done

echo "============================================================"
if [ ${#failures[@]} -eq 0 ]; then
    echo "All ${#models[@]} models swept — every step succeeded."
else
    echo "${#failures[@]} step(s) FAILED (the rest completed):"
    for f in "${failures[@]}"; do
        echo "  - ${f}"
    done
fi
echo "============================================================"

# Non-zero exit if anything failed, so callers/CI can detect it — but only
# AFTER the full sweep has run, never mid-way.
[ ${#failures[@]} -eq 0 ]
