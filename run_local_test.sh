#!/bin/bash
# Local smoke sweep: run every registry model through base + reports + qa inference.
#
# Deliberately NO `set -e`: we want the sweep to keep going even when one
# model/variant fails (OOM, corrupt merge, unsupported arch, ...). Each step is
# run via run_step(), which logs the failure and continues; a summary of every
# failed step is printed at the end so nothing fails silently.

# currently running on full test dataset
# LIMIT=5000

# Tee all stdout+stderr to a timestamped log file (under logs/, matching the
# SLURM log convention) while still printing to the console.
mkdir -p logs
LOG_FILE="logs/run_local_test_$(date +%Y%m%d-%H%M%S).log"
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

for model in "${models[@]}"; do
    # skip if model == google/medgemma-4b-it
    # if [ "$model" == "google/medgemma-4b-it" ]; then
    #     # this model is for CLI tests only
    #     continue
    # fi
    slug=$(echo "$model" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')  # match helpers.sanitize_slug

    # # skip models that are already _reports.parqet
    # echo "Checking if data/inference_${slug}__reports.parquet exists..."
    # if [ -f "data/inference_${slug}__reports.parquet" ]; then
    #     echo "Skipping $model because data/inference_${slug}__reports.parquet already exists."
    #     continue
    # fi

    # baseline
    run_step "${model} :: base :: $(date '+%F %T')" \
    python -m src.inference --model "$model"
    baseline_parquet="data/inference_${slug}.parquet"
    # if [ -f "$baseline_parquet" ]; then
    #     run_step "${model} :: base :: eval :: $(date '+%F %T')" \
    #         python -m src.evaluate --model "$model"
    # else
    #     echo "=== [SKIP]  ${model} :: base :: eval — no inference parquet at ${baseline_parquet} ==="
    #     echo
    # fi

    # reports
    # python -m src.merge_adapter --model "$model" --variant reports
    run_step "${model} :: reports :: $(date '+%F %T')" \
        python -m src.inference --model "$model" --variant reports
    reports_parquet="data/inference_${slug}__reports.parquet"
    # if [ -f "$reports_parquet" ]; then
    #     run_step "${model} :: reports :: eval :: $(date '+%F %T')" \
    #         python -m src.evaluate --model "$model" --variant reports
    # else
    #     echo "=== [SKIP]  ${model} :: reports :: eval — no inference parquet at ${reports_parquet} ==="
    #     echo
    # fi

    # qba
    # python -m src.merge_adapter --model "$model" --variant qa
    run_step "${model} :: qa :: $(date '+%F %T')" \
        python -m src.inference --model "$model" --variant qa
    qa_parquet="data/inference_${slug}__qa.parquet"
    # if [ -f "$qa_parquet" ]; then
    #     run_step "${model} :: qa :: eval :: $(date '+%F %T')" \
    #         python -m src.evaluate --model "$model" --variant qa
    # else
    #     echo "=== [SKIP]  ${model} :: qa :: eval — no inference parquet at ${qa_parquet} ==="
    #     echo
    # fi

    # qba + reports
    # python -m src.merge_adapter --model "$model" --variant qa_reports
    run_step "${model} :: qa_reports :: $(date '+%F %T')" \
        python -m src.inference --model "$model" --variant qa_reports
    qa_reports_parquet="data/inference_${slug}__qa_reports.parquet"
    # if [ -f "$qa_reports_parquet" ]; then
    #     run_step "${model} :: qa_reports :: eval :: $(date '+%F %T')" \
    #         python -m src.evaluate --model "$model" --variant qa_reports
    # else
    #     echo "=== [SKIP]  ${model} :: qa_reports :: eval — no inference parquet at ${qa_reports_parquet} ==="
    #     echo
    # fi
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
