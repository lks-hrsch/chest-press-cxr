#!/usr/bin/env bash
# Submit one sbatch inference job per model in src/model_registry.py.
#
#   ./run_all_inference.sh                       # base (untrained) models
#   ./run_all_inference.sh --variant reports     # apply each model's reports_adapter via vLLM LoRA
#   ./run_all_inference.sh --variant qa          # ditto for qa / qa_reports adapters
#
# Skips models whose output parquet already exists in data/.  For adapter variants, also
# skips models that have no benchmark/checkpoints/<slug>/<variant>_merged on disk.
set -e

source .hpc.env
source .env

# --- argument parsing ---
VARIANT=""
CHECKPOINTS_ROOT="benchmark/checkpoints"  # fixed merged-checkpoint root
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant) VARIANT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--variant {base,qa,reports,qa_reports}]"; exit 1 ;;
    esac
done

# Derive model list by grepping "name" entries from the registry — no Python import needed.
mapfile -t models < <(grep -oP '"name":\s*"\K[^"]+' src/model_registry.py)

mkdir -p logs

# Output suffix + per-model adapter check only apply to the trained variants.
suffix=""
case "$VARIANT" in
    qa|reports|qa_reports) suffix="__${VARIANT}" ;;
esac

for model in "${models[@]}"; do
    slug=$(echo "$model" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')  # match helpers.sanitize_slug
    out="data/inference_${slug}${suffix}.parquet"
    if [[ -f "$out" ]]; then
        echo "skip: $out already exists"
        continue
    fi
    # Adapter variants are served as MERGED full checkpoints — require the merged dir before
    # submitting (run ./run_all_merge.sh --variant "$VARIANT" first to produce them).
    if [[ -n "$suffix" ]]; then
        merged="${CHECKPOINTS_ROOT}/${slug}/${VARIANT}_merged"
        if [[ ! -f "${merged}/config.json" ]]; then
            echo "skip: no merged checkpoint at ${merged} (run ./run_all_merge.sh --variant ${VARIANT} first)"
            continue
        fi
    fi
    echo "submitting: $model${VARIANT:+ (variant=$VARIANT)}"
    if [[ -n "$VARIANT" ]]; then
        sbatch ./run_inference.sh --model "$model" --variant "$VARIANT"
    else
        sbatch ./run_inference.sh --model "$model"
    fi
done
