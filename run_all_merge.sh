#!/usr/bin/env bash
# Submit one sbatch merge job per model that has a trained adapter for a given variant.
#
#   ./run_all_merge.sh --variant reports
#
# Trained adapters are served uniformly as merged full checkpoints (see src/merge_adapter.py),
# so EVERY model with an adapter is merged.  Skips models with no adapter dir, and models whose
# merged checkpoint already exists.  Run this BEFORE ./run_all_inference.sh --variant <v>.
set -e

source .hpc.env
source .env

# --- argument parsing ---
VARIANT=""
CHECKPOINTS_ROOT="benchmark/checkpoints"  # fixed merged/adapter root
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant) VARIANT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 --variant {qa,reports,qa_reports}"; exit 1 ;;
    esac
done
if [[ -z "$VARIANT" ]]; then
    echo "Error: --variant is required"; exit 1
fi

# Derive model list by grepping "name" entries from the registry — no Python import needed.
mapfile -t models < <(grep -oP '"name":\s*"\K[^"]+' src/model_registry.py)

mkdir -p logs

for model in "${models[@]}"; do
    slug=$(echo "$model" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')  # match helpers.sanitize_slug
    merged="${CHECKPOINTS_ROOT}/${slug}/${VARIANT}_merged"
    adapter="${CHECKPOINTS_ROOT}/${slug}/${VARIANT}_adapter"
    if [[ -f "${merged}/config.json" ]]; then
        echo "skip: merged checkpoint already exists at ${merged}"
        continue
    fi
    if [[ ! -f "${adapter}/adapter_config.json" ]]; then
        echo "skip: no adapter at ${adapter} (if only checkpoint-N/ subdirs exist, merge it manually with --adapter-path)"
        continue
    fi
    echo "submitting merge: $model (variant=$VARIANT)"
    sbatch ./run_merge.sh --model "$model" --variant "$VARIANT"
done
