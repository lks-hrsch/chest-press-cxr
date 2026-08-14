#!/usr/bin/env bash
# Submit one sbatch training-stage-eval job per model in src/model_registry.py.
# Skips models that already have a benchmark/results_training_stages/<slug>.parquet.
set -e

source .hpc.env
source .env

# Derive model list by grepping "name" entries from the registry — no Python import needed.
mapfile -t models < <(grep -oP '"name":\s*"\K[^"]+' src/model_registry.py)

mkdir -p logs benchmark/results_training_stages

for model in "${models[@]}"; do
    slug=$(echo "$model" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')  # match helpers.sanitize_slug
    if [[ -f "benchmark/results_training_stages/${slug}.parquet" ]]; then
        echo "skip: prior training-stage scores exist for $model"
        continue
    fi
    echo "submitting: $model"
    sbatch ./run_evaluate_stages.sh --model "$model" "$@"
done
