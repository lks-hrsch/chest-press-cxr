#!/usr/bin/env bash
# Submit one sbatch eval job per model in src/model_registry.py.
# Skips models that already have a summary.csv in benchmark/results/.
set -e

source .hpc.env
source .env

# Derive model list by grepping "name" entries from the registry — no Python import needed.
mapfile -t models < <(grep -oP '"name":\s*"\K[^"]+' src/model_registry.py)

mkdir -p logs benchmark/results

for model in "${models[@]}"; do
    slug=$(echo "$model" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')  # match helpers.sanitize_slug
    if compgen -G "benchmark/results/${slug}_*/summary.csv" > /dev/null 2>&1; then
        echo "skip: prior summary exists for $model"
        continue
    fi
    echo "submitting: $model"
    sbatch ./run_eval.sh --model "$model"
done
