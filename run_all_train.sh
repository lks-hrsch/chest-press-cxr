#!/usr/bin/env bash
# Submit ONE training stage for every model in src/model_registry.py.
#
# Usage:
#   ./run_all_train.sh --stage qa           # fresh LoRA on base, VQA grounding → qa_adapter/
#   ./run_all_train.sh --stage reports      # fresh LoRA on base, reports       → reports_adapter/
#   ./run_all_train.sh --stage qa+reports   # continue qa_adapter on reports    → qa_reports_adapter/
#
# Stages are triggered INDIVIDUALLY — there is no sbatch dependency chain.  Each invocation
# submits one independent job per model for the chosen stage.  Adapters are kept as adapters
# (no merging).
#
# qa+reports is a qa→reports curriculum via PEFT continuation: it loads the trained qa adapter
# as trainable LoRA weights and continues it on the report data.  PREREQUISITE: the qa stage
# must already be done — models without a qa_adapter/ are warned and skipped.
#
# Idempotent: a model is skipped if the target adapter directory for the stage already exists.
set -e

source .hpc.env
source .env

# --- argument parsing ---
STAGE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --stage {qa|reports|qa+reports}"
            exit 1
            ;;
    esac
done

case "$STAGE" in
    qa|reports|qa+reports) ;;
    *)
        echo "Error: --stage {qa|reports|qa+reports} is required"
        exit 1
        ;;
esac

mapfile -t models < <(grep -oP '"name":\s*"\K[^"]+' src/model_registry.py)

mkdir -p logs benchmark/checkpoints

for model in "${models[@]}"; do
    slug=$(echo "$model" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')  # match helpers.sanitize_slug
    qa_adapter="benchmark/checkpoints/${slug}/qa_adapter"
    reports_adapter="benchmark/checkpoints/${slug}/reports_adapter"
    qa_reports_adapter="benchmark/checkpoints/${slug}/qa_reports_adapter"

    echo "=== $model ==="

    case "$STAGE" in
        qa)
            if [[ -d "$qa_adapter" ]]; then
                echo "  skip: $qa_adapter exists"
            else
                echo "  submit qa (VQA grounding)"
                jid=$(sbatch --parsable ./run_train.sh --model "$model" --stage qa)
                echo "    job id: $jid"
            fi
            ;;
        reports)
            if [[ -d "$reports_adapter" ]]; then
                echo "  skip: $reports_adapter exists"
            else
                echo "  submit reports (reports on raw base)"
                jid=$(sbatch --parsable ./run_train.sh --model "$model" --stage reports)
                echo "    job id: $jid"
            fi
            ;;
        qa+reports)
            if [[ ! -d "$qa_adapter" ]]; then
                echo "  skip: prerequisite $qa_adapter missing — run '--stage qa' first"
            elif [[ -d "$qa_reports_adapter" ]]; then
                echo "  skip: $qa_reports_adapter exists"
            else
                echo "  submit qa+reports (continue qa_adapter on reports)"
                jid=$(sbatch --parsable ./run_train.sh --model "$model" --stage qa+reports)
                echo "    job id: $jid"
            fi
            ;;
    esac

    echo ""
done

echo "All '$STAGE' jobs submitted."
