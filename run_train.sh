#!/bin/bash
#SBATCH --job-name=chest-press-cxr-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-cpu=12288
#SBATCH --gres=gpu:4
#SBATCH --licenses horse
#SBATCH --mail-user=lukas.hirsch1@mailbox.tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --account=p_scads_gnn_electricity
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail

# set train time to 12h for reports or 24h for qba

module purge
module load release/25.06  GCCcore/13.3.0 Python/3.12.3

source .hpc.env
source .env

nvidia-smi

# --- argument parsing ---
MODEL=""
STAGE=""           # qa | reports | qa+reports
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --init-adapter|--output-dir|--max-steps)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <hf-model-id> --stage {qa|reports|qa+reports} [extra args]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    exit 1
fi
if [[ -z "$STAGE" ]]; then
    echo "Error: --stage {qa|reports|qa+reports} is required"
    exit 1
fi

mkdir -p logs benchmark/checkpoints

# --- setup training venv (separate from inference venv) ---
MODEL_SAFE="${MODEL//\//_}"
export UV_PROJECT_ENVIRONMENT="/tmp/luhi161d-chest-press-cxr-train-${MODEL_SAFE}-venv"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT
date
uv sync --frozen
date
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# --- run ---
date
case "$STAGE" in
    qa)
        accelerate launch --config_file accelerate_ddp.yaml \
            -m src.train.train_qba --model "$MODEL" "${EXTRA_ARGS[@]}"
        ;;
    reports)
        accelerate launch --config_file accelerate_ddp.yaml \
            -m src.train.train_reports --model "$MODEL" "${EXTRA_ARGS[@]}"
        ;;
    qa+reports)
        # Continue the qa adapter on the report data (PEFT continuation, no merge).
        # The slug must match Python's sanitize_slug (non-[A-Za-z0-9._-] runs → "_", ends stripped).
        SLUG=$(echo "$MODEL" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')
        QA_ADAPTER="benchmark/checkpoints/${SLUG}/qa_adapter"
        if [[ ! -d "$QA_ADAPTER" ]]; then
            echo "Error: stage qa+reports requires the qa adapter at $QA_ADAPTER — run --stage qa first"
            exit 1
        fi
        accelerate launch --config_file accelerate_ddp.yaml \
            -m src.train.train_reports --model "$MODEL" --init-adapter "$QA_ADAPTER" "${EXTRA_ARGS[@]}"
        ;;
    *)
        echo "Error: unknown stage '$STAGE' — must be qa, reports, or qa+reports"
        exit 1
        ;;
esac
date
