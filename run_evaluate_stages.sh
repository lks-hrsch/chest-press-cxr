#!/bin/bash
#SBATCH --job-name=chest-press-cxr-evaluate-stages
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=12288
#SBATCH --gres=gpu:1
#SBATCH --licenses horse
#SBATCH --mail-user=lukas.hirsch1@mailbox.tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00   # adjust per model if needed; BERTScore + RadGraph + SRR-BERT require GPU
#SBATCH --account=p_scads_gnn_electricity
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail

module purge
module load release/25.06  GCCcore/13.3.0 Python/3.12.3

source .hpc.env
source .env

nvidia-smi

# --- argument parsing ---
MODEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir|--sample-size|--sample-seed)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --debug-dev-subset)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <hf-model-id> [--output-dir <dir>] [--sample-size <n>] [--sample-seed <n>] [--debug-dev-subset]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    exit 1
fi

# --- setup venv ---
export UV_PROJECT_ENVIRONMENT=/tmp/luhi161d-chest-press-cxr-venv
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT
date
uv sync --frozen
date
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# --- run training-stage evaluation ---
date
python -m src.evaluate_stages --model "$MODEL" "${EXTRA_ARGS[@]}"
date
