#!/bin/bash
#SBATCH --job-name=chest-press-cxr-merge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=12288
#SBATCH --gres=gpu:1
#SBATCH --licenses horse
#SBATCH --mail-user=lukas.hirsch1@mailbox.tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
#SBATCH --account=p_scads_gnn_electricity
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail

module purge
module load release/25.06  GCCcore/13.3.0 Python/3.12.3

source .hpc.env
source .env

# --- argument parsing ---
MODEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --variant|--adapter-path|--output-dir)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --force)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <hf-model-id> --variant {qa,reports,qa_reports} [--adapter-path <dir>] [--output-dir <dir>] [--force]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    exit 1
fi

# --- setup venv (same env as inference: peft/transformers are present) ---
MODEL_SAFE="${MODEL//\//_}"
export UV_PROJECT_ENVIRONMENT="/tmp/luhi161d-chest-press-cxr-${MODEL_SAFE}-venv"
echo "Using virtual environment at $UV_PROJECT_ENVIRONMENT"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT
date
uv sync --frozen
date
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# --- merge ---
date
python -m src.merge_adapter --model "$MODEL" "${EXTRA_ARGS[@]}"
date
