#!/bin/bash
#SBATCH --job-name=chest-press-cxr-benchmark-create-datasets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-cpu=12288
#SBATCH --gres=gpu:4
#SBATCH --licenses horse
#SBATCH --mail-user=lukas.hirsch1@mailbox.tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --time=07:00:00
#SBATCH --account=p_scads_gnn_electricity
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -e

module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3

source .hpc.env
source .env

nvidia-smi

export UV_PROJECT_ENVIRONMENT=/tmp/luhi161d-chest-press-cxr-venv
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT
date
uv sync --frozen
date
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

date
# already done
python -m src.dataset.generate_patient_report_with_lables 
date
python -m src.dataset.generate_chest_press_cxr
date
python -m src.dataset.generate_chest_press_cxr_splits
date
