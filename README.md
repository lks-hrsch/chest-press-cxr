# Chest-PRESS CXR

Patient-conditioned chest X-ray report generation: LoRA fine-tuning of general-purpose and medical LVLMs, compared on MIMIC-CXR and MIMIC-IV.

Repository: https://github.com/lks-hrsch/chest-press-cxr
Report: [docs/01-report-for-university-project.pdf](docs/01-report-for-university-project.pdf)

## project

**Chest-PRESS** — Patient-Conditioned Chest X-Ray Report Generation: comparing LoRA fine-tuned general-purpose vs. medical-domain LVLMs. TU Dresden Komplexpraktikum, author Lukas Hirsch.

**Research questions:**
1. Does LoRA fine-tuning of general-purpose LVLMs match or beat fine-tuning medical-domain LVLMs for CXR report generation?
2. How does rich patient clinical context (priors, indication, labs, microbiology) influence generated reports?
3. How do LVLMs differ in strengths/weaknesses across CXR subdomains (per-CheXpert-label, priors vs. no-priors, single vs. multi-view)?

Output contract: **FINDINGS + IMPRESSION only**, in that order.

## setup

``` bash
git clone https://github.com/lks-hrsch/chest-press-cxr.git
cd chest-press-cxr
uv sync
```

Set your own Hugging Face token in `src/settings.py` (`HF_ACCESS_TOKEN`) — required for gated models (Gemma, MedGemma). Get one at https://huggingface.co/settings/tokens. Never commit a real token; the file ships with a `CHANGEME` placeholder.

## data

### on local machine

``` bash
❯ ls -la /mnt/mars/datasets/physionet.org/files                    
total 0
drwxr-xr-x 2 lkshrsch users 0 Apr 22 15:47 .
drwxr-xr-x 2 lkshrsch users 0 Apr 22 13:52 ..
drwxr-xr-x 2 lkshrsch users 0 Apr 22 16:20 mimic-cxr
drwxr-xr-x 2 lkshrsch users 0 Apr 22 15:47 mimic-cxr-jpg
drwxr-xr-x 2 lkshrsch users 0 Apr 22 13:52 mimic-ext-cxr-qba
drwxr-xr-x 2 lkshrsch users 0 Apr 22 15:35 mimiciv
```

### on HPC

directory for custom jupyter kernel: `/data/horse/ws/luhi161d-custom-jupyter-kernels/chest-press-cxr-kernel`
directory for downloaded datasets: `/data/horse/ws/luhi161d-chest-press-cxr/datasets`

``` bash
[luhi161d@c155.capella university]$ ls -la /data/horse/ws/luhi161d-chest-press-cxr/datasets/physionet.org/files/
total 32
drwxr-xr-x. 8 luhi161d         203518 4096 May 27 20:41 .
drwxr-xr-x. 3 luhi161d         203518 4096 Oct 21  2025 ..
drwxr-xr-x. 3 luhi161d         203518 4096 Oct 22  2025 cxr-pro
drwxr-xr-x. 3 luhi161d         203518 4096 Nov  6  2025 mimic-cxr
drwxr-xr-x. 3 luhi161d         203518 4096 Oct 23  2025 mimic-cxr-jpg
drwxr-xr-x. 3 luhi161d p_scads_lv_llm 4096 May 27 20:41 mimic-ext-cxr-qba
drwxr-xr-x. 3 luhi161d         203518 4096 Oct 21  2025 mimiciv
drwxr-xr-x. 3 luhi161d         203518 4096 Oct 22  2025 vindr-cxr
```

> Note: the `luhi161d-chest-press-cxr` HPC workspace still exists under its old name (`luhi161d-medical-image-text-retrieval`) unless renamed on Capella to match.

### datasets used

- [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/)
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [MIMIC-EXT-CXR-QBA](https://physionet.org/content/mimic-ext-cxr-qba/1.0.0/)

All require PhysioNet credentialed access — not redistributed in this repo.

``` bash
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimiciv/2.2/hosp/patients.csv.gz
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimiciv/2.2/hosp/labevents.csv.gz
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimiciv/2.2/hosp/d_labitems.csv.gz
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimiciv/2.2/hosp/microbiologyevents.csv.gz
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimiciv/2.2/hosp/omr.csv.gz
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-negbio.csv.gz
wget -r -N -c -np --user lkshrsch --ask-password https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip
```

``` bash
# copy just the CSV files to a faster storage
dtrsync -av --progress --prune-empty-dirs \\
    --exclude='mimic-cxr/2.1.0/files/' --exclude='mimic-cxr-jpg/2.1.0/files/' --exclude='*' \\
    --include='*/' --include='*.csv' \\
    /data/horse/ws/luhi161d-chest-press-cxr/datasets/physionet.org/files/ \\
    /data/cat/ws/luhi161d-chest-press-cxr-fast/datasets/physionet.org/files/
```

## project structure

```
.
├── src/
│   ├── inference.py          # entry point: batch-generate reports via vLLM
│   ├── evaluate.py           # entry point: score inference parquets
│   ├── evaluate_stages.py    # entry point: score LoRA training-stage checkpoints
│   ├── evaluate_ablation.py  # entry point: score context-ablation runs
│   ├── merge_adapter.py      # merge a trained LoRA adapter into base weights
│   ├── model_registry.py     # single source of truth for HF model IDs
│   ├── settings.py           # HF token, random seed
│   ├── helpers.py            # sanitize_slug(), gpu_label()
│   ├── dataset/
│   │   ├── constants.py      # hard-coded data paths, OUTPUT_LABELS contract
│   │   ├── split_report.py   # shared report-section parser
│   │   ├── generate_patient_report_with_lables.py    # pipeline step 1
│   │   ├── generate_chest_press_cxr.py                # pipeline step 2 (vLLM, multi-GPU)
│   │   ├── generate_chest_press_cxr_splits.py          # pipeline step 3
│   │   └── context_ablation.py                        # context-ablation variant generator
│   ├── train/
│   │   ├── common.py         # shared LoRA/SFT training setup
│   │   ├── train_reports.py  # Stage-B: report-generation LoRA fine-tuning
│   │   ├── train_qba.py      # Stage-A: VQA grounding fine-tuning
│   │   ├── data_reports.py   # Stage-B dataset loading
│   │   └── data_qba.py       # Stage-A dataset loading
│   └── eval/
│       ├── sections.py         # FINDINGS/IMPRESSION extraction
│       ├── text_metrics.py     # BLEU, ROUGE-L, BERTScore
│       ├── clinical_metrics.py # RadGraph F1
│       ├── category_score.py   # F1-SRR-BERT / per-CheXpert-category scoring
│       ├── ssr_bert.py         # SRR-BERT scorer
│       └── optional_metrics.py # METEOR and other optional metrics
├── data/                                                # not tracked — populate from PhysioNet (see above)
│   ├── chest_press_cxr.parquet            # full dataset
│   ├── chest_press_cxr_{train,valid,test}.parquet
│   ├── patient_report_with_labels.parquet # pipeline step 1 intermediate
│   └── inference_<model-slug>.parquet     # inference output, one per model
├── benchmark/                     # not tracked — created by run_all_eval.sh
│   ├── results/<model>_<ts>/summary.csv
│   └── SUMMARY.md
├── run_inference.sh      # SLURM worker: run inference for one model
├── run_all_inference.sh  # launcher: sbatch one inference job per model
├── run_eval.sh           # SLURM worker: evaluate one model
├── run_all_eval.sh       # launcher: sbatch one eval job per model
├── run_train.sh          # SLURM worker: LoRA fine-tuning for one model
├── run_all_train.sh      # launcher: sbatch one training job per model
├── run_merge.sh          # SLURM worker: merge a LoRA adapter into base weights
├── run_create_datasets.sh  # SLURM: regenerate full dataset (multi-GPU, vLLM)
├── accelerate_ddp.yaml    # accelerate config for multi-GPU LoRA training
└── pyproject.toml         # uv project
```

## connect to compute cluster

Capella
156 nodes, each with
4 x NVIDIA H100-SXM5 Tensor Core-GPUs (94 GB HBM2e)
2 x AMD EPYC CPU 9334 (32 cores) @ 2.7 GHz, Multithreading disabled
768 GB RAM (12 x 32 GB DDR5-4800 MT/s per socket)

``` bash
srun --pty --nodes=1 --cpus-per-task=14 --mem-per-cpu=12288 --gres=gpu:1 --licenses horse --time=04:00:00 --account=p_bird04 bash -l
srun --pty --nodes=1 --cpus-per-task=28 --mem-per-cpu=12288 --gres=gpu:2 --licenses horse --time=04:00:00 --account=p_bird04 bash -l
srun --pty --nodes=1 --cpus-per-task=56 --mem-per-cpu=12288 --gres=gpu:4 --licenses horse --time=04:00:00 --account=p_bird04 bash -l
srun --pty --nodes=1 --cpus-per-task=2 --mem-per-cpu=2048 --gres=gpu:1 --licenses horse --partition=capella-interactive --time=1:00:00 bash -l
```

## load python tools (HPC)

environment variables

``` bash
source .hpc.env
```

create kernel

``` bash
module load release/25.06  GCCcore/13.3.0 Python/3.12.3
cd /data/horse/ws/luhi161d-custom-jupyter-kernels

python3 -m venv --system-site-packages chest-press-cxr-kernel
source chest-press-cxr-kernel/bin/activate
pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name chest-press-cxr-kernel --display-name="chest-press-cxr-kernel"
```

use kernel

``` bash  
source /data/horse/ws/luhi161d-custom-jupyter-kernels/chest-press-cxr-kernel/bin/activate
python -m jupyter server --no-browser --port=8888
```

## running the pipeline

``` bash
# 1. Generate the dataset (multi-GPU vLLM step — run on Capella)
sbatch run_create_datasets.sh
#   or step-by-step:
python -m src.dataset.generate_patient_report_with_lables
python -m src.dataset.generate_chest_press_cxr
python -m src.dataset.generate_chest_press_cxr_splits

# 2. Fine-tune adapters (optional — skip to use base models)
./run_all_train.sh                            # HPC: one sbatch job per model
sbatch ./run_train.sh --model <hf-id>         # HPC: single model
sbatch ./run_merge.sh --model <hf-id>         # merge a trained LoRA adapter

# 3. Run inference  →  data/inference_<model-slug>.parquet
./run_all_inference.sh                        # HPC: one sbatch job per model
sbatch ./run_inference.sh --model <hf-id>     # HPC: single model
python -m src.inference --model <hf-id>       # local, no SLURM

# 4. Evaluate  →  benchmark/results/<model>_<ts>/summary.csv
./run_all_eval.sh                             # HPC: one sbatch job per model
sbatch ./run_eval.sh --model <hf-id>          # HPC: single model
python -m src.evaluate --model <hf-id>        # local, no SLURM
```

Model IDs come from `src/model_registry.py` (single source of truth). SLURM logs: `logs/%x-%j.{out,err}`.

## license

- **Code** — MIT, see `LICENSE`.
- **Data** — MIMIC-CXR, MIMIC-CXR-JPG, MIMIC-IV, MIMIC-EXT-CXR-QBA require PhysioNet credentialed access and are not redistributable. Nothing under `data/` is tracked in this repo; the `chest_press_cxr_*.parquet` files carry MIMIC report text and are restricted derivatives, not original data — obtain access via PhysioNet.
- **Model weights** — referenced by Hugging Face ID only, never vendored. Gemma and MedGemma ship under their own model licenses, not an OSI license — review each model's terms on Hugging Face before use.
