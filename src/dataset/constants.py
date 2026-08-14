# DATA_BASE_PATH = "data/physionet.org/files"  # on local machine
# DATA_BASE_PATH = "/data/horse/ws/luhi161d-chest-press-cxr/datasets/physionet.org/files"  # on HPC
# DATA_BASE_PATH_FAST = (
#     "/data/cat/ws/luhi161d-chest-press-cxr-fast/datasets/physionet.org/files"  # on HPC with fast storage
# )


DATA_BASE_PATH = "/mnt/mars/datasets/physionet.org/files/"  # on workstation
DATA_BASE_PATH_FAST = DATA_BASE_PATH

PATIENTS_CSV_PATH = f"{DATA_BASE_PATH_FAST}/mimiciv/3.1/hosp/patients.csv"
REPORT_METADATA_CSV_PATH = f"{DATA_BASE_PATH_FAST}/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv"
CHEXPERT_LABELS_CSV_PATH = f"{DATA_BASE_PATH_FAST}/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv"
NEGBIO_LABELS_CSV_PATH = f"{DATA_BASE_PATH_FAST}/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-negbio.csv"
MICROBIOLOGY_EVENTS_CSV_PATH = f"{DATA_BASE_PATH_FAST}/mimiciv/3.1/hosp/microbiologyevents.csv"
LAB_EVENTS_CSV_PATH = f"{DATA_BASE_PATH_FAST}/mimiciv/3.1/hosp/labevents.csv"
LAB_D_EVENTS_CSV_PATH = f"{DATA_BASE_PATH_FAST}/mimiciv/3.1/hosp/d_labitems.csv"

REPORTS_BASE_PATH = f"{DATA_BASE_PATH}/mimic-cxr/2.1.0/files/"
XRAY_BASE_PATH = f"{DATA_BASE_PATH}/mimic-cxr-jpg/2.1.0/files/"

# MIMIC-Ext-CXR-QBA — used for Stage-A VQA grounding fine-tuning.
# Edit per machine alongside DATA_BASE_PATH above.
QBA_BASE_PATH = f"{DATA_BASE_PATH}/mimic-ext-cxr-qba/1.0.0/"
# A_frontal App_q1M subset: a random 1M-pair sample of the A++ (App) quality tier. Same tier as
# the full App set, but small enough that the heavy backbones (Lingshu/LLaVA) finish a 1-epoch
# Stage-A run within the 24h SLURM wall. Swap back to `App/` (full ~1.46M) if the wall allows.
QBA_EXPORTS_APP = f"{QBA_BASE_PATH}exports/A_frontal/metadata/App_q1M/"
# Shared per-study QA JSON files (all A_frontal quality tiers):
QBA_QA_DIR = f"{QBA_BASE_PATH}exports/A_frontal/qa/"

INPUT_LABELS = [
    "HISTORY",
    "INDICATION",
    "EXAMINATION",
    "COMPARISON",
    "COMPARISONS",
    "TECHNIQUE",
    "NOTIFICATION",
]

OUTPUT_LABELS = ["FINDINGS", "IMPRESSION"]

LABELS = INPUT_LABELS + OUTPUT_LABELS
