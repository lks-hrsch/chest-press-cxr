import os
from functools import lru_cache
import polars
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .constants import (
    INPUT_LABELS,
    OUTPUT_LABELS,
    REPORTS_BASE_PATH,
    PATIENTS_CSV_PATH,
    REPORT_METADATA_CSV_PATH,
    CHEXPERT_LABELS_CSV_PATH,
    NEGBIO_LABELS_CSV_PATH,
)

from .helpers import format_datetime
from ..sink_monitor import SinkMonitor

MAX_WORKERS = getattr(os, "process_cpu_count", os.cpu_count)() or 1


@lru_cache(maxsize=1024)
def get_presence_of_labels(subject_id: str, study_id: str):
    with open(f"{REPORTS_BASE_PATH}/p{subject_id[:2]}/p{subject_id}/s{study_id}.txt", "r") as report:
        report_text = report.read()
        presence = {}
        for label in INPUT_LABELS + OUTPUT_LABELS:
            presence[label] = label in report_text
        return presence


def generate_patient_report_with_labels() -> polars.LazyFrame:
    print("Loading CSV files with Polars...")
    patients_df = polars.scan_csv(PATIENTS_CSV_PATH)
    reports_df = polars.scan_csv(REPORT_METADATA_CSV_PATH)
    chexpert_labels_df = polars.scan_csv(CHEXPERT_LABELS_CSV_PATH)
    negbio_labels_df = polars.scan_csv(NEGBIO_LABELS_CSV_PATH)
    print("CSV files loaded.")

    grouped_patient_df = (
        patients_df.join(reports_df, on="subject_id")
        .group_by("subject_id", "study_id", "anchor_age", "gender", "StudyDate", "StudyTime")
        .agg(
            polars.col("dicom_id"),
            polars.col("ProcedureCodeSequence_CodeMeaning"),
            polars.col("ViewCodeSequence_CodeMeaning"),
        )
        .with_columns(
            (
                polars.struct(["StudyDate", "StudyTime"]).map_elements(
                    lambda x: format_datetime(x["StudyDate"], x["StudyTime"]), return_dtype=polars.Utf8
                )
            ).alias("StudyDateTime")
        )
        .drop(["StudyDate", "StudyTime"])
    )

    ####
    print("Comparing CheXpert and NegBio labels for agreement...")

    comparison_df = chexpert_labels_df.join(negbio_labels_df, on=["subject_id", "study_id"], suffix="_negbio")
    columns = chexpert_labels_df.collect_schema().names()[2:]
    exprs = [
        (
            (polars.col(column) == polars.col(f"{column}_negbio"))
            | (polars.col(column).is_null() & polars.col(f"{column}_negbio").is_null())
        )
        for column in columns
    ]

    agreement_df = comparison_df.filter(polars.all_horizontal(exprs)).select(
        [polars.col("subject_id"), polars.col("study_id")] + [polars.col(column) for column in columns]
    )

    patient_report_with_labels_df = grouped_patient_df.join(agreement_df, on=["subject_id", "study_id"])

    ####

    pairs = patient_report_with_labels_df.select(["subject_id", "study_id"]).collect().iter_rows()

    def build_row(subject_id: int, study_id: int) -> dict:
        presence = get_presence_of_labels(str(subject_id), str(study_id))
        return {
            "subject_id": subject_id,
            "study_id": study_id,
            **{label: presence[label] for label in INPUT_LABELS + OUTPUT_LABELS},
        }

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(build_row, subject_id, study_id) for subject_id, study_id in pairs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking label presence"):
            rows.append(future.result())

    report_labels_presence_df = polars.DataFrame(rows).lazy()

    # filter the dataset we just want to keep studies where at least INDICATION, FINDINGS and IMPRESSION is present
    report_labels_presence_df = report_labels_presence_df.filter(
        polars.col("INDICATION") & polars.col("FINDINGS") & polars.col("IMPRESSION")
    )

    # cast subject_id and study_id to i64 to match patient_report_with_labels_df
    report_labels_presence_df = report_labels_presence_df.with_columns(
        [polars.col("subject_id").cast(polars.Int64), polars.col("study_id").cast(polars.Int64)]
    )

    # now join with patient_report_with_labels_df to filter it
    patient_report_with_labels_df = patient_report_with_labels_df.join(
        report_labels_presence_df, on=["subject_id", "study_id"], how="inner"
    )

    ####
    print("Building report_path and xray_image_paths columns...")

    patient_report_with_labels_df = patient_report_with_labels_df.with_columns(
        [
            # Build report_path using native Polars string operations (fully vectorized!)
            # No REPORTS_BASE_PATH included need to be added later
            (
                polars.lit("p")
                + polars.col("subject_id").cast(polars.Utf8).str.slice(0, 2)
                + polars.lit("/p")
                + polars.col("subject_id").cast(polars.Utf8)
                + polars.lit("/s")
                + polars.col("study_id").cast(polars.Utf8)
                + polars.lit(".txt")
            ).alias("report_path"),
        ]
    )

    # Build xray image paths row-wise because list.eval cannot access named columns.
    patient_report_with_labels_df = patient_report_with_labels_df.with_columns(
        polars.struct(["subject_id", "study_id", "dicom_id"])
        .map_elements(
            lambda row: [
                f"p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{dicom_id}.jpg"
                for dicom_id in row["dicom_id"]
            ],
            return_dtype=polars.List(polars.Utf8),
        )
        .alias("xray_image_paths")
    )

    print(patient_report_with_labels_df.collect_schema().names())
    print(patient_report_with_labels_df.head().collect())

    return patient_report_with_labels_df


if __name__ == "__main__":
    SAVE_PATH = "data/patient_report_with_labels.parquet"

    print("Start generating patient_report_with_labels_df...")
    patient_report_with_labels_df = generate_patient_report_with_labels()

    print("Start saving patient_report_with_labels_df to Parquet...")
    with SinkMonitor(SAVE_PATH):
        patient_report_with_labels_df.sink_parquet(SAVE_PATH)
