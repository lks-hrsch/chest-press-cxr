import os
from bisect import bisect_right

import polars
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from ..sink_monitor import SinkMonitor
from .constants import (
    LAB_D_EVENTS_CSV_PATH,
    LAB_EVENTS_CSV_PATH,
    MICROBIOLOGY_EVENTS_CSV_PATH,
    OUTPUT_LABELS,
)
from .generate_patient_report_with_lables import generate_patient_report_with_labels
from .split_report import split_report


NUM_LAB_ITEMS = 100
NUM_GPUS = torch.cuda.device_count()
# DEBUG_ROW_LIMIT = 5120

PATH = "data/patient_report_with_labels.parquet"
SAVE_PATH = "data/chest_press_cxr.parquet"


def build_events_index(df: polars.DataFrame, drop_cols: set[str]) -> dict[int, dict[str, list]]:
    index: dict[int, dict[str, list]] = {}
    for row in df.sort(["subject_id", "storetime"]).iter_rows(named=True):
        subject_id = row["subject_id"]
        subject_bucket = index.setdefault(subject_id, {"storetimes": [], "events": []})
        subject_bucket["storetimes"].append(row["storetime"])
        subject_bucket["events"].append({k: v for k, v in row.items() if v is not None and k not in drop_cols})
    return index


def get_events_from_index(index: dict[int, dict[str, list]], subject_id, study_datetime) -> list[dict]:
    subject_bucket = index.get(subject_id)
    if not subject_bucket:
        return []

    cutoff = bisect_right(subject_bucket["storetimes"], study_datetime)
    start = max(0, cutoff - NUM_LAB_ITEMS)
    return subject_bucket["events"][start:cutoff]


def create_summary_prompt(events, indication):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": 'You are a laboratory technician. You need to summarize lab findings in maximum 10 sentences so only keep relevant information. If there are no findings, say "Summary: No significant findings.". Otherwise start with "Summary: <summary>".',
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
{indication}
Summarize while keeping the INDICATION in mind the following lab findings:
{events}
""",
                },
            ],
        },
    ]


def generate_system_prompt():
    template = """
You are an expert radiologist.
You are given some clinical context about a patient and one or two chest X-Ray images from the patient in different views.
Write the IMPRESSION and FINDINGS sections of the report.
Consider the available clinical context about the patient when generating the report.
    """
    return template.strip()


def fetch_prompts_for_row(row, lab_events_index, micro_events_index):
    indication = row.get("INDICATION", "")

    res = {"lab_prompt": None, "micro_prompt": None}

    # Lab
    lab_events = get_events_from_index(lab_events_index, row["subject_id"], row["StudyDateTime"])
    if lab_events:
        res["lab_prompt"] = create_summary_prompt(lab_events, indication)

    # Micro
    micro_events = get_events_from_index(micro_events_index, row["subject_id"], row["StudyDateTime"])
    if micro_events:
        res["micro_prompt"] = create_summary_prompt(micro_events, indication)

    return res


def main():
    print(f"Detected {NUM_GPUS} GPUs for inference.")

    # check if patient_report_with_labels.parquet exists, if so load it, else generate it
    if os.path.exists(PATH):
        print(f"Loading patient_report_with_labels_df from {PATH}...")
        patient_report_with_labels_df = polars.scan_parquet(PATH).collect()
    else:
        print("Generating patient_report_with_labels_df...")
        patient_report_with_labels_df = generate_patient_report_with_labels().collect()

    # Keep the debug pipeline small so local runs stay fast.
    # patient_report_with_labels_df = patient_report_with_labels_df.head(DEBUG_ROW_LIMIT)

    # --- OPTIMIZED DATA LOADING ---

    print("Optimizing data loading...")
    # 1. Get relevant subject IDs to filter events (optimization)
    relevant_subjects = patient_report_with_labels_df.select("subject_id").unique()["subject_id"].to_list()

    labitems_df = polars.scan_csv(
        LAB_D_EVENTS_CSV_PATH,
        truncate_ragged_lines=True,
        schema_overrides={"itemid": polars.Utf8},
    )

    print("loading lab events into memory...")
    lab_events_df_all = (
        polars.scan_csv(
            LAB_EVENTS_CSV_PATH,
            truncate_ragged_lines=True,
            ignore_errors=True,
            schema_overrides={"itemid": polars.Utf8},
        )
        .filter(
            polars.col("storetime").is_not_null()
            & polars.col("valuenum").is_not_null()
            & (polars.col("ref_range_lower").is_not_null() & polars.col("ref_range_upper").is_not_null())
            & polars.col("subject_id").is_in(relevant_subjects)
        )
        .join(labitems_df, on="itemid")
        .collect()
    )

    print("loading microbiology events into memory...")
    micro_events_df_all = (
        polars.scan_csv(MICROBIOLOGY_EVENTS_CSV_PATH, truncate_ragged_lines=True, ignore_errors=True)
        .filter(polars.col("storetime").is_not_null() & polars.col("subject_id").is_in(relevant_subjects))
        .collect()
    )

    lab_events_index = build_events_index(
        lab_events_df_all,
        {"labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id", "charttime"},
    )
    micro_events_index = build_events_index(
        micro_events_df_all,
        {
            "microevent_id",
            "subject_id",
            "hadm_id",
            "micro_specimen_id",
            "order_provider_id",
            "chartdate",
            "charttime",
            "spec_itemid",
        },
    )

    # --- MAIN EXECUTION ---
    rows = patient_report_with_labels_df.to_dicts()

    report_sections = [
        split_report(row["report_path"]) for row in tqdm(rows, desc="Splitting reports", total=len(rows))
    ]
    for row, sections in zip(rows, report_sections):
        row.update(sections)

    report_sections_by_path = {row["report_path"]: sections for row, sections in zip(rows, report_sections)}

    print(f"Preparing prompts for {len(rows)} rows using cached event indexes...")

    prompt_results = [
        fetch_prompts_for_row(row, lab_events_index, micro_events_index)
        for row in tqdm(rows, desc="Preparing prompts", total=len(rows))
    ]

    prompts = []
    prompt_indices = []  # Stores (row_index, type) tuples

    # Initialize summaries
    lab_summaries = ["No significant findings."] * len(rows)
    micro_summaries = ["No significant findings."] * len(rows)

    # Unpack results
    for i, res in enumerate(prompt_results):
        if res["lab_prompt"]:
            prompts.append(res["lab_prompt"])
            prompt_indices.append((i, "lab"))
        if res["micro_prompt"]:
            prompts.append(res["micro_prompt"])
            prompt_indices.append((i, "micro"))

    # Run Inference
    if prompts:
        llm = None
        try:
            print("Initializing vLLM...")
            model = "google/gemma-4-26B-A4B-it"  # google/gemma-4-E2B-it for faster testing; google/gemma-4-26B-A4B-it for better performance

            llm = LLM(
                model=model,
                tensor_parallel_size=NUM_GPUS,
                quantization="fp8",
                gpu_memory_utilization=0.85,  # utilizing 85% of GPU memory
                max_model_len=24576,
                limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
                # speculative_config={"model": f"{model}-assistant", "num_speculative_tokens": 2},
                # Disabled: vLLM 0.22.0 Gemma4-MTP + TRITON_ATTN asserts uniform num_heads per
                # attention group, but the E2B target (8 heads) and assistant draft (4 heads)
                # end up in the same group → AssertionError: got {8, 4}. Re-enable once fixed.
            )

            # Set sampling parameters (Greedy decoding for summaries)
            sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

            print(f"Running vLLM inference on {len(prompts)} prompts...")
            outputs = llm.chat(prompts, sampling_params)

            # 5. Process Results
            for i, output in enumerate(outputs):
                summary = output.outputs[0].text.strip()

                # Simple cleanup if the model repeats the prefix
                if "Summary:" in summary:
                    summary = summary.split("Summary:")[-1].strip()

                row_idx, type_ = prompt_indices[i]
                if type_ == "lab":
                    lab_summaries[row_idx] = summary
                else:
                    micro_summaries[row_idx] = summary
        finally:
            if llm is not None:
                engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
                shutdown = getattr(engine_core, "shutdown", None)
                if callable(shutdown):
                    shutdown()

    # Assemble Final Dataframe
    print("Assembling final dataset...")
    system_prompts = []
    input_texts = []
    expected_output_findings_texts = []
    expected_output_impressions_texts = []

    for i, row in tqdm(enumerate(rows), total=len(rows), desc="Building Rows"):
        sections = report_sections_by_path[row["report_path"]]
        study_datetime = row["StudyDateTime"]
        subject_id = row["subject_id"]

        # Past clinical data
        subject_id_past_events_df = (
            patient_report_with_labels_df.filter(
                (polars.col("subject_id") == subject_id) & (polars.col("StudyDateTime") < study_datetime)
            )
            .sort("StudyDateTime")
            .tail(5)
        )

        past_reports = ""
        if not subject_id_past_events_df.is_empty():
            past_reports += "\n## Clinical Context (last report is most recent):"
            for past_row in subject_id_past_events_df.iter_rows(named=True):
                past_reports += f"\n### {past_row['StudyDateTime']}"
                for key, value in report_sections_by_path[past_row["report_path"]].items():
                    if value.strip() != "" and key in ["INDICATION"] + OUTPUT_LABELS:
                        past_reports += f"\n{value}"
            past_reports = "\n" + past_reports.strip() + "\n"

        template = f"""
## Patient Information
GENDER: {row["gender"]}
AGE: {row["anchor_age"]}
LAST_LAB_FINDINGS: {lab_summaries[i]}
LAST_MICROBIOLOGY_FINDINGS: {micro_summaries[i]}
{past_reports}
## Current Study ({study_datetime}):
    """.strip()

        template += "\nClinical Context (last report is most recent):"
        for key, value in sections.items():
            if value.strip() != "" and key in [
                "HISTORY",
                "COMPARISON",
                "COMPARISONS",
                "TECHNIQUE",
                "EXAMINATION",
                "INDICATION",
            ]:
                template += f"\n{value}"

        findings = sections.get("FINDINGS", "").strip()
        impression = sections.get("IMPRESSION", "").strip()

        input_texts.append(template.strip())
        system_prompts.append(generate_system_prompt())
        expected_output_findings_texts.append(findings)
        expected_output_impressions_texts.append(impression)

    # Add columns to the collected dataframe
    patient_report_with_labels_df = patient_report_with_labels_df.with_columns(
        [
            polars.Series("system_prompt", system_prompts),
            polars.Series("input_text", input_texts),
            polars.Series("expected_output_findings_text", expected_output_findings_texts),
            polars.Series("expected_output_impressions_text", expected_output_impressions_texts),
        ]
    )

    # remove all rows with empty input_text or expected output sections
    patient_report_with_labels_df = patient_report_with_labels_df.filter(
        (polars.col("input_text").str.len_chars() > 1)
        & (polars.col("expected_output_findings_text").str.len_chars() > 1)
        & (polars.col("expected_output_impressions_text").str.len_chars() > 1)
    )

    print(patient_report_with_labels_df.collect_schema().names())
    # print(df_collected.head().collect())

    # save final dataframe to parquet (supports nested data like lists)
    with SinkMonitor(SAVE_PATH):
        patient_report_with_labels_df.write_parquet(SAVE_PATH)


if __name__ == "__main__":
    main()
