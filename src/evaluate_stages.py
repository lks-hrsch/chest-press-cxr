"""Per-row scoring for the training-stage comparison grid (RQ1 x QBA-curriculum benefit).

Computes ``src/eval/per_example.py``'s per-row metrics for one ``--model`` across all 4 training-stage
variants -- ``base`` (untrained checkpoint), ``qa`` (QBA-only LoRA), ``reports`` (report-only LoRA),
``qa_reports`` (QBA-then-reports curriculum LoRA) -- see ``src/adapters.py``'s ``ADAPTER_VARIANTS`` and
``src/inference.py --variant``. This is a different axis from ``src/evaluate_ablation.py`` (clinical
-context tiers): every row here is scored at the single default ``rich_clinical_context`` tier (the
unsuffixed inference files), so there is no tier ablation in this grid. Writes one long-format parquet
per model to ``benchmark/results_training_stages/<slug>.parquet``. Notebooks read that output only --
no metrics are computed in the notebooks themselves (mirrors ``src/evaluate_ablation.py`` /
``notebooks/evaluation-context-ablation-*.ipynb``'s compute/visualize split).

Each row also carries ``anchor_age``, ``gender``, ``model_family`` (``"general"`` or ``"medical"`` --
``src/model_registry.py`` has no domain field, so this mirrors the classification documented in
``CLAUDE.md``: 3 general-purpose backbones vs. 4 medical-domain backbones), and the 14 CheXpert label
columns from the test split (unaffected by which checkpoint generated the prediction).

Usage::

    python -m src.evaluate_stages --model google/gemma-3-4b-it
    python -m src.evaluate_stages --model google/gemma-3-4b-it --sample-size 250 --sample-seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

from .eval import compute_clinical_metrics_per_row, compute_text_metrics_per_row, extract_report_sections
from .evaluate import (
    TASK_CHOICES,
    _pred_ref_for_task,
    collect_reference_sections,
    inference_path_for,
    should_evaluate_reference,
)
from .evaluate_ablation import _dedup_keys
from .helpers import sanitize_slug
from .model_registry import MODELS


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

DATASET_FILE = Path("data/chest_press_cxr_test.parquet")
DEFAULT_OUTPUT_DIR = Path("benchmark/results_training_stages")
STAGES = ("base", "qa", "reports", "qa_reports")
METRIC_COLUMNS = ("bleu", "rougeL", "bertscore_f1", "radgraph_f1", "f1_ssr_bert", "category_score")

CHEXPERT_LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# src/model_registry.py carries no domain/category field -- this mirrors the 3-general/4-medical
# split documented in CLAUDE.md's "Architecture" -> "Shared components" section (the src/model_registry.py bullet).
GENERAL_MODELS = frozenset(
    {
        "google/gemma-3-4b-it",
        "google/gemma-4-E2B-it",
        "llava-hf/llava-v1.6-mistral-7b-hf",
    }
)
MEDICAL_MODELS = frozenset(
    {
        "google/medgemma-4b-it",
        "google/medgemma-1.5-4b-it",
        "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        "lingshu-medical-mllm/Lingshu-I-8B",
    }
)


def model_family(model_name: str) -> str:
    """``"general"`` or ``"medical"`` -- see the GENERAL_MODELS/MEDICAL_MODELS module docstring note."""
    if model_name in GENERAL_MODELS:
        return "general"
    if model_name in MEDICAL_MODELS:
        return "medical"
    raise ValueError(f"Unknown model family for {model_name!r} -- add it to GENERAL_MODELS or MEDICAL_MODELS")


def stage_path(model_name: str, stage: str) -> Path:
    """The default (rich_clinical_context, unsuffixed) inference parquet for a (model, stage).

    ``inference_path_for`` already normalises ``variant="base"`` to the unsuffixed path and
    ``"qa"``/``"reports"``/``"qa_reports"`` to their ``__<variant>`` suffix -- no extra logic needed.
    """
    return inference_path_for(model_name, stage)


def matched_keys(model_name: str) -> tuple[pl.DataFrame | None, dict, dict[str, Path]]:
    """4-way key intersection across the 4 stages, or ``(None, ...)`` if any stage file is missing.

    Every stage must be scored on the exact same matched rows for a fair comparison -- see
    ``evaluate_ablation.matched_keys`` for the identical rationale on the tier axis.
    """
    paths = {stage: stage_path(model_name, stage) for stage in STAGES}
    missing = [stage for stage, path in paths.items() if not path.exists()]
    if missing:
        return None, {"missing_stages": missing}, paths

    common = None
    for stage in STAGES:
        keys = _dedup_keys(paths[stage]).select("subject_id", "study_id")
        common = keys if common is None else common.join(keys, on=["subject_id", "study_id"], how="inner")
    return common.collect(), {"missing_stages": []}, paths


def score_stages(
    model_name: str,
    keys: pl.DataFrame,
    paths: dict[str, Path],
    gt_by_key: dict[tuple[int, int], dict],
) -> list[dict]:
    """Score all 4 stages x 3 tasks for one model on the given matched ``keys``."""
    rows: list[dict] = []
    family = model_family(model_name)
    for stage in STAGES:
        infer_rows = _dedup_keys(paths[stage]).join(keys.lazy(), on=["subject_id", "study_id"], how="inner").collect()

        # Build (prediction, reference) pairs per task, reusing the exact scoring primitives
        # score_predictions() uses (src/evaluate.py) so the pred/ref text is identical to the
        # aggregate pipeline -- only the metric reduction differs (per-row vs corpus/mean).
        per_task: dict[str, dict[str, list]] = {
            task: {"subject_id": [], "study_id": [], "pred": [], "ref": []} for task in TASK_CHOICES
        }
        for row in infer_rows.iter_rows(named=True):
            key = (row["subject_id"], row["study_id"])
            gt_row = gt_by_key.get(key)
            if gt_row is None:
                continue
            prediction = extract_report_sections(row["generated_output_text"] or "")
            reference = collect_reference_sections(gt_row)
            for task in TASK_CHOICES:
                pred, ref = _pred_ref_for_task(task, prediction, reference)
                bucket = per_task[task]
                bucket["subject_id"].append(row["subject_id"])
                bucket["study_id"].append(row["study_id"])
                bucket["pred"].append(pred)
                bucket["ref"].append(ref)

        for task in TASK_CHOICES:
            bucket = per_task[task]
            n = len(bucket["pred"])
            scored_mask = [should_evaluate_reference(ref) for ref in bucket["ref"]]
            scored_idx = [i for i, is_scored in enumerate(scored_mask) if is_scored]
            scored_preds = [bucket["pred"][i] for i in scored_idx]
            scored_refs = [bucket["ref"][i] for i in scored_idx]

            per_row_metrics = {
                **compute_text_metrics_per_row(scored_preds, scored_refs),
                **compute_clinical_metrics_per_row(scored_preds, scored_refs, task=task),
            }
            full_metrics = {col: [None] * n for col in METRIC_COLUMNS}
            for out_i, src_i in enumerate(scored_idx):
                for col in METRIC_COLUMNS:
                    full_metrics[col][src_i] = per_row_metrics[col][out_i]

            for i in range(n):
                key = (bucket["subject_id"][i], bucket["study_id"][i])
                gt_row = gt_by_key[key]
                rows.append(
                    {
                        "model": model_name,
                        "model_family": family,
                        "stage": stage,
                        "task": task,
                        "subject_id": key[0],
                        "study_id": key[1],
                        "scored": scored_mask[i],
                        **{col: full_metrics[col][i] for col in METRIC_COLUMNS},
                        "anchor_age": gt_row.get("anchor_age"),
                        "gender": gt_row.get("gender"),
                        **{col: gt_row.get(col) for col in CHEXPERT_LABEL_COLUMNS},
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the base/qa/reports/qa_reports training-stage grid for one model"
    )
    parser.add_argument("--model", type=str, required=True, choices=[m["name"] for m in MODELS], help="Model to score")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the per-model parquet"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Cap matched rows to this many (deterministic sample). Default: no cap (full matched set).",
    )
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for --sample-size sampling")
    parser.add_argument(
        "--debug-dev-subset", action="store_true", help='Filter to subject_id starting with "10" for fast debugging'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_slug(args.model)

    gt = pl.scan_parquet(DATASET_FILE)
    if args.debug_dev_subset:
        gt = gt.filter(pl.col("subject_id").cast(pl.Utf8).str.starts_with("10"))

    keys, info, paths = matched_keys(args.model)
    if keys is None:
        LOGGER.warning("skip model=%s: missing stages %s", args.model, info["missing_stages"])
        all_rows: list[dict] = []
        coverage = {"available": False, "missing_stages": info["missing_stages"], "matched_rows": None}
    else:
        if args.sample_size is not None and keys.height > args.sample_size:
            keys = keys.sample(n=args.sample_size, seed=args.sample_seed)

        gt_matched = gt.join(keys.lazy(), on=["subject_id", "study_id"], how="inner").collect()
        gt_by_key = {(row["subject_id"], row["study_id"]): row for row in gt_matched.iter_rows(named=True)}

        LOGGER.info("model=%s matched_rows=%d", args.model, keys.height)
        all_rows = score_stages(args.model, keys, paths, gt_by_key)
        coverage = {"available": True, "missing_stages": [], "matched_rows": keys.height}

    out_path = args.output_dir / f"{slug}.parquet"
    if all_rows:
        # Explicit overrides for the metric + CheXpert label columns -- a rare label can have zero
        # non-null values across a single model's capped sample, in which case polars infers that
        # column as Null (not Float64); forcing the dtype keeps every model's output parquet
        # schema-identical so pl.concat() across models in the notebooks doesn't break. Same fix as
        # evaluate_ablation.py's final write (see that module for the full incident writeup).
        schema_overrides = {col: pl.Float64 for col in (*METRIC_COLUMNS, *CHEXPERT_LABEL_COLUMNS)}
        pl.DataFrame(all_rows, schema_overrides=schema_overrides, infer_schema_length=None).write_parquet(out_path)
        LOGGER.info("Wrote %d scored rows to %s", len(all_rows), out_path)
    else:
        LOGGER.warning("No stages available for %s -- nothing written", args.model)

    coverage_path = args.output_dir / f"{slug}_coverage.json"
    coverage_path.write_text(json.dumps({"model": args.model, **coverage}, indent=2), encoding="utf-8")
    LOGGER.info("Wrote coverage summary to %s", coverage_path)


if __name__ == "__main__":
    main()
