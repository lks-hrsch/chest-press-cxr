from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from .dataset.split_report import split_report
from .eval import compute_clinical_metrics, compute_text_metrics, extract_report_sections, normalize_report_text
from .helpers import sanitize_slug
from .model_registry import MODELS


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

DATASET_FILE = Path("data/chest_press_cxr.parquet")
INFERENCE_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("benchmark/results")
TASK_CHOICES = ("findings", "impression", "combined")


def inference_path_for(model_name: str, variant: str | None = None) -> Path:
    """Return the expected inference parquet path, optionally with a variant suffix.

    Variant suffix uses double-underscore to avoid collisions with the model slug
    (which uses single underscore for path separators):
        qa           → inference_<slug>__qa.parquet
        reports      → inference_<slug>__reports.parquet
        qa_reports   → inference_<slug>__qa_reports.parquet
        base / None  → inference_<slug>.parquet  (untrained baseline, no suffix)

    "base" is normalised to None to match src/inference.py, which writes the untrained
    baseline to the suffix-less path.
    """
    if variant == "base":
        variant = None
    slug = sanitize_slug(model_name)
    suffix = f"__{variant}" if variant else ""
    return INFERENCE_DIR / f"inference_{slug}{suffix}.parquet"


def should_evaluate_reference(reference_text: str) -> bool:
    return bool(normalize_report_text(reference_text))


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _strip_leading_section_heading(name: str, text: str) -> str:
    """Drop a leading ``<name>:`` heading (e.g. ``FINDINGS:``/``IMPRESSION:``) from a section value.

    ``split_report`` keeps each section's own heading inside the value (its zero-width lookahead splits
    *before* the heading), and the ``expected_output_*`` columns carry it too.  Predictions, by contrast,
    are parsed by :func:`extract_report_sections`, whose section values are heading-less.  Stripping the
    heading here makes the reference symmetric with the prediction so the metrics score body-vs-body
    instead of penalising the prediction for a missing ``FINDINGS:``/``IMPRESSION:`` token.
    """
    return re.sub(rf"^\s*(?:#{{1,6}}\s*)?{re.escape(name)}\s*[:\-–—]\s*", "", text, count=1, flags=re.IGNORECASE)


def collect_reference_sections(row: dict[str, Any]) -> dict[str, str]:
    report_path = row.get("report_path")
    if report_path:
        try:
            sections = split_report(str(report_path))
            return {k.upper(): _strip_leading_section_heading(k, normalize_report_text(v)) for k, v in sections.items()}
        except Exception as error:  # pragma: no cover - defensive fallback for malformed source reports.
            LOGGER.warning("Failed to split reference report %s: %s", report_path, error)
    return {
        "FINDINGS": _strip_leading_section_heading(
            "FINDINGS", normalize_report_text(row.get("expected_output_findings_text") or "")
        ),
        "IMPRESSION": _strip_leading_section_heading(
            "IMPRESSION", normalize_report_text(row.get("expected_output_impressions_text") or "")
        ),
    }


def _pred_ref_for_task(task: str, prediction: Any, reference: dict[str, str]) -> tuple[str, str]:
    if task == "combined":
        findings_pred = normalize_report_text(prediction.sections.get("FINDINGS", ""))
        impression_pred = normalize_report_text(prediction.sections.get("IMPRESSION", ""))
        findings_ref = normalize_report_text(reference.get("FINDINGS", ""))
        impression_ref = normalize_report_text(reference.get("IMPRESSION", ""))
        pred = "\n\n".join(s for s in (findings_pred, impression_pred) if s)
        ref = "\n\n".join(s for s in (findings_ref, impression_ref) if s)
    else:
        key = task.upper()
        pred = normalize_report_text(prediction.sections.get(key, ""))
        ref = normalize_report_text(reference.get(key, ""))
    return pred, ref


def score_predictions(
    rows: pl.DataFrame,
    *,
    model_name: str,
    tasks: Sequence[str],
    include_meteor: bool,
    output_dir: Path,
) -> list[dict[str, Any]]:
    task_pairs: dict[str, list[tuple[str, str]]] = {task: [] for task in tasks}
    raw_rows: list[dict[str, Any]] = []

    for row in rows.iter_rows(named=True):
        full_prediction = row["generated_output_text"] or ""
        prediction = extract_report_sections(full_prediction)
        reference = collect_reference_sections(row)

        pairs: dict[str, tuple[str, str]] = {}
        for task in tasks:
            pred, ref = _pred_ref_for_task(task, prediction, reference)
            pairs[task] = (pred, ref)
            task_pairs[task].append((pred, ref))

        raw_rows.append(
            {
                "subject_id": row.get("subject_id"),
                "study_id": row.get("study_id"),
                "report_path": row.get("report_path"),
                "full_prediction": full_prediction,
                "prediction_sections_method": prediction.method,
                "prediction_sections_headings": list(prediction.headings),
                "reference_sections_method": "split_report" if row.get("report_path") else "expected_output_columns",
                "tasks": {
                    task: {
                        "prediction": pairs[task][0],
                        "reference": pairs[task][1],
                        "prediction_is_empty": not pairs[task][0],
                        "reference_is_empty": not should_evaluate_reference(pairs[task][1]),
                    }
                    for task in tasks
                },
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for task in tasks:
        pairs_list = task_pairs[task]
        scored = [(p, r) for p, r in pairs_list if should_evaluate_reference(r)]
        preds = [p for p, _ in scored]
        refs = [r for _, r in scored]

        text_metrics = compute_text_metrics(preds, refs, include_meteor=include_meteor)
        clinical_metrics = compute_clinical_metrics(preds, refs, task=task)

        summary: dict[str, Any] = {
            "task": task,
            "model": model_name,
            "total_rows": len(pairs_list),
            "scored_rows": len(scored),
            "missing_reference_rows": len(pairs_list) - len(scored),
            "empty_prediction_rows": sum(1 for p, _ in scored if not normalize_report_text(p)),
        }
        summary.update(text_metrics)
        summary.update(clinical_metrics)
        summary_rows.append(summary)

    run_dir = output_dir / f"{sanitize_slug(model_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        pl.DataFrame(summary_rows).write_csv(run_dir / "summary.csv")
    else:
        (run_dir / "summary.csv").write_text("", encoding="utf-8")
    write_jsonl(run_dir / "raw_predictions.jsonl", raw_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote summary metrics to %s", run_dir / "summary.csv")
    LOGGER.info("Wrote raw predictions to %s", run_dir / "raw_predictions.jsonl")
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vision-language models on CXR report retrieval")
    parser.add_argument(
        "--model",
        type=str,
        default="google/medgemma-4b-it",
        choices=[model["name"] for model in MODELS],
        help="Model to score (must have a matching data/inference_<slug>.parquet)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=("all", *TASK_CHOICES),
        help="Which report section(s) to score",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for raw predictions and summary files",
    )
    parser.add_argument(
        "--include-meteor",
        action="store_true",
        help="Include METEOR as a legacy metric",
    )
    parser.add_argument(
        "--debug-dev-subset",
        action="store_true",
        help='Filter to subject_id starting with "10" for fast debugging',
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["base", "qa", "reports", "qa_reports"],
        help=(
            "Training variant to evaluate.  Reads inference_<slug>__<variant>.parquet and "
            "writes results to benchmark/results/<slug>__<variant>_<ts>/.  "
            "Omit for the legacy (no-variant) output filename."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # "base" is the untrained baseline — normalise to None so it maps to the suffix-less
    # inference_<slug>.parquet (matching src/inference.py), not inference_<slug>__base.parquet.
    variant = None if args.variant == "base" else args.variant

    inf_path = inference_path_for(args.model, variant)
    if not inf_path.exists():
        variant_flag = f" --variant {variant}" if variant else ""
        raise FileNotFoundError(
            f"No inference parquet for {args.model} at {inf_path}. "
            f"Run: python -m src.inference --model {args.model}{variant_flag}"
        )

    source = pl.scan_parquet(DATASET_FILE)
    inference = pl.scan_parquet(inf_path)
    joined = source.join(inference, on=["subject_id", "study_id"], how="inner")
    if args.debug_dev_subset:
        joined = joined.filter(pl.col("subject_id").cast(pl.Utf8).str.starts_with("10"))
    rows = joined.collect()
    LOGGER.info("Scoring %s rows for %s", len(rows), args.model)

    tasks = TASK_CHOICES if args.task == "all" else (args.task,)
    # When a variant is active, embed it in the run directory name so all four
    # variant results sit side-by-side under benchmark/results/ without collisions.
    variant_label = f"__{variant}" if variant else ""
    effective_model_label = f"{args.model}{variant_label}"
    summary_rows = score_predictions(
        rows=rows,
        model_name=effective_model_label,
        tasks=tasks,
        include_meteor=args.include_meteor,
        output_dir=args.output_dir,
    )

    for row in summary_rows:
        LOGGER.info(
            "task=%s scored=%s bleu=%s rougeL=%s bertscore_f1=%s radgraph_f1=%s",
            row["task"],
            row["scored_rows"],
            row.get("bleu"),
            row.get("rougeL"),
            row.get("bertscore_f1"),
            row.get("radgraph_f1"),
        )


if __name__ == "__main__":
    main()
