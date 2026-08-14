"""Per-row scoring for the clinical-context ablation grid (RQ2 x RQ3).

Computes ``src/eval/per_example.py``'s per-row metrics for one ``--model`` across all 3 context
tiers (``baseline`` / ``clinical_context`` / ``rich_clinical_context`` -- see
``src/dataset/context_ablation.py``) and both training stages (``base`` untrained checkpoint,
``reports`` LoRA fine-tuned checkpoint), and writes one long-format parquet per model to
``benchmark/results_ablation/<slug>.parquet``. Notebooks read that output only -- no metrics are
computed in the notebooks themselves (mirrors the existing ``src/evaluate.py`` compute /
``notebooks/evaluation-base-vs-sft-*.ipynb`` visualize split).

Each row also carries ``anchor_age``, ``gender``, and the 14 CheXpert label columns from the test
split (unaffected by the ablation -- only ``input_text``/``system_prompt`` differ per tier) so the
subgroup notebook can stratify without any further joins.

Usage::

    python -m src.evaluate_ablation --model google/gemma-3-4b-it
    python -m src.evaluate_ablation --model google/gemma-3-4b-it --sample-size 250 --sample-seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

from .dataset.context_ablation import TIERS
from .eval import compute_clinical_metrics_per_row, compute_text_metrics_per_row, extract_report_sections
from .evaluate import (
    TASK_CHOICES,
    _pred_ref_for_task,
    collect_reference_sections,
    inference_path_for,
    should_evaluate_reference,
)
from .helpers import sanitize_slug
from .model_registry import MODELS


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

DATASET_FILE = Path("data/chest_press_cxr_test.parquet")
DEFAULT_OUTPUT_DIR = Path("benchmark/results_ablation")
STAGES = ("base", "reports")
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


def tier_path(model_name: str, stage: str, tier: str) -> Path:
    """The inference parquet for a (model, stage, context tier) combination.

    ``rich_clinical_context`` is today's default (unsuffixed) tier -- ``inference_path_for``
    already returns that path for ``variant=None``/``"reports"``; the other two tiers add the
    ``__ablation_<tier>`` suffix ``src/inference.py --context`` composes after any variant suffix.
    """
    variant = None if stage == "base" else "reports"
    path = inference_path_for(model_name, variant)
    if tier == "rich_clinical_context":
        return path
    return path.with_name(f"{path.stem}__ablation_{tier}{path.suffix}")


def _dedup_keys(path: Path) -> pl.LazyFrame:
    """Inference parquet deduped on (subject_id, study_id), keep-first (files carry ~3% duplicate
    keys -- confirmed on llava-v1.6's base/baseline tier: 1000 rows, 971 unique).
    """
    return pl.scan_parquet(path).unique(subset=["subject_id", "study_id"], keep="first")


def matched_keys(model_name: str, stage: str) -> tuple[pl.DataFrame | None, dict, dict[str, Path]]:
    """3-way key intersection across the stage's 3 tiers, or ``(None, ...)`` if any tier file is
    missing. Tier key-sets differ even within the same model/stage (confirmed), so every tier must
    be scored on the exact same matched rows for a fair ablation comparison.
    """
    paths = {tier: tier_path(model_name, stage, tier) for tier in TIERS}
    missing = [tier for tier, path in paths.items() if not path.exists()]
    if missing:
        return None, {"missing_tiers": missing}, paths

    common = None
    for tier in TIERS:
        keys = _dedup_keys(paths[tier]).select("subject_id", "study_id")
        common = keys if common is None else common.join(keys, on=["subject_id", "study_id"], how="inner")
    return common.collect(), {"missing_tiers": []}, paths


#: The tier whose inference parquet *is* the unsuffixed main file -- i.e. exactly what
#: ``src/evaluate_stages.py`` already scores for stages ``base`` and ``reports``. Verified
#: identical (max abs diff 0.0 on radgraph_f1 across all 4 840 rows of both stages).
REUSABLE_TIER = "rich_clinical_context"


def reusable_rows_from_stages(
    model_name: str,
    stage: str,
    keys: pl.DataFrame,
    gt_by_key: dict[tuple[int, int], dict],
    stages_dir: Path,
) -> list[dict] | None:
    """Rows for the ``rich_clinical_context`` tier, lifted from the already-scored stages parquet.

    Returns ``None`` (meaning "compute it normally") unless the stages parquet exists AND covers
    every matched key for this stage -- a partial reuse would silently mix coverage between the
    two files, which is exactly the bug this whole re-scoring pass exists to fix.

    The two schemas differ by one column each way: stages carries ``model_family``, ablation
    carries ``context_tier``. Everything else -- all 6 metric columns, the demographics and the 14
    CheXpert labels -- is shared and copied verbatim.
    """
    path = stages_dir / f"{sanitize_slug(model_name)}.parquet"
    if not path.exists():
        LOGGER.info("no stages parquet at %s -- computing %s normally", path, REUSABLE_TIER)
        return None

    scored = pl.read_parquet(path).filter(pl.col("stage") == stage)
    if scored.is_empty():
        LOGGER.info("stages parquet has no stage=%s -- computing %s normally", stage, REUSABLE_TIER)
        return None

    # Compare against the keys score_stage would actually EMIT, not the raw matched set: it skips
    # any key absent from gt_by_key, and --debug-dev-subset filters the ground truth without
    # touching the matched keys. Using the raw set here made the flag always fall back.
    wanted = {
        key for key in zip(keys["subject_id"].to_list(), keys["study_id"].to_list(), strict=True) if key in gt_by_key
    }
    have = set(zip(scored["subject_id"].to_list(), scored["study_id"].to_list(), strict=True))
    if not wanted <= have:
        LOGGER.warning(
            "stages parquet covers %d/%d matched keys for stage=%s -- computing %s normally instead of mixing coverage",
            len(wanted & have),
            len(wanted),
            stage,
            REUSABLE_TIER,
        )
        return None

    scored = scored.join(keys, on=["subject_id", "study_id"], how="inner")
    rows: list[dict] = []
    for row in scored.iter_rows(named=True):
        key = (row["subject_id"], row["study_id"])
        gt_row = gt_by_key.get(key)
        if gt_row is None:  # mirrors score_stage's skip, so both paths emit the same row set
            continue
        rows.append(
            {
                "model": model_name,
                "stage": stage,
                "context_tier": REUSABLE_TIER,
                "task": row["task"],
                "subject_id": key[0],
                "study_id": key[1],
                "scored": row["scored"],
                **{col: row[col] for col in METRIC_COLUMNS},
                "anchor_age": gt_row.get("anchor_age"),
                "gender": gt_row.get("gender"),
                **{col: gt_row.get(col) for col in CHEXPERT_LABEL_COLUMNS},
            }
        )
    LOGGER.info("reused %d scored rows for stage=%s tier=%s from %s", len(rows), stage, REUSABLE_TIER, path)
    return rows


def score_stage(
    model_name: str,
    stage: str,
    keys: pl.DataFrame,
    paths: dict[str, Path],
    gt_by_key: dict[tuple[int, int], dict],
    tiers: tuple[str, ...] = TIERS,
) -> list[dict]:
    """Score ``tiers`` x 3 tasks for one (model, stage) on the given matched ``keys``."""
    rows: list[dict] = []
    for tier in tiers:
        infer_rows = _dedup_keys(paths[tier]).join(keys.lazy(), on=["subject_id", "study_id"], how="inner").collect()

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
                        "stage": stage,
                        "context_tier": tier,
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
    parser = argparse.ArgumentParser(description="Score the clinical-context ablation grid for one model")
    parser.add_argument("--model", type=str, required=True, choices=[m["name"] for m in MODELS], help="Model to score")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the per-model parquet"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Cap matched rows per stage to this many (deterministic sample). Default: no cap (full matched set).",
    )
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for --sample-size sampling")
    parser.add_argument(
        "--debug-dev-subset", action="store_true", help='Filter to subject_id starting with "10" for fast debugging'
    )
    parser.add_argument(
        "--stages-dir",
        type=Path,
        default=Path("benchmark/results_training_stages"),
        help=f"Where evaluate_stages wrote its per-model parquets, reused for the {REUSABLE_TIER} tier",
    )
    parser.add_argument(
        "--no-reuse-stages",
        action="store_true",
        help=f"Recompute the {REUSABLE_TIER} tier instead of lifting it from --stages-dir (slower; "
        "use to reproduce the pre-reuse behaviour or when the stages parquets are stale)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_slug(args.model)

    gt = pl.scan_parquet(DATASET_FILE)
    if args.debug_dev_subset:
        gt = gt.filter(pl.col("subject_id").cast(pl.Utf8).str.starts_with("10"))

    all_rows: list[dict] = []
    coverage: list[dict] = []

    for stage in STAGES:
        keys, info, paths = matched_keys(args.model, stage)
        if keys is None:
            LOGGER.warning("skip model=%s stage=%s: missing tiers %s", args.model, stage, info["missing_tiers"])
            coverage.append(
                {"stage": stage, "available": False, "missing_tiers": info["missing_tiers"], "matched_rows": None}
            )
            continue

        if args.sample_size is not None and keys.height > args.sample_size:
            keys = keys.sample(n=args.sample_size, seed=args.sample_seed)

        gt_matched = gt.join(keys.lazy(), on=["subject_id", "study_id"], how="inner").collect()
        gt_by_key = {(row["subject_id"], row["study_id"]): row for row in gt_matched.iter_rows(named=True)}

        LOGGER.info("model=%s stage=%s matched_rows=%d", args.model, stage, keys.height)

        # The rich_clinical_context tier IS the main inference file, already scored by
        # evaluate_stages for exactly these stages. Recomputing it costs 1/3 of this script's
        # runtime for a bit-identical result, so lift it instead -- but only on an exact coverage
        # match (see reusable_rows_from_stages).
        reused = (
            None
            if args.no_reuse_stages
            else reusable_rows_from_stages(args.model, stage, keys, gt_by_key, args.stages_dir)
        )
        tiers = tuple(t for t in TIERS if t != REUSABLE_TIER) if reused is not None else TIERS
        all_rows.extend(score_stage(args.model, stage, keys, paths, gt_by_key, tiers=tiers))
        if reused is not None:
            all_rows.extend(reused)
        coverage.append(
            {
                "stage": stage,
                "available": True,
                "missing_tiers": [],
                "matched_rows": keys.height,
                "reused_tier": REUSABLE_TIER if reused is not None else None,
            }
        )

    out_path = args.output_dir / f"{slug}.parquet"
    if all_rows:
        # Explicit overrides for the metric + CheXpert label columns: a rare label (e.g. "Pleural
        # Other", ~1% positive in the full test split) can have ZERO non-null values across a
        # single model's capped sample, in which case polars infers that column as Null (not
        # Float64) -- harmless alone, but it then breaks pl.concat() across models in the notebooks
        # once one model's file has a Null-typed column where the others have Float64. Forcing the
        # dtype here (rather than just widening infer_schema_length, which only helps when SOME row
        # has a non-null value) keeps every model's output parquet schema-identical regardless of
        # which labels happen to be unobserved in its sample.
        schema_overrides = {col: pl.Float64 for col in (*METRIC_COLUMNS, *CHEXPERT_LABEL_COLUMNS)}
        pl.DataFrame(all_rows, schema_overrides=schema_overrides, infer_schema_length=None).write_parquet(out_path)
        LOGGER.info("Wrote %d scored rows to %s", len(all_rows), out_path)
    else:
        LOGGER.warning("No stage available for %s -- nothing written", args.model)

    coverage_path = args.output_dir / f"{slug}_coverage.json"
    coverage_path.write_text(json.dumps({"model": args.model, "stages": coverage}, indent=2), encoding="utf-8")
    LOGGER.info("Wrote coverage summary to %s", coverage_path)


if __name__ == "__main__":
    main()
