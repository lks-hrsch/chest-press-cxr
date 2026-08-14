"""Stage-A dataset: MIMIC-Ext-CXR-QBA A++ (App) subset.

Builds a PyTorch Dataset of VQA examples from the App quality-tier export.
The App subset (1,318,885 QA-pairs) lives in:
    exports/A_frontal/metadata/App/  — parquet index (patient/study/question/image IDs)
    exports/A_frontal/qa/            — per-study JSON files (question + answer text)

Example output format (matches TRL SFTTrainer VLM convention):
    {
        "messages": [
            {"role": "system",    "content": [{"type": "text",  "text": SYSTEM_PROMPT}]},
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text",  "text": main_answer}]},
        ],
        "images": [PIL.Image],   # one frontal CXR image per QA-pair
    }

First call builds a flat index parquet (data/qba_app_index.parquet) by loading all per-study
QA JSONs; subsequent calls use the cache.  Use cache_path=None to skip caching.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
from PIL import Image
from torch.utils.data import Dataset

from ..dataset.constants import QBA_EXPORTS_APP, QBA_QA_DIR, XRAY_BASE_PATH


LOGGER = logging.getLogger(__name__)

VQA_SYSTEM_PROMPT = "You are an expert radiologist. Answer the question about the chest X-ray concisely and accurately."

# Default on-disk cache for the flat question/answer index.
DEFAULT_CACHE_PATH = "data/qba_app_q1m_index.parquet"


class QBADataset(Dataset):
    """PyTorch Dataset of A++ CXR VQA pairs.

    Images are loaded from disk lazily on each __getitem__ call.

    Parameters
    ----------
    app_dir:
        Path to ``exports/A_frontal/metadata/App/`` (contains the App parquets).
    qa_dir:
        Path to ``exports/A_frontal/qa/`` (contains per-study JSON files).
    xray_base_path:
        Root directory of MIMIC-CXR-JPG ``files/`` hierarchy.
    cache_path:
        Parquet path to write/read the flattened index.  Pass None to disable caching.
    """

    def __init__(
        self,
        app_dir: str = QBA_EXPORTS_APP,
        qa_dir: str = QBA_QA_DIR,
        xray_base_path: str = XRAY_BASE_PATH,
        cache_path: str | None = DEFAULT_CACHE_PATH,
        max_studies: int | None = None,
    ) -> None:
        self.xray_base_path = xray_base_path.rstrip("/") + "/"
        self._records: list[dict[str, Any]] = []

        if cache_path and Path(cache_path).exists():
            LOGGER.info("loading QBA index from cache %s", cache_path)
            self._records = pl.read_parquet(cache_path).to_dicts()
        else:
            LOGGER.info("building QBA index from %s — this may take a few minutes", app_dir)
            index_df = _build_index(app_dir, qa_dir, max_studies=max_studies)
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                index_df.write_parquet(cache_path)
                LOGGER.info("wrote QBA index cache to %s (%d rows)", cache_path, len(index_df))
            self._records = index_df.to_dicts()

        LOGGER.info("QBADataset ready: %d examples", len(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        r = self._records[idx]
        img_path = self.xray_base_path + r["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            LOGGER.warning("cannot open image %s: %s — using blank", img_path, exc)
            img = Image.new("RGB", (224, 224), color=0)

        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": VQA_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": r["question_text"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": r["answer_text"]}]},
            ],
            "images": [img],
        }


# ---------------------------------------------------------------------------
# Index-building internals
# ---------------------------------------------------------------------------


def _build_index(app_dir: str, qa_dir: str, max_studies: int | None = None) -> pl.DataFrame:
    """Load all QA JSONs for App-indexed questions; return flat polars DataFrame.

    Columns: patient_id, study_id, question_id, image_id, image_path,
             question_text, answer_text.

    The per-study JSON reads are I/O-bound (one small file open each, often over a
    network/ZFS mount), so they are fanned out across a thread pool.  Pass
    ``max_studies`` to cap the build for smoke tests (do not cache a truncated index).
    """
    app_dir = app_dir.rstrip("/")
    qa_dir = qa_dir.rstrip("/")

    LOGGER.info("reading App metadata parquets from %s", app_dir)
    q_meta = pl.read_parquet(f"{app_dir}/question_metadata.parquet").select(["patient_id", "study_id", "question_id"])
    qi_meta = pl.read_parquet(f"{app_dir}/question_image_metadata.parquet").select(
        ["patient_id", "study_id", "question_id", "image_id"]
    )

    # Left-join: questions without an image_id entry get null (handled below)
    joined = q_meta.join(qi_meta, on=["patient_id", "study_id", "question_id"], how="left")

    # Group questions by study so we load each JSON exactly once
    studies = (
        joined.group_by(["patient_id", "study_id"])
        .agg(
            pl.col("question_id").alias("question_ids"),
            pl.col("image_id").alias("image_ids"),
        )
        .to_dicts()
    )
    if max_studies is not None:
        studies = studies[:max_studies]
        LOGGER.info("limiting build to first %d studies", len(studies))
    LOGGER.info("processing %d studies", len(studies))

    records: list[dict[str, Any]] = []
    failed_studies = 0
    total_studies = len(studies)

    def _process_study(study: dict) -> tuple[list[dict[str, Any]], bool, str]:
        """Read one study's QA JSON; return (records, failed, qa_path)."""
        patient_id: str = study["patient_id"]
        study_id: str = study["study_id"]
        question_ids: list[str] = study["question_ids"]
        image_ids: list[str | None] = study["image_ids"]

        qa_path = f"{qa_dir}/{patient_id[:3]}/{patient_id}/{study_id}.qa.json"
        qa_lookup: dict[str, dict] = {}
        try:
            with open(qa_path) as fh:
                qa_data = json.load(fh)
            # Build {question_id: {question_text, main_answer_text, first_image_id}}
            for q in qa_data.get("questions", []):
                qid = q.get("question_id", "")
                qa_lookup[qid] = {
                    "question_text": q.get("question", ""),
                    "answer_text": _extract_main_answer(q.get("answers", [])),
                    "json_image_id": _extract_first_image_id(q.get("answers", [])),
                }
        except Exception:
            return [], True, qa_path

        study_records: list[dict[str, Any]] = []
        for qid, img_id in zip(question_ids, image_ids):
            entry = qa_lookup.get(qid)
            if entry is None:
                continue  # question not in this study's JSON (shouldn't happen but be safe)
            if not entry["answer_text"]:
                continue  # no usable main answer

            # Prefer the metadata image_id (per question_image_metadata); fall back to JSON
            resolved_image_id = img_id or entry["json_image_id"]
            if not resolved_image_id:
                continue  # no image reference — skip

            # Build relative path matching XRAY_BASE_PATH convention:
            # patient_id[:3]/patient_id/study_id/image_id.jpg
            image_path = f"{patient_id[:3]}/{patient_id}/{study_id}/{resolved_image_id}.jpg"

            study_records.append(
                {
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "question_id": qid,
                    "image_id": resolved_image_id,
                    "image_path": image_path,
                    "question_text": entry["question_text"],
                    "answer_text": entry["answer_text"],
                }
            )
        return study_records, False, qa_path

    # Fan out the (I/O-bound) JSON reads across threads; the GIL is released during file I/O.
    with ThreadPoolExecutor(max_workers=32) as pool:
        for i, (study_records, failed, qa_path) in enumerate(pool.map(_process_study, studies)):
            if i and i % 10_000 == 0:
                LOGGER.info(
                    "indexing studies %d/%d (%d records, %d failed)", i, total_studies, len(records), failed_studies
                )
            if failed:
                failed_studies += 1
                if failed_studies <= 5:
                    LOGGER.warning("QA JSON load failed: %s", qa_path)
                continue
            records.extend(study_records)

    if failed_studies:
        LOGGER.warning("%d studies failed JSON load (total records built: %d)", failed_studies, len(records))

    return pl.DataFrame(records)


def _extract_main_answer(answers: list[dict]) -> str:
    """Return the text of the first top-level main_answer (answer_level==0)."""
    for ans in answers:
        if ans.get("answer_type") == "main_answer" and ans.get("answer_level", 0) == 0:
            text = ans.get("text", "").strip()
            if text:
                return text
    return ""


def _extract_first_image_id(answers: list[dict]) -> str | None:
    """Extract the first image_id referenced in any answer's localization dict."""
    for ans in answers:
        loc = ans.get("localization", {})
        if isinstance(loc, dict) and loc:
            return next(iter(loc.keys()))
    return None
