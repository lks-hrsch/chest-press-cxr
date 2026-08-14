"""Stage-B dataset: CXR report generation from chest_press_cxr_train.parquet.

Mirrors the message layout of src/inference.py:build_conversation exactly, but uses
{"type":"image"} placeholders (PIL images in the ``images`` list) instead of
{"type":"image_url"} file:// URLs so that the transformers processor can handle them.

Appends an assistant turn with the concatenated FINDINGS + IMPRESSION sections
(FINDINGS first, matching the OUTPUT_LABELS contract in src/dataset/constants.py).

Output format per example:
    {
        "messages": [
            {"role": "system",    "content": [{"type": "text",  "text": system_prompt}]},
            {"role": "user",      "content": [
                {"type": "image"}, ...N images...,
                {"type": "text",  "text": input_text},
            ]},
            {"role": "assistant", "content": [{"type": "text",  "text": "FINDINGS: ...\nIMPRESSION: ..."}]},
        ],
        "images": [PIL.Image, ...],
    }
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl
from PIL import Image
from torch.utils.data import Dataset

from ..dataset.constants import XRAY_BASE_PATH


LOGGER = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = "data/chest_press_cxr_train.parquet"


class ReportDataset(Dataset):
    """PyTorch Dataset wrapping the CXR report generation training parquet.

    Images are loaded lazily on each __getitem__ call.

    Parameters
    ----------
    parquet_path:
        Path to the training split parquet.
    xray_base_path:
        Root directory of MIMIC-CXR-JPG ``files/`` hierarchy.
    max_images:
        Cap the number of images per study (None = no cap). Useful for smoke tests.
    """

    def __init__(
        self,
        parquet_path: str = DEFAULT_TRAIN_PATH,
        xray_base_path: str = XRAY_BASE_PATH,
        max_images: int | None = None,
    ) -> None:
        self.xray_base_path = xray_base_path.rstrip("/") + "/"
        self.max_images = max_images
        self._records: list[dict[str, Any]] = pl.read_parquet(parquet_path).to_dicts()
        LOGGER.info("ReportDataset: loaded %d rows from %s", len(self._records), parquet_path)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        row = self._records[idx]

        # --- build image list ---
        image_paths: list[str] = row["xray_image_paths"]
        if self.max_images is not None:
            image_paths = image_paths[: self.max_images]

        images: list[Image.Image] = []
        for rel_path in image_paths:
            full_path = self.xray_base_path + rel_path
            try:
                images.append(Image.open(full_path).convert("RGB"))
            except Exception as exc:
                LOGGER.warning("cannot open image %s: %s — using blank", full_path, exc)
                images.append(Image.new("RGB", (224, 224), color=0))

        # --- build messages (mirror build_conversation + assistant turn) ---
        messages: list[dict] = []

        if row.get("system_prompt"):
            messages.append({"role": "system", "content": [{"type": "text", "text": row["system_prompt"]}]})

        user_content: list[dict] = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": row["input_text"]})
        messages.append({"role": "user", "content": user_content})

        # FINDINGS then IMPRESSION — locked by OUTPUT_LABELS contract
        findings = (row.get("expected_output_findings_text") or "").strip()
        impression = (row.get("expected_output_impressions_text") or "").strip()
        # The fields are already prefixed ("FINDINGS: ..." / "IMPRESSION: ...")
        assistant_text = findings + "\n" + impression if impression else findings
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

        return {
            "messages": messages,
            "images": images,
        }
