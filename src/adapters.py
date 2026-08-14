"""Shared helpers for serving trained LoRA adapters at inference time.

Trained adapters are served UNIFORMLY as merged full checkpoints — one path for every
architecture.  ``src/merge_adapter.py`` merges the PEFT adapter into the base weights and
``src/inference.py`` serves the result via ``--variant``.  (vLLM 0.22 only
supports *runtime* LoRA for a subset of archs; merging sidesteps that and vLLM serves the
merged checkpoint like any other model.)

Layout produced by ``src/train/*``:
    <checkpoints-root>/<model-slug>/<variant>_adapter/   (PEFT adapter dir)
and produced by the merge step:
    <checkpoints-root>/<model-slug>/<variant>_merged/    (full merged checkpoint)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any


LOGGER = logging.getLogger(__name__)

# Root holding per-model adapter / merged checkpoint directories.
CHECKPOINTS_ROOT_DEFAULT = "benchmark/checkpoints"
# Variants that correspond to a trained LoRA adapter (i.e. not the untrained base model).
ADAPTER_VARIANTS = ("qa", "reports", "qa_reports")
_ADAPTER_SUFFIX = "_adapter"


def adapter_dir_for(checkpoints_root: str, slug: str, variant: str) -> str:
    return os.path.join(checkpoints_root, slug, f"{variant}{_ADAPTER_SUFFIX}")


def merged_dir_for(checkpoints_root: str, slug: str, variant: str) -> str:
    return os.path.join(checkpoints_root, slug, f"{variant}_merged")


def infer_variant(adapter_path: str) -> str | None:
    """Infer the training variant from an adapter directory path.

    Handles both the final adapter dir (``.../reports_adapter``) and a raw trainer
    checkpoint dir (``.../reports_adapter/checkpoint-10374`` — used when only
    checkpoint-N subdirs were copied).  Returns None when it cannot be determined.
    """
    norm = os.path.normpath(adapter_path)
    base = os.path.basename(norm)
    if base.endswith(_ADAPTER_SUFFIX):
        return base[: -len(_ADAPTER_SUFFIX)]
    if base.startswith("checkpoint-"):
        parent = os.path.basename(os.path.dirname(norm))
        if parent.endswith(_ADAPTER_SUFFIX):
            return parent[: -len(_ADAPTER_SUFFIX)]
    return None


def read_adapter_config(adapter_path: str) -> dict[str, Any] | None:
    """Return the parsed adapter_config.json, or None if the dir has no top-level config.

    A missing config typically means only ``checkpoint-N/`` subdirs were copied (no final
    ``save_model`` output) — point at the latest checkpoint dir instead.
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as handle:
        return json.load(handle)
