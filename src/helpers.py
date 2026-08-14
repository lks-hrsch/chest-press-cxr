from __future__ import annotations

import re

from torch import cuda, mps


# Check GPU availability
def gpu_label():
    if cuda.is_available():
        return "cuda"
    elif mps.is_available():
        return "mps"
    else:
        return "cpu"


def sanitize_slug(value: str) -> str:
    """Return a filesystem-safe slug for a model name or any arbitrary string.

    Replaces every run of characters outside ``[A-Za-z0-9._-]`` with a single
    underscore, strips leading/trailing underscores, and falls back to
    ``"model"`` for the empty string.

    Examples
    --------
    >>> sanitize_slug("google/medgemma-1.5-4b-it")
    'google_medgemma-1.5-4b-it'
    >>> sanitize_slug("Qwen/Qwen3-VL-4B-Instruct")
    'Qwen_Qwen3-VL-4B-Instruct'
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"
