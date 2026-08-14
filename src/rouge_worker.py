"""Standalone ROUGE-L worker, deliberately kept free of heavy imports.

This module exists **only** so ``spawn``-ed worker processes stay small. It must not import
``src.eval`` (whose ``__init__`` pulls in torch, RadGraph and bert_score) or anything else that
loads a model: a spawn worker re-imports the module its target function lives in, so putting the
worker inside ``src/eval/text_metrics.py`` gave every one of the 12 workers a full torch +
RadGraph import -- roughly 1 GB each, which exhausted the 30 GB box and got the pool OOM-killed
mid-run. ``src/__init__.py`` is empty, so importing ``src.rouge_worker`` costs only ``evaluate``.

Keep it that way: no ``from .eval import ...``, no torch, no transformers.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _rouge():
    import evaluate

    return evaluate.load("rouge")


def rouge_l_chunk(payload: tuple[list[str], list[str]]) -> list[float]:
    """ROUGE-L F1 per row for one contiguous slice of **already normalized** text.

    Identical call to the serial path (``use_aggregator=False``, ``use_stemmer=True``) so the
    concatenated per-row output matches computing the whole batch in one go.
    """
    predictions, references = payload
    scores = _rouge().compute(
        predictions=predictions,
        references=references,
        use_aggregator=False,
        use_stemmer=True,
    )
    return [float(v.fmeasure) if hasattr(v, "fmeasure") else float(v) for v in scores["rougeL"]]
