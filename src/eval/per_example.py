"""Per-row (non-aggregated) metric primitives.

``score_predictions`` (via :func:`src.eval.compute_text_metrics` / :func:`compute_clinical_metrics`)
only returns metrics aggregated over an entire ``(predictions, references)`` batch -- corpus BLEU,
mean ROUGE-L, mean BERTScore-F1, mean RadGraph-F1 -- with no per-row breakdown. Label/age-stratified
analysis needs a score per row instead, without paying the cost of re-running the underlying models
once per stratification subgroup. Every metric here reuses the *exact same* underlying model/call as
its aggregate counterpart in ``text_metrics.py`` / ``clinical_metrics.py`` / ``ssr_bert.py`` -- only
the reduction step (mean / corpus-level aggregation) is skipped or replaced with a per-row equivalent:

    BERTScore       -- ``bert_score_score(...)`` already returns per-example tensors; the aggregate
                       version (``compute_bertscore``) just calls ``.mean()`` on them.
    RadGraph         -- ``F1RadGraph.forward()`` returns ``(mean_reward, reward_list, ...)``; the
                       aggregate version (``compute_radgraph_metrics``) discards ``reward_list``,
                       which already holds a per-row ``(rg_e, rg_er, rg_bar_er)`` tuple.
    ROUGE-L          -- ``rouge.compute(..., use_aggregator=False)`` returns a per-example list.
    BLEU             -- corpus BLEU has no clean per-row decomposition; this uses sentence-level BLEU
                       (``nltk.translate.bleu_score.sentence_bleu``, same ``method7`` smoothing
                       function as the aggregate) as a per-row proxy. **Not** numerically identical
                       to the aggregate corpus BLEU, and -- a known ``method7`` quirk on short
                       sentences -- can occasionally exceed 1.0 (e.g. an exact-match five-token
                       sentence scores ~1.12); this washes out at the corpus level but is visible
                       per row, so don't treat 1.0 as a hard ceiling for this column.
    F1-SRR-BERT /
    Category Score   -- sklearn's "samples avg" F1 (used by ``ssr_bert.score`` / ``category_score.score``)
                       is already a per-sample F1 averaged across samples, so the per-row value is
                       just that single sample's F1 -- no sklearn aggregation call needed.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .clinical_metrics import F1RadGraph
from .sections import normalize_report_texts
from .text_metrics import _tokenize_for_bleu, bert_score_score, rouge_l_f1_per_row


LOGGER = logging.getLogger(__name__)
_SMOOTHING_FUNCTION = SmoothingFunction().method7


@lru_cache(maxsize=4)
def _load_radgraph(model_type: str, reward_level: str):
    """Load+cache F1RadGraph once per (model_type, reward_level), mirroring the ``@lru_cache``
    pattern ``ssr_bert._load()`` already uses. The aggregate ``compute_radgraph_metrics``
    instantiates a fresh ``F1RadGraph`` on every call (fine there -- it's called a handful of
    times); the per-row path here is called once per (tier x task) in ``evaluate_ablation.py``'s
    grid, so reloading RadGraph-XL every time would dominate runtime for no benefit.
    """
    if F1RadGraph is None:
        return None
    return F1RadGraph(reward_level=reward_level, model_type=model_type)


def compute_bleu_per_row(predictions: Sequence[str], references: Sequence[str]) -> list[float | None]:
    """Sentence-level BLEU per row. NOT the same reduction as the aggregate corpus BLEU."""
    if not predictions:
        return []
    tokenized_predictions = _tokenize_for_bleu(predictions)
    tokenized_references = _tokenize_for_bleu(references)
    return [
        float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=_SMOOTHING_FUNCTION))
        for pred_tokens, ref_tokens in zip(tokenized_predictions, tokenized_references, strict=True)
    ]


def compute_rouge_l_per_row(predictions: Sequence[str], references: Sequence[str]) -> list[float | None]:
    """Per-row ROUGE-L F1.

    Normalization stays here (cheap, and it keeps the exact text the serial path used); the scoring
    itself goes through ``rouge_l_f1_per_row``, which fans out across worker processes above
    ``_ROUGE_PARALLEL_MIN_ROWS``. Chunks are contiguous and re-concatenated in order, so the
    per-row output is identical to computing the whole batch in one call.
    """
    if not predictions:
        return []
    return rouge_l_f1_per_row(
        normalize_report_texts(predictions),
        normalize_report_texts(references),
    )


def compute_bertscore_per_row(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "roberta-large",
    lang: str = "en",
    batch_size: int = 16,
    device: str | None = None,
    rescale_with_baseline: bool = False,
) -> dict[str, list[float | None]]:
    n = len(predictions)
    result: dict[str, list[float | None]] = {
        "bertscore_precision": [None] * n,
        "bertscore_recall": [None] * n,
        "bertscore_f1": [None] * n,
    }
    if n == 0 or bert_score_score is None:
        return result

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        precision, recall, f1 = bert_score_score(
            normalize_report_texts(predictions),
            normalize_report_texts(references),
            model_type=model_type,
            lang=lang,
            batch_size=batch_size,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
    except Exception as error:  # pragma: no cover - optional backend may fail at runtime.
        # Mirrors compute_bertscore's graceful degradation (text_metrics.py): one bad batch
        # (transient CUDA OOM, etc.) shouldn't crash the whole scoring grid and lose every
        # already-accumulated row for this model -- return None for this batch and keep going.
        LOGGER.warning("Skipping per-row BERTScore for this batch after backend failure: %s", error)
        return result
    return {
        "bertscore_precision": [float(v) for v in precision.tolist()],
        "bertscore_recall": [float(v) for v in recall.tolist()],
        "bertscore_f1": [float(v) for v in f1.tolist()],
    }


def compute_text_metrics_per_row(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    bertscore_model_type: str = "roberta-large",
    bertscore_lang: str = "en",
    bertscore_batch_size: int = 16,
    bertscore_device: str | None = None,
) -> dict[str, list[float | None]]:
    metrics: dict[str, list[float | None]] = {
        "bleu": compute_bleu_per_row(predictions, references),
        "rougeL": compute_rouge_l_per_row(predictions, references),
    }
    metrics.update(
        compute_bertscore_per_row(
            predictions,
            references,
            model_type=bertscore_model_type,
            lang=bertscore_lang,
            batch_size=bertscore_batch_size,
            device=bertscore_device,
        )
    )
    return metrics


# RadGraph-XL peaks hard on long reports. Scoring a whole (model, stage, tier, task) cell in one
# call made it try to allocate 16.43 GiB for gemma-4-E2B-it's verbose `base`/`baseline` output on a
# 24 GB card; the blanket except then nulled all 4 840 rows of that cell at once. Chunking bounds
# the peak allocation and keeps a failure local. 256 is comfortably inside budget for the longest
# reports observed while still amortising the per-call overhead.
DEFAULT_RADGRAPH_BATCH_SIZE = 256


def compute_radgraph_metrics_per_row(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "radgraph-xl",
    reward_level: str = "all",
    batch_size: int = DEFAULT_RADGRAPH_BATCH_SIZE,
) -> dict[str, list[float | None]]:
    """Per-row RadGraph scores, computed in bounded chunks.

    A chunk that raises (typically CUDA OOM on unusually long reports) is retried at half the size,
    recursively down to single rows, so a failure costs only the rows that genuinely cannot be
    scored rather than the entire batch. Unscored rows stay ``None``, matching the graceful
    degradation contract of ``compute_radgraph_metrics``.
    """
    n = len(predictions)
    result: dict[str, list[float | None]] = {
        "radgraph_rg_e": [None] * n,
        "radgraph_f1": [None] * n,  # rg_er, matching the aggregate's "radgraph_f1" == rg_er
        "radgraph_rg_bar_er": [None] * n,
    }
    if n == 0:
        return result
    metric = _load_radgraph(model_type, reward_level)
    if metric is None:
        return result

    hyps = normalize_report_texts(predictions)
    refs = normalize_report_texts(references)

    def _run(lo: int, hi: int) -> bool:
        """Score rows [lo, hi) in a single backend call; True on success."""
        try:
            _, reward_list, _, _ = metric(hyps=hyps[lo:hi], refs=refs[lo:hi])
            rg_e, rg_er, rg_bar_er = reward_list
        except Exception as error:  # pragma: no cover - optional backend may fail at runtime.
            LOGGER.warning("Per-row RadGraph failed on rows [%d, %d): %s", lo, hi, error)
            return False
        for offset in range(hi - lo):
            result["radgraph_rg_e"][lo + offset] = float(rg_e[offset])
            result["radgraph_f1"][lo + offset] = float(rg_er[offset])
            result["radgraph_rg_bar_er"][lo + offset] = float(rg_bar_er[offset])
        return True

    def _score_span(lo: int, hi: int, size: int) -> None:
        """Score [lo, hi) in slices of `size`, halving any slice that fails (down to 1 row)."""
        start = lo
        while start < hi:
            end = min(start + size, hi)
            if not _run(start, end) and end - start > 1:
                _score_span(start, end, max(1, (end - start) // 2))
            start = end

    _score_span(0, n, max(1, batch_size))
    unscored = sum(1 for v in result["radgraph_f1"] if v is None)
    if unscored:
        LOGGER.warning("Per-row RadGraph left %d/%d rows unscored after chunk retries", unscored, n)
    return result


def _per_row_label_f1(pred_row: np.ndarray, true_row: np.ndarray) -> float:
    """Single-sample F1 over two multi-hot label vectors -- matches sklearn's per-sample term in
    ``classification_report(..., zero_division=0)``'s "samples avg" (verified in
    ``tests/test_per_example.py``): F1 = 2*TP / (|pred| + |true|), 0.0 when both are empty.
    """
    pred_sum = int(pred_row.sum())
    true_sum = int(true_row.sum())
    denom = pred_sum + true_sum
    if denom == 0:
        return 0.0
    tp = int(np.logical_and(pred_row, true_row).sum())
    return 2.0 * tp / denom


def compute_f1_ssr_bert_per_row(predictions: Sequence[str], references: Sequence[str]) -> list[float | None]:
    if not predictions:
        return []
    from .ssr_bert import _load, predict_label_matrix

    if _load() is None:
        return [None] * len(predictions)
    try:
        pred_matrix = predict_label_matrix(predictions)
        ref_matrix = predict_label_matrix(references)
    except Exception as error:  # pragma: no cover - optional backend may fail at runtime.
        # Mirrors ssr_bert.score's graceful degradation via compute_optional_metric -- see the
        # BERTScore try/except above for why this matters for the ablation grid's long run.
        LOGGER.warning("Skipping per-row F1-SRR-BERT for this batch after backend failure: %s", error)
        return [None] * len(predictions)
    return [_per_row_label_f1(pred_matrix[i], ref_matrix[i]) for i in range(len(predictions))]


def compute_category_score_per_row(predictions: Sequence[str], references: Sequence[str]) -> list[float | None]:
    if not predictions:
        return []
    from .ssr_bert import _load, label_matrix_to_category, predict_label_matrix

    if _load() is None:
        return [None] * len(predictions)
    try:
        pred_cat = label_matrix_to_category(predict_label_matrix(predictions))
        ref_cat = label_matrix_to_category(predict_label_matrix(references))
    except Exception as error:  # pragma: no cover - optional backend may fail at runtime.
        LOGGER.warning("Skipping per-row Category Score for this batch after backend failure: %s", error)
        return [None] * len(predictions)
    return [_per_row_label_f1(pred_cat[i], ref_cat[i]) for i in range(len(predictions))]


def compute_clinical_metrics_per_row(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    task: str,
    radgraph_model_type: str = "radgraph-xl",
    radgraph_reward_level: str = "all",
) -> dict[str, list[float | None]]:
    n = len(predictions)
    metrics: dict[str, list[float | None]] = {}
    metrics.update(
        compute_radgraph_metrics_per_row(
            predictions, references, model_type=radgraph_model_type, reward_level=radgraph_reward_level
        )
    )
    metrics["f1_ssr_bert"] = (
        compute_f1_ssr_bert_per_row(predictions, references) if task in ("findings", "impression") else [None] * n
    )
    metrics["category_score"] = (
        compute_category_score_per_row(predictions, references) if task == "findings" else [None] * n
    )
    return metrics
