"""Text metric computation helpers."""

from __future__ import annotations

import logging
from functools import lru_cache
from numbers import Number
from typing import Sequence

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

import evaluate

from .sections import normalize_report_text, normalize_report_texts


LOGGER = logging.getLogger(__name__)
_SMOOTHING_FUNCTION = SmoothingFunction().method7

try:
    from bert_score import score as bert_score_score
except ImportError:  # pragma: no cover - exercised only when the optional dependency is missing.
    bert_score_score = None


@lru_cache(maxsize=1)
def _load_rouge_metric():
    return evaluate.load("rouge")


# --- parallel ROUGE-L --------------------------------------------------------
# ``rouge.compute`` is pure-CPU, single-threaded, and 13% of total scoring runtime (measured:
# 3.3 ms/row of 26.2 ms/row). The box has 24 cores and the GPU metrics leave them idle, so the
# per-row path fans out across processes. Workers run the *same* ``rouge.compute`` call on a
# contiguous slice and the slices are re-concatenated in order, so results are identical to the
# serial path -- rouge_score has no cross-row state.
#: Measured 722 MB RSS per worker (``evaluate`` + datasets/pyarrow/nltk; torch is deliberately kept
#: out, see src/rouge_worker.py). 12 workers = 8.5 GB on top of a parent already holding
#: RadGraph-XL + roberta-large + SRR-BERT, which is too tight on this 30 GB box for an unattended
#: run. 8 costs ~10% of the speedup for 2.9 GB of headroom.
_ROUGE_WORKERS = 8
#: Below this, pool dispatch costs more than it saves; the serial path is used instead. Also keeps
#: the unit tests (which score a handful of rows) off the pool entirely.
_ROUGE_PARALLEL_MIN_ROWS = 512
#: More chunks than workers, so a slice of unusually long reports cannot stall one worker while
#: the rest idle.
_ROUGE_CHUNKS_PER_WORKER = 4
#: Refuse to start the pool when host RAM is already tight. The parent holds RadGraph-XL,
#: roberta-large and SRR-BERT; adding workers on top of that is what OOM-killed the first attempt.
_ROUGE_MIN_AVAILABLE_GB = 6.0

_ROUGE_POOL = None


def _available_ram_gb() -> float:
    """MemAvailable in GiB, or ``inf`` when it cannot be read (never block the pool on a parse)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:  # pragma: no cover - non-Linux or restricted /proc
        pass
    return float("inf")


def _rouge_l_chunk(payload: tuple[list[str], list[str]]) -> list[float]:
    """Serial ROUGE-L for one slice. The *worker* entry point lives in ``src.rouge_worker``."""
    from ..rouge_worker import rouge_l_chunk

    return rouge_l_chunk(payload)


def _get_rouge_pool():
    """Lazily build a persistent worker pool, or return ``None`` to fall back to serial.

    ``spawn`` (not fork) is mandatory: the parent has CUDA initialized by the time ROUGE runs, and
    forking a CUDA-initialized process is undefined behaviour. Persisting the pool means each
    worker pays ``evaluate.load("rouge")`` once for the whole grid rather than once per cell.

    The target function deliberately lives in ``src.rouge_worker``: a spawn worker re-imports the
    module its target is defined in, and importing anything under ``src.eval`` drags in torch +
    RadGraph (~1 GB per worker). That is what exhausted RAM and got the pool killed the first time.
    """
    global _ROUGE_POOL
    if _ROUGE_POOL is not None:
        return _ROUGE_POOL
    available = _available_ram_gb()
    if available < _ROUGE_MIN_AVAILABLE_GB:
        LOGGER.warning(
            "only %.1f GB RAM available (< %.1f GB); running ROUGE-L serially instead of risking an "
            "OOM-killed worker pool",
            available,
            _ROUGE_MIN_AVAILABLE_GB,
        )
        return None
    try:
        import atexit
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        from ..rouge_worker import rouge_l_chunk  # noqa: F401  -- fail here, not inside a worker

        _ROUGE_POOL = ProcessPoolExecutor(
            max_workers=_ROUGE_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
        )
        atexit.register(_shutdown_rouge_pool)
    except Exception as error:  # pragma: no cover - environment-dependent (no /dev/shm, etc.)
        LOGGER.warning("ROUGE worker pool unavailable, falling back to serial: %s", error)
        _ROUGE_POOL = None
    return _ROUGE_POOL


def _shutdown_rouge_pool() -> None:
    """Tear the pool down. Registered via atexit -- note SIGKILL skips it and orphans workers."""
    global _ROUGE_POOL
    pool, _ROUGE_POOL = _ROUGE_POOL, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def rouge_l_f1_per_row(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    """Per-row ROUGE-L F1 over **already normalized** text, parallel above the row threshold."""
    n = len(predictions)
    if n == 0:
        return []
    predictions = list(predictions)
    references = list(references)

    pool = _get_rouge_pool() if n >= _ROUGE_PARALLEL_MIN_ROWS else None
    if pool is None:
        return _rouge_l_chunk((predictions, references))

    n_chunks = min(n, _ROUGE_WORKERS * _ROUGE_CHUNKS_PER_WORKER)
    size = -(-n // n_chunks)  # ceil, so the last chunk is the short one
    payloads = [(predictions[i : i + size], references[i : i + size]) for i in range(0, n, size)]
    from ..rouge_worker import rouge_l_chunk

    try:
        return [value for chunk in pool.map(rouge_l_chunk, payloads) for value in chunk]
    except Exception as error:  # pragma: no cover - a dead pool must not lose the whole cell.
        # Drop the pool so later cells go straight to serial rather than re-paying a failed
        # parallel attempt on every call -- the retry is pure waste when the cause is OOM.
        LOGGER.warning("Parallel ROUGE-L failed, falling back to serial for the rest of this run: %s", error)
        _shutdown_rouge_pool()
        return _rouge_l_chunk((predictions, references))


def _tokenize_for_bleu(texts: Sequence[str]) -> list[list[str]]:
    return [normalize_report_text(text).split() for text in texts]


def compute_bleu(predictions: Sequence[str], references: Sequence[str]) -> float | None:
    if not predictions:
        return None

    tokenized_predictions = _tokenize_for_bleu(predictions)
    tokenized_references = [[reference_tokens] for reference_tokens in _tokenize_for_bleu(references)]
    return float(corpus_bleu(tokenized_references, tokenized_predictions, smoothing_function=_SMOOTHING_FUNCTION))


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float | None:
    if not predictions:
        return None

    rouge = _load_rouge_metric()
    rouge_scores = rouge.compute(
        predictions=normalize_report_texts(predictions),
        references=normalize_report_texts(references),
        use_aggregator=True,
        use_stemmer=True,
    )
    rouge_l = rouge_scores["rougeL"]
    if hasattr(rouge_l, "fmeasure"):
        return float(rouge_l.fmeasure)
    if isinstance(rouge_l, Number):
        return float(rouge_l)
    return None


def compute_meteor(predictions: Sequence[str], references: Sequence[str]) -> float | None:
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:  # pragma: no cover - METEOR is optional.
        return None

    if not predictions:
        return None

    scores: list[float] = []
    for prediction, reference in zip(predictions, references, strict=False):
        try:
            scores.append(
                float(
                    meteor_score([normalize_report_text(reference).split()], normalize_report_text(prediction).split())
                )
            )
        except LookupError:
            LOGGER.warning("METEOR resources are unavailable; skipping METEOR.")
            return None

    return float(sum(scores) / len(scores)) if scores else None


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "roberta-large",
    lang: str = "en",
    batch_size: int = 16,
    device: str | None = None,
    rescale_with_baseline: bool = False,
) -> dict[str, float | str | bool | None]:
    result: dict[str, float | str | bool | None] = {
        "bertscore_available": False,
        "bertscore_backend": None,
        "bertscore_precision": None,
        "bertscore_recall": None,
        "bertscore_f1": None,
    }

    if not predictions:
        return result

    if bert_score_score is None:
        LOGGER.warning("bert_score is not installed; skipping BERTScore.")
        return result

    try:
        bert_score_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        precision, recall, f1 = bert_score_score(
            normalize_report_texts(predictions),
            normalize_report_texts(references),
            model_type=model_type,
            lang=lang,
            batch_size=batch_size,
            device=bert_score_device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
    except Exception as error:  # pragma: no cover - optional backend may fail at runtime.
        LOGGER.warning("Skipping BERTScore after backend failure: %s", error)
        return result

    result.update(
        {
            "bertscore_available": True,
            "bertscore_backend": model_type,
            "bertscore_precision": float(precision.mean().item()),
            "bertscore_recall": float(recall.mean().item()),
            "bertscore_f1": float(f1.mean().item()),
        }
    )
    return result


def compute_text_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    include_meteor: bool = False,
    bertscore_model_type: str = "roberta-large",
    bertscore_lang: str = "en",
    bertscore_batch_size: int = 16,
    bertscore_device: str | None = None,
    bertscore_rescale_with_baseline: bool = False,
) -> dict[str, float | str | bool | None]:
    metrics: dict[str, float | str | bool | None] = {
        "bleu": compute_bleu(predictions, references),
        "rougeL": compute_rouge_l(predictions, references),
    }

    if include_meteor:
        metrics["meteor"] = compute_meteor(predictions, references)

    metrics.update(
        compute_bertscore(
            predictions,
            references,
            model_type=bertscore_model_type,
            lang=bertscore_lang,
            batch_size=bertscore_batch_size,
            device=bertscore_device,
            rescale_with_baseline=bertscore_rescale_with_baseline,
        )
    )
    return metrics
