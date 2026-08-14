"""Clinical metric computation helpers."""

from __future__ import annotations

import logging
from typing import Sequence

from .optional_metrics import compute_optional_metric
from .sections import normalize_report_texts


LOGGER = logging.getLogger(__name__)

# transformers 5.x removed several methods that radgraph's bundled allennlp still calls.
# We patch PreTrainedTokenizerBase (the universal ancestor in transformers 5.x, including the new
# TokenizersBackend that BertTokenizer now inherits from) so that all tokenizer subclasses get
# these methods regardless of which branch of the class hierarchy they sit in.
try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

    # --- encode_plus ---
    def _encode_plus_compat(self, text, text_pair=None, **kwargs):
        if text_pair is not None:
            return self(text, text_pair=text_pair, **kwargs)
        return self(text, **kwargs)

    if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
        PreTrainedTokenizerBase.encode_plus = _encode_plus_compat  # type: ignore[method-assign]
    # Belt-and-suspenders: also patch the legacy subclasses in case they shadow the base.
    if not hasattr(PreTrainedTokenizer, "encode_plus"):
        PreTrainedTokenizer.encode_plus = _encode_plus_compat  # type: ignore[method-assign]
    if not hasattr(PreTrainedTokenizerFast, "encode_plus"):
        PreTrainedTokenizerFast.encode_plus = _encode_plus_compat  # type: ignore[method-assign]

    # --- build_inputs_with_special_tokens ---
    # Called by pretrained_transformer_indexer.py when re-adding [CLS]/[SEP] around each chunk
    # of a long sequence.  Matches BERT-style behaviour: [CLS] A [SEP] or [CLS] A [SEP] B [SEP].
    def _build_inputs_with_special_tokens_compat(self, token_ids_0, token_ids_1=None):
        cls_id = getattr(self, "cls_token_id", None)
        sep_id = getattr(self, "sep_token_id", None)
        bos_id = getattr(self, "bos_token_id", None)
        eos_id = getattr(self, "eos_token_id", None)
        start = [cls_id] if cls_id is not None else ([bos_id] if bos_id is not None else [])
        end = [sep_id] if sep_id is not None else ([eos_id] if eos_id is not None else [])
        if token_ids_1 is None:
            return start + list(token_ids_0) + end
        return start + list(token_ids_0) + end + list(token_ids_1) + end

    if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
        PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs_with_special_tokens_compat  # type: ignore[method-assign]
    if not hasattr(PreTrainedTokenizer, "build_inputs_with_special_tokens"):
        PreTrainedTokenizer.build_inputs_with_special_tokens = _build_inputs_with_special_tokens_compat  # type: ignore[method-assign]
    if not hasattr(PreTrainedTokenizerFast, "build_inputs_with_special_tokens"):
        PreTrainedTokenizerFast.build_inputs_with_special_tokens = _build_inputs_with_special_tokens_compat  # type: ignore[method-assign]
except ImportError:
    pass

try:
    from radgraph import F1RadGraph
except ImportError:  # pragma: no cover - exercised only when the optional dependency is missing.
    F1RadGraph = None


def compute_radgraph_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "radgraph-xl",
    reward_level: str = "all",
) -> dict[str, float | str | bool | None]:
    result: dict[str, float | str | bool | None] = {
        "radgraph_available": False,
        "radgraph_backend": None,
        "radgraph_rg_e": None,
        "radgraph_rg_er": None,
        "radgraph_rg_bar_er": None,
        "radgraph_f1": None,
    }

    if not predictions:
        return result

    if F1RadGraph is None:
        LOGGER.warning("radgraph is not installed; skipping F1RadGraph.")
        return result

    try:
        metric = F1RadGraph(reward_level=reward_level, model_type=model_type)
        mean_reward, _, _, _ = metric(hyps=normalize_report_texts(predictions), refs=normalize_report_texts(references))
        rg_e, rg_er, rg_bar_er = (float(value) for value in mean_reward)
    except Exception as error:  # pragma: no cover - optional backend may fail at runtime.
        LOGGER.warning("Skipping F1RadGraph after backend failure: %s", error)
        return result

    result.update(
        {
            "radgraph_available": True,
            "radgraph_backend": model_type,
            "radgraph_rg_e": rg_e,
            "radgraph_rg_er": rg_er,
            "radgraph_rg_bar_er": rg_bar_er,
            "radgraph_f1": rg_er,
        }
    )
    return result


def compute_clinical_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    task: str,
    radgraph_model_type: str = "radgraph-xl",
    radgraph_reward_level: str = "all",
) -> dict[str, float | str | bool | None]:
    metrics: dict[str, float | str | bool | None] = {}
    metrics.update(
        compute_radgraph_metrics(
            predictions, references, model_type=radgraph_model_type, reward_level=radgraph_reward_level
        )
    )
    if task in ("findings", "impression"):
        metrics.update(
            compute_optional_metric(
                "f1_ssr_bert",
                predictions,
                references,
                module_candidates=(
                    "src.eval.ssr_bert",
                    "src.eval.f1_ssr_bert",
                    "ssr_bert",
                    "f1_ssr_bert",
                    "f1_srr_bert",
                ),
                task=task,
            )
        )
    if task == "findings":
        metrics.update(
            compute_optional_metric(
                "category_score",
                predictions,
                references,
                module_candidates=("src.eval.category_score", "category_score"),
                task=task,
            )
        )
    return metrics
