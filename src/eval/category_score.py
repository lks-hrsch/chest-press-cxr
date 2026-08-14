"""Category Score — organ-level structured-label agreement (SRRG; findings only).

Delbrouck et al., 2025 / C-SRRG (arXiv:2510.00428, Table 10) report a **findings-only**
"Category Score" over the 8 SRRG anatomical organ categories (P, A, H/M, O, L/A, C, M/C,
T/C/S) "for the correctness of organ-section headers". StructEval computes it by parsing
organ-section headers out of the *structured* report and scoring header presence — which
our *free-text* MIMIC-CXR reports do not contain.

So we compute the same organ-level correctness that the metric is meant to capture, in a
way that applies to free text: run SRR-BERT on the section, roll its 55 leaf labels up to
the 8 organ categories (:data:`src.eval.ssr_bert.ORGAN_CATEGORIES`), and report sklearn
*samples-averaged* precision / recall / F1 over per-report organ-category presence — the
same aggregation StructEval uses for its section-presence score. SRR-BERT inference is
shared with :mod:`src.eval.ssr_bert` (cached), so this adds no extra model pass.

The optional-backend loader (``src.eval.optional_metrics``) calls :func:`score`.
"""

from __future__ import annotations

from typing import Sequence

from .ssr_bert import (
    _load,
    _samples_avg,
    label_matrix_to_category,
    predict_label_matrix,
)


def score(
    predictions: Sequence[str],
    references: Sequence[str],
    task: str | None = None,
) -> dict[str, float | int | None]:
    """Category Score for a list of (prediction, reference) findings sections.

    Returns ``{"score", "precision", "recall", "f1", "n"}``; ``score`` == ``f1`` is picked
    up as ``category_score`` by the optional-backend normalizer.
    """
    if not predictions or _load() is None:
        return {"score": None, "precision": None, "recall": None, "f1": None, "n": 0}

    pred_categories = label_matrix_to_category(predict_label_matrix(predictions))
    ref_categories = label_matrix_to_category(predict_label_matrix(references))
    precision, recall, f1 = _samples_avg(ref_categories, pred_categories)
    return {"score": f1, "precision": precision, "recall": recall, "f1": f1, "n": len(predictions)}
