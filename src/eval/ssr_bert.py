"""F1-SRR-BERT clinical metric (SRRG / StructEval; Delbrouck et al., 2025).

This is a self-contained reimplementation of the **F1-SRR-BERT** structured-label metric
used by "Automated Structured Radiology Report Generation" (Delbrouck et al., 2025,
arXiv:2505.24223) and reused by "Automated Structured Radiology Report Generation with
Rich Clinical Context" (C-SRRG, arXiv:2510.00428). The reference implementation is
``jbdel/StructEval`` (not on PyPI; pins ``transformers==4.39`` plus the heavy StanfordAIMI
metric stack, which conflicts with this project's ``transformers 5.x`` env), so we mirror
its model + math directly instead of installing it.

Model / logic (identical to ``StructEval.StructBert``):
    * checkpoint : ``StanfordAIMI/SRR-BERT-Leaves`` — ``BertForSequenceClassification`` with
      55 multi-label leaf disease labels.
    * tokenizer  : ``microsoft/BiomedVLP-CXR-BERT-general`` (CXR-BERT, Boecking et al., 2022).
    * decision   : multi-label ``sigmoid(logits) > 0.5`` (fixed threshold, no tuning).

**F1-SRR-BERT** = run SRR-BERT on the prediction and the reference text, turn each into a
55-dim multi-hot label vector, and report sklearn *samples-averaged* precision / recall /
F1 (``classification_report(y_true=ref, y_pred=pred)['samples avg']``), matching StructEval.

**Free-text adaptation.** StructEval targets the SRRG *structured* format (findings under
organ-section headers with bulleted utterances; numbered impressions) and predicts per
utterance. This project generates *free-text* MIMIC-CXR reports (no organ headers), so we
split a section into sentences, predict per sentence, and OR-merge the per-sentence label
vectors into one vector per report (``np.any`` — exactly StructEval's ``aligned=False``
section-merge), then take the same samples-averaged F1 across reports. Same model, same
55-label taxonomy, same aggregation; only the utterance segmentation differs (sentences
instead of the structured format's explicit utterances).

The optional-backend loader (``src.eval.optional_metrics``) calls :func:`score`.
:mod:`src.eval.category_score` reuses :func:`predict_label_matrix` +
:func:`label_matrix_to_category` so SRR-BERT runs once per (text-list) even though both
metrics are computed for the findings task.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Sequence

import numpy as np


LOGGER = logging.getLogger(__name__)

MODEL_ID = "StanfordAIMI/SRR-BERT-Leaves"
TOKENIZER_ID = "microsoft/BiomedVLP-CXR-BERT-general"
THRESHOLD = 0.5
MAX_LENGTH = 512
_BATCH_SIZE = 64

# 55 leaf labels in checkpoint index order (StructEval structeval/leaves_mapping.json).
LEAF_LABELS: tuple[str, ...] = (
    "No Finding",  # 0
    "Lung Lesion",  # 1
    "Edema",  # 2
    "Pneumonia",  # 3
    "Atelectasis",  # 4
    "Aspiration",  # 5
    "Lung collapse",  # 6
    "Perihilar airspace opacity",  # 7
    "Air space opacity–multifocal",  # 8
    "Mass/Solitary lung mass",  # 9
    "Nodule/Solitary lung nodule",  # 10
    "Cavitating mass with content",  # 11
    "Cavitating masses",  # 12
    "Emphysema",  # 13
    "Fibrosis",  # 14
    "Pulmonary congestion",  # 15
    "Hilar lymphadenopathy",  # 16
    "Bronchiectasis",  # 17
    "Simple pneumothorax",  # 18
    "Loculated pneumothorax",  # 19
    "Tension pneumothorax",  # 20
    "Simple pleural effusion",  # 21
    "Loculated pleural effusion",  # 22
    "Pleural scarring",  # 23
    "Hydropneumothorax",  # 24
    "Pleural Other",  # 25
    "Cardiomegaly",  # 26
    "Pericardial effusion",  # 27
    "Inferior mediastinal mass",  # 28
    "Superior mediastinal mass",  # 29
    "Tortuous Aorta",  # 30
    "Calcification of the Aorta",  # 31
    "Enlarged pulmonary artery",  # 32
    "Hernia",  # 33
    "Pneumomediastinum",  # 34
    "Tracheal deviation",  # 35
    "Acute humerus fracture",  # 36
    "Acute rib fracture",  # 37
    "Acute clavicle fracture",  # 38
    "Acute scapula fracture",  # 39
    "Compression fracture",  # 40
    "Shoulder dislocation",  # 41
    "Subcutaneous Emphysema",  # 42
    "Suboptimal central line",  # 43
    "Suboptimal endotracheal tube",  # 44
    "Suboptimal nasogastric tube",  # 45
    "Suboptimal pulmonary arterial catheter",  # 46
    "Pleural tube",  # 47
    "PICC line",  # 48
    "Port catheter",  # 49
    "Pacemaker",  # 50
    "Implantable defibrillator",  # 51
    "LVAD",  # 52
    "Intraaortic balloon pump",  # 53
    "Pneumoperitoneum",  # 54
)
NUM_LABELS = len(LEAF_LABELS)

# --- Category Score taxonomy -------------------------------------------------------------
# The paper's Category Score is an *organ-level* score over the 8 SRRG anatomical categories
# (StructEval ``VALID_ORGANS``; C-SRRG Table 10: P, A, H/M, O, L/A, C, M/C, T/C/S). StructEval
# derives them by parsing organ-section HEADERS, which free-text reports do not have, so we
# roll the 55 leaf labels up to their organ category with this curated map (each leaf has an
# unambiguous anatomical home, and the SRRG structured format files it under exactly this
# header). Order matches ``VALID_ORGANS``.
ORGAN_CATEGORIES: tuple[str, ...] = (
    "Lungs and Airways",
    "Pleura",
    "Cardiovascular",
    "Hila and Mediastinum",
    "Tubes, Catheters, and Support Devices",
    "Musculoskeletal and Chest Wall",
    "Abdominal",
    "Other",
)

# leaf index -> organ category index (into ORGAN_CATEGORIES).
_ORGAN_LEAF_INDICES: dict[int, tuple[int, ...]] = {
    0: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17),  # Lungs and Airways
    1: (18, 19, 20, 21, 22, 23, 24, 25),  # Pleura
    2: (26, 27, 30, 31, 32),  # Cardiovascular
    3: (16, 28, 29, 34, 35),  # Hila and Mediastinum
    4: (43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53),  # Tubes, Catheters, and Support Devices
    5: (36, 37, 38, 39, 40, 41, 42),  # Musculoskeletal and Chest Wall
    6: (33, 54),  # Abdominal
    7: (0,),  # Other (No Finding — so a normal report matches a normal reference)
}
# [NUM_LABELS x len(ORGAN_CATEGORIES)] boolean membership matrix.
_LEAF_TO_ORGAN = np.zeros((NUM_LABELS, len(ORGAN_CATEGORIES)), dtype=bool)
for _organ_idx, _leaf_idxs in _ORGAN_LEAF_INDICES.items():
    for _leaf_idx in _leaf_idxs:
        _LEAF_TO_ORGAN[_leaf_idx, _organ_idx] = True
assert _LEAF_TO_ORGAN.sum() == NUM_LABELS, "every leaf label must map to exactly one organ category"


@lru_cache(maxsize=1)
def _load():
    """Load SRR-BERT once. Returns ``(model, tokenizer, device)`` or ``None`` if unavailable."""
    try:
        import torch
        from transformers import AutoTokenizer, BertForSequenceClassification
    except ImportError as error:  # pragma: no cover - torch/transformers are core deps.
        LOGGER.warning("SRR-BERT unavailable (missing torch/transformers): %s", error)
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(MODEL_ID, num_labels=NUM_LABELS)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as error:  # pragma: no cover - network/model-load failure.
        LOGGER.warning("Failed to load SRR-BERT (%s) / tokenizer (%s): %s", MODEL_ID, TOKENIZER_ID, error)
        return None
    return model, tokenizer, device


# Split whitespace-normalized (single-line) section text into sentence-like utterances.
# Text reaching the metric has already been run through ``normalize_report_text`` (newlines
# collapsed to spaces), so we segment on sentence punctuation, avoiding splits inside
# decimals ("1.5 cm") by requiring the delimiter to follow an alphanumeric.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[A-Za-z0-9])[.;?!]+(?:\s+|$)")
_LEADING_MARKER_RE = re.compile(r"^[\s\-\*•\d.)]+")


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences: list[str] = []
    for chunk in _SENTENCE_SPLIT_RE.split(text):
        chunk = _LEADING_MARKER_RE.sub("", chunk).strip()
        if chunk:
            sentences.append(chunk)
    # A section with no sentence-ending punctuation still gets scored as one utterance.
    if not sentences:
        cleaned = _LEADING_MARKER_RE.sub("", text).strip()
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _predict_sentences(sentences: Sequence[str]) -> np.ndarray:
    """Multi-label predict for a flat list of sentences -> ``[len(sentences), NUM_LABELS]`` int8."""
    loaded = _load()
    if loaded is None or not sentences:
        return np.zeros((len(sentences), NUM_LABELS), dtype=np.int8)
    import torch

    model, tokenizer, device = loaded
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sentences), _BATCH_SIZE):
            batch = list(sentences[start : start + _BATCH_SIZE])
            enc = tokenizer(
                batch,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits
            preds = (torch.sigmoid(logits) > THRESHOLD).to(torch.int8).cpu().numpy()
            rows.append(preds)
    return np.concatenate(rows, axis=0)


@lru_cache(maxsize=16)
def _predict_matrix_cached(texts: tuple[str, ...]) -> np.ndarray:
    """Per-report merged label matrix ``[len(texts), NUM_LABELS]`` (cached by text tuple).

    Each report is split into sentences, predicted per sentence, and OR-merged (``np.any``)
    into a single multi-hot vector. Empty reports yield an all-zero row. Cached so the
    findings-task ``f1_ssr_bert`` and ``category_score`` calls share one SRR-BERT pass.
    """
    per_report_sentences = [_split_sentences(text) for text in texts]
    flat = [sentence for sentences in per_report_sentences for sentence in sentences]
    flat_preds = _predict_sentences(flat)

    matrix = np.zeros((len(texts), NUM_LABELS), dtype=np.int8)
    cursor = 0
    for row_idx, sentences in enumerate(per_report_sentences):
        count = len(sentences)
        if count:
            matrix[row_idx] = np.any(flat_preds[cursor : cursor + count], axis=0).astype(np.int8)
            cursor += count
    return matrix


def predict_label_matrix(texts: Sequence[str]) -> np.ndarray:
    """Public, cache-friendly wrapper returning the per-report 55-label matrix."""
    return _predict_matrix_cached(tuple(texts))


def label_matrix_to_category(label_matrix: np.ndarray) -> np.ndarray:
    """Roll a ``[N, NUM_LABELS]`` leaf matrix up to ``[N, len(ORGAN_CATEGORIES)]`` presence."""
    return (label_matrix.astype(bool) @ _LEAF_TO_ORGAN).astype(np.int8)


def _samples_avg(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """sklearn samples-averaged (precision, recall, f1) over multi-label indicator matrices."""
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    avg = report["samples avg"]
    return float(avg["precision"]), float(avg["recall"]), float(avg["f1-score"])


def score(
    predictions: Sequence[str],
    references: Sequence[str],
    task: str | None = None,
) -> dict[str, float | int | None]:
    """F1-SRR-BERT for a list of (prediction, reference) report sections.

    Returns ``{"score", "precision", "recall", "f1", "n"}``. ``score`` == ``f1`` is the
    headline value picked up as ``f1_ssr_bert`` by the optional-backend normalizer.
    """
    if not predictions or _load() is None:
        return {"score": None, "precision": None, "recall": None, "f1": None, "n": 0}

    pred_matrix = predict_label_matrix(predictions)
    ref_matrix = predict_label_matrix(references)
    precision, recall, f1 = _samples_avg(ref_matrix, pred_matrix)
    return {"score": f1, "precision": precision, "recall": recall, "f1": f1, "n": len(predictions)}
