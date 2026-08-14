"""Helpers for optional metric backends."""

from __future__ import annotations

import importlib
import inspect
import logging
from numbers import Number
from typing import Any, Sequence


LOGGER = logging.getLogger(__name__)


def _load_optional_backend(module_candidates: Sequence[str]) -> tuple[Any, str] | None:
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        for attribute_name in ("score", "compute", "evaluate", "calculate"):
            backend = getattr(module, attribute_name, None)
            if callable(backend):
                return backend, module_name

    return None


def _invoke_optional_backend(backend: Any, predictions: Sequence[str], references: Sequence[str], task: str | None) -> Any:
    attempts = [
        lambda: backend(predictions=predictions, references=references, task=task),
        lambda: backend(predictions, references, task=task),
        lambda: backend(predictions=predictions, refs=references, task=task),
        lambda: backend(hyps=predictions, refs=references, task=task),
        lambda: backend(predictions, references),
        lambda: backend(hyps=predictions, refs=references),
    ]

    for attempt in attempts:
        try:
            return attempt()
        except TypeError:
            continue

    try:
        signature = inspect.signature(backend)
    except (TypeError, ValueError):
        return backend(predictions, references)

    keyword_arguments: dict[str, Any] = {}
    if "predictions" in signature.parameters:
        keyword_arguments["predictions"] = predictions
    if "references" in signature.parameters:
        keyword_arguments["references"] = references
    if "refs" in signature.parameters:
        keyword_arguments["refs"] = references
    if "hyps" in signature.parameters:
        keyword_arguments["hyps"] = predictions
    if task is not None and "task" in signature.parameters:
        keyword_arguments["task"] = task

    return backend(**keyword_arguments)


def _normalize_optional_metric_result(metric_name: str, result: Any) -> dict[str, float | str | bool | None]:
    normalized: dict[str, float | str | bool | None] = {
        f"{metric_name}_available": True,
        f"{metric_name}_backend": None,
        metric_name: None,
        f"{metric_name}_precision": None,
        f"{metric_name}_recall": None,
        f"{metric_name}_f1": None,
    }

    if isinstance(result, dict):
        scalar_value = result.get("score", result.get("value"))
        if isinstance(scalar_value, Number):
            normalized[metric_name] = float(scalar_value)

        for key, value in result.items():
            if isinstance(value, Number):
                if key in {"precision", "recall", "f1"}:
                    normalized[f"{metric_name}_{key}"] = float(value)
                else:
                    normalized[f"{metric_name}_{key}"] = float(value)
        return normalized

    if isinstance(result, (tuple, list)):
        numeric_values = [float(value) for value in result if isinstance(value, Number)]
        if len(numeric_values) == 1:
            normalized[metric_name] = numeric_values[0]
        elif len(numeric_values) >= 3:
            normalized[f"{metric_name}_precision"] = numeric_values[0]
            normalized[f"{metric_name}_recall"] = numeric_values[1]
            normalized[f"{metric_name}_f1"] = numeric_values[2]
        return normalized

    if isinstance(result, Number):
        normalized[metric_name] = float(result)

    return normalized


def compute_optional_metric(
    metric_name: str,
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    module_candidates: Sequence[str],
    task: str | None = None,
) -> dict[str, float | str | bool | None]:
    result: dict[str, float | str | bool | None] = {
        f"{metric_name}_available": False,
        f"{metric_name}_backend": None,
        metric_name: None,
        f"{metric_name}_precision": None,
        f"{metric_name}_recall": None,
        f"{metric_name}_f1": None,
    }

    if not predictions:
        return result

    backend = _load_optional_backend(module_candidates)
    if backend is None:
        LOGGER.warning("No backend found for optional metric '%s'.", metric_name)
        return result

    backend_callable, backend_module_name = backend
    try:
        backend_result = _invoke_optional_backend(backend_callable, predictions, references, task)
    except Exception as error:  # pragma: no cover - optional backends may fail at runtime.
        LOGGER.warning("Skipping optional metric '%s' after backend failure: %s", metric_name, error)
        return result

    result.update(_normalize_optional_metric_result(metric_name, backend_result))
    result[f"{metric_name}_available"] = True
    result[f"{metric_name}_backend"] = backend_module_name
    return result
