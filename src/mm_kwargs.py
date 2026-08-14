"""Multimodal processor-kwarg translation shared by training and inference.

Tiling lives once in ``src/model_registry.py`` in vLLM's vocabulary
(``mm_processor_kwargs``).  vLLM and the transformers image processor name the same quantity
differently, so it has to be translated before it reaches a transformers processor:

    InternVL / Lingshu   ``max_dynamic_patch=N``  ->  ``crop_to_patches=True, max_patches=N``

Training already did this (``src/train/common.mm_to_hf_image_kwargs``).  **Inference did not** — it
passed ``max_dynamic_patch`` straight through to vLLM, and because ``Lingshu-I-8B`` resolves to the
*transformers* ``InternVLForConditionalGeneration`` backend (not vLLM's native ``internvl.py``, which
is the only path that consumes ``max_dynamic_patch``), the kwarg was silently dropped:

    {}                                          -> 3335 tok/img
    {'max_dynamic_patch': 4}                    -> 3335 tok/img   (ignored)
    {'crop_to_patches': True, 'max_patches': 4} -> 1287 tok/img   (intended)

So Lingshu was **trained at 4 tiles and served at 13** — 2.6x the intended prefill and KV cost.  An
ignored processor kwarg is only a warning, which is exactly why this survived unnoticed; see
:func:`assert_image_kwargs_supported` for the fail-fast guard that now makes it an error.
"""

from __future__ import annotations

from typing import Any


def mm_to_hf_image_kwargs(mm_processor_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Translate a registry ``mm_processor_kwargs`` entry to transformers processor image kwargs.

    Returns ``{}`` when there is nothing to translate (fixed-resolution backbones / no kwargs).
    """
    mm = mm_processor_kwargs or {}
    max_patches = mm.get("max_dynamic_patch")
    if max_patches:
        return {"crop_to_patches": True, "max_patches": max_patches}
    return {}


def _valid_image_kwarg_names(image_processor: Any) -> set[str]:
    """Best-effort set of image-kwarg names the processor actually honours.

    transformers exposes this as a ``valid_kwargs`` TypedDict (annotations) and/or a
    ``_valid_kwargs_names`` sequence, depending on version.  Returns an empty set when neither is
    present, which callers must treat as "cannot verify" rather than "nothing is valid".
    """
    names: set[str] = set()
    valid = getattr(type(image_processor), "valid_kwargs", None) or getattr(image_processor, "valid_kwargs", None)
    annotations = getattr(valid, "__annotations__", None)
    if annotations:
        names |= set(annotations)
    explicit = getattr(type(image_processor), "_valid_kwargs_names", None) or getattr(
        image_processor, "_valid_kwargs_names", None
    )
    if explicit:
        names |= set(explicit)
    return names


def assert_image_kwargs_supported(image_processor: Any, image_kwargs: dict[str, Any], model_name: str) -> None:
    """Raise if the processor would silently ignore any of ``image_kwargs``.

    transformers only *warns* on an unknown processor kwarg ("Keyword argument `X` is not a valid
    argument for this processor and will be ignored"), so a mistranslated tiling setting degrades
    silently into a train/serve mismatch.  Turn that into a hard failure.

    No-ops when the processor exposes no kwarg schema — we refuse to guess and block a valid run.
    """
    if not image_kwargs:
        return
    valid = _valid_image_kwarg_names(image_processor)
    if not valid:
        return
    unsupported = sorted(set(image_kwargs) - valid)
    if unsupported:
        raise ValueError(
            f"{model_name}: image processor {type(image_processor).__name__} would IGNORE "
            f"{unsupported} — it accepts {sorted(valid)}. This silently changes the served image "
            "tiling and desyncs inference from training; fix the translation in src/mm_kwargs.py."
        )
