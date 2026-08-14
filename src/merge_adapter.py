"""Merge a trained LoRA adapter into its base model for vLLM inference.

All trained adapters are served as MERGED full checkpoints — one uniform path for every
architecture.  (vLLM 0.22 only supports *runtime* LoRA for a subset of archs; merging the
adapter into the base weights sidesteps that entirely and vLLM serves the merged checkpoint
like any other model.)  The merged checkpoint is then served via:

    python -m src.inference --model <id> --variant <v>     # auto-finds <slug>/<variant>_merged

Usage
-----
    # variant-derived adapter (benchmark/checkpoints/<slug>/<variant>_adapter)
    python -m src.merge_adapter --model google/medgemma-4b-it --variant reports

    # explicit adapter dir (e.g. a raw checkpoint when no final save exists)
    python -m src.merge_adapter --model llava-hf/llava-v1.6-mistral-7b-hf \
        --adapter-path benchmark/checkpoints/llava-hf_llava-v1.6-mistral-7b-hf/reports_adapter/checkpoint-10374 \
        --variant reports

Idempotent: an existing merged checkpoint is reused unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from .adapters import (
    ADAPTER_VARIANTS,
    CHECKPOINTS_ROOT_DEFAULT,
    adapter_dir_for,
    infer_variant,
    merged_dir_for,
    read_adapter_config,
)
from .helpers import sanitize_slug


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def _assert_module_finite(model, *, context: str) -> None:
    """Raise if any parameter of the merged model is non-finite (NaN/inf).

    A diverged adapter merges into a full-size checkpoint that *looks* healthy — correct file size,
    valid safetensors — but is entirely NaN. File size is not a health signal; only a finiteness scan
    catches it. Run one before writing so a poisoned checkpoint fails loudly at merge time instead of
    silently entering the inference sweep. Scans parameters only — a LoRA merge touches linear weights,
    not buffers — so the cost is trivial next to the multi-GB save.
    """
    import torch

    bad = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        head = ", ".join(bad[:8])
        more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
        raise ValueError(
            f"merged checkpoint for {context} contains {len(bad)} non-finite parameter tensor(s) "
            f"(NaN/inf) — the adapter almost certainly diverged during training; refusing to write a "
            f"poisoned checkpoint. First offenders: {head}{more}"
        )


def _local_checkpoint_weight_map(model_name: str) -> dict[str, str] | None:
    """Return ``{weight_key: absolute_shard_path}`` for ``model_name``'s cached raw checkpoint.

    Local-cache-only (no network, no token needed) via ``huggingface_hub.try_to_load_from_cache``,
    which resolves "main" the same way ``from_pretrained`` already did and respects the
    ``HF_HOME``/``HUGGINGFACE_HUB_CACHE`` pin from ``.hpc.env`` on Capella. Handles both a sharded
    original (``model.safetensors.index.json``) and a single-file original. Returns ``None`` if
    neither is cached — the caller treats that as "skip the graft check", not a hard failure.
    """
    from huggingface_hub import try_to_load_from_cache

    index_path = try_to_load_from_cache(repo_id=model_name, filename="model.safetensors.index.json")
    if isinstance(index_path, str):
        with open(index_path) as fh:
            weight_map: dict[str, str] = json.load(fh)["weight_map"]
        shard_dir = os.path.dirname(index_path)
        return {key: os.path.join(shard_dir, shard) for key, shard in weight_map.items()}

    single_path = try_to_load_from_cache(repo_id=model_name, filename="model.safetensors")
    if isinstance(single_path, str):
        from safetensors import safe_open

        with safe_open(single_path, framework="pt") as fh:
            return {key: single_path for key in fh.keys()}

    return None


def _graft_weights_into_checkpoint(original_map: dict[str, str], missing_keys: list[str], checkpoint_file: str) -> None:
    """Copy ``missing_keys`` verbatim from their source shards (``original_map``) into
    ``checkpoint_file``, casting dtype to match the checkpoint's existing tensors.

    safetensors has no in-place "append" API, so this reads the full existing checkpoint into
    memory, adds the missing tensors, and rewrites it whole via ``safetensors.torch.save_file``.
    Assumes a single-file ``checkpoint_file`` (true for every registry model's merged output today,
    up to ~16GB) — a hypothetical sharded destination would need to patch the correct shard via a
    sibling ``model.safetensors.index.json`` instead; not implemented since no current model needs it.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(checkpoint_file, framework="pt") as fh:
        tensors = {key: fh.get_tensor(key) for key in fh.keys()}
        metadata = fh.metadata()
    target_dtype = next(iter(tensors.values())).dtype  # merged checkpoints are dtype-uniform (bf16)

    for key in missing_keys:
        with safe_open(original_map[key], framework="pt") as fh:
            tensor = fh.get_tensor(key)
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"grafted tensor {key!r} from the original checkpoint at {original_map[key]} "
                "contains non-finite values — refusing to write it into the merged checkpoint"
            )
        tensors[key] = tensor

    save_file(tensors, checkpoint_file, metadata=metadata)


def _graft_missing_base_weights(model_name: str, output_dir: str) -> None:
    """Graft into ``output_dir`` any weight key the ORIGINAL ``model_name`` checkpoint has that the
    just-saved merged checkpoint doesn't (generic — not architecture-specific).

    Why this is needed: some architectures (currently: Gemma 3n/4's ``gemma-4-E2B-it``, whose
    ``Gemma4TextAttention`` skips allocating ``k_norm``/``k_proj``/``v_proj`` for KV-shared layers)
    publish MORE per-layer tensors on the Hub than the live ``nn.Module`` graph has slots for.
    ``transformers``' own ``from_pretrained`` silently drops the unmatched checkpoint keys at load
    time (not an error), so ``model.save_pretrained()`` faithfully omits them from the merged output
    too. vLLM's own from-scratch reimplementation of the same architecture allocates a parameter per
    layer regardless of sharing, so its weight loader then raises "Following weights were not
    initialized from checkpoint" against the merged checkpoint. A safe no-op everywhere else — the
    diff is empty for every other registry backbone.

    Deliberately reads ``merged_keys`` back from the just-written ``model.safetensors`` file rather
    than from ``model.state_dict()`` pre-save: some architectures' live ``state_dict()`` key names do
    NOT match what ``save_pretrained()`` actually writes to disk (e.g. a stripped/renamed shared
    prefix), which would otherwise misreport the entire checkpoint as "missing" and silently
    overwrite every trained weight with the untrained base checkpoint — reading the real file is the
    only way to compare against what will actually be loaded.
    """
    checkpoint_file = os.path.join(output_dir, "model.safetensors")
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(
            f"expected a single-file {checkpoint_file} after save_pretrained(); sharded merged "
            "output is not supported by the graft step yet (see _graft_weights_into_checkpoint)"
        )

    original_map = _local_checkpoint_weight_map(model_name)
    if original_map is None:
        LOGGER.warning(
            "could not locate a local model.safetensors[.index.json] for %s in the HF cache — "
            "skipping the post-merge weight-graft check (if this model needed it, vLLM will still "
            "raise loudly at load time, same failure mode as before this check existed)",
            model_name,
        )
        return

    from safetensors import safe_open

    with safe_open(checkpoint_file, framework="pt") as fh:
        merged_keys = set(fh.keys())

    missing = sorted(set(original_map) - merged_keys)
    if not missing:
        LOGGER.info("post-merge graft check: no weight keys missing for %s (nothing to graft)", model_name)
        return

    sample = ", ".join(missing[:8]) + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
    LOGGER.info(
        "post-merge graft check: %d weight key(s) exist in the original %s checkpoint but not in "
        "the merged module graph — grafting verbatim: %s",
        len(missing),
        model_name,
        sample,
    )
    _graft_weights_into_checkpoint(original_map, missing, checkpoint_file)
    LOGGER.info("grafted %d weight key(s) into %s", len(missing), checkpoint_file)


def merge_adapter(model_name: str, adapter_path: str, output_dir: str, *, force: bool = False) -> str:
    """Merge ``adapter_path`` into base ``model_name`` and save a full checkpoint to ``output_dir``.

    Returns ``output_dir``.  Heavy imports (torch/transformers/peft) are deferred so that
    importing this module stays cheap for callers that only need the path helpers.
    """
    if os.path.isfile(os.path.join(output_dir, "config.json")) and not force:
        LOGGER.info("merged checkpoint already exists: %s (use --force to rebuild)", output_dir)
        return output_dir

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from .settings import HF_ACCESS_TOKEN

    LOGGER.info("loading base %s (cpu, eager attn) for merge", model_name)
    # eager attn avoids importing flash-attn during a CPU-side load; attention is never run here.
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN,
    )
    LOGGER.info("applying adapter %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    LOGGER.info("merging adapter into base weights (merge_and_unload)")
    model = model.merge_and_unload()

    LOGGER.info("scanning merged weights for NaN/inf")
    _assert_module_finite(model, context=f"{model_name} + {adapter_path}")

    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info("saving merged checkpoint to %s", output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    del model  # free the live module graph before the graft step reloads the saved file into RAM

    # Some architectures (e.g. gemma-4-E2B-it's KV-sharing) publish more per-layer weights on the
    # Hub than the live module graph has slots for; restore any such keys before vLLM ever sees this
    # checkpoint. A verified no-op for every other registry backbone.
    _graft_missing_base_weights(model_name, output_dir)

    # vLLM needs the processor/tokenizer alongside the weights; take them from the base id.
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=HF_ACCESS_TOKEN)
    processor.save_pretrained(output_dir)
    LOGGER.info("merged checkpoint written to %s", output_dir)

    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model for vLLM serving")
    parser.add_argument("--model", type=str, required=True, help="Base HuggingFace model ID")
    parser.add_argument("--variant", type=str, choices=list(ADAPTER_VARIANTS), help="Training variant")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Explicit adapter dir; overrides the --variant-derived path. Requires --variant for naming.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override the merged output dir.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the merged checkpoint exists.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    slug = sanitize_slug(args.model)

    variant = args.variant or (infer_variant(args.adapter_path) if args.adapter_path else None)
    if variant is None:
        parser.error("--variant is required (could not infer it from --adapter-path)")

    adapter_path = args.adapter_path or adapter_dir_for(CHECKPOINTS_ROOT_DEFAULT, slug, variant)
    config = read_adapter_config(adapter_path)
    if config is None:
        parser.error(
            f"no adapter_config.json in {adapter_path}\n"
            "  - check the path, or\n"
            "  - if only checkpoint-N/ subdirs exist, pass --adapter-path <dir>/checkpoint-N --variant <v>"
        )
    base_in_adapter = config.get("base_model_name_or_path")
    if base_in_adapter and base_in_adapter != args.model:
        parser.error(f"adapter was trained on {base_in_adapter!r} but --model is {args.model!r} — base mismatch")

    output_dir = args.output_dir or merged_dir_for(CHECKPOINTS_ROOT_DEFAULT, slug, variant)
    merge_adapter(args.model, adapter_path, output_dir, force=args.force)


if __name__ == "__main__":
    main()
