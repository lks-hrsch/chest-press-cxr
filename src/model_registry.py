"""Shared model registry for inference and benchmarking."""

from __future__ import annotations

from typing import Any


# Each MODELS entry also carries a top-level `use_liger_kernel` bool (consumed by the train
# entry points as SFTConfig.use_liger_kernel).  It is False on the Gemma-based backbones because
# liger-kernel 0.8.0's gemma3 vision-tower patch crashes under transformers 5.x (see _GEMMA_TRAIN
# note below); True elsewhere.
#
# training_config is consumed by src/train/common.py; ignored by inference.py/evaluate.py.
# Keys:
#   attn_implementation  — passed to from_pretrained; "sdpa" for Gemma (fast, low-memory: no
#                          materialised seq×seq scores tensor; verified to train Gemma-3 with
#                          gradient checkpointing); "flash_attention_2" for LLaVA/Lingshu.
#   vision_tower_attr    — module attribute to freeze during LoRA training.
#   lora_target_modules  — list of linear layer names in the LM backbone to LoRA-adapt.
#   lora_mlp_modules     — MLP projection names appended to lora_target_modules by build_lora_config.
#                          Populated on all backbones (Source: "Fine-tuning VLMs with LoRA" guide —
#                          training attention+MLP outperforms attention-only on medical VLM tasks).
#
# Adapter INFERENCE is uniform across all archs: trained adapters are merged into the base
# (src/merge_adapter.py) and served as full checkpoints (src/inference.py --variant).
# vLLM 0.22 only supports runtime LoRA for some of these archs, so merging is the single safe path.

_MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]

_GEMMA_TRAIN = {
    "attn_implementation": "sdpa",  # fast + low-memory; avoids eager's full attention-scores tensor
    "vision_tower_attr": "vision_tower",
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_mlp_modules": _MLP_MODULES,
    # Keep LoRA on the language model only. Gemma 3n (gemma-4-E2B-it) wraps the vision AND
    # audio encoders' q/k/v/o/gate/up/down in Gemma4ClippableLinear, which PEFT cannot
    # inject — so exclude both towers (and the embed_* bridges) from target matching.
    "lora_exclude_modules": r".*(vision_tower|audio_tower|embed_vision|embed_audio).*",
    # NOTE: Liger is DISABLED for Gemma via the per-model `use_liger_kernel: False` flag on each
    # MODELS entry (not here).  liger-kernel 0.8.0's `apply_liger_kernel_to_gemma3` dereferences
    # `vision_tower.vision_model.post_layernorm` unconditionally, but transformers 5.x removed that
    # nested `.vision_model` attribute (hoisted onto SiglipVisionModel) → fatal AttributeError
    # before step 0.  `liger_kernel_config={"layer_norm": False}` does NOT prevent it (that flag
    # only gates the encoder-internal layernorms, not the top-level post_layernorm access).
}

_LLAVA_TRAIN = {
    "attn_implementation": "flash_attention_2",
    "vision_tower_attr": "vision_tower",
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_mlp_modules": _MLP_MODULES,
}

_LINGSHU_TRAIN = {
    "attn_implementation": "flash_attention_2",
    # Lingshu-I-8B loads as transformers' InternVL architecture: the encoder lives at
    # model.model.vision_tower (NOT .visual / .vision_model). The wrong attr left the tower
    # unfrozen AND let LoRA inject into the ViT attention (its layers expose q/k/v_proj),
    # since the derived exclude regex matched nothing.
    "vision_tower_attr": "vision_tower",
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_mlp_modules": _MLP_MODULES,
}

# Per-model SFT batch sizing, read by both training stages via model_registry.get_sft_batch().
# Shape: {stage: (per_device_batch_size, grad_accum_steps)} for stage in {"qba", "reports"}; both
# stages currently use the SAME tuple per model (Stage A and Stage B train at the same batch).
#   - per_device_batch_size : micro-batch per GPU
#   - grad_accum_steps      : micro-batches accumulated per optimizer step
#   - effective batch       = per_device_batch_size x 4 GPUs x grad_accum_steps
#     (Gemma 16x4x1 = 64; LLaVA-Med & Lingshu 8x4x1 = 32; LLaVA-v1.6 4x4x2 = 32 -- NOT uniform).
# Stage B (reports) is heavier than Stage A (qba) -- up to 4 images + ~16k context vs a single image --
# so the same tuple can fit qba yet OOM reports; smoke-test reports before a full sweep. Stage-B EVAL
# is sized in train_reports.py as per_device_batch_size // 2 (grad_accum 1); eval materialises the
# full fp32 logit tensor [batch x seq_len x vocab x 4B], so on the large-vocab backbones
# (Gemma 262k, Lingshu 152k) even the halved batch can OOM on a max-length sample.
# Measured-safe so far: medgemma reports per_device=4, qba per_device=8 (medgemma + lingshu). Values
# above those are unverified -- smoke-test before a full run, and lower a model's tuple if it OOMs.
_BATCH_GEMMA = {"qba": (16, 1), "reports": (8, 2)}  # 4B Gemma/MedGemma — SigLIP 896, 256 tok/img
_BATCH_LLAVA = {"qba": (8, 1), "reports": (8, 1)}  # LLaVA-Med — CLIP 336, 576 tok/img
_BATCH_LLAVA_NEXT = {"qba": (4, 2), "reports": (4, 2)}  # LLaVA-v1.6 — AnyRes up to ~2880 tok/img (heavy)
_BATCH_LINGSHU = {"qba": (8, 1), "reports": (8, 1)}  # Lingshu-8B — InternVL, up to 1280 tok/img

MODELS: list[dict[str, Any]] = [
    {
        "name": "google/gemma-3-4b-it",
        "mm_processor_kwargs": None,
        "training_config": _GEMMA_TRAIN,
        "sft_batch": _BATCH_GEMMA,
        "use_liger_kernel": False,  # liger-kernel 0.8.0 gemma3 vision patch crashes on transformers 5.x
    },
    {
        "name": "google/gemma-4-E2B-it",
        "mm_processor_kwargs": None,
        "training_config": _GEMMA_TRAIN,
        "sft_batch": _BATCH_GEMMA,
        "use_liger_kernel": False,  # liger-kernel 0.8.0 gemma3 vision patch crashes on transformers 5.x
    },
    {
        "name": "google/medgemma-4b-it",
        "mm_processor_kwargs": None,
        "training_config": _GEMMA_TRAIN,
        "sft_batch": _BATCH_GEMMA,
        "use_liger_kernel": False,  # liger-kernel 0.8.0 gemma3 vision patch crashes on transformers 5.x
    },
    {
        "name": "google/medgemma-1.5-4b-it",
        "mm_processor_kwargs": None,
        "training_config": _GEMMA_TRAIN,
        "sft_batch": _BATCH_GEMMA,
        "use_liger_kernel": False,  # liger-kernel 0.8.0 gemma3 vision patch crashes on transformers 5.x
    },
    {
        "name": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        "mm_processor_kwargs": None,
        "training_config": _LLAVA_TRAIN,
        "sft_batch": _BATCH_LLAVA,
        # Liger's LLaVA vision patch may hit the same transformers-5.x bug (Liger-Kernel#1011);
        # flip to False if a first training run crashes in apply_liger_kernel_to_llava.
        "use_liger_kernel": True,
    },
    {
        "name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "mm_processor_kwargs": None,
        "training_config": _LLAVA_TRAIN,
        "sft_batch": _BATCH_LLAVA_NEXT,
        "use_liger_kernel": True,  # see llava-med note: flip to False if liger's llava patch crashes
    },
    # InternVL tile cap: 4 tiles × ~256 tokens keeps 4 images within budget. Single source of
    # truth for tiling — inference.py passes it to vLLM; training translates it to the
    # transformers processor's keys via common.mm_to_hf_image_kwargs.
    {
        "name": "lingshu-medical-mllm/Lingshu-I-8B",
        "mm_processor_kwargs": {"max_dynamic_patch": 4},
        "training_config": _LINGSHU_TRAIN,
        "sft_batch": _BATCH_LINGSHU,
        "use_liger_kernel": True,  # InternVL/InternViT — verify liger patch on first training run
    },
]

MODELS_BY_NAME = {model["name"]: model for model in MODELS}


def get_model_config(model_name: str) -> dict[str, Any]:
    try:
        return MODELS_BY_NAME[model_name]
    except KeyError as error:
        available = ", ".join(model["name"] for model in MODELS)
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}") from error


def get_sft_batch(model_name: str, stage: str) -> tuple[int, int]:
    """Return ``(per_device_batch_size, grad_accum_steps)`` for a model and training stage.

    ``stage`` is ``"qba"`` (Stage A) or ``"reports"`` (Stage B). Sizing lives per model in the
    registry (``sft_batch``) so both training entry points read a single source of truth.
    """
    batch = get_model_config(model_name)["sft_batch"]
    if stage not in batch:
        raise KeyError(f"Unknown training stage {stage!r}; expected one of {sorted(batch)}.")
    return batch[stage]
