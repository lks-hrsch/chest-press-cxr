"""Shared utilities for the LoRA fine-tuning pipeline.

Provides:
- silence_non_main_ranks()    mute INFO logs on non-main DDP ranks; return the PartialState
- load_model_and_processor()  load & configure a registry model for training
- build_lora_config()         assemble a LoraConfig from the registry training_config
- get_image_token_ids()       resolve image-structural token IDs per architecture
- make_collate_fn()           build the custom VLM data collator closure
- build_collate_fn()          resolve token IDs + tiling kwargs and build the collator
- compute_save_steps()        shared save/eval cadence (full-run ratio vs smoke-test int)
- default_sft_config_kwargs() shared SFTConfig defaults for both training stages
- NanSkipCallback             zero non-finite grads pre-step so a bad batch can't poison the run
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any

import torch
from accelerate import PartialState
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback

from ..mm_kwargs import mm_to_hf_image_kwargs
from ..settings import HF_ACCESS_TOKEN, RANDOM_SEED_VALUE


# ``mm_to_hf_image_kwargs`` now lives in src/mm_kwargs.py so inference shares one mapper with
# training (inference used to pass the untranslated vLLM spelling straight through, which the
# transformers InternVL processor silently ignored).  Imported here for use below, which also keeps
# the historical ``from .common import mm_to_hf_image_kwargs`` import path working.

LOGGER = logging.getLogger(__name__)

# LoRA defaults — reference-paper (Table 2) values. r=32 with alpha=64 keeps the alpha/r scale at
# 2.0 (same per-step update magnitude as the former 16/32), so doubling the rank buys capacity without
# changing training dynamics; dropout 0.1 adds regularisation for the small report set.
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

# Distributed collective timeout. NCCL's default is 10 min, but Stage-A builds the QBA index from
# ~194k study JSONs on rank 0 inside ``main_process_first()`` (~13 min on a cold cache) while ranks
# 1–3 block on the entry barrier — overrunning the default and crashing with a c10d wait-timeout
# (DistBackendError). A generous timeout lets the barrier wait through the cold-cache build (and any
# cold model-load rank skew). It governs every collective, so a genuine training hang still fails
# the job — just after this window rather than 10 min.
DIST_TIMEOUT = timedelta(hours=2)


def silence_non_main_ranks() -> PartialState:
    """Return the distributed PartialState, muting INFO logs on non-main ranks.

    ``accelerate launch`` runs one process per GPU, so without this every rank emits the same INFO
    logs n×. Errors/warnings on any rank still show. Under a plain ``python -m ...`` launch this is
    a no-op single-process state (``is_main_process`` True).

    The process group is initialised with :data:`DIST_TIMEOUT` (forwarded to
    ``torch.distributed.init_process_group``) so the long rank-0-only QBA index build inside
    ``main_process_first()`` cannot trip NCCL's 10-min default barrier timeout. The kwarg is inert
    on the single-process path.

    Invariant: this must be the **first** accelerate-state / process-group construction in the
    program. ``PartialState`` shares a borg state guarded by ``if not self.initialized``, so the
    ``timeout`` kwarg is honoured only on the first construction — if anything (an ``Accelerator``,
    an ``SFTConfig``/``TrainingArguments`` triggering ``_setup_devices``, …) initialises the process
    group earlier, ``DIST_TIMEOUT`` is silently dropped back to NCCL's 10-min default and the
    cold-build barrier crash returns. We fail loudly here rather than let that regress silently.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed is already initialised before silence_non_main_ranks(); the "
            "DIST_TIMEOUT passed to PartialState would be ignored (its borg state honours kwargs "
            "only on first construction), reintroducing the Stage-A NCCL barrier-timeout crash. "
            "This helper must be the first accelerate-state / process-group construction — do not "
            "build an Accelerator/PartialState or an SFTConfig/TrainingArguments before it."
        )
    state = PartialState(timeout=DIST_TIMEOUT)
    if not state.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)
    return state


def load_model_and_processor(
    model_name: str,
    training_config: dict[str, Any],
    *,
    from_path: str | None = None,
) -> tuple[Any, Any]:
    """Load model + processor from HF hub (or a local checkpoint directory).

    The vision tower is frozen in-place.  Attention implementation and dtype
    are taken from *training_config* as recorded in src/model_registry.py.

    Parameters
    ----------
    model_name:
        HF model ID — used for the processor and as a fallback when *from_path* is None.
    training_config:
        The ``training_config`` dict from the registry entry.
    from_path:
        If not None, load base weights from this local directory instead of HF hub.
        The processor is always loaded from *model_name*.
    """
    load_path = from_path or model_name
    attn_impl = training_config.get("attn_implementation", "sdpa")

    LOGGER.info("loading model %s (attn=%s, from_path=%s)", model_name, attn_impl, from_path)
    model = AutoModelForImageTextToText.from_pretrained(
        load_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN,
        # do NOT set device_map — let accelerate/SFTTrainer handle placement for DDP
    )

    _freeze_vision_tower(model, training_config)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=HF_ACCESS_TOKEN)
    # right-padding is required: left-pad batches break image token positions
    processor.tokenizer.padding_side = "right"

    return model, processor


def _freeze_vision_tower(model: Any, training_config: dict[str, Any]) -> None:
    """Freeze the vision encoder weights so LoRA only adapts the LM.

    Uses substring matching over ``model.named_parameters()`` rather than a
    fixed ``getattr`` chain, so it is robust to transformers v5 nesting depth
    (e.g. ``model.model.vision_tower.blocks.0.attn.q_proj``).  The key is the
    ``vision_tower_attr`` string from the registry training_config.
    """
    key = training_config.get("vision_tower_attr")
    if not key:
        LOGGER.warning("vision_tower_attr not set in training_config — vision tower NOT frozen")
        return

    frozen = 0
    for name, param in model.named_parameters():
        if key in name:
            param.requires_grad = False
            frozen += param.numel()

    if frozen == 0:
        LOGGER.warning("vision tower key '%s' matched no params — NOT frozen", key)
    else:
        LOGGER.info("froze vision tower '%s' (%d params)", key, frozen)


def build_lora_config(training_config: dict[str, Any]) -> LoraConfig:
    """Build a LoraConfig from the registry training_config."""
    target_modules = list(training_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
    # Optionally add MLP modules if specified
    mlp_modules = training_config.get("lora_mlp_modules")
    if mlp_modules:
        target_modules = target_modules + list(mlp_modules)

    # Scope LoRA to the language model. target_modules match by name suffix across the
    # *whole* model, so bare q_proj/gate_proj/... also hit the (frozen) vision/audio
    # encoders — which on Gemma 3n are Gemma4ClippableLinear wrappers PEFT can't inject,
    # and on the other VLMs would silently LoRA-adapt the vision tower. Exclude those
    # subtrees. An explicit regex wins; otherwise derive one from vision_tower_attr.
    exclude_modules = training_config.get("lora_exclude_modules")
    if exclude_modules is None:
        vision_attr = training_config.get("vision_tower_attr")
        if vision_attr:
            exclude_modules = rf".*{re.escape(vision_attr)}.*"

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        exclude_modules=exclude_modules,
    )


def log_trainable_parameters(model: Any) -> None:
    """Log trainable vs. total parameter counts (sanity-check LoRA + frozen towers)."""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    LOGGER.info("trainable params: %s || all params: %s || trainable%%: %.4f", f"{trainable:,}", f"{total:,}", pct)


def assert_no_vision_lora(model: Any, vision_tower_attr: str | None) -> None:
    """Raise if LoRA adapters leaked onto the vision tower.

    LoRA ``target_modules`` are matched by name suffix across the *whole* model
    graph.  If ``exclude_modules`` in the LoraConfig is wrong or empty, vision
    tower layers that share names with LM layers (e.g. ``q_proj``, ``gate_proj``)
    will silently receive LoRA adapters — inflating the adapter size and corrupting
    the frozen-tower assumption.  Call this immediately after the PEFT model is
    built (after ``SFTTrainer`` wraps the model) to catch the mistake at step 0.

    Safe to call for the ``--init-adapter`` continuation path: the PEFT model
    already exists before ``SFTTrainer`` and the check applies equally.

    Parameters
    ----------
    model:
        The (PEFT-wrapped) model, typically ``trainer.model``.
    vision_tower_attr:
        The ``vision_tower_attr`` key from the registry training_config
        (e.g. ``"vision_tower"``).  Used as a substring filter.
    """
    key = vision_tower_attr or "vision"
    leaked = [n for n, _ in model.named_modules() if "lora" in n.lower() and key in n]
    if leaked:
        raise RuntimeError(
            f"LoRA adapters leaked onto vision tower (key={key!r}): {leaked[:5]!r}. "
            "Check lora_exclude_modules in the registry training_config."
        )
    LOGGER.info("LoRA leak check passed — no adapters found under vision key '%s'", key)


def get_image_token_ids(processor: Any) -> list[int]:
    """Return token IDs for image-structural tokens (should be masked with -100 in labels).

    These are architecture-specific tokens inserted by the chat template to encode image
    position/boundaries (BOI, soft-image tokens, etc.) — NOT the visual embeddings themselves.
    Setting labels=-100 on them avoids training the LM to "predict" these structural tokens.
    """
    tokenizer = processor.tokenizer
    unk_id = tokenizer.unk_token_id
    ids: set[int] = set()

    # Candidate special token strings across known architectures
    candidates = [
        "<image>",  # LLaVA 1.5 / 1.6
        "<img>",  # InternVL / Lingshu
        "<IMG_CONTEXT>",  # InternVL context token
        "<|image_pad|>",  # Qwen2-VL / Qwen3-VL
        "<|vision_start|>",  # Qwen-VL boundary
        "<|vision_end|>",  # Qwen-VL boundary
        "<image_soft_token>",  # Gemma-3 (string form; numeric id added below)
    ]
    for name in candidates:
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != unk_id:
            ids.add(tid)

    # Gemma-3: the image soft token lives at a fixed ID beyond the text vocab
    # (TRL VLM SFT guide explicitly masks 262144).  Only add when the model's
    # extended vocab reaches that range to avoid false positives on smaller models.
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size > 260_000:
        ids.add(262144)

    # Catch tokens declared in special_tokens_map under image-related keys
    stm = getattr(tokenizer, "special_tokens_map", {})
    for key in ("boi_token", "eoi_token", "image_token"):
        if key in stm:
            tok = stm[key]
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != unk_id:
                ids.add(tid)

    if ids:
        LOGGER.info("image token IDs to mask: %s", sorted(ids))
    else:
        LOGGER.warning("no image token IDs found — labels will only mask padding")

    return list(ids)


def make_collate_fn(
    processor: Any,
    image_token_ids: list[int],
    image_kwargs: dict[str, Any] | None = None,
    max_length: int | None = None,
):
    """Return a collate_fn suitable for SFTTrainer's data_collator argument.

    Each example must have:
        messages: list of role/content dicts (OpenAI chat format with {"type":"image"} blocks)
        images:   list of PIL.Image instances (one per {"type":"image"} block in messages)

    The collator:
    1. Applies the processor's chat template to render text + image placeholder tokens.
    2. Calls processor(images, texts) to get input_ids, attention_mask, pixel_values.
    3. Builds labels = input_ids, then masks padding + image structural tokens with -100.

    image_kwargs (e.g. {"crop_to_patches": True, "max_patches": 4} for InternVL/Lingshu) are
    forwarded to the processor call to cap image tiling — None for fixed-resolution backbones.

    max_length: when set, truncation is applied inside the processor call (truncation_side="right",
    i.e. keep_start — safe for VLMs where image tokens appear early in the sequence).
    NOTE: SFTConfig.max_length has no effect when skip_prepare_dataset=True (TRL's tokenisation
    pipeline is bypassed entirely); truncation must happen here, in the processor call.
    """
    image_kwargs = image_kwargs or {}
    truncation_kwargs: dict[str, Any] = {"truncation": max_length is not None}
    if max_length is not None:
        truncation_kwargs["max_length"] = max_length

    def collate_fn(examples: list[dict]) -> dict:
        texts = [
            processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False).strip()
            for ex in examples
        ]
        images = [[img.convert("RGB") for img in ex["images"]] for ex in examples]

        batch = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            **truncation_kwargs,
            **image_kwargs,
        )

        labels = batch["input_ids"].clone()
        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Mask image-structural tokens (BOI, soft-image, vision boundaries, etc.)
        for tid in image_token_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


# Default max_length 20480 (= inference max_model_len); both stages rely on this default (see CLAUDE.md).
def build_collate_fn(processor: Any, model_config: dict[str, Any], *, max_length: int = 20 * 1024):
    """Build the SFT data collator for a registry model (shared by both training stages).

    Resolves the architecture's image-structural token IDs (masked in the labels) and the
    training-time image-tiling kwargs (translated from the registry ``mm_processor_kwargs``).
    """
    image_token_ids = get_image_token_ids(processor)
    return make_collate_fn(
        processor,
        image_token_ids,
        image_kwargs=mm_to_hf_image_kwargs(model_config.get("mm_processor_kwargs")),
        max_length=max_length,
    )


def compute_save_steps(max_steps: int, saves_per_epoch: int, num_epochs: float) -> tuple[bool, float | int]:
    """Resolve ``(is_smoke_test, save/eval step cadence)`` shared by both training stages.

    Full run: a float ratio ``1/(saves_per_epoch × num_epochs)`` — transformers reads ``save_steps``
    < 1 as a fraction of the total (GPU- and grad-accum-aware) step count, giving N saves per epoch
    with no manual math. Smoke run (``max_steps`` > 0): a small int, since a float ratio of a
    ~10-step run rounds to 0 and is rejected.
    """
    smoke_test = bool(max_steps and max_steps > 0)
    steps = max(1, max_steps // 2) if smoke_test else 1 / (saves_per_epoch * num_epochs)
    return smoke_test, steps


def default_sft_config_kwargs(
    output_dir: str,
    *,
    liger_kernel_config: dict | None = None,
    **overrides,
) -> dict:
    """Return a dict of SFTConfig keyword arguments with sensible VLM training defaults.

    Pass as ``SFTConfig(**default_sft_config_kwargs(...), num_train_epochs=1)``.

    Parameters
    ----------
    liger_kernel_config:
        Per-component override dict forwarded to ``SFTConfig(liger_kernel_config=...)``,
        which transformers passes as ``**kwargs`` to the architecture-specific
        ``apply_liger_kernel_to_<arch>()`` call.  Generic hook for future per-arch tuning;
        currently unset by every registry entry (resolves to ``None`` → all defaults).
        NOTE: this is NOT how Gemma avoids the liger crash — ``{"layer_norm": False}`` does
        not prevent it (liger-kernel 0.8.0 dereferences the vision tower's missing
        ``.vision_model`` unconditionally).  Gemma disables Liger wholesale via the per-model
        ``use_liger_kernel: False`` flag in model_registry.py instead.
    """
    base = {
        "output_dir": output_dir,
        # VLM: never truncate by default (would chop image tokens). Stage-B
        # (train_reports.py) overrides this to max_length=20480 to prevent eval OOM.
        "max_length": None,
        # Memory & precision
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Every trainable param is a LoRA adapter on attention+MLP that runs in every forward
        # (vision tower frozen), so none is ever unused. The PEFT-wrapped model otherwise makes
        # the Trainer default this to True (an extra autograd-graph traversal each micro-step).
        # Flip back to True only if a future change adds conditional / MoE / partial-layer
        # trainable modules — else DDP will hang waiting for grads that never arrive.
        "ddp_find_unused_parameters": False,
        # Fused linear-cross-entropy. The cross-entropy loss is computed in fp32 for
        # numerical stability (standard under bf16 training — a bf16 softmax over a large
        # vocab loses precision); this is the *loss* dtype, not the weights/activations,
        # which are bf16. Liger avoids materialising the full [seq × vocab] fp32 logit
        # tensor during training — the dominant per-step allocation (e.g. ~14 GB for
        # Gemma-3's 262k vocab at 13k tokens).
        # CAVEAT: this fused path only activates while model.training=True.  TRL's eval
        # path (compute_loss / shift_logits) still materialises the full fp32 logit tensor,
        # so Stage-B eval requires max_length + eval batch 1 to stay within VRAM budget.
        # This is only a DEFAULT — the train entry points pass an explicit use_liger_kernel
        # from the per-model `use_liger_kernel` flag (model_registry.py), False on Gemma.
        "use_liger_kernel": True,
        "optim": "adamw_torch_fused",
        # VLM dataset handling — skip TRL's automatic tokenisation pass
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        # Dataloader IO (both stages): workers run the PIL open + processor resize inside the
        # collator; prefetch hides storage latency.
        "dataloader_num_workers": 8,
        "dataloader_pin_memory": True,
        "dataloader_prefetch_factor": 4,
        # Logging & checkpointing
        "report_to": "tensorboard",
        "logging_steps": 50,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        # Reproducibility
        "seed": RANDOM_SEED_VALUE,
        # LR (adapters tolerate higher rates than full FT). Kept at 1e-4 — the reference paper uses
        # 2e-4, but that doubles the per-step update and re-invites the bf16 overflow that diverged
        # gemma-3-4b-it Stage-A; NanSkipCallback guards the residual risk.
        "learning_rate": 1e-4,
        # Cosine decay (reference-paper Table 2): near-peak longer early, gentle near-zero at the end.
        "lr_scheduler_type": "cosine",
        # warmup_steps < 1 is interpreted as a ratio of total steps (transformers 5.x merged
        # warmup_ratio into warmup_steps; the standalone warmup_ratio arg is deprecated).
        "warmup_steps": 0.03,
    }
    if liger_kernel_config is not None:
        base["liger_kernel_config"] = liger_kernel_config
    base.update(overrides)
    return base


class NanSkipCallback(TrainerCallback):
    """Zero non-finite gradients before the optimizer step so one bad batch can't poison the run.

    A single bf16 activation/logit overflow yields a NaN/inf gradient. Applied by the optimizer it
    permanently poisons every weight (NaN propagates forever) — this is how gemma-3-4b-it Stage-A
    diverged: healthy convergence for ~4k steps, then an instant NaN spike that ran on to a 100%-NaN
    adapter. transformers fires ``on_pre_optimizer_step`` *after* gradient clipping and *before*
    ``optimizer.step()`` (clipping can't rescue a NaN norm), so zeroing the gradient here makes the
    step a no-op (at most a tiny, *finite* stale-momentum nudge at weight_decay=0) instead of NaN.

    This is the in-training first line of defense; the merge-time finiteness scan in merge_adapter.py
    is the hard backstop. A one-off outlier batch is skipped and training continues; sustained
    divergence (``max_consecutive`` non-finite steps in a row) aborts, rather than silently burning
    compute on a run that can no longer learn.
    """

    def __init__(self, max_consecutive: int = 25) -> None:
        self.max_consecutive = max_consecutive
        self.total_skips = 0
        self._consecutive = 0

    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):  # noqa: ARG002
        if optimizer is None:
            return
        params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if all(torch.isfinite(p.grad).all() for p in params):
            self._consecutive = 0
            return

        self.total_skips += 1
        self._consecutive += 1
        LOGGER.warning(
            "non-finite gradient at step %s — zeroing grad to skip update (total=%s, consecutive=%s)",
            state.global_step,
            self.total_skips,
            self._consecutive,
        )
        for p in params:
            p.grad.zero_()

        if self._consecutive >= self.max_consecutive:
            raise RuntimeError(
                f"training diverged: {self._consecutive} consecutive non-finite gradients "
                f"(step {state.global_step}). Aborting instead of burning compute on a dead run."
            )
