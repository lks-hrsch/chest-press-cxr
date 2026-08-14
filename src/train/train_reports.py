"""Stage-B training: CXR report generation fine-tuning.

Trains a LoRA report-generation adapter on data/chest_press_cxr_train.parquet.
Run per model to produce the report-generation variants the ablation requires:

  reports (base + reports):
    python -m src.train.train_reports --model <hf-id>

  qa+reports (base + QA + reports):
    python -m src.train.train_reports --model <hf-id> \\
        --init-adapter benchmark/checkpoints/<slug>/qa_adapter

When --init-adapter is given the existing qa adapter is loaded as trainable LoRA weights
(PEFT continuation, is_trainable=True) and training CONTINUES that same adapter on the
report data — no merging.  Omit --init-adapter to train a fresh adapter on the raw base.

Usage (HPC via run_train.sh, 4×H100 DDP):
    # reports
    accelerate launch --config_file accelerate_ddp.yaml \\
        -m src.train.train_reports --model google/medgemma-4b-it

    # qa+reports (after the qa stage has produced qa_adapter)
    accelerate launch --config_file accelerate_ddp.yaml \\
        -m src.train.train_reports --model google/medgemma-4b-it \\
            --init-adapter benchmark/checkpoints/google_medgemma-4b-it/qa_adapter

Usage (local single-GPU smoke test):
    python -m src.train.train_reports --model google/medgemma-4b-it --max-steps 10

Output:
  --init-adapter absent → benchmark/checkpoints/<slug>/reports_adapter/
  --init-adapter given  → benchmark/checkpoints/<slug>/qa_reports_adapter/

Hyperparameters are fixed constants below — edit here, not via CLI.
Eval is always on: once per epoch on the full valid split (eval_strategy="epoch").
"""

from __future__ import annotations

import argparse
import logging
import os

from peft import PeftModel
from trl import SFTConfig, SFTTrainer

from ..helpers import sanitize_slug
from ..model_registry import get_model_config, get_sft_batch
from .common import (
    NanSkipCallback,
    assert_no_vision_lora,
    build_collate_fn,
    build_lora_config,
    compute_save_steps,
    default_sft_config_kwargs,
    load_model_and_processor,
    log_trainable_parameters,
    silence_non_main_ranks,
)
from .data_reports import ReportDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed training hyperparameters — edit here rather than at the command line.
# ---------------------------------------------------------------------------
TRAIN_PARQUET = "data/chest_press_cxr_train.parquet"
VALID_PARQUET = "data/chest_press_cxr_valid.parquet"
# Per-device batch size + grad-accum are per-model in model_registry.py (sft_batch["reports"]);
# eval is always forced to batch 1 below (full fp32 logits), independent of the train batch.
NUM_EPOCHS = 1  # small dataset so low epoch size
MAX_IMAGES = 4  # cap images/study: trims the rare 4+-image tail (≈1% of rows) that spikes image tokens
EVALS_PER_EPOCH = 10  # run eval + save this many times per epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-B: CXR report generation fine-tuning")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID from model_registry.py")
    parser.add_argument(
        "--init-adapter",
        type=str,
        default=None,
        help=(
            "Path to a qa-stage LoRA adapter directory.  "
            "When given, that adapter is loaded as trainable LoRA weights and training "
            "continues it on the report data (→ qa+reports), no merging.  "
            "Omit for a fresh adapter on the raw base (→ reports)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for adapter (default auto-derived from --init-adapter presence)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override NUM_EPOCHS: stop after this many steps (useful for smoke tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # accelerate launch spawns one process per GPU (DDP); mute non-main ranks so logs aren't n×.
    # (Stage B reads parquets — no cache build — so no main_process_first barrier is needed here.)
    silence_non_main_ranks()

    model_config = get_model_config(args.model)
    training_config = model_config.get("training_config", {})
    per_device_batch, grad_accum = get_sft_batch(args.model, "reports")
    slug = sanitize_slug(args.model)

    variant = "qa_reports" if args.init_adapter else "reports"
    output_dir = args.output_dir or f"benchmark/checkpoints/{slug}/{variant}_adapter"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    LOGGER.info("=== Stage B — report generation (%s): %s → %s ===", variant, args.model, output_dir)
    LOGGER.info(
        "hyperparameters: batch=%d accum=%d epochs=%.1f max_images=%d train=%s valid=%s",
        per_device_batch,
        grad_accum,
        NUM_EPOCHS,
        MAX_IMAGES,
        TRAIN_PARQUET,
        VALID_PARQUET,
    )

    # --- data ---
    LOGGER.info("loading train dataset from %s", TRAIN_PARQUET)
    dataset = ReportDataset(parquet_path=TRAIN_PARQUET, max_images=MAX_IMAGES)
    LOGGER.info("loading eval dataset from %s", VALID_PARQUET)
    eval_dataset = ReportDataset(parquet_path=VALID_PARQUET, max_images=MAX_IMAGES)

    # --- eval/save cadence: EVALS_PER_EPOCH times/epoch on the full valid split ---
    smoke_test, eval_save_steps = compute_save_steps(args.max_steps, EVALS_PER_EPOCH, NUM_EPOCHS)

    LOGGER.info(
        "eval/save cadence: eval_steps=%s (evals_per_epoch=%d, epochs=%.1f)",
        eval_save_steps,
        EVALS_PER_EPOCH,
        NUM_EPOCHS,
    )

    # --- model + processor (vision tower frozen in load_model_and_processor) ---
    model, processor = load_model_and_processor(args.model, training_config)

    # --- LoRA: fresh adapter on the raw base, or continue an existing qa adapter ---
    if args.init_adapter:
        # qa+reports: load the qa adapter as trainable LoRA weights and CONTINUE training
        # it on the report data (no merging).  The model already carries the adapter, so
        # SFTTrainer must NOT add a new one → peft_config stays None.
        LOGGER.info("continuing qa adapter %s (PEFT continuation, is_trainable=True)", args.init_adapter)
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        lora_config = None
    else:
        lora_config = build_lora_config(training_config)
        LOGGER.info(
            "LoRA config: r=%d alpha=%d modules=%s",
            lora_config.r,
            lora_config.lora_alpha,
            lora_config.target_modules,
        )

    # --- collator (max_length applies in the processor call: skip_prepare_dataset bypasses TRL's
    # tokenisation, so SFTConfig.max_length doesn't reach our custom collator) ---
    collate_fn = build_collate_fn(processor, model_config)

    # --- trainer config ---
    sft_kwargs = default_sft_config_kwargs(
        output_dir=output_dir,
        liger_kernel_config=training_config.get("liger_kernel_config"),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=NUM_EPOCHS,
        max_steps=args.max_steps,
        # Truncate long sequences to fit eval VRAM budget.  keep_start is mandatory for VLMs:
        # keep_end can discard image tokens from the prompt and desync pixel_values from
        # input_ids (raises "Image features and image tokens do not match" in the processor).
        # 20480 matches the max_model_len used in inference and is comfortably above the image-token
        # budget (4 images × ~256 tokens = 1024 tokens), leaving plenty of room for reports.
        # (Vestigial — the truncation that binds is build_collate_fn's default in common.py; kept in sync.)
        max_length=20 * 1024,
        truncation_mode="keep_start",
        # Eval: prediction_loss_only (no logit accumulation across the split); eval batch = half the
        # train batch at accum 1. Eval materialises the full fp32 logit tensor
        # [batch × max_length × vocab × 4B] — e.g. Gemma3 at batch 8 × 20480 × 262,144 ≈ 172 GiB at the
        # 20k cap; even halved (eval batch 4) ≈ 86 GiB, which OVERSHOOTS an 80 GiB H100 — so the
        # large-vocab Gemma-family reports sft_batch likely needs lowering (e.g. (4,4)) to keep eval in VRAM.
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        per_device_eval_batch_size=max(1, per_device_batch // 2),
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        # Per-model Liger toggle (model_registry.py).  False on Gemma (liger-kernel 0.8.0's gemma3
        # vision-tower patch crashes on transformers 5.x); then eval/train lean on the image cap +
        # alloc tuning (batch=1, max_length=20480, keep_start) for the non-fused-CE fp32 logits.
        use_liger_kernel=model_config.get("use_liger_kernel", True),
        # Save aligned to eval so best-checkpoint selection is valid (HF requires
        # save_steps to be a multiple of eval_steps when load_best_model_at_end=True).
        save_strategy="steps",
        save_steps=eval_save_steps,
        # Keep the best eval-loss checkpoint as the final saved adapter.  Disabled for
        # smoke tests so a short run with no completed eval can't fail the best-model reload.
        load_best_model_at_end=(not smoke_test),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        data_collator=collate_fn,
        processing_class=processor,
        callbacks=[NanSkipCallback()],
    )
    log_trainable_parameters(trainer.model)
    assert_no_vision_lora(trainer.model, training_config.get("vision_tower_attr"))

    # --- train ---
    trainer.train()

    # --- save adapter ---
    trainer.save_model(output_dir)
    LOGGER.info("adapter saved to %s", output_dir)


if __name__ == "__main__":
    main()
