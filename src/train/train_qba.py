"""Stage-A training: VQA grounding on MIMIC-Ext-CXR-QBA A++ subset.

Trains a LoRA adapter on the full App quality tier (1,318,885 pairs, 1 epoch) to ground
each model on CXR visual question answering before Stage-B report fine-tuning.

Usage (local single-GPU smoke test):
    python -m src.train.train_qba --model google/medgemma-4b-it --max-steps 10

Usage (HPC via run_train.sh / run_all_train.sh, 4×H100 DDP):
    accelerate launch --config_file accelerate_ddp.yaml \\
        -m src.train.train_qba --model google/medgemma-4b-it

Output: benchmark/checkpoints/<model-slug>/qa_adapter/

Hyperparameters are fixed constants below — edit here, not via CLI.
"""

from __future__ import annotations

import argparse
import logging
import os

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
from .data_qba import QBADataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed training hyperparameters — edit here rather than at the command line.
# ---------------------------------------------------------------------------
QBA_CACHE = "data/qba_app_q1m_index.parquet"  # flat index for the App_q1M subset (built on first run)
# Per-device batch size + grad-accum are per-model in model_registry.py (sft_batch["qba"]).
NUM_EPOCHS = 1.0  # one pass over the App_q1M subset
# Stage A: no eval / no best-checkpoint selection by design (1 epoch over 1.3M unique
# pairs → overfitting N/A; divergence shows in train loss). Log sparsely; saves use Stage B's
# step-ratio cadence (SAVES_PER_EPOCH) but with no eval / no best-model selection (no valid split).
LOGGING_STEPS = 1000  # sparse logs suffice for the long single-epoch Stage-A run
SAVES_PER_EPOCH = 10  # save this many times per epoch (mirrors Stage B's cadence); ratio computed in main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-A: VQA grounding on MIMIC-Ext-CXR-QBA A++")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID from model_registry.py")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for adapter (default: benchmark/checkpoints/<slug>/qa_adapter)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override NUM_EPOCHS: stop after this many steps (useful for smoke tests)",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Cap QBA index build to first N studies (smoke tests). Skips the canonical cache so a "
        "truncated index is never persisted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # accelerate launch spawns one process per GPU (DDP); mute non-main ranks so logs aren't n×.
    state = silence_non_main_ranks()

    model_config = get_model_config(args.model)
    training_config = model_config.get("training_config", {})
    per_device_batch, grad_accum = get_sft_batch(args.model, "qba")
    slug = sanitize_slug(args.model)

    output_dir = args.output_dir or f"benchmark/checkpoints/{slug}/qa_adapter"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    LOGGER.info("=== Stage A — VQA grounding: %s → %s ===", args.model, output_dir)
    LOGGER.info(
        "hyperparameters: batch=%d accum=%d epochs=%.1f cache=%s",
        per_device_batch,
        grad_accum,
        NUM_EPOCHS,
        QBA_CACHE,
    )

    # --- data ---
    # A capped build is a smoke test: skip the canonical cache so a truncated index is never persisted.
    cache_path = None if args.max_studies else QBA_CACHE
    LOGGER.info("loading QBA dataset (cache: %s, max_studies: %s)", cache_path, args.max_studies)

    # Build the index once on rank 0 (which writes the cache); the other ranks wait at the barrier
    # then load it. Avoids 4× redundant builds + a write race on the cache file, and guarantees every
    # rank sees an identical dataset so the DistributedSampler stays balanced for the full run.
    with state.main_process_first():
        dataset = QBADataset(cache_path=cache_path, max_studies=args.max_studies)

    # --- model + processor ---
    model, processor = load_model_and_processor(args.model, training_config)

    # --- LoRA config ---
    lora_config = build_lora_config(training_config)
    LOGGER.info(
        "LoRA config: r=%d alpha=%d modules=%s", lora_config.r, lora_config.lora_alpha, lora_config.target_modules
    )

    # --- collator ---
    collate_fn = build_collate_fn(processor, model_config)

    # --- save cadence: SAVES_PER_EPOCH saves/epoch (Stage A has no eval) ---
    _, save_steps = compute_save_steps(args.max_steps, SAVES_PER_EPOCH, NUM_EPOCHS)

    # --- trainer config ---
    sft_kwargs = default_sft_config_kwargs(
        output_dir=output_dir,
        liger_kernel_config=training_config.get("liger_kernel_config"),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=NUM_EPOCHS,
        max_steps=args.max_steps,
        # Per-model Liger toggle (model_registry.py).  False on Gemma: liger-kernel 0.8.0's
        # gemma3 vision-tower patch crashes on transformers 5.x (see _GEMMA_TRAIN note).
        use_liger_kernel=model_config.get("use_liger_kernel", True),
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=save_steps,
    )

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
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
