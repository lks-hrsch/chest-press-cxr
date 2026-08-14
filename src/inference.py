import argparse
import logging
import math
import os

import polars as pl
from vllm import LLM, SamplingParams

from .adapters import ADAPTER_VARIANTS, CHECKPOINTS_ROOT_DEFAULT, merged_dir_for
from .dataset.constants import XRAY_BASE_PATH
from .helpers import sanitize_slug
from .mm_kwargs import assert_image_kwargs_supported, mm_to_hf_image_kwargs
from .model_registry import get_model_config
from .settings import HF_ACCESS_TOKEN, RANDOM_SEED_VALUE


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

# vLLM's SamplingParams defaults max_tokens=16, which truncates every generation to ~one clause
# (so reports never reach IMPRESSION and impression/combined metrics collapse to ~0). The explicit
# cap was dropped in 34e19ee; restore a report-sized budget. Greedy decoding (temperature=0) stops
# at EOS, so this generous cap costs nothing on normal reports; 1024 covers the full FINDINGS+
# IMPRESSION reference length distribution (max ~710 tokens) with headroom under max_model_len.
MAX_NEW_TOKENS = 1024

parser = argparse.ArgumentParser(description="Run vLLM inference on medical image-text-to-text models")
parser.add_argument("--model", type=str, default="google/medgemma-1.5-4b-it", help="HuggingFace model ID")
parser.add_argument(
    "--variant",
    type=str,
    default=None,
    choices=["base", "qa", "reports", "qa_reports"],
    help=(
        "Training variant.  For the adapter variants (qa/reports/qa_reports) the matching MERGED "
        f"checkpoint at {CHECKPOINTS_ROOT_DEFAULT}/<slug>/<variant>_merged is served (run "
        "`python -m src.merge_adapter` / `./run_all_merge.sh` first).  Also suffixes the output "
        "parquet: data/inference_<slug>__<variant>.parquet.  Use 'base' (or omit) for the "
        "untrained baseline (data/inference_<slug>.parquet)."
    ),
)
parser.add_argument(
    "--context",
    type=str,
    default=None,
    choices=["baseline", "clinical_context", "rich_clinical_context"],
    help=(
        "Clinical-context ablation tier (see src/dataset/context_ablation.py). Reads "
        "data/chest_press_cxr_{test,valid}__<context>.parquet instead of the "
        "plain (rich, unsuffixed) split files, and suffixes the output parquet: "
        "data/inference_<slug>__ablation_<context>.parquet (composed after any --variant suffix). "
        "Omit for today's default (unsuffixed) input/output."
    ),
)
parser.add_argument("--subject_id", type=str, default=None, help="filter by subject_id")
parser.add_argument("--study_id", type=str, default=None, help="filter by study_id")
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Cap to the first N rows (debug/smoke runs). Default: no cap — run the full split(s).",
)
parser.add_argument("--max_dynamic_patch", type=int, default=None, help="InternVL: max image tiles per image")
parser.add_argument("--max_pixels", type=int, default=None, help="Qwen-VL: max pixels per image before tiling")
parser.add_argument(
    "--chat_batch_size",
    type=int,
    default=1000,
    help=(
        "Chunk the row-set into batches of this size and call llm.chat() once per chunk. vLLM's "
        "offline LLM.chat() fully renders (decodes every image + enqueues) the ENTIRE row-set "
        "before any generation starts, so a single call over a large row-set materializes every "
        "row's multimodal payload in host RAM at once. Chunking bounds peak host RAM to roughly "
        "chat_batch_size x max_images worth of payload instead of the whole split. Default 500 is "
        "conservative for 8B-class multi-tile vision models (InternVL/AnyRes) on a RAM-constrained "
        "host; raise it for lighter/fixed-resolution models or more RAM."
    ),
)


TEST_PATH = "data/chest_press_cxr_test.parquet"
VALID_PATH = "data/chest_press_cxr_valid.parquet"
# Splits to run inference over.  CLAUDE.md documents the full benchmark as test + valid; set
# INCLUDE_VALID_SPLIT = True for that.  Kept test-only by default for fast local iteration.
INCLUDE_VALID_SPLIT = False


def _with_context_suffix(path: str, context: str | None) -> str:
    if not context:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}__{context}{ext}"


def split_paths(context: str | None) -> list[str]:
    """Return the split parquet path(s) to scan for this run, optionally suffixed by a
    --context ablation tier. See src/dataset/context_ablation.py: it writes
    data/chest_press_cxr_{test,valid}__<tier>.parquet as drop-in replacements for the
    plain split files (identical schema; only input_text/system_prompt differ per tier).
    """
    paths = [_with_context_suffix(TEST_PATH, context)]
    if INCLUDE_VALID_SPLIT:
        paths.append(_with_context_suffix(VALID_PATH, context))
    return paths


def build_conversation(row: dict) -> list[dict]:
    messages = []
    if row["system_prompt"]:
        messages.append({"role": "system", "content": [{"type": "text", "text": row["system_prompt"]}]})
    content = [
        {"type": "image_url", "image_url": {"url": f"file://{os.path.normpath(XRAY_BASE_PATH + path)}"}}
        for path in row["xray_image_paths"]
    ]
    content.append({"type": "text", "text": row["input_text"]})
    messages.append({"role": "user", "content": content})
    return messages


def inference_all(
    dataset: pl.LazyFrame,
    model_name: str,
    output_path: str = "data/inference_model.parquet",
    mm_processor_kwargs: dict | None = None,
    model_path: str | None = None,
    chat_batch_size: int = 250,
):
    rows = list(dataset.collect().iter_rows(named=True))
    if not rows:
        LOGGER.warning("no rows after filtering — nothing to do")
        return

    max_images = max(len(r["xray_image_paths"]) for r in rows)
    # Serve a local full checkpoint (base or merged adapter) when given; else the HF hub model.
    # Trained adapters are always served as MERGED checkpoints (src/merge_adapter.py) — uniform
    # across all architectures, since vLLM only supports runtime LoRA for a subset of archs.
    load_target = model_path or model_name
    LOGGER.info(
        "loading %s (path=%s) — %s rows, max_images=%s, writing to %s",
        model_name,
        model_path,
        len(rows),
        max_images,
        output_path,
    )

    # Fail fast if the processor would silently ignore the tiling kwargs.  transformers only *warns*
    # on an unknown processor kwarg, which is how Lingshu ended up trained at 4 tiles and served at
    # 13 without anyone noticing.  Best-effort: skipped when the processor exposes no kwarg schema.
    if mm_processor_kwargs:
        try:
            from transformers import AutoProcessor

            probe = AutoProcessor.from_pretrained(load_target, trust_remote_code=True, token=HF_ACCESS_TOKEN)
            image_processor = getattr(probe, "image_processor", None)
            if image_processor is not None:
                assert_image_kwargs_supported(image_processor, mm_processor_kwargs, model_name)
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            LOGGER.warning("could not verify mm_processor_kwargs against the processor: %s", error)

    llm = LLM(
        model=load_target,
        trust_remote_code=True,
        hf_token=HF_ACCESS_TOKEN,
        seed=RANDOM_SEED_VALUE,
        # Uniform precision across the benchmark: some models (the LLaVA models) ship a
        # float16 config; pin bf16 so every model runs in the same dtype on H100.
        dtype="bfloat16",
        # Inference context window. Sized to clear Lingshu's longest studies (a handful render to
        # ~17k tokens: max_dynamic_patch=4 images + long history) while still fitting the 7B/8B
        # merged checkpoints' KV cache on the 24 GB workstation card at the util below (24576 did
        # not — vLLM estimated only ~15-21k fit at 0.80). Matches the 20480 training-truncation
        # length, so no study is dropped; every served model supports >=20480.
        max_model_len=20 * 1024,  # 20480
        gpu_memory_utilization=0.85,  # 24 GB card: fits the 7B/8B KV cache at the window above; 0.89 crowded the display baseline
        limit_mm_per_prompt={"image": max_images},
        allowed_local_media_path=os.path.normpath(XRAY_BASE_PATH),
        mm_processor_kwargs=mm_processor_kwargs,
    )
    sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    # vLLM's offline LLM.chat() fully renders (decodes every image + enqueues) its ENTIRE input
    # before any generation starts (see _run_chat in vllm/entrypoints/offline_utils.py: it runs
    # _add_chat_requests() to completion, THEN _run_engine()). Submitting the whole split in one
    # call materializes every row's multimodal payload in host RAM simultaneously — for 8B-class
    # multi-tile vision models (InternVL/AnyRes) that's enough to swap a RAM-constrained host.
    # Chunking bounds peak host RAM to one batch's worth, independent of total row count.
    num_chunks = math.ceil(len(rows) / chat_batch_size)
    LOGGER.info("starting inference: %d rows in %d batches of up to %d", len(rows), num_chunks, chat_batch_size)
    results: list[dict] = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chat_batch_size
        chunk_rows = rows[start : start + chat_batch_size]
        LOGGER.info(
            "chunk %d/%d: building %d conversations (rows %d-%d)",
            chunk_idx + 1,
            num_chunks,
            len(chunk_rows),
            start,
            start + len(chunk_rows) - 1,
        )
        conversations = [build_conversation(row) for row in chunk_rows]
        outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=True)
        results.extend(
            {
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
                "generated_output_text": out.outputs[0].text,
            }
            for row, out in zip(chunk_rows, outputs, strict=True)
        )
        LOGGER.info(
            "chunk %d/%d done — %d/%d rows completed so far", chunk_idx + 1, num_chunks, len(results), len(rows)
        )

    # Any chunk failure (OOM, CUDA error, etc.) propagates uncaught: the whole run aborts with a
    # non-zero exit and NO output parquet is written, matching the pre-chunking fail-fast behavior
    # (a single llm.chat() failure previously also produced zero output rows). Do not catch
    # per-chunk exceptions here — run_local_test.sh already retries/continues at the process level.
    pl.DataFrame(results).write_parquet(output_path)
    LOGGER.info("wrote %s rows to %s", len(results), output_path)


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = pl.concat([pl.scan_parquet(p) for p in split_paths(args.context)])

    if args.subject_id:
        LOGGER.info("Filtering for subject_id: %s", args.subject_id)
        dataset = dataset.filter(pl.col("subject_id") == int(args.subject_id))

    if args.study_id:
        LOGGER.info("Filtering for study_id: %s", args.study_id)
        dataset = dataset.filter(pl.col("study_id") == int(args.study_id))

    if args.limit is not None:
        LOGGER.warning("DEBUG: limiting inference to the first %d rows (--limit)", args.limit)
        dataset = dataset.head(args.limit)

    # Start from registry defaults, then apply any CLI overrides
    try:
        registry_kwargs = get_model_config(args.model).get("mm_processor_kwargs") or {}
    except KeyError:
        registry_kwargs = {}
    mm_kwargs = dict(registry_kwargs)
    if args.max_dynamic_patch is not None:
        mm_kwargs["max_dynamic_patch"] = args.max_dynamic_patch
    if args.max_pixels is not None:
        mm_kwargs["max_pixels"] = args.max_pixels

    # Translate vLLM's tiling spelling to the one the transformers image processor honours.  Models
    # like Lingshu-I-8B resolve to the *transformers* InternVLForConditionalGeneration backend, whose
    # processor takes crop_to_patches/max_patches; `max_dynamic_patch` is only consumed by vLLM's
    # native internvl.py and is silently DROPPED here.  Untranslated, Lingshu served 13 tiles
    # (3335 tok/img) while training used 4 (1287 tok/img) — a train/serve mismatch costing 2.6x
    # prefill and KV.  See src/mm_kwargs.py.
    tiling_kwargs = mm_to_hf_image_kwargs(mm_kwargs)
    if tiling_kwargs:
        mm_kwargs = {k: v for k, v in mm_kwargs.items() if k != "max_dynamic_patch"} | tiling_kwargs

    slug = sanitize_slug(args.model)

    # --- Determine which trained variant (if any) we are serving, and the filename suffix. ---
    # "base" is the untrained baseline: normalise it to "no variant" so the output is the
    # canonical data/inference_<slug>.parquet (NOT __base) — matches run_all_inference.sh / evaluate.py.
    variant = None if args.variant == "base" else args.variant

    # The served checkpoint is determined solely by --model + --variant: base/none serves the HF
    # hub --model; a trained variant serves its merged full checkpoint (uniform across all archs).
    model_path = None
    if variant in ADAPTER_VARIANTS:
        merged = merged_dir_for(CHECKPOINTS_ROOT_DEFAULT, slug, variant)
        if not os.path.isfile(os.path.join(merged, "config.json")):
            parser.error(
                f"no merged checkpoint at {merged} — merge the adapter first:\n"
                f"  python -m src.merge_adapter --model {args.model} --variant {variant}\n"
                "  (or sweep all models: ./run_all_merge.sh --variant " + variant + ")"
            )
        model_path = merged
        LOGGER.info("serving merged checkpoint: %s (variant=%s)", merged, variant)

    # Variant suffix: data/inference_<slug>.parquet (base/none) or
    #                 data/inference_<slug>__<variant>.parquet (trained variant)
    variant_suffix = f"__{variant}" if variant else ""
    # Context suffix (orthogonal to variant): data/inference_<slug>__ablation_<context>.parquet,
    # or data/inference_<slug>__<variant>__ablation_<context>.parquet when both are set. Omitted
    # entirely when --context is not given, matching today's default output path exactly.
    context_suffix = f"__ablation_{args.context}" if args.context else ""
    output_parquet = f"data/inference_{slug}{variant_suffix}{context_suffix}.parquet"

    inference_all(
        dataset,
        args.model,
        output_parquet,
        mm_processor_kwargs=mm_kwargs or None,
        model_path=model_path,
        chat_batch_size=args.chat_batch_size,
    )
