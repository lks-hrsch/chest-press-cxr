"""Clinical-context ablation for RQ2 ("How does rich patient clinical context influence
generated reports?"): derive baseline / clinical_context / rich_clinical_context variants of
the already-rendered `input_text` / `system_prompt` columns and write one parquet per tier.

Each tier is a literal block-drop from the unchanged `input_text` produced by
`generate_chest_press_cxr.py` -- no reordering, no regenerated text:

    rich_clinical_context   unchanged input_text (today's default)
    clinical_context        drop the "## Patient Information" block (demographics/labs/microbiology)
    baseline                additionally drop the optional prior-studies block

Current-study exam metadata (indication/technique/comparison) is present in all three tiers -- it
is never ablated. This isolates RQ2's two named components: baseline -> clinical_context isolates
the effect of prior studies; clinical_context -> rich_clinical_context isolates the effect of
patient demographics + labs + microbiology.

Design and the anchor-based split logic were prototyped and validated against all 7,860 rows of
the test split in notebooks/build-baseline-test.ipynb before being ported here unchanged.
"""

import os

import polars

from ..sink_monitor import SinkMonitor


PATIENT_INFO_HEADER = "## Patient Information"
LAST_LAB_FINDINGS_PREFIX = "LAST_LAB_FINDINGS:"
PRIOR_BLOCK_HEADER = "## Clinical Context (last report is most recent):"
CURRENT_STUDY_MARKER = "## Current Study ("
CURRENT_STUDY_CONTEXT_HEADER = "Clinical Context (last report is most recent):"

# rich_clinical_context and clinical_context intentionally share the SAME system_prompt -- both
# tiers genuinely have clinical context (current study, optionally priors), so keeping the wording
# constant isolates the ablation to input content, not prompt wording. Only baseline's prompt is
# edited, since it lacks priors/demographics/labs/microbiology entirely.
BASELINE_SYSTEM_PROMPT = (
    "You are an expert radiologist.\n"
    "You are given a chest X-Ray study (one or two images in different views) along with the "
    "current study's exam details (indication, technique, comparison).\n"
    "Write the IMPRESSION and FINDINGS sections of the report."
)

TIERS = ("baseline", "clinical_context", "rich_clinical_context")

_TIERS_STRUCT_DTYPE = polars.Struct(
    {"rich_clinical_context": polars.Utf8, "clinical_context": polars.Utf8, "baseline": polars.Utf8}
)


def split_input_text(input_text: str) -> dict:
    """Split a rich_clinical_context input_text into its structural pieces.

    Must anchor off CURRENT_STUDY_MARKER first, not the no-## context header text directly --
    that text is a literal substring of PRIOR_BLOCK_HEADER, so searching for it naively would
    match inside the prior block on has-prior rows.
    """
    if not input_text.startswith(PATIENT_INFO_HEADER):
        raise ValueError("missing '## Patient Information' header -- unexpected input_text shape")
    if LAST_LAB_FINDINGS_PREFIX not in input_text:
        raise ValueError("missing 'LAST_LAB_FINDINGS:' anchor -- unexpected input_text shape")
    if input_text.count(CURRENT_STUDY_MARKER) != 1:
        raise ValueError("'## Current Study (' anchor is not unique -- unexpected input_text shape")

    cur_idx = input_text.index(CURRENT_STUDY_MARKER)
    current_study = input_text[cur_idx:].rstrip("\n")

    close_idx = current_study.index("):")
    if not current_study[close_idx + 2 :].startswith("\n" + CURRENT_STUDY_CONTEXT_HEADER):
        raise ValueError("current-study header does not immediately follow '):' -- unexpected input_text shape")

    prior_idx = input_text.find(PRIOR_BLOCK_HEADER)
    has_prior = prior_idx != -1 and prior_idx < cur_idx
    prior_studies = input_text[prior_idx:cur_idx].rstrip("\n") if has_prior else ""

    return {"current_study": current_study, "prior_studies": prior_studies, "has_prior": has_prior}


def build_context_tiers(input_text: str) -> dict:
    """Derive the three context tiers as pure slices of the unchanged `input_text`."""
    parts = split_input_text(input_text)
    clinical_context = (
        parts["prior_studies"] + "\n\n" + parts["current_study"] if parts["has_prior"] else parts["current_study"]
    )
    return {
        "rich_clinical_context": input_text,
        "clinical_context": clinical_context,
        "baseline": parts["current_study"],
    }


def validate_input_texts(input_texts) -> None:
    """Fail fast if any row's `input_text` doesn't match the expected template shape."""
    failures = []
    for i, text in enumerate(input_texts):
        try:
            split_input_text(text)
        except ValueError as exc:
            failures.append((i, str(exc)))
    if failures:
        raise ValueError(
            f"{len(failures)} / {len(input_texts)} rows do not match the expected input_text "
            f"template shape -- first failures: {failures[:5]}"
        )


def build_context_ablation_frames(df: polars.DataFrame) -> dict[str, polars.DataFrame]:
    """Return `{tier: DataFrame}` for all three tiers.

    Each returned frame is a drop-in replacement for `df`: every column is unchanged except
    `input_text`/`system_prompt`, which are swapped to that tier's variant -- so downstream code
    (e.g. src/inference.py's `build_conversation`) needs no changes, only a different source path.
    """
    validate_input_texts(df["input_text"].to_list())
    if df["system_prompt"].n_unique() != 1:
        raise ValueError("expected a single fixed system_prompt across all rows")
    rich_system_prompt = df["system_prompt"][0]

    tiers_struct = polars.col("input_text").map_elements(build_context_tiers, return_dtype=_TIERS_STRUCT_DTYPE)
    with_tiers = df.with_columns(tiers_struct.alias("_tiers"))

    system_prompt_by_tier = {
        "rich_clinical_context": rich_system_prompt,
        "clinical_context": rich_system_prompt,
        "baseline": BASELINE_SYSTEM_PROMPT,
    }

    return {
        tier: with_tiers.with_columns(
            polars.col("_tiers").struct.field(tier).alias("input_text"),
            polars.lit(system_prompt_by_tier[tier]).alias("system_prompt"),
        ).drop("_tiers")
        for tier in TIERS
    }


def write_context_ablation_splits(source_path: str, output_dir: str = "data") -> dict[str, str]:
    """Read `source_path`, derive the 3 tiers, and write one parquet per tier to `output_dir`,
    named `<source_stem>__<tier>.parquet`. Returns `{tier: output_path}`.
    """
    df = polars.read_parquet(source_path)
    frames = build_context_ablation_frames(df)

    stem = os.path.splitext(os.path.basename(source_path))[0]
    output_paths = {}
    for tier, frame in frames.items():
        output_path = os.path.join(output_dir, f"{stem}__{tier}.parquet")
        with SinkMonitor(output_path):
            frame.write_parquet(output_path)
        output_paths[tier] = output_path
    return output_paths


def main() -> None:
    # Test split only, matching the current scope. Pass a different source_path (e.g. the valid
    # split) to write_context_ablation_splits() directly for other splits.
    source_path = "data/chest_press_cxr_test.parquet"
    print(f"Building context-ablation splits from {source_path}...")
    output_paths = write_context_ablation_splits(source_path)
    for tier, path in output_paths.items():
        print(f"  {tier:<22} -> {path}")


if __name__ == "__main__":
    main()
