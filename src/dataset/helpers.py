from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


# unfold StudyDate and StudyTime into a single StudyDateTime column like "2180-03-23 11:51:00"
# current format is StudyDate: "21800323", StudyTime: "115100.000000"
# which is "YYYYMMDD", "HHMMSS.ssssss"
# NOTE: polars infers StudyTime as a float, so a pre-10:00 time like "091500.000000" arrives as
# 91500.0 → "91500" with the leading zero already dropped.  Pad on the LEFT (zfill) so it becomes
# "091500" → "09:15:00".  ljust(6, "0") would pad on the RIGHT → "915000" → a bogus "91:50:00",
# which both corrupts the prompt timestamp and breaks the storetime bisect (the inflated time sorts
# after every real same-day event, leaking post-study labs into the context).
def format_datetime(study_date: int | str, study_time: float | str) -> str:
    date_str = str(study_date)
    padded_study_time = str(study_time).split(".")[0].zfill(6)  # left-pad to HHMMSS

    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    time_str = f"{padded_study_time[:2]}:{padded_study_time[2:4]}:{padded_study_time[4:6]}"
    return f"{date_str} {time_str}"


def assign_subjects_to_splits(
    subject_ids: Sequence[T],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> tuple[set[T], set[T], set[T]]:
    """Partition (already-ordered) subject IDs into train/valid/test sets by position.

    ``train`` takes the first ``train_ratio`` of the list, ``valid`` the next ``valid_ratio``, and
    ``test`` the remainder — so the ratios need not sum to 1 and any int-flooring leftover lands in
    test.  Splitting by ``subject_id`` keeps every study for a patient in a single split (no
    leakage).  Boundaries use the same ``int(...)`` flooring as the original split script, so
    re-running over the same shuffled order reproduces the identical split.  Pure and deterministic
    given the input order; the caller performs any (seeded) shuffle beforehand.
    """
    n = len(subject_ids)
    train_end = int(train_ratio * n)
    valid_end = int((train_ratio + valid_ratio) * n)
    return (
        set(subject_ids[:train_end]),
        set(subject_ids[train_end:valid_end]),
        set(subject_ids[valid_end:]),
    )
