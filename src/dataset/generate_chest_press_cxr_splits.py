"""
Create train, validation, and test splits for the CXR report retrieval dataset.
Uses an 80-10-10 split based on unique patient IDs to ensure no data leakage between splits.
"""

import random

import polars

from ..settings import RANDOM_SEED_VALUE
from ..sink_monitor import SinkMonitor
from .helpers import assign_subjects_to_splits


INPUT_DATA_PATH = "data/chest_press_cxr.parquet"
TRAIN_OUTPUT_PATH = "data/chest_press_cxr_train.parquet"
VALID_OUTPUT_PATH = "data/chest_press_cxr_valid.parquet"
TEST_OUTPUT_PATH = "data/chest_press_cxr_test.parquet"

random.seed(RANDOM_SEED_VALUE)


# load the dataset
df = polars.scan_parquet(INPUT_DATA_PATH)

# get unique patient IDs
unique_patient_ids = df.select("subject_id").unique().collect().to_series().to_list()
# shuffle patient IDs
random.shuffle(unique_patient_ids)

# split patient IDs into train, valid, test (80-10-10) — boundary/flooring contract lives in
# assign_subjects_to_splits (kept identical so a re-run reproduces the same split).
train_ids, valid_ids, test_ids = assign_subjects_to_splits(unique_patient_ids)

# create splits
train_df = df.filter(polars.col("subject_id").is_in(train_ids))
valid_df = df.filter(polars.col("subject_id").is_in(valid_ids))
test_df = df.filter(polars.col("subject_id").is_in(test_ids))

# save splits to parquet
with SinkMonitor(TRAIN_OUTPUT_PATH):
    train_df.sink_parquet(TRAIN_OUTPUT_PATH)

with SinkMonitor(VALID_OUTPUT_PATH):
    valid_df.sink_parquet(VALID_OUTPUT_PATH)

with SinkMonitor(TEST_OUTPUT_PATH):
    test_df.sink_parquet(TEST_OUTPUT_PATH)
