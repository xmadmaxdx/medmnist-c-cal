"""Shared named constants. No magic numbers in code."""
from __future__ import annotations

IMAGE_SIZE = 28
NUM_BLOOD_CLASSES = 8
NUM_DERMA_CLASSES = 7
ECE_BINS = 15
LOG_EPS = 1e-8
RANDOM_SEED_DEFAULT = 0
VAL_FRACTION = 0.1
SMOKE_SUBSET = 200
SMOKE_EPOCHS = 1
CORRUPTION_SEVERITIES = (1, 2, 3, 4, 5)
CALIBRATION_COVERAGE_NOTE = "ECE-15 + NLL + Brier on clean and corrupted splits."
