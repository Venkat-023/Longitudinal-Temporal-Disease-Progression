from __future__ import annotations

import sys

import numpy as np


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Fail early if model-ready arrays contain invalid values."""
    problems: list[str] = []
    for name, arr in arrays.items():
        if arr.size == 0:
            problems.append(f"{name} is empty")
            continue
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            problems.append(f"{name} has {n_nan} NaN and {n_inf} infinite values")

    for label_name in ["y_train", "y_val", "y_test"]:
        if label_name in arrays:
            labels = arrays[label_name]
            bad = set(np.unique(labels).tolist()) - {0, 1}
            if bad:
                problems.append(f"{label_name} contains non-binary labels: {sorted(bad)}")

    if problems:
        print("\nInvalid preprocessed arrays:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nRebuild the dataset with:")
        print("  python src/preprocessing/build_cardiac_progression_dataset.py --zip data/mimic_iv_raw/mimic-iv-2-1.zip --chunk 100000 --seq_len 6")
        sys.exit(1)
