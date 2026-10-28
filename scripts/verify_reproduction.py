"""Compare regenerated experiment outputs with committed reference outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compare_json(expected: Any, actual: Any, location: str = "root") -> None:
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"keys differ at {location}")
        for key in expected:
            compare_json(expected[key], actual[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"length differs at {location}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            compare_json(expected_item, actual_item, f"{location}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isclose(expected, actual, rtol=1e-7, atol=1e-9):
            raise AssertionError(f"numeric value differs at {location}: {expected} != {actual}")
    elif expected != actual:
        raise AssertionError(f"value differs at {location}: {expected!r} != {actual!r}")


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: verify_reproduction.py EXPECTED_JSON ACTUAL_JSON "
            "EXPECTED_CSV ACTUAL_CSV"
        )
    expected_json, actual_json, expected_csv, actual_csv = map(Path, sys.argv[1:])
    compare_json(json.loads(expected_json.read_text()), json.loads(actual_json.read_text()))
    expected_frame = pd.read_csv(expected_csv)
    actual_frame = pd.read_csv(actual_csv)
    pd.testing.assert_frame_equal(
        expected_frame,
        actual_frame,
        check_exact=False,
        rtol=1e-7,
        atol=1e-9,
    )
    print("Regenerated metrics and predictions match the committed reference outputs.")


if __name__ == "__main__":
    main()
