"""
Feature processing utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 19/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import numbers
from typing import Any


def normalize_to_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def as_float(value: Any) -> float | None:
    if isinstance(value, numbers.Number):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    return None
