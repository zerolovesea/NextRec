"""
Feature processing utilities for NextRec

Date: create on 03/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""


def normalize_to_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
