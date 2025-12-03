"""
Model-related utilities for NextRec

Date: create on 03/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from collections import OrderedDict


def merge_features(primary, secondary) -> list:
    merged: OrderedDict[str, object] = OrderedDict()
    for feat in list(primary or []) + list(secondary or []):
        merged.setdefault(feat.name, feat)
    return list(merged.values())


def get_mlp_output_dim(params: dict, fallback: int) -> int:
    dims = params.get("dims")
    if dims:
        return dims[-1]
    return fallback
