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


def select_features(
    available_features: list,
    names: list[str],
    param_name: str,
) -> list:
    if not names:
        return []

    if len(names) != len(set(names)):
        raise ValueError(f"{param_name} contains duplicate feature names: {names}")

    feature_map = {feat.name: feat for feat in available_features}
    missing = [name for name in names if name not in feature_map]
    if missing:
        raise ValueError(
            f"{param_name} contains unknown feature names {missing}. "
            f"Available features: {list(feature_map)}"
        )

    return [feature_map[name] for name in names]
