import torch
import platform
from collections import OrderedDict
from typing import Sequence, Union, TYPE_CHECKING


def resolve_device() -> str:
    """Select a usable device with graceful fallback."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        mac_ver = platform.mac_ver()[0]
        try:
            major, minor = (int(x) for x in mac_ver.split(".")[:2])
        except Exception:
            major, minor = 0, 0
        if major >= 14:
            return "mps"
    return "cpu"


def merge_features(primary, secondary) -> list:
    """
    Merge two feature lists while preserving order and deduplicating by feature name.
    Later duplicates are skipped.
    """
    merged: OrderedDict[str, object] = OrderedDict()
    for feat in list(primary or []) + list(secondary or []):
        merged.setdefault(feat.name, feat)
    return list(merged.values())


def get_mlp_output_dim(params: dict, fallback: int) -> int:
    """
    Get the output dimension of an MLP-like config.
    If dims are provided, use the last dim; otherwise fall back to input dim.
    """
    dims = params.get("dims")
    if dims:
        return dims[-1]
    return fallback
