import torch
import platform
from collections import OrderedDict


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


def normalize_to_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
    

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

def to_tensor(value, dtype: torch.dtype, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert any value to a tensor with the desired dtype/device."""
    if value is None:
        raise ValueError("[Tensor Utils Error] Cannot convert None to tensor.")
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        target_device = device if isinstance(device, torch.device) else torch.device(device)
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
    return tensor
