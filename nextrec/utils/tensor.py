"""
Tensor manipulation utilities for NextRec

Date: create on 03/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch
from typing import Any


def to_tensor(
    value: Any, dtype: torch.dtype, device: torch.device | str | None = None
) -> torch.Tensor:
    if value is None:
        raise ValueError("[Tensor Utils Error] Cannot convert None to tensor.")
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)

    if device is not None:
        target_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
    return tensor


def stack_tensors(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if not tensors:
        raise ValueError("[Tensor Utils Error] Cannot stack empty list of tensors.")
    return torch.stack(tensors, dim=dim)


def concat_tensors(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if not tensors:
        raise ValueError(
            "[Tensor Utils Error] Cannot concatenate empty list of tensors."
        )
    return torch.cat(tensors, dim=dim)


def pad_sequence_tensors(
    tensors: list[torch.Tensor],
    max_len: int | None = None,
    padding_value: float = 0.0,
    padding_side: str = "right",
) -> torch.Tensor:
    if not tensors:
        raise ValueError("[Tensor Utils Error] Cannot pad empty list of tensors.")
    if max_len is None:
        max_len = max(t.size(0) for t in tensors)
    batch_size = len(tensors)
    padded = torch.full(
        (batch_size, max_len),
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    for i, tensor in enumerate(tensors):
        length = min(tensor.size(0), max_len)
        if padding_side == "right":
            padded[i, :length] = tensor[:length]
        elif padding_side == "left":
            padded[i, -length:] = tensor[:length]
        else:
            raise ValueError(
                f"[Tensor Utils Error] padding_side must be 'right' or 'left', got {padding_side}"
            )
    return padded
