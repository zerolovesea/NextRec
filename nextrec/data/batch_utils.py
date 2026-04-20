"""
Batch collation utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 20/04/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from typing import Literal

import numpy as np
import torch


def stack_section(batch: list[dict], section: Literal["features", "labels", "keys"]):
    """
    input example:
    batch = [
        {"features": {"f1": tensor1, "f2": tensor2}, "labels": {"label": tensor3}},
        {"features": {"f1": tensor4, "f2": tensor5}, "labels": {"label": tensor6}},
        ...
    ]
    output example:
    {
        "f1": torch.stack([tensor1, tensor4], dim=0),
        "f2": torch.stack([tensor2, tensor5], dim=0),
    }

    """
    entries = [item.get(section) for item in batch if item.get(section) is not None]
    if not entries:
        return None
    merged: dict = {}
    for name in entries[0]:  # type: ignore
        tensors = [item[section][name] for item in batch if item.get(section) is not None and name in item[section]]
        tensor_sample = tensors[0]
        if isinstance(tensor_sample, torch.Tensor):
            merged[name] = torch.stack(tensors, dim=0)
        elif isinstance(tensor_sample, np.ndarray):
            merged[name] = np.stack(tensors, axis=0)
        else:
            merged[name] = tensors
    return merged


def collate_fn(batch):
    """
    Collate a list of sample dicts into the unified batch format:
    {
        "features": {name: Tensor(B, ...)},
        "labels": {target: Tensor(B, ...)} or None,
        "keys": {key_name: Tensor(B, ...)} or None,
    }
    Args: batch: List of samples from DataLoader

    Returns: dict: Batched data in unified format
    """
    if not batch:
        return {"features": {}, "labels": None, "keys": None, "schema": None}

    first = batch[0]
    # Streaming dataset yields already-batched chunks; avoid adding an extra dim.
    if first.get("stream_mode") and len(batch) == 1:
        return {
            "features": first["features"],
            "labels": first["labels"],
            "keys": first["keys"],
            "schema": first["schema"],
        }
    return {
        "features": stack_section(batch, "features") or {},
        "labels": stack_section(batch, "labels"),
        "keys": stack_section(batch, "keys"),
        "schema": first["schema"],
    }
