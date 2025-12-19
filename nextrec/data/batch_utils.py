"""
Batch collation utilities for NextRec

Date: create on 03/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from typing import Any, Mapping

import numpy as np
import torch


def stack_section(batch: list[dict], section: str):
    entries = [item.get(section) for item in batch if item.get(section) is not None]
    if not entries:
        return None
    merged: dict = {}
    for name in entries[0]:  # type: ignore
        tensors = [
            item[section][name]
            for item in batch
            if item.get(section) is not None and name in item[section]
        ]
        merged[name] = torch.stack(tensors, dim=0)
    return merged


def collate_fn(batch):
    """
    Collate a list of sample dicts into the unified batch format:
    {
        "features": {name: Tensor(B, ...)},
        "labels": {target: Tensor(B, ...)} or None,
        "ids": {id_name: Tensor(B, ...)} or None,
    }
    Args: batch: List of samples from DataLoader

    Returns: dict: Batched data in unified format
    """
    if not batch:
        return {"features": {}, "labels": None, "ids": None}

    first = batch[0]
    if isinstance(first, dict) and "features" in first:
        # Streaming dataset yields already-batched chunks; avoid adding an extra dim.
        if first.get("_already_batched") and len(batch) == 1:
            return {
                "features": first.get("features", {}),
                "labels": first.get("labels"),
                "ids": first.get("ids"),
            }
        return {
            "features": stack_section(batch, "features") or {},
            "labels": stack_section(batch, "labels"),
            "ids": stack_section(batch, "ids"),
        }

    # Fallback: stack tuples/lists of tensors
    num_tensors = len(first)
    result = []
    for i in range(num_tensors):
        tensor_list = [item[i] for item in batch]
        first_item = tensor_list[0]
        if isinstance(first_item, torch.Tensor):
            stacked = torch.cat(tensor_list, dim=0)
        elif isinstance(first_item, np.ndarray):
            stacked = np.concatenate(tensor_list, axis=0)
        elif isinstance(first_item, list):
            combined = []
            for entry in tensor_list:
                combined.extend(entry)
            stacked = combined
        else:
            stacked = tensor_list
        result.append(stacked)
    return tuple(result)


def batch_to_dict(batch_data: Any, include_ids: bool = True) -> dict:
    if not (isinstance(batch_data, Mapping) and "features" in batch_data):
        raise TypeError(
            "[BaseModel-batch_to_dict Error] Batch data must be a dict with 'features' produced by the current DataLoader."
        )
    return {
        "features": batch_data.get("features", {}),
        "labels": batch_data.get("labels"),
        "ids": batch_data.get("ids") if include_ids else None,
    }
