"""
Data processing utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 07/02/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import polars as pl


def take_split_values(values: Any, indices: np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values[indices]
    if isinstance(values, pd.Series):
        return values.iloc[indices].to_numpy()
    return np.asarray(values, dtype=object)[indices]


def get_column_data(data: dict | pd.DataFrame | pl.DataFrame, name: str):
    """
    Get a column from dict/pandas/polars input.
    """
    if isinstance(data, dict):
        return data[name] if name in data else None
    if isinstance(data, pd.DataFrame):
        return data[name].values
    if isinstance(data, pl.DataFrame):
        return data.get_column(name).to_numpy()
    raise KeyError(f"Only dict or DataFrame supported, got {type(data)}")


def to_column_names(columns: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Convert optional column name inputs to a list of strings."""
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return [str(column) for column in columns]


def has_column(data: dict | pd.DataFrame | pl.DataFrame, name: str) -> bool:
    """Return True when the input contains the specified column/key."""
    if isinstance(data, dict):
        return name in data
    if isinstance(data, pd.DataFrame):
        return name in data.columns
    if isinstance(data, pl.DataFrame):
        return name in data.columns
    return False


def get_col_numpy(data: dict | pd.DataFrame | pl.DataFrame, name: str) -> np.ndarray:
    """
    Get a column as numpy array, raising error if not found.
    """
    values = get_column_data(data, name)
    if values is None:
        raise KeyError(f"Column '{name}' not found")
    return np.asarray(values)


def slice_data_by_indices(
    data: dict | pd.DataFrame | pl.DataFrame, indices: np.ndarray
) -> dict | pd.DataFrame | pl.DataFrame:
    """Slice dict/pandas/polars data by row indices."""

    indices = np.asarray(indices)

    if isinstance(data, pd.DataFrame):
        return data.iloc[indices].reset_index(drop=True)

    if isinstance(data, pl.DataFrame):
        return data[indices]

    if isinstance(data, dict):
        return {k: take_split_values(v, indices) for k, v in data.items()}

    raise TypeError(f"Unsupported data type: {type(data)}")


def get_data_length(data: Any) -> int | None:
    if isinstance(data, pd.DataFrame):
        return len(data)
    if isinstance(data, pl.DataFrame):
        return data.height
    if isinstance(data, dict):
        if not data:
            return None
        sample_key = next(iter(data))
        return len(data[sample_key])
    return None


def split_train_valid(
    train_data: dict | pd.DataFrame | pl.DataFrame,
    valid_split: float,
    split_stratify_by: str | None = None,
    split_group_by: str | None = None,
    random_state: int = 42,
) -> tuple[dict | pd.DataFrame | pl.DataFrame, dict | pd.DataFrame | pl.DataFrame]:
    if not (0 < valid_split < 1):
        raise ValueError(f"[BaseModel-validation Error] valid_split must be between 0 and 1, got {valid_split}")

    rng = np.random.default_rng(random_state)

    if isinstance(train_data, pd.DataFrame):
        total_length = len(train_data)
    elif isinstance(train_data, pl.DataFrame):
        total_length = train_data.height
    elif isinstance(train_data, dict):
        if not train_data:
            raise ValueError("[BaseModel-validation Error] train_data dict is empty.")
        sample_key = next(iter(train_data))
        total_length = len(train_data[sample_key])
        for key, values in train_data.items():
            if len(values) != total_length:
                raise ValueError(f"[BaseModel-validation Error] Length mismatch: {key} != {sample_key}")
    else:
        raise TypeError(f"[BaseModel-validation Error] Unsupported data type: {type(train_data)}")

    if split_group_by is not None:
        group_values = get_col_numpy(train_data, split_group_by)
        unique_groups, inverse_group_idx = np.unique(group_values, return_inverse=True)
        group_indices = np.arange(len(unique_groups))

        if split_stratify_by is None:
            shuffled = rng.permutation(group_indices)
            split_idx = int(len(shuffled) * (1 - valid_split))
            train_group_idx = shuffled[:split_idx]
            valid_group_idx = shuffled[split_idx:]
        else:
            stratify_values = get_col_numpy(train_data, split_stratify_by)
            group_labels = []
            for group_idx in group_indices:
                mask = inverse_group_idx == group_idx
                labels = np.unique(stratify_values[mask])
                if len(labels) == 1:
                    group_labels.append(str(labels[0]))
                else:
                    group_labels.append("|".join(sorted(map(str, labels))))
            group_labels = np.asarray(group_labels)

            train_parts, valid_parts = [], []
            for label in np.unique(group_labels):
                idxs = group_indices[group_labels == label]
                shuffled = rng.permutation(idxs)
                split_idx = int(len(shuffled) * (1 - valid_split))
                train_parts.append(shuffled[:split_idx])
                valid_parts.append(shuffled[split_idx:])

            train_group_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
            valid_group_idx = np.concatenate(valid_parts) if valid_parts else np.array([], dtype=int)

        train_indices = np.flatnonzero(np.isin(inverse_group_idx, train_group_idx))
        valid_indices = np.flatnonzero(np.isin(inverse_group_idx, valid_group_idx))

    elif split_stratify_by is not None:
        stratify_values = get_col_numpy(train_data, split_stratify_by)
        all_indices = np.arange(total_length)
        train_parts, valid_parts = [], []

        for label in np.unique(stratify_values):
            idxs = all_indices[stratify_values == label]
            shuffled = rng.permutation(idxs)
            split_idx = int(len(shuffled) * (1 - valid_split))
            train_parts.append(shuffled[:split_idx])
            valid_parts.append(shuffled[split_idx:])

        train_indices = np.concatenate(train_parts)
        valid_indices = np.concatenate(valid_parts)

    else:
        indices = rng.permutation(total_length)
        split_idx = int(total_length * (1 - valid_split))
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]

    if len(train_indices) == 0 or len(valid_indices) == 0:
        raise ValueError("Split produced empty train or valid set")

    train_indices = rng.permutation(train_indices)
    valid_indices = rng.permutation(valid_indices)

    train_split_data = slice_data_by_indices(train_data, train_indices)
    valid_split_data = slice_data_by_indices(train_data, valid_indices)
    return train_split_data, valid_split_data


def split_dict_random(data_dict, test_size=0.2, random_state=None):

    lengths = [len(v) for v in data_dict.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Length mismatch: {lengths}")

    n = lengths[0]
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    cut = int(round(n * (1 - test_size)))
    train_idx, test_idx = perm[:cut], perm[cut:]

    train_dict = {k: take_split_values(v, train_idx) for k, v in data_dict.items()}
    test_dict = {k: take_split_values(v, test_idx) for k, v in data_dict.items()}
    return train_dict, test_dict


def build_eval_candidates(
    df_all: pd.DataFrame,
    user_col: str,
    item_col: str,
    label_col: str,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    num_pos_per_user: int = 5,
    num_neg_per_pos: int = 50,
    random_seed: int = 2025,
) -> pd.DataFrame:
    """
    Build evaluation candidates with positive and negative samples for each user.

    Args:
        df_all: Full interaction DataFrame
        user_col: Name of the user ID column
        item_col: Name of the item ID column
        label_col: Name of the label column
        user_features: DataFrame containing user features
        item_features: DataFrame containing item features
        num_pos_per_user: Number of positive samples per user (default: 5)
        num_neg_per_pos: Number of negative samples per positive (default: 50)
        random_seed: Random seed for reproducibility (default: 2025)

    Returns:
        pd.DataFrame: Evaluation candidates with features
    """
    rng = np.random.default_rng(random_seed)

    users = df_all[user_col].unique()
    all_items = item_features[item_col].unique()
    rows = []
    user_hist_items = {u: df_all[df_all[user_col] == u][item_col].unique() for u in users}

    for u in users:
        df_user = df_all[df_all[user_col] == u]
        pos_items = df_user[df_user[label_col] == 1][item_col].unique()
        if len(pos_items) == 0:
            continue
        pos_items = pos_items[:num_pos_per_user]
        seen_items = set(user_hist_items[u])
        neg_pool = np.setdiff1d(all_items, np.fromiter(seen_items, dtype=all_items.dtype))
        if len(neg_pool) == 0:
            continue
        for pos in pos_items:
            if len(neg_pool) <= num_neg_per_pos:
                neg_items = neg_pool
            else:
                neg_items = rng.choice(neg_pool, size=num_neg_per_pos, replace=False)
            rows.append((u, pos, 1))
            for ni in neg_items:
                rows.append((u, ni, 0))

    eval_df = pd.DataFrame(rows, columns=[user_col, item_col, label_col])
    eval_df = eval_df.merge(user_features, on=user_col, how="left")
    eval_df = eval_df.merge(item_features, on=item_col, how="left")
    return eval_df


def get_group_ids(data: Any, group_id: str | None = None) -> np.ndarray | None:
    """
    Extract group IDs from various data structures.

    Args:
        data: Data source (DataFrame, dict, or batch dict)
        group_id: Key column name used as the group id.

    Returns:
        np.ndarray | None: Group IDs as numpy array, or None if not found
    """
    if not group_id:
        return None

    if isinstance(data, pd.DataFrame) and group_id in data.columns:
        arr = np.asarray(data[group_id].values)
        return arr.reshape(arr.shape[0])

    if isinstance(data, pl.DataFrame) and group_id in data.columns:
        arr = data.get_column(group_id).to_numpy()
        return arr.reshape(arr.shape[0])

    if isinstance(data, dict):
        keys_container = data.get("keys")
        if isinstance(keys_container, dict) and group_id in keys_container:
            val = keys_container[group_id]
            val = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
            return val.reshape(val.shape[0])
        if group_id in data:
            arr = np.asarray(data[group_id])
            return arr.reshape(arr.shape[0])

    return None
