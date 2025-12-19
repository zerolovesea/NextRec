"""
Data processing utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 19/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch


def get_column_data(data: dict | pd.DataFrame, name: str):

    if isinstance(data, dict):
        return data[name] if name in data else None
    elif isinstance(data, pd.DataFrame):
        if name not in data.columns:
            return None
        return data[name].values
    else:
        if hasattr(data, name):
            return getattr(data, name)
        raise KeyError(f"Unsupported data type for extracting column {name}")


def split_dict_random(data_dict, test_size=0.2, random_state=None):

    lengths = [len(v) for v in data_dict.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Length mismatch: {lengths}")

    n = lengths[0]
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    cut = int(round(n * (1 - test_size)))
    train_idx, test_idx = perm[:cut], perm[cut:]

    def take(v, idx):
        if isinstance(v, np.ndarray):
            return v[idx]
        elif isinstance(v, pd.Series):
            return v.iloc[idx].to_numpy()
        else:
            v_arr = np.asarray(v, dtype=object)
            return v_arr[idx]

    train_dict = {k: take(v, train_idx) for k, v in data_dict.items()}
    test_dict = {k: take(v, test_idx) for k, v in data_dict.items()}
    return train_dict, test_dict


def split_data(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    valid_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, valid_df


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
    user_hist_items = {
        u: df_all[df_all[user_col] == u][item_col].unique() for u in users
    }

    for u in users:
        df_user = df_all[df_all[user_col] == u]
        pos_items = df_user[df_user[label_col] == 1][item_col].unique()
        if len(pos_items) == 0:
            continue
        pos_items = pos_items[:num_pos_per_user]
        seen_items = set(user_hist_items[u])
        neg_pool = np.setdiff1d(
            all_items, np.fromiter(seen_items, dtype=all_items.dtype)
        )
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


def get_user_ids(
    data: Any, id_columns: list[str] | str | None = None
) -> np.ndarray | None:
    """
    Extract user IDs from various data structures.

    Args:
        data: Data source (DataFrame, dict, or batch dict)
        id_columns: List or single ID column name(s) (default: None)

    Returns:
        np.ndarray | None: User IDs as numpy array, or None if not found
    """
    id_columns = (
        id_columns
        if isinstance(id_columns, list)
        else [id_columns] if isinstance(id_columns, str) else []
    )
    if not id_columns:
        return None

    main_id = id_columns[0]
    if isinstance(data, pd.DataFrame) and main_id in data.columns:
        arr = np.asarray(data[main_id].values)
        return arr.reshape(arr.shape[0])

    if isinstance(data, dict):
        ids_container = data.get("ids")
        if isinstance(ids_container, dict) and main_id in ids_container:
            val = ids_container[main_id]
            val = (
                val.detach().cpu().numpy()
                if isinstance(val, torch.Tensor)
                else np.asarray(val)
            )
            return val.reshape(val.shape[0])
        if main_id in data:
            arr = np.asarray(data[main_id])
            return arr.reshape(arr.shape[0])

    return None


def hash_md5_mod(value: str, hash_size: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % hash_size
