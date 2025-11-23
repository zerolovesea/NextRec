"""Data processing utilities for NextRec."""

import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def collate_fn(batch):
    """Collate a list of tensor tuples from ``FileDataset`` into batched tensors."""
    if not batch:
        return tuple()

    num_tensors = len(batch[0])
    result = []
    
    for i in range(num_tensors):
        tensor_list = [item[i] for item in batch]
        first = tensor_list[0]

        if isinstance(first, torch.Tensor):
            stacked = torch.cat(tensor_list, dim=0)
        elif isinstance(first, np.ndarray):
            stacked = np.concatenate(tensor_list, axis=0)
        elif isinstance(first, list):
            combined = []
            for entry in tensor_list:
                combined.extend(entry)
            stacked = combined
        else:
            stacked = tensor_list

        result.append(stacked)
    
    return tuple(result)


def get_column_data(data: dict | pd.DataFrame, name: str):
    """Extract column data from various data structures."""
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


def resolve_file_paths(path: str) -> tuple[list[str], str]:
    """Resolve file or directory path into a sorted list of files and file type."""
    path_obj = Path(path)

    if path_obj.is_file():
        file_type = path_obj.suffix.lower().lstrip(".")
        assert file_type in ["csv", "parquet"], f"Unsupported file extension: {file_type}"
        return [str(path_obj)], file_type

    if path_obj.is_dir():
        collected_files = [p for p in path_obj.iterdir() if p.is_file()]
        csv_files = [str(p) for p in collected_files if p.suffix.lower() == ".csv"]
        parquet_files = [str(p) for p in collected_files if p.suffix.lower() == ".parquet"]

        if csv_files and parquet_files:
            raise ValueError("Directory contains both CSV and Parquet files. Please keep a single format.")
        file_paths = csv_files if csv_files else parquet_files
        if not file_paths:
            raise ValueError(f"No CSV or Parquet files found in directory: {path}")
        file_paths.sort()
        file_type = "csv" if csv_files else "parquet"
        return file_paths, file_type

    raise ValueError(f"Invalid path: {path}")


def iter_file_chunks(file_path: str, file_type: str, chunk_size: int):
    """Yield DataFrame chunks for CSV/Parquet without loading the whole file."""
    if file_type == "csv":
        yield from pd.read_csv(file_path, chunksize=chunk_size)
        return
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()


def read_table(file_path: str, file_type: str) -> pd.DataFrame:
    """Read a single CSV/Parquet file."""
    if file_type == "csv":
        return pd.read_csv(file_path)
    return pd.read_parquet(file_path)


def load_dataframes(file_paths: list[str], file_type: str) -> list[pd.DataFrame]:
    """Load multiple files of the same type into DataFrames."""
    return [read_table(fp, file_type) for fp in file_paths]


def default_output_dir(path: str) -> Path:
    """Generate a default output directory path based on the input path."""
    path_obj = Path(path)
    if path_obj.is_file():
        return path_obj.parent / f"{path_obj.stem}_preprocessed"
    return path_obj.with_name(f"{path_obj.name}_preprocessed")


def split_dict_random(data_dict: dict, test_size: float=0.2, random_state:int|None=None):
    """Randomly split a dictionary of data into training and testing sets."""
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
    test_dict  = {k: take(v, test_idx)  for k, v in data_dict.items()}
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
    """Build evaluation candidates with positive and negative samples for each user.   """
    rng = np.random.default_rng(random_seed)

    users = df_all[user_col].unique()
    all_items = item_features[item_col].unique()

    rows = []

    user_hist_items = {
        u: df_all[df_all[user_col] == u][item_col].unique()
        for u in users
    }

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
    eval_df = eval_df.merge(user_features, on=user_col, how='left')
    eval_df = eval_df.merge(item_features, on=item_col, how='left')
    return eval_df
