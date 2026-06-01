"""
Dataloader definitions

Date: create on 27/10/2025
Checkpoint: edit on 20/04/2026
Author: Yang Zhou,zyaztec@gmail.com
"""

from __future__ import annotations

import logging
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from nextrec.basic.features import (
    DenseFeature,
    FeatureSet,
    SemanticIdFeature,
    SequenceFeature,
    SparseFeature,
)
from nextrec.data.batch_utils import collate_fn
from nextrec.data.data_processing import (
    get_column_data,
    has_column,
    parse_sequence_value,
    to_column_names,
    to_object_array,
)
from nextrec.data.preprocessor import DataProcessor
from nextrec.utils.data import (
    expand_tabular_rows,
    get_file_paths,
    is_path_data,
    iter_file_chunks,
    read_table,
)
from nextrec.utils.timing import StageTimer
from nextrec.utils.torch_utils import to_tensor
from nextrec.utils.types import BatchSchema


class TensorDictDataset(Dataset):
    """Dataset returning sample-level dicts matching the unified batch schema."""

    def __init__(self, tensors: dict):
        self.features = tensors.get("features", {})
        self.labels = tensors.get("labels")
        self.keys = tensors.get("keys")
        self.schema = tensors.get("schema")
        if not self.features:
            raise ValueError("[TensorDictDataset Error] Dataset requires at least one feature tensor.")
        lengths = [tensor.shape[0] for tensor in self.features.values()]
        if not lengths:
            raise ValueError("[TensorDictDataset Error] Feature tensors are empty.")
        self.length = lengths[0]
        for length in lengths[1:]:
            if length != self.length:
                raise ValueError("[TensorDictDataset Error] All feature tensors must have the same length.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        sample_features = {name: tensor[idx] for name, tensor in self.features.items()}
        sample_labels = {name: tensor[idx] for name, tensor in self.labels.items()} if self.labels else None
        sample_keys = {name: tensor[idx] for name, tensor in self.keys.items()} if self.keys else None
        return {"features": sample_features, "labels": sample_labels, "keys": sample_keys, "schema": self.schema}


class FileDataset(FeatureSet, IterableDataset):
    def __init__(
        self,
        file_paths: list[str],
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        semantic_id_features: list[SemanticIdFeature] | None,
        target_columns: list[str],
        key_columns: list[str] | None = None,
        chunk_size: int = 10000,
        file_type: str = "csv",
        processor: DataProcessor | None = None,
        shard_rank: int = 0,
        shard_count: int = 1,
        profiler: StageTimer | None = None,
        expand: dict[str, list] | None = None,
        schema: BatchSchema | None = None,
    ):
        """Streaming dataset for reading files in chunks.

        Args:
            file_paths: List of file paths to read
            dense_features: Dense feature definitions
            sparse_features: Sparse feature definitions
            sequence_features: Sequence feature definitions
            target_columns: Target column names
            key_columns: Key columns to carry through
            chunk_size: Number of rows per chunk
            file_type: Format type (csv, parquet, etc.)
            processor: Optional DataProcessor for transformation
        """
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.file_type = file_type
        self.processor = processor
        self.shard_rank = int(shard_rank)
        self.shard_count = int(shard_count)
        self.profiler = profiler
        self.expand = expand or {}
        self.schema = dict(schema or {})

        self.set_all_features(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target_columns,
            key_columns=key_columns,
            semantic_id_features=semantic_id_features,
        )
        self.total_files = len(file_paths)

    def __iter__(self):
        base_shard_count = max(int(self.shard_count), 1)
        base_shard_rank = int(self.shard_rank) if base_shard_count > 1 else 0
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1
        shard_count = max(base_shard_count * worker_count, 1)
        shard_rank = base_shard_rank * worker_count + worker_id

        # assign files to each worker
        file_indices_all = list(range(self.total_files))
        # For a single input file, keep it on every shard and split by chunk/row-group below.
        # Otherwise, non-zero shards would receive no files and produce empty outputs.
        if shard_count > 1 and self.total_files > 1:
            file_indices_all = [idx for idx in file_indices_all if (idx % shard_count) == shard_rank]

        if not file_indices_all:
            return

        for file_index in file_indices_all:
            file_path = self.file_paths[file_index]
            chunk_index = 0
            use_row_group_shard = shard_count > 1 and self.total_files == 1 and self.file_type == "parquet"
            if (
                shard_count > 1
                and self.total_files == 1
                and self.file_type == "csv"
                and shard_rank == 0
                and worker_id == 0
                and base_shard_rank == 0
            ):
                logging.info(
                    "[RecDataLoader Info] Streaming with a single CSV file and multiple shards (processes/workers) will scan the full file in each shard. "
                    "Consider splitting the file into multiple shards or converting to parquet for better parallelism."
                )
            for chunk in iter_file_chunks(
                file_path,
                self.file_type,
                self.chunk_size,
                shard_rank=shard_rank if use_row_group_shard else 0,
                shard_count=shard_count if use_row_group_shard else 1,
                profiler=self.profiler,
            ):
                if self.expand:
                    chunk = expand_tabular_rows(chunk, self.expand)
                if shard_count > 1 and self.total_files == 1 and not use_row_group_shard:
                    if (chunk_index % shard_count) != shard_rank:
                        chunk_index += 1
                        continue
                chunk_index += 1
                if self.processor is not None:
                    if not self.processor.is_fitted:
                        raise ValueError(
                            "[DataLoader Error] DataProcessor must be fitted before using in streaming mode"
                        )
                    start = time.perf_counter()
                    transformed_data = self.processor.transform(chunk, return_dict=True)
                    if self.profiler is not None:
                        self.profiler.add("preprocess", time.perf_counter() - start)
                else:
                    transformed_data = chunk
                # if data=str|os.pathlike;  processor.transform(data, return_dict=False) will return file paths list
                # which will casue error in build_tensors_from_data
                if isinstance(transformed_data, list):
                    raise TypeError(
                        "[DataLoader Error] DataProcessor.transform returned file paths; use return_dict=True with in-memory data for streaming."
                    )
                start = time.perf_counter()
                batch = build_tensors_from_data(
                    data=transformed_data,
                    raw_data=chunk,
                    features=self.all_features,
                    target_columns=self.target_columns,
                    key_columns=self.key_columns,
                )
                if self.profiler is not None:
                    self.profiler.add("tensorize", time.perf_counter() - start)
                # Indicate streaming mode for collate_fn to avoid extra batching.
                batch["stream_mode"] = True
                yield batch
                del chunk, transformed_data


class RecDataLoader(FeatureSet):
    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        semantic_id_features: list[SemanticIdFeature] | None = None,
        target: list[str] | None | str = None,
        key_columns: str | list[str] | None = None,
        processor: DataProcessor | None = None,
        expand: dict[str, list] | None = None,
    ):
        """
        RecDataLoader is a unified dataloader for supporting in-memory and streaming data.
        Basemodel will accept RecDataLoader to create dataloaders for training/evaluation/prediction.

        Args:
            dense_features: list of DenseFeature definitions
            sparse_features: list of SparseFeature definitions
            sequence_features: list of SequenceFeature definitions
            target: target column name(s), e.g. 'label' or ['ctr', 'ctcvr']
            key_columns: key column name(s) to carry through (not used for model inputs), e.g. 'user_id' or ['user_id', 'item_id']
            processor: an instance of DataProcessor, if provided, will be used to transform data before creating tensors.
            expand: dictionary specifying additional columns to expand, e.g. {'user_id': ['age', 'gender']}
        """
        self.processor = processor
        self.expand = expand or {}
        self.set_all_features(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            key_columns=key_columns,
            semantic_id_features=semantic_id_features,
        )

    def create_dataloader(
        self,
        data: (
            dict | pd.DataFrame | pl.DataFrame | str | list[str] | os.PathLike | list[os.PathLike] | DataLoader | None
        ),
        batch_size: int = 32,
        shuffle: bool = True,
        streaming: bool = False,
        chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        shard_rank: int = 0,
        shard_count: int = 1,
        profiler: StageTimer | None = None,
    ) -> DataLoader:
        """
        Create a DataLoader from various data sources: dict, pd.DataFrame, file path(s), or existing DataLoader.

        Args:
            data: Data source, can be a dict, pd.DataFrame, file path (str), or existing DataLoader.
            batch_size: Batch size for DataLoader.
            shuffle: Whether to shuffle the data (ignored in streaming mode).
            streaming: If True, use streaming mode for large files; if False, load full data into memory.
            chunk_size: Chunk size for streaming mode (number of rows per chunk).
            num_workers: Number of worker processes for data loading.
            prefetch_factor: Number of batches loaded in advance by each worker.
        Returns:
            DataLoader instance.
        """

        if isinstance(data, DataLoader):
            return data

        if is_path_data(data):
            return self.create_from_path(
                path=data,
                batch_size=batch_size,
                shuffle=shuffle,
                streaming=streaming,
                chunk_size=chunk_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                shard_rank=shard_rank,
                shard_count=shard_count,
                profiler=profiler,
            )

        if isinstance(data, (dict, pd.DataFrame, pl.DataFrame)):
            return self.create_from_memory(
                data=data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

        raise ValueError(f"[RecDataLoader Error] Unsupported data type: {type(data)}")

    def create_from_memory(
        self,
        data: dict | pd.DataFrame | pl.DataFrame,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
    ) -> DataLoader:
        """
        Create a DataLoader from in-memory data. It builds a TensorDictDataset
        that implements __getitem__ and __len__, allowing PyTorch DataLoader to
        assign data to each worker.
        """

        # Keep a copy of raw data for key columns.
        raw_data = data
        if self.expand:
            raw_data = expand_tabular_rows(raw_data, self.expand)
            data = raw_data

        if self.processor is not None:
            if not self.processor.is_fitted:
                raise ValueError(
                    "[RecDataLoader Error] DataProcessor must be fitted before transforming data in memory"
                )
            data = self.processor.transform(data, return_dict=True)  # type: ignore

        tensors = build_tensors_from_data(
            data=data,
            raw_data=raw_data,
            features=self.all_features,
            target_columns=self.target_columns,
            key_columns=self.key_columns,
        )
        dataset = TensorDictDataset(tensors)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    def create_from_path(
        self,
        path: str | os.PathLike | list[str] | list[os.PathLike],
        batch_size: int,
        shuffle: bool,
        streaming: bool,
        chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        shard_rank: int = 0,
        shard_count: int = 1,
        profiler: StageTimer | None = None,
    ) -> DataLoader:
        """
        Create a DataLoader from file paths. It builds either a streaming
        IterableDataset (via __iter__) or an in-memory map-style dataset
        (via __getitem__/__len__).
        """

        if isinstance(path, (str, os.PathLike)):
            file_paths, file_type = get_file_paths(str(Path(path)))
        else:
            file_paths = [str(Path(p)) for p in path]
            if not file_paths:
                raise ValueError("[RecDataLoader Error] Empty file path list provided.")

            file_formats = set()
            for p in file_paths:
                name = Path(p).name
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext in {"csv", "txt"}:
                    fmt = "csv"
                elif ext == "parquet":
                    fmt = "parquet"
                else:
                    fmt = None
                if fmt is None:
                    raise ValueError(f"[RecDataLoader Error] Unsupported file extension: {Path(p).suffix}")
                file_formats.add(fmt)

            if len(file_formats) != 1:
                raise ValueError(
                    f"[RecDataLoader Error] Mixed file types in provided list: {', '.join(file_formats)}. "
                    "Please use a single format per DataLoader."
                )
            file_type = file_formats.pop()

        if streaming:
            return self.load_files_streaming(
                file_paths,
                file_type,
                batch_size,
                chunk_size,
                shuffle,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                shard_rank=shard_rank,
                shard_count=shard_count,
                profiler=profiler,
            )
        else:
            dfs = []
            for file_path in file_paths:
                df = read_table(file_path, data_format=file_type, engine="polars")
                dfs.append(df)

            if not dfs:
                raise ValueError("[RecDataLoader Error] No files loaded.")

            combined_df = dfs[0] if len(dfs) == 1 else pl.concat(dfs, how="vertical_relaxed")

            return self.create_from_memory(
                combined_df,
                batch_size,
                shuffle,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

    def load_files_streaming(
        self,
        file_paths: list[str],
        file_type: str,
        batch_size: int,
        chunk_size: int,
        shuffle: bool,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        shard_rank: int = 0,
        shard_count: int = 1,
        profiler: StageTimer | None = None,
    ) -> DataLoader:
        if shuffle:
            logging.info("[RecDataLoader Info] Shuffle is ignored in streaming mode (IterableDataset).")
        if batch_size != 1:
            logging.info(
                "[RecDataLoader Info] Streaming mode enforces batch_size=1; tune chunk_size to control memory/throughput."
            )
        # iterable dataset for streaming, implements __iter__
        dataset = FileDataset(
            file_paths=file_paths,
            dense_features=self.dense_features,
            sparse_features=self.sparse_features,
            sequence_features=self.sequence_features,
            semantic_id_features=self.semantic_id_features,
            target_columns=self.target_columns,
            key_columns=self.key_columns,
            chunk_size=chunk_size,
            file_type=file_type,
            processor=self.processor,
            shard_rank=shard_rank,
            shard_count=shard_count,
            profiler=profiler,
            expand=self.expand,
            schema=build_batch_schema(
                target_columns=self.target_columns,
            ),
        )
        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )


def prepare_sequence_column(column, feature: SequenceFeature) -> np.ndarray:
    """
    Normalize a sequence feature column into a padded int64 numpy array.
    Converts scalars/lists/arrays into a consistent 2D shape and applies
    truncation/padding based on the feature definition.
    """

    if isinstance(column, pd.Series):
        column = column.tolist()
    if isinstance(column, (list, tuple)):
        column = np.array(column, dtype=object)
    if not isinstance(column, np.ndarray):
        column = np.array([column], dtype=object)
    if column.ndim == 0:
        column = column.reshape(1)
    # transform object dtype to sequences
    if column.dtype == object:
        column = np.array([parse_sequence_value(v, feature) for v in column], dtype=object)
        sequences = []
        for seq in column:
            if isinstance(seq, (list, tuple, np.ndarray)):
                sequences.append(np.asarray(seq, dtype=np.int64))
            else:
                sequences.append(np.asarray([seq], dtype=np.int64))
        max_len = feature.max_len if feature.max_len is not None else 0
        if max_len <= 0:
            max_len = max((len(seq) for seq in sequences), default=1)
        pad_value = feature.padding_idx if feature.padding_idx is not None else 0
        padded = [
            (seq[:max_len] if len(seq) > max_len else np.pad(seq, (0, max_len - len(seq)), constant_values=pad_value))
            for seq in sequences
        ]
        column = np.stack(padded)
    elif column.ndim == 1:
        column = column.reshape(-1, 1)
    return np.asarray(column, dtype=np.int64)


def prepare_array_column(column, dtype) -> np.ndarray:
    values = to_object_array(column)
    if values.size > 0 and isinstance(values[0], (list, tuple, np.ndarray)):
        return np.stack([np.asarray(item, dtype=dtype) for item in values])
    return np.asarray(column, dtype=dtype)


def prepare_semantic_id_column(column, feature: SemanticIdFeature) -> np.ndarray:
    """
    Transform semantic-id into [N, L, K] int64 arrays.
    A single target semantic id [K] is normalized to [N, K] and left unpadded.
    """
    values = to_object_array(column)
    arrays = [np.asarray(item, dtype=np.int64) for item in values]
    if not arrays:
        return np.empty((0, feature.max_len, feature.num_codebooks), dtype=np.int64)

    num_codebooks = feature.num_codebooks
    pad_values = np.asarray(
        [
            feature.padding_idx if feature.padding_idx is not None else codebook_size
            for codebook_size in feature.codebook_sizes
        ],
        dtype=np.int64,
    )

    first_ndim = arrays[0].ndim
    if first_ndim <= 1:
        normalized = []
        for arr in arrays:
            flat = arr.reshape(-1)
            if flat.size != num_codebooks:
                raise ValueError(
                    f"[RecDataLoader Error] SemanticIdFeature '{feature.name}' expects {num_codebooks} code levels, "
                    f"got shape {tuple(arr.shape)}."
                )
            normalized.append(flat)
        return np.stack(normalized).astype(np.int64)

    padded = []
    for arr in arrays:
        if arr.ndim != 2 or arr.shape[1] != num_codebooks:
            raise ValueError(
                f"[RecDataLoader Error] SemanticIdFeature '{feature.name}' expects shape [seq_len, {num_codebooks}], "
                f"got {tuple(arr.shape)}."
            )
        seq = arr[: feature.max_len]
        if seq.shape[0] < feature.max_len:
            pad = np.tile(pad_values.reshape(1, -1), (feature.max_len - seq.shape[0], 1))
            seq = np.concatenate([seq, pad], axis=0)
        padded.append(seq)
    return np.stack(padded).astype(np.int64)


def build_batch_schema(target_columns: list[str]) -> BatchSchema:
    schema: BatchSchema = {
        "feature_scopes": {},
        "feature_layout": "flat",
        "label_format": "none" if not target_columns else "pointwise",
        "list_size": None,
    }
    return schema


def build_tensors_from_data(
    data: dict | pd.DataFrame | "pl.DataFrame",
    raw_data: dict | pd.DataFrame | "pl.DataFrame",
    features: list,
    target_columns: list[str],
    key_columns: str | list[str] | None,
) -> dict:
    """
    Build feature, label, and key tensors from raw input using feature definitions.
    This is used by RecDataLoader to construct model-ready batches.
    """

    effective_key_columns = to_column_names(key_columns)

    schema = build_batch_schema(target_columns=target_columns)
    feature_tensors = {}
    for feature in features:
        column = get_column_data(data, feature.name)
        if column is None:
            raise ValueError(f"[RecDataLoader Error] Feature column '{feature.name}' not found in data")
        if isinstance(feature, SequenceFeature):
            arr = prepare_sequence_column(column, feature)
            tensor = to_tensor(arr, dtype=torch.long)
        elif isinstance(feature, SemanticIdFeature):
            arr = prepare_semantic_id_column(column, feature)
            tensor = to_tensor(arr, dtype=torch.long)
        elif isinstance(feature, DenseFeature):
            arr = prepare_array_column(column, dtype=np.float32)
            tensor = to_tensor(arr, dtype=torch.float32)
        else:
            arr = prepare_array_column(column, dtype=np.int64)
            tensor = to_tensor(arr, dtype=torch.long)
        feature_tensors[feature.name] = tensor
    label_tensors = None
    if target_columns:
        label_tensors = {}
        for target_name in target_columns:
            if not has_column(data, target_name):
                column = None
            else:
                column = get_column_data(data, target_name)
            if column is None:
                continue
            label_format = schema.get("label_format", "pointwise")
            label_array = prepare_array_column(column, dtype=np.float32)
            label_tensor = to_tensor(label_array, dtype=torch.float32)
            if label_tensor.dim() == 2 and label_tensor.shape[0] == 1 and label_tensor.shape[1] > 1:
                label_tensor = label_tensor.t()
            if label_format == "pointwise" and label_tensor.shape[1:] == (1,):
                label_tensor = label_tensor.squeeze(1)
            label_tensors[target_name] = label_tensor
        if not label_tensors:
            label_tensors = None
    key_tensors = None
    if effective_key_columns:
        key_tensors = {}
        for key_col in effective_key_columns:
            column = get_column_data(raw_data, key_col)
            if column is None:
                column = get_column_data(data, key_col)
            if column is None:
                raise KeyError(f"[RecDataLoader Error] key column '{key_col}' not found in provided data.")
            # Normalize all key columns to strings for consistent downstream handling.
            key_tensors[key_col] = np.asarray(column, dtype=str)
    if not feature_tensors:
        raise ValueError("[RecDataLoader Error] No valid tensors could be built from the provided data.")
    return {"features": feature_tensors, "labels": label_tensors, "keys": key_tensors, "schema": schema}
