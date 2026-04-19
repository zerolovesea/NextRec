"""
Dataloader definitions

Date: create on 27/10/2025
Checkpoint: edit on 13/03/2026
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
from nextrec.utils.types import (
    BatchSchema,
    FeatureScopeName,
    ModelFamilyName,
    SamplingModeName,
    TaskTypeName,
    TrainingModeName,
)


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
        target_columns: list[str],
        key_columns: list[str] | None = None,
        target_source: (
            str | None
        ) = None,  # source column for generating next-item labels in sequence modeling; must be a SequenceFeature if specified
        target_shift_steps: int = 1,  # number of steps to shift for next-item prediction; used only if target_source is specified
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
        self.target_source = target_source
        self.target_shift_steps = target_shift_steps
        self.shard_rank = int(shard_rank)
        self.shard_count = int(shard_count)
        self.profiler = profiler
        self.expand = expand or {}
        self.schema = dict(schema or {})

        self.set_all_features(
            dense_features,
            sparse_features,
            sequence_features,
            target_columns,
            key_columns,
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
                    target_source=self.target_source,
                    target_shift_steps=self.target_shift_steps,
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
        target: list[str] | None | str = None,
        key_columns: str | list[str] | None = None,
        target_source: str | None = None,
        target_shift_steps: int = 1,
        processor: DataProcessor | None = None,
        expand: dict[str, list] | None = None,
        task: TaskTypeName | list[TaskTypeName] = "binary",
        model_family: ModelFamilyName = "ranking",
        training_mode: TrainingModeName = "pointwise",
        sampling_mode: SamplingModeName = "explicit",
        feature_scopes: dict[str, FeatureScopeName] | None = None,
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
        """
        self.processor = processor
        self.expand = expand or {}
        self.target_source = target_source
        self.target_shift_steps = target_shift_steps
        self.task = task
        self.model_family = model_family
        self.training_mode = training_mode
        self.sampling_mode = sampling_mode
        self.set_all_features(dense_features, sparse_features, sequence_features, target, key_columns)
        if feature_scopes is not None:
            self.feature_scopes = dict(feature_scopes)

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
            target_source=self.target_source,
            target_shift_steps=self.target_shift_steps,
            task=self.task,
            model_family=self.model_family,
            training_mode=self.training_mode,
            sampling_mode=self.sampling_mode,
            feature_scopes=self.feature_scopes,
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
            target_columns=self.target_columns,
            key_columns=self.key_columns,
            target_source=self.target_source,
            target_shift_steps=self.target_shift_steps,
            chunk_size=chunk_size,
            file_type=file_type,
            processor=self.processor,
            shard_rank=shard_rank,
            shard_count=shard_count,
            profiler=profiler,
            expand=self.expand,
            schema=build_batch_schema(
                features=self.all_features,
                target_columns=self.target_columns,
                task=self.task,
                model_family=self.model_family,
                training_mode=self.training_mode,
                sampling_mode=self.sampling_mode,
                feature_scopes=self.feature_scopes,
                data=None,
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


def prepare_candidate_feature_column(
    column,
    feature: DenseFeature | SparseFeature | SequenceFeature,
) -> np.ndarray:
    values = to_object_array(column)
    if values.size == 0:
        raise ValueError(f"[RecDataLoader Error] Candidate feature '{feature.name}' is empty.")
    if not isinstance(values[0], (list, tuple, np.ndarray)):
        raise ValueError(
            f"[RecDataLoader Error] Explicit candidate feature '{feature.name}' must be provided as per-row candidate lists."
        )

    if isinstance(feature, SequenceFeature):
        candidate_rows = []
        list_size = None
        for row in values:
            row_candidates = list(row)
            if list_size is None:
                list_size = len(row_candidates)
            elif len(row_candidates) != list_size:
                raise ValueError(
                    f"[RecDataLoader Error] Candidate feature '{feature.name}' must use a consistent list_size per row."
                )
            candidate_rows.append(prepare_sequence_column(row_candidates, feature))
        return np.stack(candidate_rows, axis=0)

    rows = []
    list_size = None
    for row in values:
        row_array = np.asarray(row, dtype=np.float32 if isinstance(feature, DenseFeature) else np.int64)
        if row_array.ndim == 0:
            row_array = row_array.reshape(1)
        if list_size is None:
            list_size = row_array.shape[0]
        elif row_array.shape[0] != list_size:
            raise ValueError(
                f"[RecDataLoader Error] Candidate feature '{feature.name}' must use a consistent list_size per row."
            )
        rows.append(row_array)
    return np.stack(rows, axis=0)


def prepare_candidate_label_column(column) -> np.ndarray:
    values = to_object_array(column)
    if values.size == 0:
        raise ValueError("[RecDataLoader Error] Candidate labels are empty.")
    if not isinstance(values[0], (list, tuple, np.ndarray)):
        raise ValueError(
            "[RecDataLoader Error] Explicit pairwise/listwise labels must be provided as per-row candidate lists."
        )
    rows = []
    list_size = None
    for row in values:
        row_array = np.asarray(row)
        if row_array.ndim == 0:
            row_array = row_array.reshape(1)
        if list_size is None:
            list_size = row_array.shape[0]
        elif row_array.shape[0] != list_size:
            raise ValueError("[RecDataLoader Error] Candidate labels must use a consistent list_size per row.")
        rows.append(row_array)
    return np.stack(rows, axis=0)


def build_batch_schema(
    features: list,
    target_columns: list[str],
    task: TaskTypeName | list[TaskTypeName],
    model_family: ModelFamilyName,
    training_mode: TrainingModeName,
    sampling_mode: SamplingModeName,
    feature_scopes: dict[str, FeatureScopeName] | None,
    data,
) -> BatchSchema:
    schema: BatchSchema = {
        "model_family": model_family,
        "task": task,
        "training_mode": training_mode,
        "sampling_mode": sampling_mode,
        "feature_scopes": dict(feature_scopes or {}),
        "feature_layout": "flat",
        "label_format": "none" if not target_columns else "pointwise",
        "list_size": None,
    }

    if model_family == "sequential":
        schema["feature_layout"] = "sequence"
        if target_columns:
            schema["label_format"] = "sequence"

    if training_mode in {"pairwise", "listwise"} and sampling_mode == "inbatch":
        schema["label_format"] = "implicit_inbatch"
        return schema

    if training_mode in {"pairwise", "listwise"} and sampling_mode == "explicit":
        schema["feature_layout"] = "candidate_list"
        schema["label_format"] = "candidate_list"
        if not schema["feature_scopes"]:
            default_scope = "candidate"
            schema["feature_scopes"] = {feature.name: default_scope for feature in features}
        if data is not None:
            candidate_list_size = None
            for feature in features:
                if schema["feature_scopes"].get(feature.name, "shared") != "candidate":
                    continue
                column = get_column_data(data, feature.name)
                if column is None:
                    continue
                values = to_object_array(column)
                if values.size == 0 or not isinstance(values[0], (list, tuple, np.ndarray)):
                    raise ValueError(
                        f"[RecDataLoader Error] Explicit {training_mode} data requires candidate feature '{feature.name}' to be nested per row."
                    )
                row_list_size = len(values[0])
                candidate_list_size = row_list_size if candidate_list_size is None else candidate_list_size
                if candidate_list_size != row_list_size:
                    raise ValueError("[RecDataLoader Error] Candidate features must share the same list_size.")
            schema["list_size"] = candidate_list_size
        return schema

    return schema


def build_shifted_sequence_column(
    column,
    feature: SequenceFeature,
    shift: int = 1,
) -> np.ndarray:
    """
    Build next-item labels by shifting a padded sequence column to the left.

    Example:
        [1, 2, 3, 0] -> [2, 3, 0, 0]
    """
    if shift < 1:
        raise ValueError("[RecDataLoader Error] sequence shift must be >= 1.")
    seq = prepare_sequence_column(column, feature)
    pad_value = feature.padding_idx if feature.padding_idx is not None else 0
    if seq.shape[1] == 0:
        return seq
    if shift >= seq.shape[1]:
        return np.full_like(seq, pad_value)
    shifted = np.empty_like(seq)
    shifted[:, :-shift] = seq[:, shift:]
    shifted[:, -shift:] = pad_value
    return shifted


def build_tensors_from_data(
    data: dict | pd.DataFrame | "pl.DataFrame",
    raw_data: dict | pd.DataFrame | "pl.DataFrame",
    features: list,
    target_columns: list[str],
    key_columns: str | list[str] | None,
    target_source: str | None = None,
    target_shift_steps: int = 1,
    task: TaskTypeName | list[TaskTypeName] = "binary",
    model_family: ModelFamilyName = "ranking",
    training_mode: TrainingModeName = "pointwise",
    sampling_mode: SamplingModeName = "explicit",
    feature_scopes: dict[str, FeatureScopeName] | None = None,
) -> dict:
    """
    Build feature, label, and key tensors from raw input using feature definitions.
    This is used by RecDataLoader to construct model-ready batches.
    """

    effective_key_columns = to_column_names(key_columns)

    schema = build_batch_schema(
        features=features,
        target_columns=target_columns,
        task=task,
        model_family=model_family,
        training_mode=training_mode,
        sampling_mode=sampling_mode,
        feature_scopes=feature_scopes,
        data=data,
    )
    feature_tensors = {}
    for feature in features:
        column = get_column_data(data, feature.name)
        if column is None:
            raise ValueError(f"[RecDataLoader Error] Feature column '{feature.name}' not found in data")
        feature_scope = schema.get("feature_scopes", {}).get(feature.name, "shared")
        if feature_scope == "candidate" and training_mode in {"pairwise", "listwise"} and sampling_mode == "explicit":
            arr = prepare_candidate_feature_column(column, feature)
            tensor_dtype = torch.float32 if isinstance(feature, DenseFeature) else torch.long
            tensor = to_tensor(arr, dtype=tensor_dtype)
        elif isinstance(feature, SequenceFeature):
            arr = prepare_sequence_column(column, feature)
            tensor = to_tensor(arr, dtype=torch.long)
        elif isinstance(feature, DenseFeature):
            arr = np.asarray(column, dtype=np.float32)
            tensor = to_tensor(arr, dtype=torch.float32)
        else:
            arr = np.asarray(column, dtype=np.int64)
            tensor = to_tensor(arr, dtype=torch.long)
        feature_tensors[feature.name] = tensor
    label_tensors = None
    feature_by_name = {feature.name: feature for feature in features}
    if target_columns:
        label_tensors = {}
        for target_name in target_columns:
            if not has_column(data, target_name):
                column = None
            else:
                column = get_column_data(data, target_name)
            if column is None and target_source is not None:
                source_feature = feature_by_name.get(target_source)
                if source_feature is None or not isinstance(source_feature, SequenceFeature):
                    raise KeyError(
                        f"[RecDataLoader Error] target_source='{target_source}' requires a matching SequenceFeature."
                    )
                source_column = get_column_data(data, target_source)
                if source_column is None:
                    raise KeyError(f"[RecDataLoader Error] target_source column '{target_source}' not found in data.")
                label_array = build_shifted_sequence_column(
                    source_column,
                    source_feature,
                    shift=target_shift_steps,
                )
                label_tensor = to_tensor(label_array, dtype=torch.long)
                label_tensors[target_name] = label_tensor
                continue
            if column is None:
                continue
            target_task = (
                task[target_columns.index(target_name)]
                if isinstance(task, list) and target_columns.index(target_name) < len(task)
                else (task[0] if isinstance(task, list) else task)
            )
            label_format = schema.get("label_format", "pointwise")
            if label_format == "candidate_list":
                label_array = prepare_candidate_label_column(column)
            elif len(column) > 0 and isinstance(column[0], (list, tuple, np.ndarray)):
                label_array = np.stack([np.asarray(item) for item in column])
            else:
                label_array = np.asarray(column)
            label_dtype = torch.long if target_task == "generative" else torch.float32
            label_tensor = to_tensor(label_array, dtype=label_dtype)
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
