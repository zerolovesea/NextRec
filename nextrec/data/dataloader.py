"""
Dataloader definitions

Date: create on 27/10/2025
Checkpoint: edit on 02/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

import os
import torch
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from typing import cast

from nextrec.basic.features import (
    DenseFeature,
    SparseFeature,
    SequenceFeature,
    FeatureSet,
)
from nextrec.data.preprocessor import DataProcessor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from nextrec.utils.tensor import to_tensor
from nextrec.utils.file import resolve_file_paths, read_table
from nextrec.data.batch_utils import collate_fn
from nextrec.data.data_processing import get_column_data


class TensorDictDataset(Dataset):
    """Dataset returning sample-level dicts matching the unified batch schema."""

    def __init__(self, tensors: dict):
        self.features = tensors.get("features", {})
        self.labels = tensors.get("labels")
        self.ids = tensors.get("ids")
        if not self.features:
            raise ValueError(
                "[TensorDictDataset Error] Dataset requires at least one feature tensor."
            )
        lengths = [tensor.shape[0] for tensor in self.features.values()]
        if not lengths:
            raise ValueError("[TensorDictDataset Error] Feature tensors are empty.")
        self.length = lengths[0]
        for length in lengths[1:]:
            if length != self.length:
                raise ValueError(
                    "[TensorDictDataset Error] All feature tensors must have the same length."
                )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        sample_features = {name: tensor[idx] for name, tensor in self.features.items()}
        sample_labels = (
            {name: tensor[idx] for name, tensor in self.labels.items()}
            if self.labels
            else None
        )
        sample_ids = (
            {name: tensor[idx] for name, tensor in self.ids.items()}
            if self.ids
            else None
        )
        return {"features": sample_features, "labels": sample_labels, "ids": sample_ids}


class FileDataset(FeatureSet, IterableDataset):
    def __init__(
        self,
        file_paths: list[str],  # file paths to read, containing CSV or Parquet files
        dense_features: list[DenseFeature],  # dense feature definitions
        sparse_features: list[SparseFeature],  # sparse feature definitions
        sequence_features: list[SequenceFeature],  # sequence feature definitions
        target_columns: list[str],  # target column names
        id_columns: (
            list[str] | None
        ) = None,  # id columns to carry through (not used for model inputs)
        chunk_size: int = 10000,
        file_type: str = "csv",
        processor: DataProcessor | None = None,
    ):  # optional DataProcessor for transformation
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.file_type = file_type
        self.processor = processor
        self.set_all_features(
            dense_features,
            sparse_features,
            sequence_features,
            target_columns,
            id_columns,
        )
        self.current_file_index = 0
        self.total_files = len(file_paths)

    def __iter__(self):
        self.current_file_index = 0
        for file_path in self.file_paths:
            self.current_file_index += 1
            if self.total_files == 1:
                file_name = os.path.basename(file_path)
                logging.info(f"Processing file: {file_name}")
            if self.file_type == "csv":
                yield from self.read_csv_chunks(file_path)
            elif self.file_type == "parquet":
                yield from self.read_parquet_chunks(file_path)

    def read_csv_chunks(self, file_path: str):
        chunk_iterator = pd.read_csv(file_path, chunksize=self.chunk_size)
        for chunk in chunk_iterator:
            tensors = self.dataframeto_tensors(chunk)
            yield tensors

    def read_parquet_chunks(self, file_path: str):
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            chunk = batch.to_pandas()
            tensors = self.dataframeto_tensors(chunk)
            yield tensors
            del chunk

    def dataframeto_tensors(self, df: pd.DataFrame) -> dict | None:
        if self.processor is not None:
            if not self.processor.is_fitted:
                raise ValueError(
                    "[DataLoader Error] DataProcessor must be fitted before using in streaming mode"
                )
            transformed_data = self.processor.transform(df, return_dict=True)
        else:
            transformed_data = df
        if isinstance(transformed_data, list):
            raise TypeError(
                "[DataLoader Error] DataProcessor.transform returned file paths; use return_dict=True with in-memory data for streaming."
            )
        safe_data = cast(dict | pd.DataFrame, transformed_data)
        batch = build_tensors_from_data(
            data=safe_data,
            raw_data=df,
            features=self.all_features,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
        )
        if batch is not None:
            batch["_already_batched"] = True
        return batch


class RecDataLoader(FeatureSet):
    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: list[str] | None | str = None,
        id_columns: str | list[str] | None = None,
        processor: DataProcessor | None = None,
    ):
        """
        RecDataLoader is a unified dataloader for supporting in-memory and streaming data.
        Basemodel will accept RecDataLoader to create dataloaders for training/evaluation/prediction.

        Args:
            dense_features: list of DenseFeature definitions
            sparse_features: list of SparseFeature definitions
            sequence_features: list of SequenceFeature definitions
            target: target column name(s), e.g. 'label' or ['ctr', 'ctcvr']
            id_columns: id column name(s) to carry through (not used for model inputs), e.g. 'user_id' or ['user_id', 'item_id']
            processor: an instance of DataProcessor, if provided, will be used to transform data before creating tensors.
        """
        self.processor = processor
        self.set_all_features(
            dense_features, sparse_features, sequence_features, target, id_columns
        )

    def create_dataloader(
        self,
        data: (
            dict
            | pd.DataFrame
            | str
            | os.PathLike
            | list[str]
            | list[os.PathLike]
            | DataLoader
        ),
        batch_size: int = 32,
        shuffle: bool = True,
        load_full: bool = True,
        chunk_size: int = 10000,
        num_workers: int = 0,
        sampler=None,
    ) -> DataLoader:
        """
        Create a DataLoader from various data sources.

        Args:
            data: Data source, can be a dict, pd.DataFrame, file path (str), or existing DataLoader.
            batch_size: Batch size for DataLoader.
            shuffle: Whether to shuffle the data (ignored in streaming mode).
            load_full: If True, load full data into memory; if False, use streaming mode for large files.
            chunk_size: Chunk size for streaming mode (number of rows per chunk).
            num_workers: Number of worker processes for data loading.
            sampler: Optional sampler for DataLoader, only used for distributed training.
        Returns:
            DataLoader instance.
        """

        if isinstance(data, DataLoader):
            return data
        elif isinstance(data, (str, os.PathLike)):
            return self.create_from_path(
                path=data,
                batch_size=batch_size,
                shuffle=shuffle,
                load_full=load_full,
                chunk_size=chunk_size,
                num_workers=num_workers,
            )
        elif (
            isinstance(data, list)
            and data
            and all(isinstance(p, (str, os.PathLike)) for p in data)
        ):
            return self.create_from_path(
                path=data,
                batch_size=batch_size,
                shuffle=shuffle,
                load_full=load_full,
                chunk_size=chunk_size,
                num_workers=num_workers,
            )
        elif isinstance(data, (dict, pd.DataFrame)):
            return self.create_from_memory(
                data=data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                sampler=sampler,
            )
        else:
            raise ValueError(
                f"[RecDataLoader Error] Unsupported data type: {type(data)}"
            )

    def create_from_memory(
        self,
        data: dict | pd.DataFrame,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        sampler=None,
    ) -> DataLoader:

        raw_data = data

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
            id_columns=self.id_columns,
        )
        if tensors is None:
            raise ValueError(
                "[RecDataLoader Error] No valid tensors could be built from the provided data."
            )
        dataset = TensorDictDataset(tensors)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if sampler is not None else shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def create_from_path(
        self,
        path: str | os.PathLike | list[str] | list[os.PathLike],
        batch_size: int,
        shuffle: bool,
        load_full: bool,
        chunk_size: int = 10000,
        num_workers: int = 0,
    ) -> DataLoader:
        if isinstance(path, (str, os.PathLike)):
            file_paths, file_type = resolve_file_paths(str(Path(path)))
        else:
            file_paths = [str(Path(p)) for p in path]
            if not file_paths:
                raise ValueError("[RecDataLoader Error] Empty file path list provided.")
            suffixes = {Path(p).suffix.lower() for p in file_paths}
            if len(suffixes) != 1:
                raise ValueError(
                    "[RecDataLoader Error] Mixed file types in provided list; please use only CSV or only Parquet."
                )
            suffix = suffixes.pop()
            if suffix not in {".csv", ".parquet"}:
                raise ValueError(
                    f"[RecDataLoader Error] Unsupported file extension in list: {suffix}"
                )
            file_type = "csv" if suffix == ".csv" else "parquet"
        # Load full data into memory
        if load_full:
            dfs = []
            total_bytes = 0
            for file_path in file_paths:
                try:
                    total_bytes += os.path.getsize(file_path)
                except OSError:
                    pass
                try:
                    df = read_table(file_path, data_format=file_type)
                    dfs.append(df)
                except MemoryError as exc:
                    raise MemoryError(
                        f"[RecDataLoader Error] Out of memory while reading {file_path}. Consider using load_full=False with streaming."
                    ) from exc
            try:
                combined_df = pd.concat(dfs, ignore_index=True)
            except MemoryError as exc:
                raise MemoryError(
                    f"[RecDataLoader Error] Out of memory while concatenating loaded data (approx {total_bytes / (1024**3):.2f} GB). Use load_full=False to stream or reduce chunk_size."
                ) from exc
            return self.create_from_memory(
                combined_df, batch_size, shuffle, num_workers=num_workers
            )
        else:
            return self.load_files_streaming(
                file_paths,
                file_type,
                batch_size,
                chunk_size,
                shuffle,
                num_workers=num_workers,
            )

    def load_files_streaming(
        self,
        file_paths: list[str],
        file_type: str,
        batch_size: int,
        chunk_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> DataLoader:
        if shuffle:
            logging.info(
                "[RecDataLoader Info] Shuffle is ignored in streaming mode (IterableDataset)."
            )
        if batch_size != 1:
            logging.info(
                "[RecDataLoader Info] Streaming mode enforces batch_size=1; tune chunk_size to control memory/throughput."
            )
        dataset = FileDataset(
            file_paths=file_paths,
            dense_features=self.dense_features,
            sparse_features=self.sparse_features,
            sequence_features=self.sequence_features,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
            chunk_size=chunk_size,
            file_type=file_type,
            processor=self.processor,
        )
        return DataLoader(
            dataset, batch_size=1, collate_fn=collate_fn, num_workers=num_workers
        )


def normalize_sequence_column(column, feature: SequenceFeature) -> np.ndarray:
    if isinstance(column, pd.Series):
        column = column.tolist()
    if isinstance(column, (list, tuple)):
        column = np.array(column, dtype=object)
    if not isinstance(column, np.ndarray):
        column = np.array([column], dtype=object)
    if column.ndim == 0:
        column = column.reshape(1)
    if column.dtype == object and any(isinstance(v, str) for v in column.ravel()):
        raise TypeError(
            f"[RecDataLoader Error] Sequence feature '{feature.name}' expects numeric sequences; found string values."
        )
    if (
        column.dtype == object
        and len(column) > 0
        and isinstance(column[0], (list, tuple, np.ndarray))
    ):
        sequences = []
        for seq in column:
            if isinstance(seq, str):
                raise TypeError(
                    f"[RecDataLoader Error] Sequence feature '{feature.name}' expects numeric sequences; found string values."
                )
            if isinstance(seq, (list, tuple, np.ndarray)):
                arr = np.asarray(seq, dtype=np.int64)
            else:
                arr = np.asarray([seq], dtype=np.int64)
            sequences.append(arr)
        max_len = getattr(feature, "max_len", 0)
        if max_len <= 0:
            max_len = max((len(seq) for seq in sequences), default=1)
        pad_value = getattr(feature, "padding_idx", 0)
        padded = []
        for seq in sequences:
            if len(seq) > max_len:
                padded.append(seq[:max_len])
            else:
                padded.append(
                    np.pad(seq, (0, max_len - len(seq)), constant_values=pad_value)
                )
        column = np.stack(padded)
    elif column.ndim == 1:
        column = column.reshape(-1, 1)
    return np.asarray(column, dtype=np.int64)


def build_tensors_from_data(
    data: dict | pd.DataFrame,
    raw_data: dict | pd.DataFrame,
    features: list,
    target_columns: list[str],
    id_columns: list[str],
) -> dict | None:
    feature_tensors = {}
    for feature in features:
        column = get_column_data(data, feature.name)
        if column is None:
            raise ValueError(
                f"[RecDataLoader Error] Feature column '{feature.name}' not found in data"
            )
        if isinstance(
            feature, SequenceFeature
        ):  # sequence feature will do padding/truncation again to avoid the case when input data is not preprocessed
            arr = normalize_sequence_column(column, feature)
            tensor = to_tensor(arr, dtype=torch.long)
        elif isinstance(feature, DenseFeature):
            arr = np.asarray(column, dtype=np.float32)
            tensor = to_tensor(arr, dtype=torch.float32)
        else:
            arr = np.asarray(column, dtype=np.int64)
            tensor = to_tensor(arr, dtype=torch.long)
        feature_tensors[feature.name] = tensor
    label_tensors = None
    if target_columns:
        label_tensors = {}
        for target_name in target_columns:
            column = get_column_data(data, target_name)
            if column is None:
                continue
            label_tensor = to_tensor(
                np.asarray(column, dtype=np.float32), dtype=torch.float32
            )
            if (
                label_tensor.dim() == 2
                and label_tensor.shape[0] == 1
                and label_tensor.shape[1] > 1
            ):
                label_tensor = label_tensor.t()
            if label_tensor.shape[1:] == (1,):
                label_tensor = label_tensor.squeeze(1)
            label_tensors[target_name] = label_tensor
        if not label_tensors:
            label_tensors = None
    id_tensors = None
    if id_columns:
        id_tensors = {}
        for id_col in id_columns:
            column = get_column_data(raw_data, id_col)
            if column is None:
                column = get_column_data(data, id_col)
            if column is None:
                raise KeyError(
                    f"[RecDataLoader Error] ID column '{id_col}' not found in provided data."
                )
            try:
                id_arr = np.asarray(column, dtype=np.int64)
            except Exception as exc:
                raise TypeError(
                    f"[RecDataLoader Error] ID column '{id_col}' must contain numeric values. Received dtype={np.asarray(column).dtype}, error: {exc}"
                ) from exc
            id_tensors[id_col] = to_tensor(id_arr, dtype=torch.long)
    if not feature_tensors:
        return None
    return {"features": feature_tensors, "labels": label_tensors, "ids": id_tensors}
