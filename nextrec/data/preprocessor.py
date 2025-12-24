"""
DataProcessor for data preprocessing including numeric, sparse, sequence features and target processing.

Date: create on 13/11/2025
Checkpoint: edit on 24/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from nextrec.__version__ import __version__
from nextrec.basic.features import FeatureSet
from nextrec.basic.loggers import colorize
from nextrec.basic.session import resolve_save_path
from nextrec.data.data_processing import hash_md5_mod
from nextrec.utils.console import progress
from nextrec.utils.data import (
    FILE_FORMAT_CONFIG,
    check_streaming_support,
    default_output_dir,
    iter_file_chunks,
    load_dataframes,
    read_table,
    resolve_file_paths,
)


class DataProcessor(FeatureSet):
    def __init__(self, hash_cache_size: int = 200_000):
        self.numeric_features: Dict[str, Dict[str, Any]] = {}
        self.sparse_features: Dict[str, Dict[str, Any]] = {}
        self.sequence_features: Dict[str, Dict[str, Any]] = {}
        self.target_features: Dict[str, Dict[str, Any]] = {}
        self.version = __version__

        self.is_fitted = False
        self._transform_summary_printed = (
            False  # Track if summary has been printed during transform
        )

        self.scalers: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoders: Dict[str, Dict[str, int]] = {}
        self.set_target_id(target=[], id_columns=[])

        # cache hash function
        self.hash_cache_size = int(hash_cache_size)
        if self.hash_cache_size > 0:
            self.hash_fn = functools.lru_cache(maxsize=self.hash_cache_size)(
                hash_md5_mod
            )
        else:
            self.hash_fn = hash_md5_mod

    def add_numeric_feature(
        self,
        name: str,
        scaler: Optional[
            Literal["standard", "minmax", "robust", "maxabs", "log", "none"]
        ] = "standard",
        fill_na: Optional[float] = None,
    ):
        self.numeric_features[name] = {"scaler": scaler, "fill_na": fill_na}

    def add_sparse_feature(
        self,
        name: str,
        encode_method: Literal["hash", "label"] = "label",
        hash_size: Optional[int] = None,
        fill_na: str = "<UNK>",
    ):
        if encode_method == "hash" and hash_size is None:
            raise ValueError(
                "[Data Processor Error] hash_size must be specified when encode_method='hash'"
            )
        self.sparse_features[name] = {
            "encode_method": encode_method,
            "hash_size": hash_size,
            "fill_na": fill_na,
        }

    def add_sequence_feature(
        self,
        name: str,
        encode_method: Literal["hash", "label"] = "label",
        hash_size: Optional[int] = None,
        max_len: Optional[int] = 50,
        pad_value: int = 0,
        truncate: Literal[
            "pre", "post"
        ] = "pre",  # pre: keep last max_len items, post: keep first max_len items
        separator: str = ",",
    ):
        if encode_method == "hash" and hash_size is None:
            raise ValueError(
                "[Data Processor Error] hash_size must be specified when encode_method='hash'"
            )
        self.sequence_features[name] = {
            "encode_method": encode_method,
            "hash_size": hash_size,
            "max_len": max_len,
            "pad_value": pad_value,
            "truncate": truncate,
            "separator": separator,
        }

    def add_target(
        self,
        name: str,  # example: 'click'
        target_type: Literal["binary", "regression"] = "binary",
        label_map: Optional[
            Dict[str, int]
        ] = None,  # example: {'click': 1, 'no_click': 0}
    ):
        self.target_features[name] = {
            "target_type": target_type,
            "label_map": label_map,
        }
        self.set_target_id(list(self.target_features.keys()), [])

    def hash_string(self, s: str, hash_size: int) -> int:
        return self.hash_fn(str(s), int(hash_size))

    def clear_hash_cache(self) -> None:
        cache_clear = getattr(self.hash_fn, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()

    def hash_cache_info(self):
        cache_info = getattr(self.hash_fn, "cache_info", None)
        if callable(cache_info):
            return cache_info()
        return None

    def process_numeric_feature_fit(self, data: pd.Series, config: Dict[str, Any]):
        name = str(data.name)
        scaler_type = config["scaler"]
        fill_na = config["fill_na"]
        if data.isna().any():
            if fill_na is None:
                # Default use mean value to fill missing values for numeric features
                fill_na = data.mean()
            config["fill_na_value"] = fill_na
        scaler_map = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "maxabs": MaxAbsScaler,
        }
        if scaler_type in ("log", "none"):
            scaler = None
        else:
            scaler_cls = scaler_map.get(scaler_type)
            if scaler_cls is None:
                raise ValueError(
                    f"[Data Processor Error] Unknown scaler type: {scaler_type}"
                )
            scaler = scaler_cls()
        if scaler is not None:
            filled_data = data.fillna(config.get("fill_na_value", 0))
            values = np.array(filled_data.values, dtype=np.float64).reshape(-1, 1)
            scaler.fit(values)
            self.scalers[name] = scaler

    def process_numeric_feature_transform(
        self, data: pd.Series, config: Dict[str, Any]
    ) -> np.ndarray:
        logger = logging.getLogger()
        name = str(data.name)
        scaler_type = config["scaler"]
        fill_na_value = config.get("fill_na_value", 0)
        filled_data = data.fillna(fill_na_value)
        values = np.array(filled_data.values, dtype=np.float64)
        if scaler_type == "log":
            result = np.log1p(np.maximum(values, 0))
        elif scaler_type == "none":
            result = values
        else:
            scaler = self.scalers.get(name)
            if scaler is None:
                logger.warning(
                    f"Scaler for {name} not fitted, returning original values"
                )
                result = values
            else:
                result = scaler.transform(values.reshape(-1, 1)).ravel()
        return result

    def process_sparse_feature_fit(self, data: pd.Series, config: Dict[str, Any]):
        _ = str(data.name)
        encode_method = config["encode_method"]
        fill_na = config["fill_na"]  # <UNK>
        filled_data = data.fillna(fill_na).astype(str)
        if encode_method == "label":
            vocab = sorted(set(filled_data.tolist()))
            if "<UNK>" not in vocab:
                vocab.append("<UNK>")
            token_to_idx = {token: idx for idx, token in enumerate(vocab)}
            config["_token_to_idx"] = token_to_idx
            config["_unk_index"] = token_to_idx["<UNK>"]
            config["vocab_size"] = len(vocab)
        elif encode_method == "hash":
            config["vocab_size"] = config["hash_size"]

    def process_sparse_feature_transform(
        self, data: pd.Series, config: Dict[str, Any]
    ) -> np.ndarray:
        name = str(data.name)
        encode_method = config["encode_method"]
        fill_na = config["fill_na"]

        sparse_series = (
            data if isinstance(data, pd.Series) else pd.Series(data, name=name)
        )
        sparse_series = sparse_series.fillna(fill_na).astype(str)
        if encode_method == "label":
            token_to_idx = config.get("_token_to_idx")
            if isinstance(token_to_idx, dict):
                unk_index = int(config.get("_unk_index", 0))
                return np.fromiter(
                    (token_to_idx.get(v, unk_index) for v in sparse_series.to_numpy()),
                    dtype=np.int64,
                    count=sparse_series.size,
                )
            raise ValueError(
                f"[Data Processor Error] Token index for {name} not fitted"
            )

        if encode_method == "hash":
            hash_size = config["hash_size"]
            hash_fn = self.hash_string
            return np.fromiter(
                (hash_fn(v, hash_size) for v in sparse_series.to_numpy()),
                dtype=np.int64,
                count=sparse_series.size,
            )
        return np.array([], dtype=np.int64)

    def process_sequence_feature_fit(self, data: pd.Series, config: Dict[str, Any]):
        _ = str(data.name)
        encode_method = config["encode_method"]
        separator = config["separator"]
        if encode_method == "label":
            all_tokens = set()
            for seq in data:
                all_tokens.update(self.extract_sequence_tokens(seq, separator))
            vocab = sorted(all_tokens)
            if not vocab:
                vocab = ["<PAD>"]
            if "<UNK>" not in vocab:
                vocab.append("<UNK>")
            token_to_idx = {token: idx for idx, token in enumerate(vocab)}
            config["_token_to_idx"] = token_to_idx
            config["_unk_index"] = token_to_idx["<UNK>"]
            config["vocab_size"] = len(vocab)
        elif encode_method == "hash":
            config["vocab_size"] = config["hash_size"]

    def process_sequence_feature_transform(
        self, data: pd.Series, config: Dict[str, Any]
    ) -> np.ndarray:
        """Optimized sequence transform with preallocation and cached vocab map."""
        name = str(data.name)
        encode_method = config["encode_method"]
        max_len = config["max_len"]
        pad_value = config["pad_value"]
        truncate = config["truncate"]
        separator = config["separator"]
        arr = np.asarray(data, dtype=object)
        n = arr.shape[0]
        output = np.full((n, max_len), pad_value, dtype=np.int64)
        # Shared helpers cached locally for speed and cross-platform consistency
        split_fn = str.split
        is_nan = np.isnan
        if encode_method == "label":
            class_to_idx = config.get("_token_to_idx")
            if class_to_idx is None:
                raise ValueError(
                    f"[Data Processor Error] Token index for {name} not fitted"
                )
            unk_index = int(config.get("_unk_index", class_to_idx.get("<UNK>", 0)))
        else:
            class_to_idx = None  # type: ignore
            unk_index = 0
        hash_fn = self.hash_string
        hash_size = config.get("hash_size")
        for i, seq in enumerate(arr):
            # normalize sequence to a list of strings
            tokens = []
            if seq is None:
                tokens = []
            elif isinstance(seq, (float, np.floating)):
                tokens = [] if is_nan(seq) else [str(seq)]
            elif isinstance(seq, str):
                seq_str = seq.strip()
                tokens = [] if not seq_str else split_fn(seq_str, separator)
            elif isinstance(seq, (list, tuple, np.ndarray)):
                tokens = [str(t) for t in seq]
            else:
                tokens = []
            if encode_method == "label":
                encoded = [
                    class_to_idx.get(token.strip(), unk_index)  # type: ignore[union-attr]
                    for token in tokens
                    if token is not None and token != ""
                ]
            elif encode_method == "hash":
                if hash_size is None:
                    raise ValueError(
                        "[Data Processor Error] hash_size must be set for hash encoding"
                    )
                encoded = [
                    hash_fn(str(token), hash_size)
                    for token in tokens
                    if str(token).strip()
                ]
            else:
                encoded = []
            if not encoded:
                continue
            if len(encoded) > max_len:
                encoded = encoded[-max_len:] if truncate == "pre" else encoded[:max_len]
            output[i, : len(encoded)] = encoded
        return output

    def process_target_fit(self, data: pd.Series, config: Dict[str, Any]):
        name = str(data.name)
        target_type = config["target_type"]
        label_map = config.get("label_map")
        if target_type == "binary":
            if label_map is None:
                unique_values = data.dropna().unique()
                sorted_values = sorted(unique_values)
                try:
                    int_values = [int(v) for v in sorted_values]
                    if int_values == list(range(len(int_values))):
                        label_map = {str(val): int(val) for val in sorted_values}
                    else:
                        label_map = {
                            str(val): idx for idx, val in enumerate(sorted_values)
                        }
                except (ValueError, TypeError):
                    label_map = {str(val): idx for idx, val in enumerate(sorted_values)}
                config["label_map"] = label_map
            self.target_encoders[name] = label_map

    def process_target_transform(
        self, data: pd.Series, config: Dict[str, Any]
    ) -> np.ndarray:
        logger = logging.getLogger()
        name = str(data.name)
        target_type = config.get("target_type")
        if target_type == "regression":
            values = np.array(data.values, dtype=np.float32)
            return values
        if target_type == "binary":
            label_map = self.target_encoders.get(name)
            if label_map is None:
                raise ValueError(
                    f"[Data Processor Error] Target encoder for {name} not fitted"
                )
            result = []
            for val in data:
                str_val = str(val)
                if str_val in label_map:
                    result.append(label_map[str_val])
                else:
                    logger.warning(f"Unknown target value: {val}, mapping to 0")
                    result.append(0)
            return np.array(result, dtype=np.float32)
        raise ValueError(
            f"[Data Processor Error] Unsupported target type: {target_type}"
        )

    def load_dataframe_from_path(self, path: str) -> pd.DataFrame:
        """Load all data from a file or directory path into a single DataFrame."""
        file_paths, file_type = resolve_file_paths(path)
        frames = load_dataframes(file_paths, file_type)
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def extract_sequence_tokens(self, value: Any, separator: str) -> list[str]:
        """Extract sequence tokens from a single value."""
        if value is None:
            return []
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [] if not stripped else stripped.split(separator)
        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(v) for v in value]
        return [str(value)]

    def fit_from_path(self, path: str, chunk_size: int) -> "DataProcessor":
        """Fit processor statistics by streaming files to reduce memory usage."""
        logger = logging.getLogger()
        logger.info(
            colorize(
                "Fitting DataProcessor (streaming path mode)...",
                color="cyan",
                bold=True,
            )
        )
        file_paths, file_type = resolve_file_paths(path)
        if not check_streaming_support(file_type):
            raise ValueError(
                f"[DataProcessor Error] Format '{file_type}' does not support streaming. "
                "fit_from_path only supports streaming formats (csv, parquet) to avoid high memory usage. "
                "Use fit(dataframe) with in-memory data or convert the data format."
            )

        numeric_acc: Dict[str, Dict[str, float]] = {}
        for name in self.numeric_features.keys():
            numeric_acc[name] = {
                "sum": 0.0,
                "sumsq": 0.0,
                "count": 0.0,
                "min": np.inf,
                "max": -np.inf,
                "max_abs": 0.0,
            }
        sparse_vocab: Dict[str, set[str]] = {
            name: set() for name in self.sparse_features.keys()
        }
        seq_vocab: Dict[str, set[str]] = {
            name: set() for name in self.sequence_features.keys()
        }
        target_values: Dict[str, set[Any]] = {
            name: set() for name in self.target_features.keys()
        }
        missing_features = set()
        for file_path in file_paths:
            for chunk in iter_file_chunks(file_path, file_type, chunk_size):
                columns = set(chunk.columns)
                feature_groups = [
                    ("numeric", self.numeric_features),
                    ("sparse", self.sparse_features),
                    ("sequence", self.sequence_features),
                ]
                for group, features in feature_groups:
                    missing_features.update(features.keys() - columns)
                    for name in features.keys() & columns:
                        config = features[name]
                        series = chunk[name]
                        if group == "numeric":
                            values = pd.to_numeric(series, errors="coerce").dropna()
                            if values.empty:
                                continue
                            acc = numeric_acc[name]
                            arr = values.to_numpy(dtype=np.float64, copy=False)
                            acc["count"] += arr.size
                            acc["sum"] += float(arr.sum())
                            acc["sumsq"] += float(np.square(arr).sum())
                            acc["min"] = min(acc["min"], float(arr.min()))
                            acc["max"] = max(acc["max"], float(arr.max()))
                            acc["max_abs"] = max(
                                acc["max_abs"], float(np.abs(arr).max())
                            )
                        elif group == "sparse":
                            fill_na = config["fill_na"]
                            series = series.fillna(fill_na).astype(str)
                            sparse_vocab[name].update(series.tolist())
                        else:
                            separator = config["separator"]
                            tokens = []
                            for val in series:
                                tokens.extend(
                                    self.extract_sequence_tokens(val, separator)
                                )
                            seq_vocab[name].update(tokens)

                # target features
                missing_features.update(self.target_features.keys() - columns)
                for name in self.target_features.keys() & columns:
                    vals = chunk[name].dropna().tolist()
                    target_values[name].update(vals)
        if missing_features:
            logger.warning(
                f"The following configured features were not found in provided files: {sorted(missing_features)}"
            )
        # finalize numeric scalers
        for name, config in self.numeric_features.items():
            acc = numeric_acc[name]
            if acc["count"] == 0:
                logger.warning(
                    f"Numeric feature {name} has no valid values in provided files"
                )
                continue
            mean_val = acc["sum"] / acc["count"]
            if config["fill_na"] is not None:
                config["fill_na_value"] = config["fill_na"]
            else:
                config["fill_na_value"] = mean_val
            scaler_type = config["scaler"]
            if scaler_type == "standard":
                var = max(acc["sumsq"] / acc["count"] - mean_val * mean_val, 0.0)
                scaler = StandardScaler()
                scaler.mean_ = np.array([mean_val], dtype=np.float64)
                scaler.var_ = np.array([var], dtype=np.float64)
                scaler.scale_ = np.array(
                    [np.sqrt(var) if var > 0 else 1.0], dtype=np.float64
                )
                scaler.n_samples_seen_ = np.array([int(acc["count"])], dtype=np.int64)
                self.scalers[name] = scaler

            elif scaler_type == "minmax":
                data_min = acc["min"] if np.isfinite(acc["min"]) else 0.0
                data_max = acc["max"] if np.isfinite(acc["max"]) else data_min
                scaler = MinMaxScaler()
                scaler.data_min_ = np.array([data_min], dtype=np.float64)
                scaler.data_max_ = np.array([data_max], dtype=np.float64)
                scaler.data_range_ = scaler.data_max_ - scaler.data_min_
                scaler.data_range_[scaler.data_range_ == 0] = 1.0
                # Manually set scale_/min_ for streaming fit to mirror sklearn's internal fit logic
                feature_min, feature_max = scaler.feature_range
                scale = (feature_max - feature_min) / scaler.data_range_
                scaler.scale_ = scale
                scaler.min_ = feature_min - scaler.data_min_ * scale
                scaler.n_samples_seen_ = np.array([int(acc["count"])], dtype=np.int64)
                self.scalers[name] = scaler

            elif scaler_type == "maxabs":
                scaler = MaxAbsScaler()
                scaler.max_abs_ = np.array([acc["max_abs"]], dtype=np.float64)
                scaler.n_samples_seen_ = np.array([int(acc["count"])], dtype=np.int64)
                self.scalers[name] = scaler

            elif scaler_type in ("log", "none", "robust"):
                # log and none do not require fitting; robust requires full data and is handled earlier
                continue
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")

        # finalize sparse label encoders
        for name, config in self.sparse_features.items():
            if config["encode_method"] == "label":
                vocab = sparse_vocab[name]
                if not vocab:
                    logger.warning(f"Sparse feature {name} has empty vocabulary")
                    continue
                vocab_list = sorted(vocab)
                if "<UNK>" not in vocab_list:
                    vocab_list.append("<UNK>")
                token_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
                config["_token_to_idx"] = token_to_idx
                config["_unk_index"] = token_to_idx["<UNK>"]
                config["vocab_size"] = len(vocab_list)
            elif config["encode_method"] == "hash":
                config["vocab_size"] = config["hash_size"]

        # finalize sequence vocabularies
        for name, config in self.sequence_features.items():
            if config["encode_method"] == "label":
                vocab_set = seq_vocab[name]
                vocab_list = sorted(vocab_set) if vocab_set else ["<PAD>"]
                if "<UNK>" not in vocab_list:
                    vocab_list.append("<UNK>")
                token_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
                config["_token_to_idx"] = token_to_idx
                config["_unk_index"] = token_to_idx["<UNK>"]
                config["vocab_size"] = len(vocab_list)
            elif config["encode_method"] == "hash":
                config["vocab_size"] = config["hash_size"]

        # finalize targets
        for name, config in self.target_features.items():
            if not target_values[name]:
                logger.warning(f"Target {name} has no valid values in provided files")
                continue
            self.process_target_fit(
                pd.Series(list(target_values[name]), name=name), config
            )

        self.is_fitted = True
        logger.info(
            colorize(
                "DataProcessor fitted successfully",
                color="green",
                bold=True,
            )
        )
        return self

    def transform_in_memory(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_dict: bool,
        persist: bool,
        save_format: Optional[str],
        output_path: Optional[str],
        warn_missing: bool = True,
    ):
        logger = logging.getLogger()
        data_dict = data if isinstance(data, dict) else None

        result_dict = {}
        if isinstance(data, pd.DataFrame):
            df = data  # type: ignore[assignment]
            for col in df.columns:
                result_dict[col] = df[col].to_numpy(copy=False)
        else:
            if data_dict is None:
                raise ValueError(
                    f"[Data Processor Error] Unsupported data type: {type(data)}"
                )
            for key, value in data_dict.items():
                if isinstance(value, pd.Series):
                    result_dict[key] = value.to_numpy(copy=False)
                else:
                    result_dict[key] = np.asarray(value)

        data_columns = data.columns if isinstance(data, pd.DataFrame) else data_dict
        feature_groups = [
            ("Numeric", self.numeric_features, self.process_numeric_feature_transform),
            ("Sparse", self.sparse_features, self.process_sparse_feature_transform),
            (
                "Sequence",
                self.sequence_features,
                self.process_sequence_feature_transform,
            ),
            ("Target", self.target_features, self.process_target_transform),
        ]
        for label, features, transform_fn in feature_groups:
            for name, config in features.items():
                present = name in data_columns  # type: ignore[operator]
                if not present:
                    if warn_missing:
                        logger.warning(f"{label} feature {name} not found in data")
                    continue
                series_data = (
                    data[name]
                    if isinstance(data, pd.DataFrame)
                    else pd.Series(result_dict[name], name=name)
                )
                result_dict[name] = transform_fn(series_data, config)

        def dict_to_dataframe(result: Dict[str, np.ndarray]) -> pd.DataFrame:
            # Convert all arrays to Series/lists at once to avoid fragmentation
            columns_dict = {}
            for key, value in result.items():
                if key in self.sequence_features:
                    columns_dict[key] = np.asarray(value).tolist()
                else:
                    columns_dict[key] = value
            return pd.DataFrame(columns_dict)

        effective_format = save_format
        if persist:
            effective_format = save_format or "parquet"
        result_df = None
        if (not return_dict) or persist:
            result_df = dict_to_dataframe(result_dict)
        if persist:
            if effective_format not in FILE_FORMAT_CONFIG:
                raise ValueError(f"Unsupported save format: {effective_format}")
            if output_path is None:
                raise ValueError(
                    "[Data Processor Error] output_path must be provided when persisting transformed data."
                )
            output_dir = Path(output_path)
            if output_dir.suffix:
                output_dir = output_dir.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            suffix = FILE_FORMAT_CONFIG[effective_format]["extension"][0]
            save_path = output_dir / f"transformed_data{suffix}"
            assert result_df is not None, "DataFrame conversion failed"

            # Save based on format
            if effective_format == "csv":
                result_df.to_csv(save_path, index=False)
            elif effective_format == "parquet":
                result_df.to_parquet(save_path, index=False)
            elif effective_format == "feather":
                result_df.to_feather(save_path)
            elif effective_format == "excel":
                result_df.to_excel(save_path, index=False)
            elif effective_format == "hdf5":
                result_df.to_hdf(save_path, key="data", mode="w")
            else:
                raise ValueError(f"Unsupported save format: {effective_format}")

            logger.info(
                colorize(
                    f"Transformed data saved to: {save_path.resolve()}", color="green"
                )
            )
        if return_dict:
            return result_dict
        assert result_df is not None, "DataFrame is None after transform"
        return result_df

    def transform_path(
        self,
        input_path: str,
        output_path: Optional[str],
        save_format: Optional[str],
        chunk_size: int = 200000,
    ):
        """Transform data from files under a path and save them to a new location.

        Uses chunked reading/writing to keep peak memory bounded for large files.
        """
        logger = logging.getLogger()
        file_paths, file_type = resolve_file_paths(input_path)
        target_format = save_format or file_type
        if target_format not in FILE_FORMAT_CONFIG:
            raise ValueError(f"Unsupported format: {target_format}")
        if chunk_size > 0 and not check_streaming_support(file_type):
            raise ValueError(
                f"Input format '{file_type}' does not support streaming reads. "
                "Set chunk_size<=0 to use full-load transform."
            )

        # Warn about streaming support
        if not check_streaming_support(target_format):
            logger.warning(
                f"[Data Processor Warning] Format '{target_format}' does not support streaming writes. "
                "Large files may require more memory. Use csv or parquet for better streaming support."
            )

        base_output_dir = (
            Path(output_path) if output_path else default_output_dir(input_path)
        )
        if base_output_dir.suffix:
            base_output_dir = base_output_dir.parent
        output_root = base_output_dir / "transformed_data"
        output_root.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for file_path in progress(file_paths, description="Transforming files"):
            source_path = Path(file_path)
            suffix = FILE_FORMAT_CONFIG[target_format]["extension"][0]
            target_file = output_root / f"{source_path.stem}{suffix}"

            # Stream transform for large files
            if chunk_size <= 0:
                # fallback to full load behavior
                df = read_table(file_path, file_type)
                transformed_df = self.transform_in_memory(
                    df,
                    return_dict=False,
                    persist=False,
                    save_format=None,
                    output_path=None,
                    warn_missing=True,
                )
                assert isinstance(
                    transformed_df, pd.DataFrame
                ), "[Data Processor Error] Expected DataFrame when return_dict=False"

                # Save based on format
                if target_format == "csv":
                    transformed_df.to_csv(target_file, index=False)
                elif target_format == "parquet":
                    transformed_df.to_parquet(target_file, index=False)
                elif target_format == "feather":
                    transformed_df.to_feather(target_file)
                elif target_format == "excel":
                    transformed_df.to_excel(target_file, index=False)
                elif target_format == "hdf5":
                    transformed_df.to_hdf(target_file, key="data", mode="w")
                else:
                    raise ValueError(f"Unsupported format: {target_format}")

                saved_paths.append(str(target_file.resolve()))
                continue

            first_chunk = True
            # Streaming write for supported formats
            if target_format == "parquet":
                parquet_writer = None
                try:
                    for chunk in iter_file_chunks(file_path, file_type, chunk_size):
                        transformed_df = self.transform_in_memory(
                            chunk,
                            return_dict=False,
                            persist=False,
                            save_format=None,
                            output_path=None,
                            warn_missing=first_chunk,
                        )
                        assert isinstance(
                            transformed_df, pd.DataFrame
                        ), "[Data Processor Error] Expected DataFrame when return_dict=False"
                        table = pa.Table.from_pandas(
                            transformed_df, preserve_index=False
                        )
                        if parquet_writer is None:
                            parquet_writer = pq.ParquetWriter(target_file, table.schema)
                        parquet_writer.write_table(table)
                        first_chunk = False
                finally:
                    if parquet_writer is not None:
                        parquet_writer.close()
            elif target_format == "csv":
                # CSV: append chunks; header only once
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with open(target_file, "w", encoding="utf-8", newline="") as f:
                    f.write("")
                for chunk in iter_file_chunks(file_path, file_type, chunk_size):
                    transformed_df = self.transform_in_memory(
                        chunk,
                        return_dict=False,
                        persist=False,
                        save_format=None,
                        output_path=None,
                        warn_missing=first_chunk,
                    )
                    assert isinstance(
                        transformed_df, pd.DataFrame
                    ), "[Data Processor Error] Expected DataFrame when return_dict=False"
                    transformed_df.to_csv(
                        target_file, index=False, mode="a", header=first_chunk
                    )
                    first_chunk = False
            else:
                # Non-streaming formats: collect all chunks and save once
                logger.warning(
                    f"Format '{target_format}' doesn't support streaming writes. "
                    f"Collecting all chunks in memory before saving."
                )
                all_chunks = []
                for chunk in iter_file_chunks(file_path, file_type, chunk_size):
                    transformed_df = self.transform_in_memory(
                        chunk,
                        return_dict=False,
                        persist=False,
                        save_format=None,
                        output_path=None,
                        warn_missing=first_chunk,
                    )
                    assert isinstance(transformed_df, pd.DataFrame)
                    all_chunks.append(transformed_df)
                    first_chunk = False

                if all_chunks:
                    combined_df = pd.concat(all_chunks, ignore_index=True)
                    if target_format == "feather":
                        combined_df.to_feather(target_file)
                    elif target_format == "excel":
                        combined_df.to_excel(target_file, index=False)
                    elif target_format == "hdf5":
                        combined_df.to_hdf(target_file, key="data", mode="w")

            saved_paths.append(str(target_file.resolve()))
        logger.info(
            colorize(
                f"Transformed {len(saved_paths)} file(s) saved to: {output_root.resolve()}",
                color="green",
            )
        )
        return saved_paths

    # fit is nothing but registering the statistics from data so that we can transform the data later
    def fit(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], str, os.PathLike],
        chunk_size: int = 200000,
    ):
        logger = logging.getLogger()
        if isinstance(data, (str, os.PathLike)):
            path_str = str(data)
            uses_robust = any(
                cfg.get("scaler") == "robust" for cfg in self.numeric_features.values()
            )
            if uses_robust:
                logger.warning(
                    "Robust scaler requires full data; loading all files into memory. Consider smaller chunk_size or different scaler if memory is limited."
                )
                data = self.load_dataframe_from_path(path_str)
            else:
                return self.fit_from_path(path_str, chunk_size)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        logger.info(colorize("Fitting DataProcessor...", color="cyan", bold=True))
        feature_groups = [
            ("Numeric", self.numeric_features, self.process_numeric_feature_fit),
            ("Sparse", self.sparse_features, self.process_sparse_feature_fit),
            ("Sequence", self.sequence_features, self.process_sequence_feature_fit),
            ("Target", self.target_features, self.process_target_fit),
        ]
        for label, features, fit_fn in feature_groups:
            for name, config in features.items():
                if name not in data.columns:
                    logger.warning(f"{label} feature {name} not found in data")
                    continue
                fit_fn(data[name], config)
        self.is_fitted = True
        return self

    def transform(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], str, os.PathLike],
        return_dict: bool = True,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
        chunk_size: int = 200000,
    ):
        if not self.is_fitted:
            raise ValueError(
                "[Data Processor Error] DataProcessor must be fitted before transform"
            )
        if isinstance(data, (str, os.PathLike)):
            if return_dict:
                raise ValueError(
                    "[Data Processor Error] Path transform writes files only; set return_dict=False when passing a path."
                )
            return self.transform_path(
                str(data), output_path, save_format, chunk_size=chunk_size
            )
        return self.transform_in_memory(
            data=data,
            return_dict=return_dict,
            persist=output_path is not None,
            save_format=save_format,
            output_path=output_path,
        )

    def fit_transform(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], str, os.PathLike],
        return_dict: bool = True,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
        chunk_size: int = 200000,
    ):
        self.fit(data, chunk_size=chunk_size)
        return self.transform(
            data,
            return_dict=return_dict,
            save_format=save_format,
            output_path=output_path,
        )

    def save(self, save_path: str | Path):
        logger = logging.getLogger()
        assert isinstance(save_path, (str, Path)), "save_path must be a string or Path"
        save_path = Path(save_path)
        if not self.is_fitted:
            logger.warning("Saving unfitted DataProcessor")
        target_path = resolve_save_path(
            path=save_path,
            default_dir=Path(os.getcwd()),
            default_name="fitted_processor",
            suffix=".pkl",
            add_timestamp=False,
        )
        state = {
            "numeric_features": self.numeric_features,
            "sparse_features": self.sparse_features,
            "sequence_features": self.sequence_features,
            "target_features": self.target_features,
            "is_fitted": self.is_fitted,
            "scalers": self.scalers,
            "label_encoders": self.label_encoders,
            "target_encoders": self.target_encoders,
            "processor_version": __version__,
        }
        with open(target_path, "wb") as f:
            pickle.dump(state, f)
        logger.info(
            f"DataProcessor saved to: {target_path}, NextRec version: {self.version}"
        )

    @classmethod
    def load(cls, load_path: str | Path) -> "DataProcessor":
        logger = logging.getLogger()
        load_path = Path(load_path)
        with open(load_path, "rb") as f:
            state = pickle.load(f)
        processor = cls()
        processor.numeric_features = state.get("numeric_features", {})
        processor.sparse_features = state.get("sparse_features", {})
        processor.sequence_features = state.get("sequence_features", {})
        processor.target_features = state.get("target_features", {})
        processor.is_fitted = state.get("is_fitted", False)
        processor.scalers = state.get("scalers", {})
        processor.label_encoders = state.get("label_encoders", {})
        processor.target_encoders = state.get("target_encoders", {})
        processor.version = state.get("processor_version", "unknown")
        logger.info(
            f"DataProcessor loaded from {load_path}, NextRec version: {processor.version}"
        )
        return processor

    def get_vocab_sizes(self) -> Dict[str, int]:
        vocab_sizes = {}
        for name, config in self.sparse_features.items():
            vocab_sizes[name] = config.get("vocab_size", 0)
        for name, config in self.sequence_features.items():
            vocab_sizes[name] = config.get("vocab_size", 0)
        return vocab_sizes

    def summary(self):
        """Print a summary of the DataProcessor configuration."""
        logger = logging.getLogger()

        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
        logger.info(colorize("DataProcessor Summary", color="bright_blue", bold=True))
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))

        logger.info("")
        logger.info(colorize("[1] Feature Configuration", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))

        if self.numeric_features:
            logger.info(f"Dense Features ({len(self.numeric_features)}):")

            max_name_len = max(len(name) for name in self.numeric_features.keys())
            name_width = max(max_name_len, 10) + 2

            logger.info(
                f"  {'#':<4} {'Name':<{name_width}} {'Scaler':>15} {'Fill NA':>10}"
            )
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*15} {'-'*10}")
            for i, (name, config) in enumerate(self.numeric_features.items(), 1):
                scaler = config["scaler"]
                fill_na = config.get("fill_na_value", config.get("fill_na", "N/A"))
                logger.info(
                    f"  {i:<4} {name:<{name_width}} {str(scaler):>15} {str(fill_na):>10}"
                )

        if self.sparse_features:
            logger.info(f"Sparse Features ({len(self.sparse_features)}):")

            max_name_len = max(len(name) for name in self.sparse_features.keys())
            name_width = max(max_name_len, 10) + 2

            logger.info(
                f"  {'#':<4} {'Name':<{name_width}} {'Method':>12} {'Vocab Size':>12} {'Hash Size':>12}"
            )
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*12} {'-'*12}")
            for i, (name, config) in enumerate(self.sparse_features.items(), 1):
                method = config["encode_method"]
                vocab_size = config.get("vocab_size", "N/A")
                hash_size = config.get("hash_size", "N/A")
                logger.info(
                    f"  {i:<4} {name:<{name_width}} {str(method):>12} {str(vocab_size):>12} {str(hash_size):>12}"
                )

        if self.sequence_features:
            logger.info(f"Sequence Features ({len(self.sequence_features)}):")

            max_name_len = max(len(name) for name in self.sequence_features.keys())
            name_width = max(max_name_len, 10) + 2

            logger.info(
                f"  {'#':<4} {'Name':<{name_width}} {'Method':>12} {'Vocab Size':>12} {'Hash Size':>12} {'Max Len':>10}"
            )
            logger.info(
                f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*12} {'-'*12} {'-'*10}"
            )
            for i, (name, config) in enumerate(self.sequence_features.items(), 1):
                method = config["encode_method"]
                vocab_size = config.get("vocab_size", "N/A")
                hash_size = config.get("hash_size", "N/A")
                max_len = config.get("max_len", "N/A")
                logger.info(
                    f"  {i:<4} {name:<{name_width}} {str(method):>12} {str(vocab_size):>12} {str(hash_size):>12} {str(max_len):>10}"
                )

        logger.info("")
        logger.info(colorize("[2] Target Configuration", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))

        if self.target_features:
            logger.info(f"Target Features ({len(self.target_features)}):")

            max_name_len = max(len(name) for name in self.target_features.keys())
            name_width = max(max_name_len, 10) + 2

            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Type':>15}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*15}")
            for i, (name, config) in enumerate(self.target_features.items(), 1):
                target_type = config["target_type"]
                logger.info(f"  {i:<4} {name:<{name_width}} {str(target_type):>15}")
        else:
            logger.info("No target features configured")

        logger.info("")
        logger.info(colorize("[3] Processor Status", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        logger.info(f"Fitted:                  {self.is_fitted}")
        logger.info(
            f"Total Features:          {len(self.numeric_features) + len(self.sparse_features) + len(self.sequence_features)}"
        )
        logger.info(f"  Dense Features:        {len(self.numeric_features)}")
        logger.info(f"  Sparse Features:       {len(self.sparse_features)}")
        logger.info(f"  Sequence Features:     {len(self.sequence_features)}")
        logger.info(f"Target Features:         {len(self.target_features)}")

        logger.info("")
        logger.info("")
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
