"""
DataProcessor for data preprocessing including numeric, sparse, sequence features and target processing.

Date: create on 13/11/2025
Checkpoint: edit on 13/03/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from nextrec.__version__ import __version__
from nextrec.basic.features import FeatureSet
from nextrec.basic.loggers import colorize
from nextrec.basic.session import get_save_path
from nextrec.utils.console import progress
from nextrec.utils.data import (
    get_file_paths,
)


class DataProcessor(FeatureSet):
    def __init__(
        self,
        hash_cache_size: int = 200_000,
    ):
        """
        DataProcessor for data preprocessing including numeric, sparse, sequence features and target processing.

        Args:
            hash_cache_size (int, optional): Cache size for string hashing. Defaults to 200,000.
        """
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
            )
        self.numeric_features = {}
        self.sparse_features = {}
        self.sequence_features = {}
        self.target_features = {}
        self.version = __version__

        self.is_fitted = False

        self.scalers = {}
        self.label_encoders = {}
        self.target_encoders = {}
        self.set_target_keys(target=[], key_columns=[])

        # cache hash function
        self.hash_cache_size = int(hash_cache_size)
        if self.hash_cache_size > 0:
            self.hash_fn = functools.lru_cache(maxsize=self.hash_cache_size)(self.hash_string)
        else:
            self.hash_fn = self.hash_string

    def __getstate__(self):
        state = self.__dict__.copy()
        # lru_cache wrappers on instance fields are not picklable under spawn
        state.pop("hash_fn", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.hash_cache_size > 0:
            self.hash_fn = functools.lru_cache(maxsize=self.hash_cache_size)(self.hash_string)
        else:
            self.hash_fn = self.hash_string

    @staticmethod
    def _build_output_name(source_name: str, method_name: str) -> str:
        return f"{source_name}_{method_name}"

    @staticmethod
    def _normalize_numeric_scalers(
        scaler: Optional[
            Union[
                Literal["standard", "minmax", "robust", "maxabs", "log", "none"],
                Iterable[Literal["standard", "minmax", "robust", "maxabs", "log", "none"]],
            ]
        ],
    ) -> list[str]:
        if scaler is None:
            values = ["none"]
        elif isinstance(scaler, str):
            values = [scaler]
        else:
            values = [str(v) for v in scaler]
        values = [str(v).strip().lower() for v in values if str(v).strip()]
        if not values:
            values = ["none"]
        allowed = {"standard", "minmax", "robust", "maxabs", "log", "none"}
        invalid = [v for v in values if v not in allowed]
        if invalid:
            raise ValueError(
                f"[Data Processor Error] Unsupported numeric scaler(s): {invalid}. " f"Supported: {sorted(allowed)}"
            )
        return values

    @staticmethod
    def _numeric_scaler_key(feature_key: str, stage_idx: int, scaler_type: str) -> str:
        return f"{feature_key}__stage{stage_idx}__{scaler_type}"

    @staticmethod
    def _config_source_name(feature_name: str, config: Dict[str, Any]) -> str:
        return str(config.get("source_name", feature_name))

    @staticmethod
    def _config_output_name(feature_name: str, config: Dict[str, Any]) -> str:
        return str(config.get("output_name", feature_name))

    @staticmethod
    def _normalize_filter_values(values: Optional[Union[str, Iterable[Any]]]) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            raw_values = [values]
        else:
            raw_values = list(values)
        normalized: list[str] = []
        for value in raw_values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                normalized.append(text)
        # Preserve order while removing duplicates
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _normalize_match_mode(match_mode: Optional[str]) -> str:
        mode = str(match_mode or "exact").strip().lower()
        if mode not in {"exact", "contains", "regex"}:
            raise ValueError(
                f"[Data Processor Error] Unsupported match_mode='{match_mode}'. " "Use one of: exact, contains, regex."
            )
        return mode

    @staticmethod
    def _build_sparse_match_expr(col, patterns: list[str], match_mode: str):
        if match_mode == "exact":
            return col.is_in(patterns)
        if match_mode == "contains":
            exprs = [col.str.contains(re.escape(pat)) for pat in patterns]
            return pl.any_horizontal(exprs)
        exprs = [col.str.contains(pat) for pat in patterns]
        return pl.any_horizontal(exprs)

    @staticmethod
    def _build_sequence_match_expr(seq_col, patterns: list[str], match_mode: str):
        if match_mode == "exact":
            exprs = [seq_col.list.contains(token) for token in patterns]
            return pl.any_horizontal(exprs)
        if match_mode == "contains":
            exprs = [
                seq_col.list.eval(pl.element().cast(pl.Utf8).str.contains(re.escape(pat))).list.any()
                for pat in patterns
            ]
            return pl.any_horizontal(exprs)
        exprs = [seq_col.list.eval(pl.element().cast(pl.Utf8).str.contains(pat)).list.any() for pat in patterns]
        return pl.any_horizontal(exprs)

    @staticmethod
    def _map_with_default(expr, mapping: Dict[str, int], default: int, dtype):
        # Compatible with older polars versions without Expr.map_dict
        return expr.map_elements(
            lambda x: mapping.get(x, default),
            return_dtype=dtype,
        )

    def add_numeric_feature(
        self,
        name: str,
        scaler: Optional[
            Union[
                Literal["standard", "minmax", "robust", "maxabs", "log", "none"],
                Iterable[Literal["standard", "minmax", "robust", "maxabs", "log", "none"]],
            ]
        ] = "standard",
        fill_na: Optional[float] = None,
    ):
        """Add a numeric feature configuration.

        Args:
            name (str): Feature name.
            scaler: Scaler type or ordered scaler list. Supported values:
                "standard", "minmax", "robust", "maxabs", "log", "none".
            fill_na (Optional[float], optional): Fill value for missing entries. Defaults to None.
        """
        scaler_pipeline = self._normalize_numeric_scalers(scaler)
        method_name = "_".join(scaler_pipeline)
        output_name = self._build_output_name(name, method_name)
        self.numeric_features[output_name] = {
            "scaler": scaler if isinstance(scaler, str) or scaler is None else list(scaler),
            "scaler_pipeline": scaler_pipeline,
            "fill_na": fill_na,
            "source_name": name,
            "output_name": output_name,
        }

    def add_sparse_feature(
        self,
        name: str,
        encode_method: Literal["hash", "label"] = "hash",
        hash_size: Optional[int] = None,
        min_freq: Optional[int] = None,
        fill_na: str = "<UNK>",
        filter_value: Optional[Union[str, Iterable[Any]]] = None,
        keep_value: Optional[Union[str, Iterable[Any]]] = None,
        match_mode: Literal["exact", "contains", "regex"] = "exact",
    ):
        """Add a sparse feature configuration.

        Args:
            name: Feature name.
            encode_method: Encoding method, including "hash encoding" and "label encoding". Defaults to "hash" because it is more scalable and much faster.
            hash_size: Hash size for hash encoding. Required if encode_method is "hash".
            min_freq: Minimum frequency for hash encoding to keep tokens; lower-frequency tokens map to unknown. Defaults to None.
            fill_na: Fill value for missing entries. Defaults to "<UNK>".
            filter_value: Drop rows where sparse value is in this list.
            keep_value: Keep only rows where sparse value is in this list.
            match_mode: Matching mode for keep/filter values: exact, contains, regex.
        """
        if encode_method == "hash" and hash_size is None:
            raise ValueError("[Data Processor Error] hash_size must be specified when encode_method='hash'")
        normalized_match_mode = self._normalize_match_mode(match_mode)
        method_name = str(encode_method)
        output_name = self._build_output_name(name, method_name)
        filter_values = self._normalize_filter_values(filter_value)
        keep_values = self._normalize_filter_values(keep_value)
        self.sparse_features[output_name] = {
            "encode_method": encode_method,
            "hash_size": hash_size,
            "min_freq": min_freq,
            "fill_na": fill_na,
            "filter_value": filter_values,
            "keep_value": keep_values,
            "match_mode": normalized_match_mode,
            "source_name": name,
            "output_name": output_name,
        }

    def add_sequence_feature(
        self,
        name: str,
        encode_method: Literal["hash", "label"] = "hash",
        hash_size: Optional[int] = None,
        min_freq: Optional[int] = None,
        max_len: Optional[int] = 50,
        pad_value: int = 0,
        truncate: Literal["pre", "post"] = "pre",  # pre: keep last max_len items, post: keep first max_len items
        separator: str = ",",
        filter_value: Optional[Union[str, Iterable[Any]]] = None,
        keep_value: Optional[Union[str, Iterable[Any]]] = None,
        match_mode: Literal["exact", "contains", "regex"] = "exact",
    ):
        """Add a sequence feature configuration.

        Args:
            name: Feature name.
            encode_method: Encoding method, including "hash encoding" and "label encoding". Defaults to "hash".
            hash_size: Hash size for hash encoding. Required if encode_method is "hash".
            min_freq: Minimum frequency for hash encoding to keep tokens; lower-frequency tokens map to unknown. Defaults to None.
            max_len: Maximum sequence length. Defaults to 50.
            pad_value: Padding value for sequences shorter than max_len. Defaults to 0.
            truncate: Truncation strategy for sequences longer than max_len, including "pre" (keep last max_len items) and "post" (keep first max_len items). Defaults to "pre".
            separator: Separator for string sequences. Defaults to ",".
            filter_value: Drop rows where sequence contains any token in this list.
            keep_value: Keep only rows where sequence contains at least one token in this list.
            match_mode: Matching mode for keep/filter values: exact, contains, regex.
        """
        if encode_method == "hash" and hash_size is None:
            raise ValueError("[Data Processor Error] hash_size must be specified when encode_method='hash'")
        normalized_match_mode = self._normalize_match_mode(match_mode)
        method_name = str(encode_method)
        output_name = self._build_output_name(name, method_name)
        filter_values = self._normalize_filter_values(filter_value)
        keep_values = self._normalize_filter_values(keep_value)
        self.sequence_features[output_name] = {
            "encode_method": encode_method,
            "hash_size": hash_size,
            "min_freq": min_freq,
            "max_len": max_len,
            "pad_value": pad_value,
            "truncate": truncate,
            "separator": separator,
            "filter_value": filter_values,
            "keep_value": keep_values,
            "match_mode": normalized_match_mode,
            "source_name": name,
            "output_name": output_name,
        }

    def add_target(
        self,
        name: str,  # example: 'click'
        target_type: Literal["binary", "regression"] = "binary",
        label_map: Optional[Dict[str, int]] = None,  # example: {'click': 1, 'no_click': 0}
    ):
        """Add a target configuration.

        Args:
            name (str): Target name.
            target_type (Literal["binary", "regression"], optional): Target type. Defaults to "binary".
            label_map (Optional[Dict[str, int]], optional): Label mapping for binary targets. Defaults to None.
        """

        self.target_features[name] = {
            "target_type": target_type,
            "label_map": label_map,
        }
        self.set_target_keys(list(self.target_features.keys()), [])

    @staticmethod
    def hash_string(value: str, hash_size: int) -> int:
        hashed = pl.Series([value], dtype=pl.Utf8).hash().cast(pl.UInt64)
        return int(hashed[0]) % int(hash_size)

    def polars_scan(self, file_paths: list[str], file_type: str):
        file_type = file_type.lower()
        if file_type == "csv":
            return pl.scan_csv(file_paths, ignore_errors=True)
        if file_type == "parquet":
            return pl.scan_parquet(file_paths)
        raise ValueError(f"[Data Processor Error] Polars backend only supports csv/parquet, got: {file_type}")

    def sequence_expr(self, source_name: str, config: Dict[str, Any], schema: Dict[str, Any]):
        """
        generate polars expression for sequence feature processing

        Example Input:
            sequence_str: "1,2,3"
            sequence_str: " 4, ,5 "
            sequence_list: ["7", "8", "9"]
            sequence_list: ["", "10", " 11 "]

        Example Output:
            sequence_str  -> ["1","2","3"]
            sequence_str  -> ["4","5"]
            sequence_list -> ["7","8","9"]
            sequence_list -> ["10","11"]
        """
        separator = config["separator"]
        dtype = schema.get(source_name)
        col = pl.col(source_name)
        if dtype is not None and isinstance(dtype, pl.List):
            seq_col = col
        else:
            seq_col = col.cast(pl.Utf8).fill_null("").str.split(separator)
        elem = pl.element().cast(pl.Utf8).str.strip_chars()
        seq_col = seq_col.list.eval(pl.when(elem == "").then(None).otherwise(elem)).list.drop_nulls()
        return seq_col

    def apply_row_filters(self, lazy_frame, schema: Dict[str, Any]):
        logger = logging.getLogger()
        keep_filters = []
        drop_filters = []

        for name, config in self.sparse_features.items():
            source_name = self._config_source_name(name, config)
            output_name = self._config_output_name(name, config)
            if source_name not in schema:
                continue
            keep_values = config.get("keep_value") or []
            filter_values = config.get("filter_value") or []
            match_mode = self._normalize_match_mode(config.get("match_mode"))
            if not keep_values and not filter_values:
                continue
            col = pl.col(source_name).cast(pl.Utf8).fill_null(config.get("fill_na", "<UNK>"))
            if keep_values:
                keep_filters.append(self._build_sparse_match_expr(col, keep_values, match_mode))
            if filter_values:
                drop_filters.append(self._build_sparse_match_expr(col, filter_values, match_mode))
            logger.info(
                f"Apply sparse row filter on {output_name}: keep_value={keep_values}, "
                f"filter_value={filter_values}, match_mode={match_mode}"
            )

        for name, config in self.sequence_features.items():
            source_name = self._config_source_name(name, config)
            output_name = self._config_output_name(name, config)
            if source_name not in schema:
                continue
            keep_values = config.get("keep_value") or []
            filter_values = config.get("filter_value") or []
            match_mode = self._normalize_match_mode(config.get("match_mode"))
            if not keep_values and not filter_values:
                continue
            seq_col = self.sequence_expr(source_name, config, schema)
            if keep_values:
                keep_filters.append(self._build_sequence_match_expr(seq_col, keep_values, match_mode))
            if filter_values:
                drop_filters.append(self._build_sequence_match_expr(seq_col, filter_values, match_mode))
            logger.info(
                f"Apply sequence row filter on {output_name}: keep_value={keep_values}, "
                f"filter_value={filter_values}, match_mode={match_mode}"
            )

        predicates = []
        if keep_filters:
            predicates.append(pl.any_horizontal(keep_filters))
        if drop_filters:
            predicates.append(~pl.any_horizontal(drop_filters))

        if not predicates:
            return lazy_frame
        return lazy_frame.filter(pl.all_horizontal(predicates))

    def apply_transforms(self, lazy_frame, schema: Dict[str, Any]):
        """
        Apply all transformations to a Polars LazyFrame.

        """
        logger = logging.getLogger()
        expressions = []
        output_aliases = set()

        # Numeric features
        for name, config in self.numeric_features.items():
            source_name = self._config_source_name(name, config)
            output_name = self._config_output_name(name, config)
            if source_name not in schema:
                logger.warning(f"Numeric feature {source_name} not found in data")
                continue
            scaler_pipeline = self._normalize_numeric_scalers(config.get("scaler_pipeline", config.get("scaler")))
            fill_na_value = config.get("fill_na_value", 0)
            col = pl.col(source_name).cast(pl.Float64).fill_null(fill_na_value)
            for idx, scaler_type in enumerate(scaler_pipeline):
                if scaler_type == "log":
                    col = col.clip(lower_bound=0).log1p()
                    continue
                if scaler_type == "none":
                    continue
                scaler_key = self._numeric_scaler_key(name, idx, scaler_type)
                scaler = self.scalers.get(scaler_key)
                # Backward compatibility for old single-scaler checkpoints.
                if scaler is None and len(scaler_pipeline) == 1:
                    scaler = self.scalers.get(name)
                if scaler is None:
                    logger.warning(
                        f"Scaler(stage={idx}, type={scaler_type}) for {output_name} not fitted, returning current values"
                    )
                    continue
                if scaler_type == "standard":
                    mean = float(scaler.mean_[0])
                    scale = float(scaler.scale_[0]) if scaler.scale_[0] != 0 else 1.0
                    col = (col - mean) / scale
                elif scaler_type == "minmax":
                    scale = float(scaler.scale_[0])
                    min_val = float(scaler.min_[0])
                    col = col * scale + min_val
                elif scaler_type == "maxabs":
                    max_abs = float(scaler.max_abs_[0]) or 1.0
                    col = col / max_abs
                elif scaler_type == "robust":
                    center = float(scaler.center_[0])
                    scale = float(scaler.scale_[0]) if scaler.scale_[0] != 0 else 1.0
                    col = (col - center) / scale
            if output_name in output_aliases:
                continue
            output_aliases.add(output_name)
            expressions.append(col.alias(output_name))

        # Sparse features
        for name, config in self.sparse_features.items():
            source_name = self._config_source_name(name, config)
            output_name = self._config_output_name(name, config)
            if source_name not in schema:
                logger.warning(f"Sparse feature {source_name} not found in data")
                continue
            encode_method = config["encode_method"]
            fill_na = config["fill_na"]
            col = pl.col(source_name).cast(pl.Utf8).fill_null(fill_na)
            if encode_method == "label":
                token_to_idx = config.get("_token_to_idx")
                if not isinstance(token_to_idx, dict):
                    raise ValueError(f"[Data Processor Error] Token index for {output_name} not fitted")
                unk_index = int(config.get("_unk_index", 0))
                col = self._map_with_default(col, token_to_idx, unk_index, pl.Int64)
            elif encode_method == "hash":
                hash_size = config["hash_size"]
                hash_expr = col.hash().cast(pl.UInt64) % int(hash_size)
                min_freq = config.get("min_freq")
                token_counts = config.get("_token_counts")
                if min_freq is not None and isinstance(token_counts, dict):
                    low_freq = [k for k, v in token_counts.items() if v < min_freq]
                    unk_hash = config.get("_unk_hash")
                    if unk_hash is None:
                        unk_hash = self.hash_fn("<UNK>", int(hash_size))
                    hash_expr = pl.when(col.is_in(low_freq)).then(int(unk_hash)).otherwise(hash_expr)
                col = hash_expr.cast(pl.Int64)
            if output_name in output_aliases:
                continue
            output_aliases.add(output_name)
            expressions.append(col.alias(output_name))

        # Sequence features
        for name, config in self.sequence_features.items():
            source_name = self._config_source_name(name, config)
            output_name = self._config_output_name(name, config)
            if source_name not in schema:
                logger.warning(f"Sequence feature {source_name} not found in data")
                continue
            encode_method = config["encode_method"]
            max_len = int(config["max_len"])
            pad_value = int(config["pad_value"])
            truncate = config["truncate"]
            seq_col = self.sequence_expr(source_name, config, schema)

            if encode_method == "label":
                token_to_idx = config.get("_token_to_idx")
                if not isinstance(token_to_idx, dict):
                    raise ValueError(f"[Data Processor Error] Token index for {output_name} not fitted")
                unk_index = int(config.get("_unk_index", 0))
                seq_col = seq_col.list.eval(self._map_with_default(pl.element(), token_to_idx, unk_index, pl.Int64))
            elif encode_method == "hash":
                hash_size = config.get("hash_size")
                if hash_size is None:
                    raise ValueError("[Data Processor Error] hash_size must be set for hash encoding")
                elem = pl.element().cast(pl.Utf8)
                hash_expr = elem.hash().cast(pl.UInt64) % int(hash_size)
                min_freq = config.get("min_freq")
                token_counts = config.get("_token_counts")
                if min_freq is not None and isinstance(token_counts, dict):
                    low_freq = [k for k, v in token_counts.items() if v < min_freq]
                    unk_hash = config.get("_unk_hash")
                    if unk_hash is None:
                        unk_hash = self.hash_fn("<UNK>", int(hash_size))
                    hash_expr = pl.when(elem.is_in(low_freq)).then(int(unk_hash)).otherwise(hash_expr)
                seq_col = seq_col.list.eval(hash_expr)

            if truncate == "pre":
                seq_col = seq_col.list.tail(max_len)
            else:
                seq_col = seq_col.list.head(max_len)
            pad_list = [pad_value] * max_len
            seq_col = pl.concat_list([seq_col, pl.lit(pad_list)]).list.head(max_len)
            if output_name in output_aliases:
                continue
            output_aliases.add(output_name)
            expressions.append(seq_col.alias(output_name))

        # Target features
        for name, config in self.target_features.items():
            if name not in schema:
                logger.warning(f"Target feature {name} not found in data")
                continue
            target_type = config.get("target_type")
            col = pl.col(name)
            if target_type == "regression":
                col = col.cast(pl.Float32)
            elif target_type == "binary":
                label_map = self.target_encoders.get(name)
                if label_map is None:
                    raise ValueError(f"[Data Processor Error] Target encoder for {name} not fitted")
                col = self._map_with_default(col.cast(pl.Utf8), label_map, 0, pl.Int64).cast(pl.Float32)
            else:
                raise ValueError(f"[Data Processor Error] Unsupported target type: {target_type}")
            expressions.append(col.alias(name))

        if not expressions:
            return lazy_frame
        return lazy_frame.with_columns(expressions)

    def process_target_fit(self, data: Iterable[Any], config: Dict[str, Any], name: str) -> None:
        target_type = config["target_type"]
        label_map = config.get("label_map")
        if target_type == "binary":
            if label_map is None:
                unique_values = {v for v in data if v is not None}
                sorted_values = sorted(v for v in unique_values if v is not None)

                int_values = [int(v) for v in sorted_values]
                if int_values == list(range(len(int_values))):
                    label_map = {str(val): int(val) for val in sorted_values}
                else:
                    label_map = {str(val): idx for idx, val in enumerate(sorted_values)}

                config["label_map"] = label_map
            self.target_encoders[name] = label_map

    def fit_from_lazy(self, lazy_frame, schema: Dict[str, Any]) -> "DataProcessor":
        logger = logging.getLogger()
        lazy_frame = self.apply_row_filters(lazy_frame, schema)

        missing_features = set()
        for name, config in self.numeric_features.items():
            source_name = self._config_source_name(name, config)
            if source_name not in schema:
                missing_features.add(source_name)
        for name, config in self.sparse_features.items():
            source_name = self._config_source_name(name, config)
            if source_name not in schema:
                missing_features.add(source_name)
        for name, config in self.sequence_features.items():
            source_name = self._config_source_name(name, config)
            if source_name not in schema:
                missing_features.add(source_name)
        for name in self.target_features.keys():
            if name not in schema:
                missing_features.add(name)
        if missing_features:
            logger.warning(
                f"The following configured features were not found in provided data: {sorted(missing_features)}"
            )

        for name, config in self.numeric_features.items():
            source_name = self._config_source_name(name, config)
            output_name = self._config_output_name(name, config)
            if source_name not in schema:
                continue
            scaler_pipeline = self._normalize_numeric_scalers(config.get("scaler_pipeline", config.get("scaler")))
            col_df = lazy_frame.select(pl.col(source_name).cast(pl.Float64).alias(source_name)).collect()
            raw = col_df[source_name].to_numpy()
            raw = np.asarray(raw, dtype=np.float64)
            valid_mask = ~np.isnan(raw)
            count = float(valid_mask.sum())
            if count == 0:
                logger.warning(f"Numeric feature {output_name} has no valid values in provided data")
                continue
            mean_val = float(np.nanmean(raw))
            if config["fill_na"] is not None:
                config["fill_na_value"] = config["fill_na"]
            else:
                config["fill_na_value"] = mean_val
            stage_values = np.where(np.isnan(raw), float(config["fill_na_value"]), raw).reshape(-1, 1)
            for idx, scaler_type in enumerate(scaler_pipeline):
                if scaler_type == "log":
                    stage_values = np.log1p(np.clip(stage_values, a_min=0.0, a_max=None))
                    continue
                if scaler_type == "none":
                    continue
                if scaler_type == "standard":
                    scaler = StandardScaler()
                elif scaler_type == "minmax":
                    scaler = MinMaxScaler()
                elif scaler_type == "maxabs":
                    scaler = MaxAbsScaler()
                elif scaler_type == "robust":
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {scaler_type}")
                scaler.fit(stage_values)
                scaler_key = self._numeric_scaler_key(name, idx, scaler_type)
                self.scalers[scaler_key] = scaler
                # Backward compatibility for old single-stage keys.
                if len(scaler_pipeline) == 1:
                    self.scalers[name] = scaler
                stage_values = scaler.transform(stage_values)

        # sparse features
        for name, config in self.sparse_features.items():
            source_name = self._config_source_name(name, config)
            output_name = self._config_output_name(name, config)
            if source_name not in schema:
                continue
            encode_method = config["encode_method"]
            fill_na = config["fill_na"]
            col = pl.col(source_name).cast(pl.Utf8).fill_null(fill_na)
            counts_df = (
                lazy_frame.select(col.alias(source_name)).group_by(source_name).agg(pl.len().alias("count")).collect()
            )
            counts = (
                dict(zip(counts_df[source_name].to_list(), counts_df["count"].to_list()))
                if counts_df.height > 0
                else {}
            )
            if encode_method == "label":
                min_freq = config.get("min_freq")
                if min_freq is not None:
                    config["_token_counts"] = counts
                    vocab = {token for token, count in counts.items() if count >= min_freq}
                    low_freq_types = sum(1 for count in counts.values() if count < min_freq)
                    total_types = len(counts)
                    kept_types = total_types - low_freq_types
                    if not config.get("_min_freq_logged"):
                        logger.info(
                            f"Sparse feature {output_name} min_freq={min_freq}: "
                            f"{total_types} token types total, "
                            f"{low_freq_types} low-frequency, "
                            f"{kept_types} kept."
                        )
                        config["_min_freq_logged"] = True
                else:
                    vocab = set(counts.keys())
                if not vocab:
                    logger.warning(f"Sparse feature {output_name} has empty vocabulary")
                    continue
                # Filter out None values before sorting to avoid comparison errors
                vocab_list = sorted(v for v in vocab if v is not None)
                if "<UNK>" not in vocab_list:
                    vocab_list.append("<UNK>")
                token_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
                config["_token_to_idx"] = token_to_idx
                config["_unk_index"] = token_to_idx["<UNK>"]
                config["vocab_size"] = len(vocab_list)
            elif encode_method == "hash":
                min_freq = config.get("min_freq")
                if min_freq is not None:
                    config["_token_counts"] = counts
                    config["_unk_hash"] = self.hash_fn("<UNK>", int(config["hash_size"]))
                    low_freq_types = sum(1 for count in counts.values() if count < min_freq)
                    total_types = len(counts)
                    kept_types = total_types - low_freq_types
                    if not config.get("_min_freq_logged"):
                        logger.info(
                            f"Sparse feature {output_name} min_freq={min_freq}: "
                            f"{total_types} token types total, "
                            f"{low_freq_types} low-frequency, "
                            f"{kept_types} kept."
                        )
                        config["_min_freq_logged"] = True
                config["vocab_size"] = config["hash_size"]

        # sequence features
        for name, config in self.sequence_features.items():
            source_name = self._config_source_name(name, config)
            output_name = self._config_output_name(name, config)
            if source_name not in schema:
                continue
            encode_method = config["encode_method"]
            seq_col = self.sequence_expr(source_name, config, schema)
            tokens_df = (
                lazy_frame.select(seq_col.alias("seq"))
                .explode("seq")
                .select(pl.col("seq").cast(pl.Utf8).alias("seq"))
                .drop_nulls("seq")
                .group_by("seq")
                .agg(pl.len().alias("count"))
                .collect()
            )
            counts = dict(zip(tokens_df["seq"].to_list(), tokens_df["count"].to_list())) if tokens_df.height > 0 else {}
            if encode_method == "label":
                min_freq = config.get("min_freq")
                if min_freq is not None:
                    config["_token_counts"] = counts
                    vocab_set = {token for token, count in counts.items() if count >= min_freq}
                    low_freq_types = sum(1 for count in counts.values() if count < min_freq)
                    total_types = len(counts)
                    kept_types = total_types - low_freq_types
                    if not config.get("_min_freq_logged"):
                        logger.info(
                            f"Sequence feature {output_name} min_freq={min_freq}: "
                            f"{total_types} token types total, "
                            f"{low_freq_types} low-frequency, "
                            f"{kept_types} kept."
                        )
                        config["_min_freq_logged"] = True
                else:
                    vocab_set = set(counts.keys())
                # Filter out None values before sorting to avoid comparison errors
                vocab_list = sorted(v for v in vocab_set if v is not None) if vocab_set else ["<PAD>"]
                if "<UNK>" not in vocab_list:
                    vocab_list.append("<UNK>")
                token_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
                config["_token_to_idx"] = token_to_idx
                config["_unk_index"] = token_to_idx["<UNK>"]
                config["vocab_size"] = len(vocab_list)
            elif encode_method == "hash":
                min_freq = config.get("min_freq")
                if min_freq is not None:
                    config["_token_counts"] = counts
                    config["_unk_hash"] = self.hash_fn("<UNK>", int(config["hash_size"]))
                    low_freq_types = sum(1 for count in counts.values() if count < min_freq)
                    total_types = len(counts)
                    kept_types = total_types - low_freq_types
                    if not config.get("_min_freq_logged"):
                        logger.info(
                            f"Sequence feature {output_name} min_freq={min_freq}: "
                            f"{total_types} token types total, "
                            f"{low_freq_types} low-frequency, "
                            f"{kept_types} kept."
                        )
                        config["_min_freq_logged"] = True
                config["vocab_size"] = config["hash_size"]

        # targets
        for name, config in self.target_features.items():
            if name not in schema:
                continue
            if config.get("target_type") == "binary":
                unique_vals = lazy_frame.select(pl.col(name).drop_nulls().unique()).collect().to_series().to_list()
                self.process_target_fit(unique_vals, config, name)

        self.is_fitted = True
        logger.info("")
        logger.info("DataProcessor fitted successfully")
        return self

    def fit_from_files(self, file_paths: list[str], file_type: str) -> "DataProcessor":
        logger = logging.getLogger()
        logger.info("Fitting DataProcessor...")

        for config in self.sparse_features.values():
            config.pop("_min_freq_logged", None)
        for config in self.sequence_features.values():
            config.pop("_min_freq_logged", None)
        lazy_frame = self.polars_scan(file_paths, file_type)
        schema = lazy_frame.collect_schema()
        return self.fit_from_lazy(lazy_frame, schema)

    def fit_from_path(self, path: str) -> "DataProcessor":
        logger = logging.getLogger()
        logger.info("Fitting DataProcessor...")

        for config in self.sparse_features.values():
            config.pop("_min_freq_logged", None)
        for config in self.sequence_features.values():
            config.pop("_min_freq_logged", None)
        file_paths, file_type = get_file_paths(path)
        return self.fit_from_files(file_paths=file_paths, file_type=file_type)

    def transform_in_memory(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, Dict[str, Any]],
        return_dict: bool,
        persist: bool,
        save_format: Optional[str],
        output_path: Optional[str],
    ):
        logger = logging.getLogger()

        if isinstance(data, dict):
            df = pl.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = pl.from_pandas(data)
        else:
            df = data

        schema = df.schema
        lazy_frame = df.lazy()
        lazy_frame = self.apply_row_filters(lazy_frame, schema)
        lazy_frame = self.apply_transforms(lazy_frame, schema)
        out_df = lazy_frame.collect()

        effective_format = save_format
        if persist:
            effective_format = save_format or "parquet"

        if persist:
            if effective_format not in {"csv", "parquet"}:
                raise ValueError(f"Unsupported save format: {effective_format}")
            if output_path is None:
                raise ValueError(
                    "[Data Processor Error] output_path must be provided when persisting transformed data."
                )
            output_dir = Path(output_path)
            if output_dir.suffix:
                output_dir = output_dir.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            suffix = ".csv" if effective_format == "csv" else ".parquet"
            save_path = output_dir / f"transformed_data{suffix}"
            if effective_format == "csv":
                out_df.write_csv(save_path)
            elif effective_format == "parquet":
                out_df.write_parquet(save_path)
            else:
                raise ValueError(f"Format '{effective_format}' is not supported by the polars-only pipeline.")
            logger.info(
                colorize(
                    f"Transformed data saved to: {save_path.resolve()}",
                    color="green",
                )
            )

        if return_dict:
            result_dict = {}
            sequence_output_names = set()
            for name, config in self.sequence_features.items():
                output_name = self._config_output_name(name, config)
                sequence_output_names.add(output_name)
            for col in out_df.columns:
                series = out_df.get_column(col)
                if col in sequence_output_names:
                    result_dict[col] = np.asarray(series.to_list(), dtype=np.int64)
                else:
                    result_dict[col] = series.to_numpy()
            return result_dict

        return out_df

    def transform_path(
        self,
        input_path: str,
        output_path: Optional[str],
        save_format: Optional[str],
    ):
        """Transform data from files under a path and save them using polars lazy pipeline."""
        logger = logging.getLogger()
        file_paths, file_type = get_file_paths(input_path)
        target_format = save_format or file_type
        if target_format not in {"csv", "parquet"}:
            raise ValueError(f"Format '{target_format}' is not supported by the polars-only pipeline.")
        if file_type not in {"csv", "parquet"}:
            raise ValueError(
                f"Input format '{file_type}' does not support streaming reads. "
                "Polars backend supports csv/parquet only."
            )

        if output_path:
            base_output_dir = Path(output_path)
        else:
            input_path_obj = Path(input_path)
            if input_path_obj.is_file():
                base_output_dir = input_path_obj.parent / f"{input_path_obj.stem}_preprocessed"
            else:
                base_output_dir = input_path_obj.with_name(f"{input_path_obj.name}_preprocessed")
        if base_output_dir.suffix:
            base_output_dir = base_output_dir.parent
        output_root = base_output_dir / "transformed_data"
        output_root.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for file_path in progress(file_paths, description="Transforming files"):
            source_path = Path(file_path)
            suffix = ".csv" if target_format == "csv" else ".parquet"
            target_file = output_root / f"{source_path.stem}{suffix}"

            lazy_frame = self.polars_scan([file_path], file_type)
            schema = lazy_frame.collect_schema()
            lazy_frame = self.apply_row_filters(lazy_frame, schema)
            lazy_frame = self.apply_transforms(lazy_frame, schema)

            if target_format == "parquet":
                lazy_frame.sink_parquet(target_file)
            elif target_format == "csv":
                # CSV doesn't support nested data (lists), so convert list columns to string
                transformed_schema = lazy_frame.collect_schema()
                list_cols = [name for name, dtype in transformed_schema.items() if isinstance(dtype, pl.List)]
                if list_cols:
                    # Convert list columns to string representation for CSV
                    # Format as [1, 2, 3] by casting elements to string, joining with ", ", and adding brackets
                    list_exprs = []
                    for name in list_cols:
                        # Convert list to string representation
                        list_exprs.append(
                            (
                                pl.lit("[")
                                + pl.col(name).list.eval(pl.element().cast(pl.String)).list.join(", ")
                                + pl.lit("]")
                            ).alias(name)
                        )
                    lazy_frame = lazy_frame.with_columns(list_exprs)
                lazy_frame.sink_csv(target_file)
            else:
                raise ValueError(f"Format '{target_format}' is not supported by the polars-only pipeline.")
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
        data: Union[pl.DataFrame, pd.DataFrame, Dict[str, Any], str, os.PathLike],
    ):
        """
        Fit the DataProcessor to the provided data.

        Args:
            data (Union[pl.DataFrame, pd.DataFrame, Dict[str, Any], str, os.PathLike]): Input data for fitting.

        Returns:
            DataProcessor: Fitted DataProcessor instance.
        """

        for config in self.sparse_features.values():
            config.pop("_min_freq_logged", None)
        for config in self.sequence_features.values():
            config.pop("_min_freq_logged", None)
        if isinstance(data, (str, os.PathLike)):
            return self.fit_from_path(str(data))
        if isinstance(data, dict):
            df = pl.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = pl.from_pandas(data)
        else:
            df = data
        lazy_frame = df.lazy()
        schema = df.schema
        return self.fit_from_lazy(lazy_frame, schema)

    @overload
    def transform(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, Dict[str, Any]],
        return_dict: Literal[True] = True,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]: ...

    @overload
    def transform(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, Dict[str, Any]],
        return_dict: Literal[False] = False,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pl.DataFrame: ...

    @overload
    def transform(
        self,
        data: str | os.PathLike,
        return_dict: Literal[False] = False,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> list[str]: ...

    def transform(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, Dict[str, Any], str, os.PathLike],
        return_dict: bool = True,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        """
        Transform the provided data using the fitted DataProcessor.

        Args:
            data (Union[pl.DataFrame, pd.DataFrame, Dict[str, Any], str, os.PathLike]): Input data to transform.
            return_dict (bool): Whether to return a dictionary of numpy arrays.
            save_format (Optional[str]): Format to save the data if output_path is provided.
            output_path (Optional[str]): Output path to save the transformed data.
        Returns:
            Union[pl.DataFrame, Dict[str, np.ndarray], List[str]]: Transformed data or list of saved file paths.
        """

        if not self.is_fitted:
            raise ValueError("[Data Processor Error] DataProcessor must be fitted before transform")
        if isinstance(data, (str, os.PathLike)):
            if return_dict:
                raise ValueError(
                    "[Data Processor Error] Path transform writes files only; set return_dict=False when passing a path."
                )
            return self.transform_path(str(data), output_path, save_format)
        return self.transform_in_memory(
            data=data,
            return_dict=return_dict,
            persist=output_path is not None,
            save_format=save_format,
            output_path=output_path,
        )

    @overload
    def fit_transform(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, Dict[str, Any]],
        return_dict: Literal[True] = True,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]: ...

    @overload
    def fit_transform(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, Dict[str, Any]],
        return_dict: Literal[False] = False,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pl.DataFrame: ...

    @overload
    def fit_transform(
        self,
        data: str | os.PathLike,
        return_dict: Literal[False] = False,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> list[str]: ...

    def fit_transform(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, Dict[str, Any], str, os.PathLike],
        return_dict: bool = True,
        save_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        """
        Fit the DataProcessor to the provided data and then transform it.

        Args:
            data (Union[pl.DataFrame, pd.DataFrame, Dict[str, Any], str, os.PathLike]): Input data for fitting and transforming.
            return_dict (bool): Whether to return a dictionary of numpy arrays.
            save_format (Optional[str]): Format to save the data if output_path is provided.
            output_path (Optional[str]): Output path to save the data.
        Returns:
            Union[pl.DataFrame, Dict[str, np.ndarray], List[str]]: Transformed data or list of saved file paths.
        """

        self.fit(data)
        if isinstance(data, (str, os.PathLike)):
            if return_dict:
                raise ValueError(
                    "[Data Processor Error] Path transform writes files only; set return_dict=False when passing a path."
                )
            return self.transform_path(str(data), output_path, save_format)
        return self.transform_in_memory(
            data=data,
            return_dict=return_dict,
            persist=output_path is not None,
            save_format=save_format,
            output_path=output_path,
        )

    def save(self, save_path: str | Path):
        """
        Save the fitted DataProcessor to a file.

        Args:
            save_path (str | Path): Path to save the DataProcessor.
        """
        logger = logging.getLogger()
        assert isinstance(save_path, (str, Path)), "save_path must be a string or Path"
        save_path = Path(save_path)
        if not self.is_fitted:
            logger.warning("Saving unfitted DataProcessor")
        target_path = get_save_path(
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
        logger.info(f"DataProcessor saved to: {target_path}, NextRec version: {self.version}")

    @classmethod
    def load(cls, load_path: str | Path) -> "DataProcessor":
        """
        Load a fitted DataProcessor from a file.

        Args:
            load_path (str | Path): Path to load the DataProcessor from.

        Returns:
            DataProcessor: Loaded DataProcessor instance.
        """

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

        logger.info("")
        logger.info(
            colorize(
                f"DataProcessor loaded from {load_path}, NextRec version: {processor.version}",
                color="green",
            )
        )
        return processor

    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for all sparse and sequence features.

        Returns:
            Dict[str, int]: Mapping of feature names to vocabulary sizes.
        """
        vocab_sizes = {}
        for name, config in self.sparse_features.items():
            output_name = self._config_output_name(name, config)
            vocab_size = config.get("vocab_size", 0)
            vocab_sizes[output_name] = vocab_size
        for name, config in self.sequence_features.items():
            output_name = self._config_output_name(name, config)
            vocab_size = config.get("vocab_size", 0)
            vocab_sizes[output_name] = vocab_size
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

            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Scaler':>15} {'Fill NA':>10}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*15} {'-'*10}")
            for i, (name, config) in enumerate(self.numeric_features.items(), 1):
                scaler = config["scaler"]
                fill_na = config.get("fill_na_value", config.get("fill_na", "N/A"))
                logger.info(f"  {i:<4} {name:<{name_width}} {str(scaler):>15} {str(fill_na):>10}")

        if self.sparse_features:
            logger.info(f"Sparse Features ({len(self.sparse_features)}):")

            max_name_len = max(len(name) for name in self.sparse_features.keys())
            name_width = max(max_name_len, 10) + 2

            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Method':>12} {'Vocab Size':>12} {'Hash Size':>12}")
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
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
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
