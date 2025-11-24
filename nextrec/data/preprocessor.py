"""
DataProcessor for data preprocessing including numeric, sparse, sequence features and target processing.

Date: create on 13/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""
from __future__ import annotations
import os
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Union, Optional, Literal, Any
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    MaxAbsScaler,
    LabelEncoder
)

from nextrec.basic.loggers import setup_logger, colorize
from nextrec.data.data_utils import (
    resolve_file_paths,
    iter_file_chunks,
    read_table,
    load_dataframes,
    default_output_dir,
)
from nextrec.basic.session import create_session, resolve_save_path
from nextrec.basic.features import FeatureSpecMixin

class DataProcessor(FeatureSpecMixin):
    """DataProcessor for data preprocessing including numeric, sparse, sequence features and target processing.
    
    Examples:
        >>> processor = DataProcessor()
        >>> processor.add_numeric_feature('age', scaler='standard')
        >>> processor.add_sparse_feature('user_id', encode_method='hash', hash_size=10000)
        >>> processor.add_sequence_feature('item_history', encode_method='label', max_len=50, pad_value=0)
        >>> processor.add_target('label', target_type='binary')
        >>> 
        >>> # Fit and transform data
        >>> processor.fit(train_df)
        >>> processed_data = processor.transform(test_df)  # Returns dict of numpy arrays
        >>> 
        >>> # Save and load processor
        >>> processor.save('processor.pkl')
        >>> loaded_processor = DataProcessor.load('processor.pkl')
        >>> 
        >>> # Get vocabulary sizes for embedding layers
        >>> vocab_sizes = processor.get_vocab_sizes()
    """
    def __init__(self, session_id: str | None = None ):
        self.numeric_features: Dict[str, Dict[str, Any]] = {}
        self.sparse_features: Dict[str, Dict[str, Any]] = {}
        self.sequence_features: Dict[str, Dict[str, Any]] = {}
        self.target_features: Dict[str, Dict[str, Any]] = {}
        self.session_id = session_id
        self.session = create_session(session_id)
    
        self.is_fitted = False
        self._transform_summary_printed = False  # Track if summary has been printed during transform
        
        self.scalers: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoders: Dict[str, Dict[str, int]] = {}
        self._set_target_config([], [])
        
        # Initialize logger if not already initialized
        self._logger_initialized = False
        if not logging.getLogger().hasHandlers():
            setup_logger(session_id=self.session_id)
            self._logger_initialized = True
        
    def add_numeric_feature(
        self, 
        name: str, 
        scaler: Optional[Literal['standard', 'minmax', 'robust', 'maxabs', 'log', 'none']] = 'standard',
        fill_na: Optional[float] = None
    ):
        self.numeric_features[name] = {
            'scaler': scaler,
            'fill_na': fill_na
        }
        
    def add_sparse_feature(
        self, 
        name: str, 
        encode_method: Literal['hash', 'label'] = 'label',
        hash_size: Optional[int] = None,
        fill_na: str = '<UNK>'
    ):
        if encode_method == 'hash' and hash_size is None:
            raise ValueError("hash_size must be specified when encode_method='hash'")
            
        self.sparse_features[name] = {
            'encode_method': encode_method,
            'hash_size': hash_size,
            'fill_na': fill_na
        }
        
    def add_sequence_feature(
        self, 
        name: str,
        encode_method: Literal['hash', 'label'] = 'label',
        hash_size: Optional[int] = None,
        max_len: Optional[int] = 50,
        pad_value: int = 0,
        truncate: Literal['pre', 'post'] = 'pre',           # pre: keep last max_len items, post: keep first max_len items
        separator: str = ','
    ):

        if encode_method == 'hash' and hash_size is None:
            raise ValueError("hash_size must be specified when encode_method='hash'")
            
        self.sequence_features[name] = {
            'encode_method': encode_method,
            'hash_size': hash_size,
            'max_len': max_len,
            'pad_value': pad_value,
            'truncate': truncate,
            'separator': separator
        }
        
    def add_target(
        self, 
        name: str,                                                                # example: 'click'
        target_type: Literal['binary', 'multiclass', 'regression'] = 'binary',
        label_map: Optional[Dict[str, int]] = None                                # example: {'click': 1, 'no_click': 0}
    ):
        self.target_features[name] = {
            'target_type': target_type,
            'label_map': label_map
        }
        self._set_target_config(list(self.target_features.keys()), [])
        
    def _hash_string(self, s: str, hash_size: int) -> int:
        return int(hashlib.md5(str(s).encode()).hexdigest(), 16) % hash_size
        
    def _process_numeric_feature_fit(self, data: pd.Series, config: Dict[str, Any]):

        name = str(data.name)
        scaler_type = config['scaler']
        fill_na = config['fill_na']
        
        if data.isna().any():
            if fill_na is None:
                # Default use mean value to fill missing values for numeric features
                fill_na = data.mean()
            config['fill_na_value'] = fill_na
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'maxabs':
            scaler = MaxAbsScaler()
        elif scaler_type == 'log':
            scaler = None  
        elif scaler_type == 'none':
            scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        if scaler is not None and scaler_type != 'log':
            filled_data = data.fillna(config.get('fill_na_value', 0))
            values = np.array(filled_data.values, dtype=np.float64).reshape(-1, 1)
            scaler.fit(values)
            self.scalers[name] = scaler
            
    def _process_numeric_feature_transform(
        self, 
        data: pd.Series, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        logger = logging.getLogger()
        
        name = str(data.name)
        scaler_type = config['scaler']
        fill_na_value = config.get('fill_na_value', 0)

        filled_data = data.fillna(fill_na_value)
        values = np.array(filled_data.values, dtype=np.float64)

        if scaler_type == 'log':
            result = np.log1p(np.maximum(values, 0))
        elif scaler_type == 'none':
            result = values
        else:
            scaler = self.scalers.get(name)
            if scaler is None:
                logger.warning(f"Scaler for {name} not fitted, returning original values")
                result = values
            else:
                result = scaler.transform(values.reshape(-1, 1)).ravel()
        
        return result
        
    def _process_sparse_feature_fit(self, data: pd.Series, config: Dict[str, Any]):

        name = str(data.name)
        encode_method = config['encode_method']
        fill_na = config['fill_na'] # <UNK>
        
        filled_data = data.fillna(fill_na).astype(str)
        
        if encode_method == 'label':
            le = LabelEncoder()
            le.fit(filled_data)
            self.label_encoders[name] = le
            config['vocab_size'] = len(le.classes_)
        elif encode_method == 'hash':
            config['vocab_size'] = config['hash_size']
            
    def _process_sparse_feature_transform(
        self, 
        data: pd.Series, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Fast path sparse feature transform using cached dict mapping or hashing."""
        name = str(data.name)
        encode_method = config['encode_method']
        fill_na = config['fill_na']
        
        sparse_series = pd.Series(data, name=name).fillna(fill_na).astype(str)

        if encode_method == 'label':
            le = self.label_encoders.get(name)
            if le is None:
                raise ValueError(f"LabelEncoder for {name} not fitted")

            class_to_idx = config.get('_class_to_idx')
            if class_to_idx is None:
                class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
                config['_class_to_idx'] = class_to_idx

            encoded = sparse_series.map(class_to_idx)
            encoded = encoded.fillna(0).astype(np.int64)
            return encoded.to_numpy()
        
        if encode_method == 'hash':
            hash_size = config['hash_size']
            hash_fn = self._hash_string
            return np.fromiter(
                (hash_fn(v, hash_size) for v in sparse_series.to_numpy()),
                dtype=np.int64,
                count=sparse_series.size,
            )
        
        return np.array([], dtype=np.int64)
            
    def _process_sequence_feature_fit(self, data: pd.Series, config: Dict[str, Any]):

        name = str(data.name)
        encode_method = config['encode_method']
        separator = config['separator']
        
        if encode_method == 'label':
            all_tokens = set()
            for seq in data:
                # Skip None, np.nan, and empty strings
                if seq is None:
                    continue
                if isinstance(seq, (float, np.floating)) and np.isnan(seq):
                    continue
                if isinstance(seq, str) and seq.strip() == '':
                    continue
                
                if isinstance(seq, str):
                    tokens = seq.split(separator)
                elif isinstance(seq, (list, tuple)):
                    tokens = [str(t) for t in seq]
                elif isinstance(seq, np.ndarray):
                    tokens = [str(t) for t in seq.tolist()]
                else:
                    continue
                
                all_tokens.update(tokens)
            
            if len(all_tokens) == 0:
                all_tokens.add('<PAD>')
            
            le = LabelEncoder()
            le.fit(list(all_tokens))
            self.label_encoders[name] = le
            config['vocab_size'] = len(le.classes_)
        elif encode_method == 'hash':
            config['vocab_size'] = config['hash_size']
            
    def _process_sequence_feature_transform(
        self, 
        data: pd.Series, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Optimized sequence transform with preallocation and cached vocab map."""
        name = str(data.name)
        encode_method = config['encode_method']
        max_len = config['max_len']
        pad_value = config['pad_value']
        truncate = config['truncate']
        separator = config['separator']

        arr = np.asarray(data, dtype=object)
        n = arr.shape[0]
        output = np.full((n, max_len), pad_value, dtype=np.int64)

        # Shared helpers cached locally for speed and cross-platform consistency
        split_fn = str.split
        is_nan = np.isnan

        if encode_method == 'label':
            le = self.label_encoders.get(name)
            if le is None:
                raise ValueError(f"LabelEncoder for {name} not fitted")
            class_to_idx = config.get('_class_to_idx')
            if class_to_idx is None:
                class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
                config['_class_to_idx'] = class_to_idx
        else:
            class_to_idx = None  # type: ignore

        hash_fn = self._hash_string
        hash_size = config.get('hash_size')

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

            if encode_method == 'label':
                encoded = [
                    class_to_idx.get(token.strip(), 0)  # type: ignore[union-attr]
                    for token in tokens
                    if token is not None and token != ''
                ]
         
            elif encode_method == 'hash':
                if hash_size is None:
                    raise ValueError("hash_size must be set for hash encoding")
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
                encoded = encoded[-max_len:] if truncate == 'pre' else encoded[:max_len]

            output[i, : len(encoded)] = encoded

        return output
        
    def _process_target_fit(self, data: pd.Series, config: Dict[str, Any]):
        name = str(data.name)
        target_type = config['target_type']
        label_map = config['label_map']
        
        if target_type in ['binary', 'multiclass']:
            if label_map is None:
                unique_values = data.dropna().unique()
                sorted_values = sorted(unique_values)
                
                try:
                    int_values = [int(v) for v in sorted_values]
                    if int_values == list(range(len(int_values))):
                        label_map = {str(val): int(val) for val in sorted_values}
                    else:
                        label_map = {str(val): idx for idx, val in enumerate(sorted_values)}
                except (ValueError, TypeError):
                    label_map = {str(val): idx for idx, val in enumerate(sorted_values)}
                
                config['label_map'] = label_map    
            
            self.target_encoders[name] = label_map
            
    def _process_target_transform(
        self, 
        data: pd.Series, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        logger = logging.getLogger()
        
        name = str(data.name)
        target_type = config['target_type']
        
        if target_type == 'regression':
            values = np.array(data.values, dtype=np.float32)
            return values
        else:
            label_map = self.target_encoders.get(name)
            if label_map is None:
                raise ValueError(f"Target encoder for {name} not fitted")
            
            result = []
            for val in data:
                str_val = str(val)
                if str_val in label_map:
                    result.append(label_map[str_val])
                else:
                    logger.warning(f"Unknown target value: {val}, mapping to 0")
                    result.append(0)
            
            return np.array(result, dtype=np.int64 if target_type == 'multiclass' else np.float32)
    
    def _load_dataframe_from_path(self, path: str) -> pd.DataFrame:
        """Load all data from a file or directory path into a single DataFrame."""
        file_paths, file_type = resolve_file_paths(path)
        frames = load_dataframes(file_paths, file_type)
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def _extract_sequence_tokens(self, value: Any, separator: str) -> list[str]:
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

    def _fit_from_path(self, path: str, chunk_size: int) -> 'DataProcessor':
        """Fit processor statistics by streaming files to reduce memory usage."""
        logger = logging.getLogger()
        logger.info(colorize("Fitting DataProcessor (streaming path mode)...", color="cyan", bold=True))
        file_paths, file_type = resolve_file_paths(path)

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

        sparse_vocab: Dict[str, set[str]] = {name: set() for name in self.sparse_features.keys()}
        seq_vocab: Dict[str, set[str]] = {name: set() for name in self.sequence_features.keys()}
        target_values: Dict[str, set[Any]] = {name: set() for name in self.target_features.keys()}

        missing_features = set()

        for file_path in file_paths:
            for chunk in iter_file_chunks(file_path, file_type, chunk_size):
                # numeric features
                for name, config in self.numeric_features.items():
                    if name not in chunk.columns:
                        missing_features.add(name)
                        continue
                    series = chunk[name]
                    values = pd.to_numeric(series, errors="coerce")
                    values = values.dropna()
                    if values.empty:
                        continue
                    acc = numeric_acc[name]
                    arr = values.to_numpy(dtype=np.float64, copy=False)
                    acc["count"] += arr.size
                    acc["sum"] += float(arr.sum())
                    acc["sumsq"] += float(np.square(arr).sum())
                    acc["min"] = min(acc["min"], float(arr.min()))
                    acc["max"] = max(acc["max"], float(arr.max()))
                    acc["max_abs"] = max(acc["max_abs"], float(np.abs(arr).max()))

                # sparse features
                for name, config in self.sparse_features.items():
                    if name not in chunk.columns:
                        missing_features.add(name)
                        continue
                    fill_na = config["fill_na"]
                    series = chunk[name].fillna(fill_na).astype(str)
                    sparse_vocab[name].update(series.tolist())

                # sequence features
                for name, config in self.sequence_features.items():
                    if name not in chunk.columns:
                        missing_features.add(name)
                        continue
                    separator = config["separator"]
                    series = chunk[name]
                    tokens = []
                    for val in series:
                        tokens.extend(self._extract_sequence_tokens(val, separator))
                    seq_vocab[name].update(tokens)

                # target features
                for name in self.target_features.keys():
                    if name not in chunk.columns:
                        missing_features.add(name)
                        continue
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
                logger.warning(f"Numeric feature {name} has no valid values in provided files")
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
                scaler.scale_ = np.array([np.sqrt(var) if var > 0 else 1.0], dtype=np.float64)
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
                le = LabelEncoder()
                le.fit(list(vocab))
                self.label_encoders[name] = le
                config["vocab_size"] = len(le.classes_)
            elif config["encode_method"] == "hash":
                config["vocab_size"] = config["hash_size"]

        # finalize sequence vocabularies
        for name, config in self.sequence_features.items():
            if config["encode_method"] == "label":
                vocab = seq_vocab[name] or {"<PAD>"}
                le = LabelEncoder()
                le.fit(list(vocab))
                self.label_encoders[name] = le
                config["vocab_size"] = len(le.classes_)
            elif config["encode_method"] == "hash":
                config["vocab_size"] = config["hash_size"]

        # finalize targets
        for name, config in self.target_features.items():
            if not target_values[name]:
                logger.warning(f"Target {name} has no valid values in provided files")
                continue

            target_type = config["target_type"]
            if target_type in ["binary", "multiclass"]:
                unique_values = list(target_values[name])
                try:
                    sorted_values = sorted(unique_values)
                except TypeError:
                    sorted_values = sorted(unique_values, key=lambda x: str(x))

                label_map = config["label_map"]
                if label_map is None:
                    try:
                        int_values = [int(v) for v in sorted_values]
                        if int_values == list(range(len(int_values))):
                            label_map = {str(val): int(val) for val in sorted_values}
                        else:
                            label_map = {str(val): idx for idx, val in enumerate(sorted_values)}
                    except (ValueError, TypeError):
                        label_map = {str(val): idx for idx, val in enumerate(sorted_values)}
                    config["label_map"] = label_map

                self.target_encoders[name] = label_map

        self.is_fitted = True
        logger.info(colorize("DataProcessor fitted successfully (streaming path mode)", color="green", bold=True))
        return self

    def _transform_in_memory(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_dict: bool,
        persist: bool,
        save_format: Optional[Literal["csv", "parquet"]],
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        logger = logging.getLogger()

        # Convert input to dict format for unified processing
        if isinstance(data, pd.DataFrame):
            data_dict = {col: data[col] for col in data.columns}
        elif isinstance(data, dict):
            data_dict = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        result_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, pd.Series):
                result_dict[key] = value.values
            elif isinstance(value, np.ndarray):
                result_dict[key] = value
            else:
                result_dict[key] = np.array(value)

        # process numeric features
        for name, config in self.numeric_features.items():
            if name not in data_dict:
                logger.warning(f"Numeric feature {name} not found in data")
                continue
            # Convert to Series for processing
            series_data = pd.Series(data_dict[name], name=name)
            processed = self._process_numeric_feature_transform(series_data, config)
            result_dict[name] = processed

        # process sparse features
        for name, config in self.sparse_features.items():
            if name not in data_dict:
                logger.warning(f"Sparse feature {name} not found in data")
                continue
            series_data = pd.Series(data_dict[name], name=name)
            processed = self._process_sparse_feature_transform(series_data, config)
            result_dict[name] = processed

        # process sequence features
        for name, config in self.sequence_features.items():
            if name not in data_dict:
                logger.warning(f"Sequence feature {name} not found in data")
                continue
            series_data = pd.Series(data_dict[name], name=name)
            processed = self._process_sequence_feature_transform(series_data, config)
            result_dict[name] = processed

        # process target features
        for name, config in self.target_features.items():
            if name not in data_dict:
                logger.warning(f"Target {name} not found in data")
                continue
            series_data = pd.Series(data_dict[name], name=name)
            processed = self._process_target_transform(series_data, config)
            result_dict[name] = processed

        def _dict_to_dataframe(result: Dict[str, np.ndarray]) -> pd.DataFrame:
            # Convert all arrays to Series/lists at once to avoid fragmentation
            columns_dict = {}
            for key, value in result.items():
                if key in self.sequence_features:
                    columns_dict[key] = [list(seq) for seq in value]
                else:
                    columns_dict[key] = value
            return pd.DataFrame(columns_dict)

        assert save_format in [None, "csv", "parquet"], "save_format must be either 'csv', 'parquet', or None"
        if persist and save_format is None:
            save_format = "parquet"

        result_df = None
        if (not return_dict) or (save_format is not None):
            result_df = _dict_to_dataframe(result_dict)
            assert result_df is not None, "DataFrame is None after transform"

        if save_format is not None:
            save_path = resolve_save_path(
                path=None,
                default_dir=self.session_dir / "processor" / "preprocessed_data",
                default_name="data_processed",
                suffix=f".{save_format}",
                add_timestamp=True,
            )

            if save_format == "parquet":
                result_df.to_parquet(save_path, index=False)
            else:
                result_df.to_csv(save_path, index=False)

            logger.info(colorize(
                f"Transformed data saved to: {save_path}",
                color="green"
            ))

        if return_dict:
            return result_dict
        return result_df

    def _transform_path(self, path: str, output_path: Optional[str]) -> list[str]:
        """Transform data from files under a path and save them to a new location."""
        logger = logging.getLogger()

        file_paths, file_type = resolve_file_paths(path)
        default_root = self.session_dir / "processor" / default_output_dir(path).name
        output_root = default_root
        target_file_override: Optional[Path] = None

        if output_path:
            output_path_obj = Path(output_path)
            if not output_path_obj.is_absolute():
                output_path_obj = self.session_dir / output_path_obj
            if output_path_obj.suffix.lower() in {".csv", ".parquet"}:
                if len(file_paths) != 1:
                    raise ValueError("output_path points to a file but multiple input files were provided.")
                target_file_override = output_path_obj
                output_root = output_path_obj.parent
            else:
                output_root = output_path_obj

        output_root.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        for file_path in file_paths:
            df = read_table(file_path, file_type)

            transformed_df = self._transform_in_memory(
                df,
                return_dict=False,
                persist=False,
                save_format=None,
            )
            assert isinstance(transformed_df, pd.DataFrame), "Expected DataFrame when return_dict=False"

            source_path = Path(file_path)
            target_file = (
                target_file_override
                if target_file_override is not None
                else output_root / f"{source_path.stem}_preprocessed{source_path.suffix}"
            )

            if file_type == "csv":
                transformed_df.to_csv(target_file, index=False)
            else:
                transformed_df.to_parquet(target_file, index=False)

            saved_paths.append(str(target_file.resolve()))

        logger.info(colorize(
            f"Transformed {len(saved_paths)} file(s) saved to: {output_root.resolve()}",
            color="green",
        ))
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
            uses_robust = any(cfg.get("scaler") == "robust" for cfg in self.numeric_features.values())
            if uses_robust:
                logger.warning(
                    "Robust scaler requires full data; loading all files into memory. "
                    "Consider smaller chunk_size or different scaler if memory is limited."
                )
                data = self._load_dataframe_from_path(path_str)
            else:
                return self._fit_from_path(path_str, chunk_size)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        logger.info(colorize("Fitting DataProcessor...", color="cyan", bold=True))

        for name, config in self.numeric_features.items():
            if name not in data.columns:
                logger.warning(f"Numeric feature {name} not found in data")
                continue
            self._process_numeric_feature_fit(data[name], config)
        
        for name, config in self.sparse_features.items():
            if name not in data.columns:
                logger.warning(f"Sparse feature {name} not found in data")
                continue
            self._process_sparse_feature_fit(data[name], config)
        
        for name, config in self.sequence_features.items():
            if name not in data.columns:
                logger.warning(f"Sequence feature {name} not found in data")
                continue
            self._process_sequence_feature_fit(data[name], config)

        for name, config in self.target_features.items():
            if name not in data.columns:
                logger.warning(f"Target {name} not found in data")
                continue
            self._process_target_fit(data[name], config)
        
        self.is_fitted = True
        logger.info(colorize("DataProcessor fitted successfully", color="green", bold=True))
        return self
        
    def transform(
        self, 
        data: Union[pd.DataFrame, Dict[str, Any], str, os.PathLike],
        return_dict: bool = True,
        persist: bool = False,
        save_format: Optional[Literal["csv", "parquet"]] = None,
        output_path: Optional[str] = None,
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray], list[str]]:
        logger = logging.getLogger()

        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before transform")

        if isinstance(data, (str, os.PathLike)):
            if return_dict or persist or save_format is not None:
                raise ValueError("Path transform writes files only; use output_path and leave return_dict/persist/save_format defaults.")
            return self._transform_path(str(data), output_path)
        
        return self._transform_in_memory(
            data=data,
            return_dict=return_dict,
            persist=persist,
            save_format=save_format,
        )
            
    def fit_transform(
        self, 
        data: Union[pd.DataFrame, Dict[str, Any], str, os.PathLike],
        return_dict: bool = True,
        save_format: Optional[Literal["csv", "parquet"]] = None,
        output_path: Optional[str] = None,
        chunk_size: int = 200000,
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray], list[str]]:
        self.fit(data, chunk_size=chunk_size)
        return self.transform(
            data,
            return_dict=return_dict,
            save_format=save_format,
            output_path=output_path,
        )
        
    def save(self, save_path: str):
        logger = logging.getLogger()

        if not self.is_fitted:
            logger.warning("Saving unfitted DataProcessor")

        target_path = resolve_save_path(
            path=save_path,
            default_dir=self.session.processor_dir,
            default_name="processor",
            suffix=".pkl",
        )

        # Prepare state dict
        state = {
            "numeric_features": self.numeric_features,
            "sparse_features": self.sparse_features,
            "sequence_features": self.sequence_features,
            "target_features": self.target_features,
            "is_fitted": self.is_fitted,
            "scalers": self.scalers,
            "label_encoders": self.label_encoders,
            "target_encoders": self.target_encoders,
        }

        # Save with pickle
        with open(target_path, "wb") as f:
            pickle.dump(state, f)

        logger.info(colorize(f"DataProcessor saved to: {target_path}", color="green"))
        
    @classmethod
    def load(cls, load_path: str) -> 'DataProcessor':
        logger = logging.getLogger()
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        processor = cls()
        processor.numeric_features = state['numeric_features']
        processor.sparse_features = state['sparse_features']
        processor.sequence_features = state['sequence_features']
        processor.target_features = state['target_features']
        processor.is_fitted = state['is_fitted']
        processor.scalers = state['scalers']
        processor.label_encoders = state['label_encoders']
        processor.target_encoders = state['target_encoders']
        
        logger.info(f"DataProcessor loaded from {load_path}")
        return processor
        
    def get_vocab_sizes(self) -> Dict[str, int]:
        vocab_sizes = {}
        
        for name, config in self.sparse_features.items():
            vocab_sizes[name] = config.get('vocab_size', 0)
        
        for name, config in self.sequence_features.items():
            vocab_sizes[name] = config.get('vocab_size', 0)
        
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
                scaler = config['scaler']
                fill_na = config.get('fill_na_value', config.get('fill_na', 'N/A'))
                logger.info(f"  {i:<4} {name:<{name_width}} {str(scaler):>15} {str(fill_na):>10}")
        
        if self.sparse_features:
            logger.info(f"Sparse Features ({len(self.sparse_features)}):")
            
            max_name_len = max(len(name) for name in self.sparse_features.keys())
            name_width = max(max_name_len, 10) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Method':>12} {'Vocab Size':>12} {'Hash Size':>12}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*12} {'-'*12}")
            for i, (name, config) in enumerate(self.sparse_features.items(), 1):
                method = config['encode_method']
                vocab_size = config.get('vocab_size', 'N/A')
                hash_size = config.get('hash_size', 'N/A')
                logger.info(f"  {i:<4} {name:<{name_width}} {str(method):>12} {str(vocab_size):>12} {str(hash_size):>12}")
        
        if self.sequence_features:
            logger.info(f"Sequence Features ({len(self.sequence_features)}):")
            
            max_name_len = max(len(name) for name in self.sequence_features.keys())
            name_width = max(max_name_len, 10) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Method':>12} {'Vocab Size':>12} {'Hash Size':>12} {'Max Len':>10}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
            for i, (name, config) in enumerate(self.sequence_features.items(), 1):
                method = config['encode_method']
                vocab_size = config.get('vocab_size', 'N/A')
                hash_size = config.get('hash_size', 'N/A')
                max_len = config.get('max_len', 'N/A')
                logger.info(f"  {i:<4} {name:<{name_width}} {str(method):>12} {str(vocab_size):>12} {str(hash_size):>12} {str(max_len):>10}")
        
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
                target_type = config['target_type']
                logger.info(f"  {i:<4} {name:<{name_width}} {str(target_type):>15}")
        else:
            logger.info("No target features configured")
        
        logger.info("")
        logger.info(colorize("[3] Processor Status", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        logger.info(f"Fitted:                  {self.is_fitted}")
        logger.info(f"Total Features:          {len(self.numeric_features) + len(self.sparse_features) + len(self.sequence_features)}")
        logger.info(f"  Dense Features:        {len(self.numeric_features)}")
        logger.info(f"  Sparse Features:       {len(self.sparse_features)}")
        logger.info(f"  Sequence Features:     {len(self.sequence_features)}")
        logger.info(f"Target Features:         {len(self.target_features)}")
        
        logger.info("")
        logger.info("")
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
