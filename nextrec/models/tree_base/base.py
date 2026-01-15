"""
Tree-based model base for NextRec.

This module provides a lightweight adapter to plug tree models (xgboost/lightgbm/catboost)
into the NextRec training/prediction pipeline.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Literal, overload

import numpy as np
import pandas as pd

from nextrec.basic.features import (
    DenseFeature,
    FeatureSet,
    SequenceFeature,
    SparseFeature,
)
from nextrec.basic.loggers import colorize, format_kv, setup_logger
from nextrec.basic.metrics import check_user_id, configure_metrics, evaluate_metrics
from nextrec.basic.session import create_session, get_save_path
from nextrec.data.dataloader import RecDataLoader
from nextrec.data.data_processing import get_column_data
from nextrec.utils.console import display_metrics_table
from nextrec.utils.data import FILE_FORMAT_CONFIG, check_streaming_support
from nextrec.utils.feature import to_list
from nextrec.utils.torch_utils import to_numpy


class TreeBaseModel(FeatureSet):
    model_file_suffix = "bin"

    @property
    def model_name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def default_task(self) -> str:
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: list[str] | str | None = None,
        id_columns: list[str] | str | None = None,
        task: str | list[str] | None = None,
        device: str = "cpu",
        session_id: str | None = None,
        model_params: dict[str, Any] | None = None,
        sequence_pooling: str = "mean",
        **kwargs: Any,
    ):
        self.device = device
        self.model_params = dict(model_params or {})  # tree model parameters
        if kwargs:
            self.model_params.update(kwargs)
        self.sequence_pooling = sequence_pooling

        self.set_all_features(
            dense_features, sparse_features, sequence_features, target, id_columns
        )
        self.task = task or self.default_task

        self.session_id = session_id
        self.session = create_session(session_id)
        self.session_path = self.session.root
        self.checkpoint_path = os.path.join(
            self.session_path,
            f"{self.model_name.upper()}_checkpoint.{self.model_file_suffix}",
        )
        self.best_path = os.path.join(
            self.session_path,
            f"{self.model_name.upper()}_best.{self.model_file_suffix}",
        )
        self.features_config_path = os.path.join(
            self.session_path, "features_config.pkl"
        )

        self.model: Any | None = None
        self._cat_feature_indices: list[int] = []

    def assert_task(self) -> None:
        if self.target_columns and len(self.target_columns) > 1:
            raise ValueError(
                f"[{self.model_name}-init Error] tree models only support a single target column."
            )
        if isinstance(self.task, list) and len(self.task) > 1:
            raise ValueError(
                f"[{self.model_name}-init Error] tree models only support a single task type."
            )

    def pool_sequence(self, arr: np.ndarray, feature: SequenceFeature) -> np.ndarray:
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        padding_value = feature.padding_idx
        mask = arr != padding_value
        if self.sequence_pooling == "sum":
            pooled = (arr * mask).sum(axis=1)
        elif self.sequence_pooling == "max":
            masked = np.where(mask, arr, -np.inf)
            pooled = np.max(masked, axis=1)
            pooled = np.where(np.isfinite(pooled), pooled, 0.0)
        elif self.sequence_pooling == "last":
            idx = np.where(mask, np.arange(arr.shape[1]), -1)
            last_idx = idx.max(axis=1)
            pooled = np.array(
                [arr[row, col] if col >= 0 else 0.0 for row, col in enumerate(last_idx)]
            )
        else:
            counts = np.maximum(mask.sum(axis=1), 1)
            pooled = (arr * mask).sum(axis=1) / counts
        return pooled.reshape(-1, 1).astype(np.float32)

    def features_to_matrix(self, features: dict[str, Any]) -> np.ndarray:
        columns: list[np.ndarray] = []
        cat_indices: list[int] = []
        feature_offset = 0
        for feat in self.all_features:
            if feat.name not in features:
                raise KeyError(
                    f"[{self.model_name}-data Error] Missing feature '{feat.name}'."
                )
            arr = to_numpy(features[feat.name])
            if isinstance(feat, SequenceFeature):
                arr = self.pool_sequence(arr, feat)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if isinstance(feat, SparseFeature):
                for col_idx in range(arr.shape[1]):
                    cat_indices.append(feature_offset + col_idx)
            feature_offset += arr.shape[1]
            columns.append(arr.astype(np.float32))
        if columns:
            self._cat_feature_indices = cat_indices
            return np.concatenate(columns, axis=1)
        return np.empty((0, 0), dtype=np.float32)

    def extract_labels(self, labels: dict[str, Any] | None) -> np.ndarray | None:
        if labels is None:
            return None
        if self.target_columns:
            target = self.target_columns[0]
            if target not in labels:
                return None
            return to_numpy(labels[target]).reshape(-1)
        first_key = next(iter(labels.keys()), None)
        if first_key is None:
            return None
        return to_numpy(labels[first_key]).reshape(-1)

    def extract_ids(
        self, ids: dict[str, Any] | None, id_column: str | None
    ) -> np.ndarray | None:
        if ids is None or id_column is None:
            return None
        if id_column not in ids:
            return None
        return np.asarray(ids[id_column]).reshape(-1)

    def collect_from_dataloader(
        self,
        data_loader: Iterable,
        require_labels: bool,
        include_ids: bool,
        id_column: str | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        feature_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        id_chunks: list[np.ndarray] = []
        for batch in data_loader:
            if not isinstance(batch, dict) or "features" not in batch:
                raise TypeError(
                    f"[{self.model_name}-data Error] Expected batches with a 'features' dict."
                )
            features = batch.get("features", {})
            labels = batch.get("labels")
            ids = batch.get("ids")
            X_batch = self.features_to_matrix(features)
            feature_chunks.append(X_batch)
            y_batch = self.extract_labels(labels)
            if require_labels and y_batch is None:
                raise ValueError(
                    f"[{self.model_name}-data Error] Labels are required but missing."
                )
            if y_batch is not None:
                label_chunks.append(y_batch)
            if include_ids and id_column:
                id_batch = self.extract_ids(ids, id_column)
                if id_batch is not None:
                    id_chunks.append(id_batch)
        X_all = (
            np.concatenate(feature_chunks, axis=0)
            if feature_chunks
            else np.empty((0, 0))
        )
        y_all = np.concatenate(label_chunks, axis=0) if label_chunks else None
        ids_all = np.concatenate(id_chunks, axis=0) if id_chunks else None
        return X_all, y_all, ids_all

    def collect_from_table(
        self,
        data: dict | pd.DataFrame,
        require_labels: bool,
        include_ids: bool,
        id_column: str | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        features: dict[str, Any] = {}
        for feat in self.all_features:
            column = get_column_data(data, feat.name)
            if column is None:
                raise KeyError(
                    f"[{self.model_name}-data Error] Missing feature '{feat.name}'."
                )
            features[feat.name] = column
        X_all = self.features_to_matrix(features)
        y_all = None
        if require_labels:
            label_payload: dict[str, Any] = {}
            for name in self.target_columns:
                column = get_column_data(data, name)
                if column is not None:
                    label_payload[name] = column
            y_all = self.extract_labels(label_payload or None)
            if y_all is None:
                raise ValueError(
                    f"[{self.model_name}-data Error] Labels are required but missing."
                )
        ids_all = None
        if include_ids and id_column:
            id_col = get_column_data(data, id_column)
            if id_col is not None:
                ids_all = np.asarray(id_col).reshape(-1)
        return X_all, y_all, ids_all

    def prepare_arrays(
        self,
        data: Any,
        require_labels: bool,
        include_ids: bool,
        id_column: str | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if isinstance(data, (str, os.PathLike)):
            raise TypeError(
                f"[{self.model_name}-data Error] File paths are not supported here. "
                "Use RecDataLoader to create a DataLoader for training."
            )
        if isinstance(data, (pd.DataFrame, dict)):
            return self.collect_from_table(data, require_labels, include_ids, id_column)
        if isinstance(data, Iterable) and hasattr(data, "__iter__"):
            return self.collect_from_dataloader(
                data, require_labels, include_ids, id_column
            )
        raise TypeError(
            f"[{self.model_name}-data Error] Unsupported data type: {type(data)}"
        )

    def build_estimator(self, model_params: dict[str, Any], epochs: int | None):
        raise NotImplementedError

    def fit_estimator(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray | None,
        y_valid: np.ndarray | None,
        cat_features: list[int],
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def predict_scores(self, model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if isinstance(proba, list):
                proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
        return np.asarray(model.predict(X)).reshape(-1)

    def compile(self, optimizer=None, loss=None, loss_params=None, **kwargs) -> None:
        del optimizer, loss, loss_params, kwargs  # not used for tree models

    def fit(
        self,
        train_data: Any,
        valid_data: Any | None = None,
        metrics: list[str] | dict[str, list[str]] | None = None,
        epochs: int = 1,
        batch_size: int = 512,
        shuffle: bool = True,
        num_workers: int = 0,
        user_id_column: str | None = None,
        ignore_label: int | float | None = None,
        **kwargs: Any,
    ) -> None:
        del batch_size, shuffle, num_workers  # not used for tree models
        self.assert_task()
        if train_data is None:
            raise ValueError(f"[{self.model_name}-fit Error] train_data is required.")

        setup_logger(session_id=self.session_path)
        logging.info("")
        logging.info(colorize("[Tree Model]", color="bright_blue", bold=True))
        logging.info(colorize("-" * 80, color="bright_blue"))
        logging.info(format_kv("Model", self.__class__.__name__))
        logging.info(format_kv("Session ID", self.session_id))
        logging.info(format_kv("Device", self.device))

        target_names = self.target_columns or ["label"]
        metrics_list, task_specific_metrics, _ = configure_metrics(
            self.task, metrics, target_names
        )
        need_user_id = check_user_id(metrics_list, task_specific_metrics)
        id_column = user_id_column or (self.id_columns[0] if self.id_columns else None)
        include_ids = need_user_id and id_column is not None

        X_train, y_train, train_ids = self.prepare_arrays(
            train_data,
            require_labels=True,
            include_ids=include_ids,
            id_column=id_column,
        )
        X_valid = y_valid = valid_ids = None
        if valid_data is not None:
            X_valid, y_valid, valid_ids = self.prepare_arrays(
                valid_data,
                require_labels=True,
                include_ids=include_ids,
                id_column=id_column,
            )

        logging.info("")
        logging.info(colorize("[Features]", color="bright_blue", bold=True))
        logging.info(colorize("-" * 80, color="bright_blue"))
        logging.info(format_kv("Dense features", len(self.dense_features)))
        logging.info(format_kv("Sparse features", len(self.sparse_features)))
        logging.info(format_kv("Sequence features", len(self.sequence_features)))
        logging.info(format_kv("Targets", len(target_names)))
        logging.info(format_kv("Train rows", X_train.shape[0]))
        if X_valid is not None:
            logging.info(format_kv("Valid rows", X_valid.shape[0]))

        model = self.build_estimator(dict(self.model_params), epochs)
        self.model = self.fit_estimator(
            model,
            X_train,
            y_train,
            X_valid,
            y_valid,
            self._cat_feature_indices,
            **kwargs,
        )

        if metrics_list and y_valid is not None and X_valid is not None:
            y_pred = self.predict_scores(self.model, X_valid)
            metrics_dict = evaluate_metrics(
                y_valid,
                y_pred,
                metrics_list,
                self.task,
                target_names,
                task_specific_metrics=task_specific_metrics,
                user_ids=valid_ids,
                ignore_label=ignore_label,
            )
            display_metrics_table(
                epoch=1,
                epochs=1,
                split="valid",
                loss=None,
                metrics=metrics_dict,
                target_names=target_names,
                base_metrics=metrics_list,
            )

        self.save_model()

    @overload
    def predict(
        self,
        data: Any,
        batch_size: int = 512,
        save_path: None = None,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: Literal[True] = True,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
    ) -> pd.DataFrame: ...

    @overload
    def predict(
        self,
        data: Any,
        batch_size: int = 512,
        save_path: None = None,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: Literal[False] = False,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        data: Any,
        batch_size: int = 512,
        *,
        save_path: str | os.PathLike,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: Literal[True] = True,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
    ) -> pd.DataFrame: ...

    @overload
    def predict(
        self,
        data: Any,
        batch_size: int = 512,
        *,
        save_path: str | os.PathLike,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: Literal[False] = False,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
    ) -> Path: ...

    def predict(
        self,
        data: Any,
        batch_size: int = 512,
        save_path: str | os.PathLike | None = None,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: bool = True,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
    ) -> pd.DataFrame | np.ndarray | Path | None:
        del batch_size, num_workers  # not used for tree models

        if self.model is None:
            raise ValueError(f"[{self.model_name}-predict Error] Model is not loaded.")

        predict_id_columns = to_list(id_columns) or self.id_columns
        if include_ids is None:
            include_ids = bool(predict_id_columns)
        include_ids = include_ids and bool(predict_id_columns)

        if save_path is not None and not return_dataframe:
            return self.predict_streaming(
                data=data,
                save_path=save_path,
                save_format=save_format,
                include_ids=include_ids,
                stream_chunk_size=stream_chunk_size,
                id_columns=predict_id_columns,
            )

        if isinstance(data, (str, os.PathLike)):
            rec_loader = RecDataLoader(
                dense_features=self.dense_features,
                sparse_features=self.sparse_features,
                sequence_features=self.sequence_features,
                target=None,
                id_columns=predict_id_columns,
            )
            data = rec_loader.create_dataloader(
                data=data,
                batch_size=stream_chunk_size,
                shuffle=False,
                streaming=True,
                chunk_size=stream_chunk_size,
            )

        X_all, _, ids_all = self.prepare_arrays(
            data,
            require_labels=False,
            include_ids=include_ids,
            id_column=predict_id_columns[0] if predict_id_columns else None,
        )
        y_pred = self.predict_scores(self.model, X_all)
        y_pred = y_pred.reshape(-1, 1)

        pred_columns = self.target_columns or ["pred"]
        pred_df = pd.DataFrame(y_pred, columns=pred_columns[:1])
        if include_ids and ids_all is not None:
            id_df = pd.DataFrame({predict_id_columns[0]: ids_all})
            output = pd.concat([id_df, pred_df], axis=1)
        else:
            output = pred_df if return_dataframe else y_pred

        if save_path is not None:
            suffix = FILE_FORMAT_CONFIG[save_format]["extension"][0]
            target_path = get_save_path(
                path=save_path,
                default_dir=self.session.predictions_dir,
                default_name="predictions",
                suffix=suffix,
                add_timestamp=True if save_path is None else False,
            )
            if isinstance(output, pd.DataFrame):
                df_to_save = output
            else:
                df_to_save = pd.DataFrame(y_pred, columns=pred_columns[:1])
                if include_ids and ids_all is not None and predict_id_columns:
                    id_df = pd.DataFrame({predict_id_columns[0]: ids_all})
                    df_to_save = pd.concat([id_df, df_to_save], axis=1)
            if save_format == "csv":
                df_to_save.to_csv(target_path, index=False)
            elif save_format == "parquet":
                df_to_save.to_parquet(target_path, index=False)
            elif save_format == "feather":
                df_to_save.to_feather(target_path)
            elif save_format == "excel":
                df_to_save.to_excel(target_path, index=False)
            elif save_format == "hdf5":
                df_to_save.to_hdf(target_path, key="predictions", mode="w")
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
            logging.info(f"Predictions saved to: {target_path}")
        return output

    def predict_streaming(
        self,
        data: Any,
        save_path: str | os.PathLike,
        save_format: str,
        include_ids: bool,
        stream_chunk_size: int,
        id_columns: list[str] | None,
    ) -> Path:
        if isinstance(data, (str, os.PathLike)):
            rec_loader = RecDataLoader(
                dense_features=self.dense_features,
                sparse_features=self.sparse_features,
                sequence_features=self.sequence_features,
                target=None,
                id_columns=id_columns,
            )
            data_loader = rec_loader.create_dataloader(
                data=data,
                batch_size=stream_chunk_size,
                shuffle=False,
                streaming=True,
                chunk_size=stream_chunk_size,
            )
        else:
            data_loader = data

        if not check_streaming_support(save_format):
            logging.warning(
                f"[{self.model_name}-predict Warning] Format '{save_format}' does not support streaming writes."
            )

        suffix = FILE_FORMAT_CONFIG[save_format]["extension"][0]
        target_path = get_save_path(
            path=save_path,
            default_dir=self.session.predictions_dir,
            default_name="predictions",
            suffix=suffix,
            add_timestamp=True if save_path is None else False,
        )

        header_written = False
        parquet_writer = None
        collected_frames: list[pd.DataFrame] = []
        id_column = id_columns[0] if id_columns else None
        for batch in data_loader:
            X_batch = self.features_to_matrix(batch.get("features", {}))
            y_pred = self.predict_scores(self.model, X_batch).reshape(-1, 1)
            pred_df = pd.DataFrame(y_pred, columns=self.target_columns or ["pred"])
            if include_ids and id_column:
                ids = self.extract_ids(batch.get("ids"), id_column)
                if ids is not None:
                    pred_df.insert(0, id_column, ids)
            if save_format == "csv":
                pred_df.to_csv(
                    target_path, mode="a", header=not header_written, index=False
                )
            elif save_format == "parquet":
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except ImportError as exc:  # pragma: no cover
                    raise ImportError(
                        f"[{self.model_name}-predict Error] Parquet streaming save requires pyarrow."
                    ) from exc
                table = pa.Table.from_pandas(pred_df, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(target_path, table.schema)
                parquet_writer.write_table(table)
            else:
                collected_frames.append(pred_df)
            header_written = True
        if parquet_writer is not None:
            parquet_writer.close()
        if collected_frames:
            combined_df = pd.concat(collected_frames, ignore_index=True)
            if save_format == "feather":
                combined_df.to_feather(target_path)
            elif save_format == "excel":
                combined_df.to_excel(target_path, index=False)
            elif save_format == "hdf5":
                combined_df.to_hdf(target_path, key="predictions", mode="w")
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
        return target_path

    def save_model(self, save_path: str | os.PathLike | None = None) -> Path:
        if self.model is None:
            raise ValueError(f"[{self.model_name}-save Error] Model is not fitted.")
        target_path = get_save_path(
            path=save_path,
            default_dir=self.session_path,
            default_name=self.model_name.upper(),
            suffix=self.model_file_suffix,
            add_timestamp=True if save_path is None else False,
        )
        self.save_model_file(self.model, target_path)
        with open(self.features_config_path, "wb") as handle:
            pickle.dump(
                {
                    "all_features": self.all_features,
                    "target": self.target_columns,
                    "id_columns": self.id_columns,
                },
                handle,
            )
        return target_path

    def save_model_file(self, model: Any, path: Path) -> None:
        raise NotImplementedError

    def load_model(
        self,
        save_path: str | os.PathLike,
        map_location: str | None = None,
        verbose: bool = True,
    ) -> None:
        del map_location
        model_path = Path(save_path)
        if model_path.is_dir():
            candidates = sorted(model_path.glob(f"*.{self.model_file_suffix}"))
            if not candidates:
                raise FileNotFoundError(
                    f"[{self.model_name}-load Error] No model file found in {model_path}"
                )
            model_path = candidates[-1]
        if not model_path.exists():
            raise FileNotFoundError(
                f"[{self.model_name}-load Error] Model file does not exist: {model_path}"
            )
        self.model = self.load_model_file(model_path)
        config_path = model_path.parent / "features_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as handle:
                cfg = pickle.load(handle)
            all_features = cfg.get("all_features", [])
            dense_features = [f for f in all_features if isinstance(f, DenseFeature)]
            sparse_features = [f for f in all_features if isinstance(f, SparseFeature)]
            sequence_features = [
                f for f in all_features if isinstance(f, SequenceFeature)
            ]
            self.set_all_features(
                dense_features=dense_features,
                sparse_features=sparse_features,
                sequence_features=sequence_features,
                target=cfg.get("target"),
                id_columns=cfg.get("id_columns"),
            )
        if verbose:
            logging.info(f"Model loaded from: {model_path}")

    def load_model_file(self, path: Path) -> Any:
        raise NotImplementedError
