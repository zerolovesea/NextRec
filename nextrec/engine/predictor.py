"""
Trainer module for NextRec models, providing training loop with support for
various optimizers, schedulers, loss functions, and metrics.

Date: create on 16/04/2025
Checkpoint: edit on 18/04/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from nextrec.basic.asserts import assert_save_format, assert_streaming_data_is_filepath
from nextrec.basic.loggers import colorize
from nextrec.basic.session import get_save_path
from nextrec.engine.backends import InferenceBackend
from nextrec.utils.console import progress
from nextrec.utils.data import get_expand_columns
from nextrec.utils.timing import StageTimer
from nextrec.utils.torch_utils import smart_inference_mode, to_numpy


class BasePredictor:

    @overload
    @smart_inference_mode()
    def predict(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: str | os.PathLike | None = None,
        save_format: str = "csv",
        return_dataframe: Literal[True] = True,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        expand: dict[str, list[Any]] | None = None,
        key_columns: str | list[str] | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def predict(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: None = None,
        save_format: str = "csv",
        return_dataframe: Literal[False] = False,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        expand: dict[str, list[Any]] | None = None,
        key_columns: str | list[str] | None = None,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        *,
        save_path: str | os.PathLike,
        save_format: str = "csv",
        return_dataframe: Literal[False] = False,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        expand: dict[str, list[Any]] | None = None,
        key_columns: str | list[str] | None = None,
    ) -> Path: ...

    def predict(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: str | os.PathLike | None = None,
        save_format: str = "csv",
        return_dataframe: bool = True,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        expand: dict[str, list[Any]] | None = None,
        key_columns: str | list[str] | None = None,
    ) -> pd.DataFrame | np.ndarray | Path:
        """
        Make predictions on the given data.

        Args:
            data: Input data for prediction (file path, dict, DataFrame, or DataLoader).
            batch_size: Batch size for prediction (per process when distributed).
            save_path: Optional path to save predictions; if None, predictions are not saved to disk.
            save_format: Format to save predictions ('csv' or 'parquet').
            return_dataframe: Whether to return predictions as a pandas DataFrame; if False, returns a NumPy array.
            stream_chunk_size: Number of rows per chunk when using streaming mode for large datasets.
            num_workers: DataLoader worker count.
            prefetch_factor: Number of batches prefetched per worker (only when num_workers > 0).
            num_processes: Number of inference processes for streaming file inference.
            processor: Optional DataProcessor for transforming input data.
            expand: Optional mapping of column -> candidate values used to expand
                each input row into multiple inference rows before prediction. e.g. {"country": ["US", "CA"]} would expand each input row into two rows with country=US and country=CA for prediction.
            key_columns: Column or list of columns to use as keys for the predictions. e.g. "user_id"

        Note:
            predict does not support distributed mode currently; streaming file inference can use
            multiple processes via num_processes > 1, which may change output order.
        """
        self.eval()

        expand = get_expand_columns(expand)
        predict_key_columns = self.get_predict_keys(key_columns, expand)

        if save_path is not None and not return_dataframe:
            if num_processes > 1:
                assert_streaming_data_is_filepath(data, model_name="BaseModel-predict")
            if num_processes > 1 and num_workers != 0:
                logging.info("[BaseModel-predict-streaming Info] Multi-process streaming enforces num_workers=0.")
                logging.info("")
                num_workers = 0

            return self.predict_streaming(
                data=data,
                batch_size=batch_size,
                save_path=save_path,
                save_format=save_format,
                stream_chunk_size=stream_chunk_size,
                return_dataframe=return_dataframe,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                num_processes=num_processes,
                processor=processor,
                expand=expand,
                key_columns=predict_key_columns,
            )

        return self.predict_with_backend(
            backend=self.inference_backend,
            data=data,
            batch_size=batch_size,
            save_path=save_path,
            save_format=save_format,
            return_dataframe=return_dataframe,
            stream_chunk_size=stream_chunk_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            processor=processor,
            expand=expand,
            key_columns=predict_key_columns,
        )

    def predict_streaming(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int,
        save_path: str | os.PathLike,
        save_format: str,
        stream_chunk_size: int,
        return_dataframe: bool,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        expand: dict[str, list[Any]] | None = None,
        key_columns: list[str] | None = None,
    ):
        expand = get_expand_columns(expand)
        predict_key_columns = self.get_predict_keys(key_columns, expand)
        assert_save_format(save_format, model_name="BaseModel-predict-streaming")
        profiler = self.get_profiler()

        if num_processes > 1:
            inference_artifact_path = self.inference_artifact_path
            if num_workers > 0:
                logging.info("[BaseModel-predict-streaming Info] Multi-process streaming enforces num_workers=0.")
            num_workers = 0
            prefetch_factor = None
            target_path = Path(
                get_save_path(
                    path=save_path,
                    default_dir=self.session.predictions_dir,
                    default_name="predictions",
                    suffix=f".{save_format}",
                    add_timestamp=False,
                )
            )
            parts_dir = target_path.parent / f".{target_path.stem}_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            part_paths = [
                parts_dir / f"{target_path.stem}.part{rank}{target_path.suffix}" for rank in range(num_processes)
            ]
            profile_paths = [parts_dir / f"{target_path.stem}.profile{rank}.json" for rank in range(num_processes)]
            error_paths = [parts_dir / f"{target_path.stem}.error{rank}.txt" for rank in range(num_processes)]

            ctx = mp.get_context("spawn")
            processes = []
            parent_backend = self.inference_backend
            self.inference_backend = None
            try:
                for rank in range(num_processes):
                    process = ctx.Process(
                        target=predict_streaming_worker,
                        args=(
                            self,
                            inference_artifact_path,
                            data,
                            batch_size,
                            part_paths[rank],
                            save_format,
                            stream_chunk_size,
                            num_workers,
                            prefetch_factor,
                            processor,
                            rank,
                            num_processes,
                            expand,
                            predict_key_columns,
                            profile_paths[rank],
                            error_paths[rank],
                        ),
                    )
                    process.start()
                    processes.append(process)
            finally:
                self.inference_backend = parent_backend

            for process in progress(iter(processes), description="Predicting...", total=None):
                process.join()

            # get multi-process worker errors
            failed_processes = [process for process in processes if process.exitcode not in (0, None)]
            if failed_processes:
                errors: list[str] = []
                for error_path in error_paths:
                    if not error_path.exists():
                        continue
                    text = error_path.read_text(encoding="utf-8").strip()
                    if text and text not in errors:
                        errors.append(text)
                if not errors:
                    errors = [
                        f"[PredictWorker Error] pid={process.pid}, exitcode={process.exitcode}, name={process.name}"
                        for process in failed_processes
                    ]
                error_text = "\n\n".join(errors)
                raise RuntimeError(
                    "[BaseModel-predict-streaming Error] One or more inference processes failed.\n" + error_text
                )

            existing_parts = [path for path in part_paths if path.exists()]
            if existing_parts:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                if save_format == "csv":
                    lazy_frames = [pl.scan_csv(path) for path in existing_parts]
                    pl.concat(lazy_frames).sink_csv(target_path)
                elif save_format == "parquet":
                    lazy_frames = [pl.scan_parquet(path) for path in existing_parts]
                    pl.concat(lazy_frames).sink_parquet(target_path)
                profiler.add("merge_parts", time.perf_counter() - start)

            for part_path in part_paths:
                if part_path.exists():
                    part_path.unlink()
            for profile_path in profile_paths:
                if profile_path.exists():
                    profiler.merge(StageTimer.load(profile_path))
                    profile_path.unlink()
            for error_path in error_paths:
                if error_path.exists():
                    error_path.unlink()
            if parts_dir.exists() and not any(parts_dir.iterdir()):
                parts_dir.rmdir()

            logging.info("")
            logging.info(colorize(f"Predictions saved to: {target_path} (merged from {num_processes} parts)"))
            return target_path

        return self.predict_streaming_with_backend(
            backend=self.inference_backend,
            data=data,
            batch_size=batch_size,
            save_path=save_path,
            save_format=save_format,
            stream_chunk_size=stream_chunk_size,
            return_dataframe=return_dataframe,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            processor=processor,
            shard_rank=0,
            shard_count=1,
            expand=expand,
            key_columns=predict_key_columns,
        )

    def predict_with_backend(
        self,
        backend: InferenceBackend,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int,
        save_path: str | os.PathLike | None,
        save_format: str,
        return_dataframe: bool,
        stream_chunk_size: int,
        num_workers: int,
        prefetch_factor: int | None,
        processor: Any | None,
        expand: dict[str, list[Any]],
        key_columns: list[str],
    ) -> pd.DataFrame | np.ndarray:
        profiler = self.get_profiler()
        # if need to include key columns in the output
        with_keys = bool(key_columns) and (return_dataframe or save_path is not None)

        data_loader = self.prepare_data_loader(
            data,
            batch_size=backend.prepare_batch_size(data, batch_size),
            shuffle=False,
            streaming=isinstance(data, (str, os.PathLike)),
            chunk_size=backend.prepare_chunk_size(batch_size, stream_chunk_size),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            processor=processor,
            key_columns=key_columns,
            expand=expand,
        )

        y_pred_list = []
        key_buffers = {name: [] for name in key_columns} if with_keys else {}

        for batch_data in progress(data_loader, description=backend.progress_description):
            X_input, _ = self.get_input(batch_data, require_labels=False)
            keys = batch_data.get("keys") if with_keys else None  # key for each sample
            start = time.perf_counter()
            predictions = list(backend.iter_predict(X_input))  # inference

            profiler.add("inference", time.perf_counter() - start)

            # concatenate predictions and keys
            for prediction in predictions:
                y_pred_list.append(prediction.values)
                if with_keys:
                    key_arrays = {}
                    if keys:
                        for key_name in key_columns:
                            if key_name not in keys:
                                continue
                            key_np = to_numpy(keys[key_name])[prediction.start : prediction.end]
                            key_arrays[key_name] = key_np.reshape(-1)
                    for key_name, key_np in key_arrays.items():
                        key_buffers[key_name].append(
                            key_np.reshape(key_np.shape[0], -1) if key_np.ndim == 1 else key_np
                        )

        y_pred_all = np.concatenate(y_pred_list, axis=0) if y_pred_list else np.array([])
        key_arrays_all = None

        # single task
        if y_pred_all.ndim == 1:
            y_pred_all = y_pred_all.reshape(-1, 1)
        # when predictions is empty
        if y_pred_all.size == 0:
            num_outputs = len(self.target_columns) if self.target_columns else 1
            y_pred_all = y_pred_all.reshape(0, num_outputs)

        pred_columns = list(self.target_columns[: y_pred_all.shape[1]]) if self.target_columns else []
        while len(pred_columns) < y_pred_all.shape[1]:
            pred_columns.append(f"pred_{len(pred_columns)}")
        if with_keys:
            key_arrays_all = {
                key_name: (
                    np.concatenate([piece.reshape(piece.shape[0], -1) for piece in pieces], axis=0).reshape(-1)
                    if pieces
                    else np.array([], dtype=np.int64)
                )
                for key_name, pieces in key_buffers.items()
            }
            output_frame = pd.DataFrame(y_pred_all, columns=pred_columns)
            output_frame = pd.concat([pd.DataFrame(key_arrays_all), output_frame], axis=1)
            output = output_frame if return_dataframe else y_pred_all
        else:
            output = pd.DataFrame(y_pred_all, columns=pred_columns) if return_dataframe else y_pred_all

        # save to disk
        if save_path is not None:
            assert_save_format(save_format, model_name="BaseModel-predict")
            target_path = get_save_path(
                path=save_path,
                default_dir=self.session.predictions_dir,
                default_name="predictions",
                suffix=f".{save_format}",
                add_timestamp=False,
            )
            df_to_save = (
                output
                if isinstance(output, pd.DataFrame)
                else pd.concat(
                    [
                        pd.DataFrame(key_arrays_all),
                        pd.DataFrame(y_pred_all, columns=pred_columns),
                    ],
                    axis=1,
                )
            )
            start = time.perf_counter()
            if save_format == "csv":
                df_to_save.to_csv(target_path, index=False)
            elif save_format == "parquet":
                df_to_save.to_parquet(target_path, index=False)
            profiler.add("write_output", time.perf_counter() - start)
            logging.info("")
            logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))

        return output

    def predict_streaming_with_backend(
        self,
        backend: InferenceBackend,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int,
        save_path: str | os.PathLike,
        save_format: str,
        stream_chunk_size: int,
        return_dataframe: bool,
        num_workers: int,
        prefetch_factor: int | None,
        processor: Any | None,
        shard_rank: int,
        shard_count: int,
        expand: dict[str, list[Any]] | None = None,
        key_columns: list[str] | None = None,
        progress_description: str | None = None,
    ) -> pd.DataFrame | Path:
        assert_save_format(save_format, model_name="BaseModel-predict-streaming")
        expand = get_expand_columns(expand)
        key_columns = self.get_predict_keys(key_columns, expand)
        profiler = self.get_profiler()
        with_keys = bool(key_columns)

        data_loader = self.prepare_data_loader(
            data,
            batch_size=backend.prepare_batch_size(data, batch_size),
            shuffle=False,
            streaming=isinstance(data, (str, os.PathLike)),
            chunk_size=backend.prepare_chunk_size(batch_size, stream_chunk_size),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            processor=processor,
            key_columns=key_columns,
            shard_rank=shard_rank,
            shard_count=shard_count,
            expand=expand,
        )

        target_path = get_save_path(
            path=save_path,
            default_dir=self.session.predictions_dir,
            default_name="predictions",
            suffix=f".{save_format}",
            add_timestamp=False,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        header_written = target_path.exists() and target_path.stat().st_size > 0
        parquet_writer = None
        pred_columns = None
        cached_frames = []

        disable_progress = shard_count > 1
        for batch_data in progress(
            data_loader,
            description=progress_description or backend.progress_description,
            disable=disable_progress,
        ):
            X_input, _ = self.get_input(batch_data, require_labels=False)
            keys = batch_data.get("keys") if with_keys else None
            start = time.perf_counter()
            predictions = list(backend.iter_predict(X_input))
            profiler.add("inference", time.perf_counter() - start)

            for prediction in predictions:
                y_pred_np = prediction.values
                if pred_columns is None:
                    pred_columns = list(self.target_columns[: y_pred_np.shape[1]]) if self.target_columns else []
                    while len(pred_columns) < y_pred_np.shape[1]:
                        pred_columns.append(f"pred_{len(pred_columns)}")

                key_arrays_batch = {}
                if with_keys and keys:
                    for key_name in key_columns:
                        if key_name not in keys:
                            continue
                        key_np = to_numpy(keys[key_name])[prediction.start : prediction.end]
                        key_arrays_batch[key_name] = key_np.reshape(-1)
                df_batch = pd.DataFrame(y_pred_np, columns=pred_columns)
                if key_arrays_batch:
                    df_batch = pd.concat([pd.DataFrame(key_arrays_batch), df_batch], axis=1)

                if return_dataframe:
                    cached_frames.append(df_batch)
                start = time.perf_counter()
                if save_format == "csv":
                    df_batch.to_csv(target_path, mode="a", header=not header_written, index=False)
                    header_written = True
                else:
                    table = pa.Table.from_pandas(df_batch, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(target_path, table.schema)
                    parquet_writer.write_table(table)
                profiler.add("write_output", time.perf_counter() - start)

        if parquet_writer is not None:
            parquet_writer.close()

        logging.info("")
        logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))
        if return_dataframe:
            return (
                pd.concat(cached_frames, ignore_index=True)
                if cached_frames
                else pd.DataFrame(columns=pred_columns or [])
            )
        return target_path

    def get_predict_keys(
        self,
        key_columns: str | list[str] | None,
        expand: dict[str, list[Any]],
    ) -> list[str]:
        if key_columns is None:
            key_columns = self.key_columns
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        return list(dict.fromkeys([*(key_columns or []), *expand.keys()]))


def predict_streaming_worker(
    model,
    inference_artifact_path: str | os.PathLike | None,
    data_path: str | os.PathLike,
    batch_size: int,
    save_path: str | os.PathLike,
    save_format: str,
    stream_chunk_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    processor: Any | None,
    shard_rank: int,
    shard_count: int,
    expand: dict[str, list[Any]] | None = None,
    key_columns: list[str] | None = None,
    profile_path: str | os.PathLike | None = None,
    error_path: str | os.PathLike | None = None,
) -> None:
    """
    Single worker function for multi-process streaming prediction.
    """

    try:
        model.eval()
        model.set_inference_backend(inference_artifact_path)
        model.profiler = StageTimer(enabled=True)
        model.predict_streaming_with_backend(
            backend=model.inference_backend,
            data=data_path,
            batch_size=batch_size,
            save_path=save_path,
            save_format=save_format,
            stream_chunk_size=stream_chunk_size,
            return_dataframe=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            processor=processor,
            shard_rank=shard_rank,
            shard_count=shard_count,
            expand=expand,
            key_columns=key_columns,
        )
        if profile_path is not None:
            model.profiler.dump(Path(profile_path))
    except BaseException:
        import traceback

        error_text = f"[PredictWorker Error] rank={shard_rank}\n{traceback.format_exc()}"
        if error_path is not None:
            path = Path(error_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(error_text, encoding="utf-8")
        raise
