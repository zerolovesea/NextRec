#!/usr/bin/env python3
"""
Profile NextRec prediction pipeline to locate bottlenecks.

Usage:
  python scripts/profile_predict.py --predict_config /path/to/predict_config.yaml

Optional:
  --max_batches N     Limit batches for faster profiling (0 = full dataset)
  --warmup_batches N  Warmup batches excluded from timing (default: 5)
  --log_every N       Log progress every N batches (default: 50)
  --skip_model        Skip model forward pass (data pipeline only)
  --skip_save         Skip writing predictions to disk
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.loggers import colorize, format_kv, setup_logger
from nextrec.data.batch_utils import batch_to_dict
from nextrec.data.dataloader import RecDataLoader
from nextrec.data.preprocessor import DataProcessor
from nextrec.utils.config import build_model_instance, resolve_path
from nextrec.utils.console import get_nextrec_version
from nextrec.utils.data import read_yaml

logger = logging.getLogger(__name__)


class Profiler:
    def __init__(self) -> None:
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - start
            self.counts[name] += 1

    def add(self, name: str, duration: float) -> None:
        self.times[name] += duration
        self.counts[name] += 1

    def summary(
        self, total_key: str | None = None
    ) -> list[tuple[str, float, int, float]]:
        total = (
            self.times.get(total_key, sum(self.times.values()))
            if total_key
            else sum(self.times.values())
        )
        rows = []
        for name, duration in sorted(
            self.times.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (duration / total * 100.0) if total > 0 else 0.0
            rows.append((name, duration, self.counts.get(name, 0), pct))
        return rows


def log_section(title: str) -> None:
    logger.info("")
    logger.info(colorize(f"[{title}]", color="bright_blue", bold=True))
    logger.info(colorize("-" * 80, color="bright_blue"))


def log_kv(items: list[tuple[str, Any]]) -> None:
    for label, value in items:
        logger.info(format_kv(label, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile NextRec prediction pipeline")
    parser.add_argument("--predict_config", required=True)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--skip_save", action="store_true")
    return parser.parse_args()


def get_batch_size(batch_dict: dict) -> int:
    features = batch_dict.get("features") or {}
    for tensor in features.values():
        if hasattr(tensor, "shape"):
            return int(tensor.shape[0])
    return 0


def wrap_iter_file_chunks(profiler: Profiler):
    from nextrec.utils import data as data_utils

    original = data_utils.iter_file_chunks

    def wrapped_iter_file_chunks(file_path: str, file_type: str, chunk_size: int):
        gen = original(file_path, file_type, chunk_size)
        while True:
            start = time.perf_counter()
            try:
                chunk = next(gen)
            except StopIteration:
                return
            profiler.add("file_read", time.perf_counter() - start)
            yield chunk

    data_utils.iter_file_chunks = wrapped_iter_file_chunks  # type: ignore[assignment]
    return original


def wrap_processor_transform(profiler: Profiler):
    original = DataProcessor.transform

    def wrapped_transform(self, *args, **kwargs):
        with profiler.track("processor_transform"):
            return original(self, *args, **kwargs)

    DataProcessor.transform = wrapped_transform  # type: ignore[assignment]
    return original


def wrap_build_tensors(profiler: Profiler):
    from nextrec.data import dataloader as dl

    original = dl.build_tensors_from_data

    def wrapped_build_tensors(*args, **kwargs):
        with profiler.track("build_tensors"):
            return original(*args, **kwargs)

    dl.build_tensors_from_data = wrapped_build_tensors  # type: ignore[assignment]
    return original


def maybe_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    config_file = Path(args.predict_config)
    config_dir = config_file.resolve().parent
    cfg = read_yaml(config_file)

    # Checkpoint path is the primary configuration
    if "checkpoint_path" not in cfg:
        session_cfg = cfg.get("session", {}) or {}
        session_id = session_cfg.get("id", "nextrec_session")
        artifact_root = Path(session_cfg.get("artifact_root", "nextrec_logs"))
        session_dir = artifact_root / session_id
    else:
        session_dir = Path(cfg["checkpoint_path"])
        session_cfg = cfg.get("session", {}) or {}
        session_id = session_cfg.get("id") or session_dir.name

    setup_logger(session_id=session_dir.resolve())

    log_section("CLI")
    log_kv(
        [
            ("Mode", "predict-profile"),
            ("Version", get_nextrec_version()),
            ("Session ID", session_id),
            ("Checkpoint", session_dir.resolve()),
            ("Config", config_file.resolve()),
        ]
    )

    profiler = Profiler()

    # Monkeypatch for deeper breakdowns
    original_iter_file_chunks = wrap_iter_file_chunks(profiler)
    original_transform = wrap_processor_transform(profiler)
    original_build_tensors = wrap_build_tensors(profiler)

    processor_path = Path(session_dir / "processor.pkl")
    if not processor_path.exists():
        processor_path = session_dir / "processor" / "processor.pkl"

    predict_cfg = cfg.get("predict", {}) or {}

    if "model_config" in cfg:
        model_cfg_path = resolve_path(cfg["model_config"], config_dir)
    else:
        auto_model_cfg = session_dir / "model_config.yaml"
        model_cfg_path = (
            auto_model_cfg
            if auto_model_cfg.exists()
            else resolve_path("model_config.yaml", config_dir)
        )

    with profiler.track("read_model_config"):
        model_cfg = read_yaml(model_cfg_path)
    model_cfg.setdefault("session_id", session_id)
    model_cfg.setdefault("params", {})

    log_section("Config")
    log_kv(
        [
            ("Predict config", config_file.resolve()),
            ("Model config", model_cfg_path),
            ("Processor", processor_path),
        ]
    )

    with profiler.track("load_processor"):
        processor = DataProcessor.load(processor_path)

    checkpoint_base = Path(session_dir)
    if checkpoint_base.is_dir():
        best_candidates = sorted(checkpoint_base.glob("*_best.pt"))
        candidates = sorted(checkpoint_base.glob("*.pt"))
        if best_candidates:
            model_file = best_candidates[-1]
        elif candidates:
            model_file = candidates[-1]
        else:
            raise FileNotFoundError(
                f"[NextRec CLI Error]: Unable to find model checkpoint: {checkpoint_base}"
            )
        config_dir_for_features = checkpoint_base
    else:
        model_file = (
            checkpoint_base.with_suffix(".pt")
            if checkpoint_base.suffix == ""
            else checkpoint_base
        )
        config_dir_for_features = model_file.parent

    features_config_path = config_dir_for_features / "features_config.pkl"
    if not features_config_path.exists():
        raise FileNotFoundError(
            f"[NextRec CLI Error]: Unable to find features_config.pkl: {features_config_path}"
        )

    with profiler.track("load_features_config"):
        with open(features_config_path, "rb") as f:
            features_config = pickle.load(f)

    all_features = features_config.get("all_features", [])
    target_cols = features_config.get("target", [])
    id_columns = features_config.get("id_columns", [])

    dense_features = [f for f in all_features if isinstance(f, DenseFeature)]
    sparse_features = [f for f in all_features if isinstance(f, SparseFeature)]
    sequence_features = [f for f in all_features if isinstance(f, SequenceFeature)]

    target_override = (
        cfg.get("targets")
        or model_cfg.get("targets")
        or model_cfg.get("params", {}).get("targets")
        or model_cfg.get("params", {}).get("target")
    )
    if target_override:
        target_cols = (
            [target_override]
            if isinstance(target_override, str)
            else list(target_override)
        )

    device = predict_cfg.get("device", "cpu")
    with profiler.track("build_model"):
        model = build_model_instance(
            model_cfg=model_cfg,
            model_cfg_path=model_cfg_path,
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target_cols,
            device=device,
        )
    model.id_columns = id_columns
    with profiler.track("load_model_weights"):
        model.load_model(model_file, map_location=device, verbose=True)

    id_columns = []
    if predict_cfg.get("id_column"):
        id_columns = [predict_cfg["id_column"]]
        model.id_columns = id_columns

    effective_id_columns = id_columns or model.id_columns

    log_section("Features")
    log_kv(
        [
            ("Dense features", len(dense_features)),
            ("Sparse features", len(sparse_features)),
            ("Sequence features", len(sequence_features)),
            ("Targets", len(target_cols)),
            ("ID columns", len(effective_id_columns)),
        ]
    )

    log_section("Model")
    use_onnx = bool(predict_cfg.get("use_onnx")) or bool(predict_cfg.get("onnx_path"))
    onnx_path = predict_cfg.get("onnx_path") or cfg.get("onnx_path")
    if onnx_path:
        onnx_path = resolve_path(onnx_path, config_dir)
    log_kv(
        [
            ("Model", model.__class__.__name__),
            ("Checkpoint", model_file),
            ("Device", device),
            ("Use ONNX", use_onnx),
            ("ONNX path", onnx_path if use_onnx else "(disabled)"),
        ]
    )

    rec_dataloader = RecDataLoader(
        dense_features=model.dense_features,
        sparse_features=model.sparse_features,
        sequence_features=model.sequence_features,
        target=None,
        id_columns=effective_id_columns,
        processor=processor,
    )

    data_path = resolve_path(predict_cfg["data_path"], config_dir)
    streaming = bool(predict_cfg.get("streaming", True))
    chunk_size = int(predict_cfg.get("chunk_size", 20000))
    batch_size = int(predict_cfg.get("batch_size", 512))
    effective_batch_size = chunk_size if streaming else batch_size

    log_section("Data")
    log_kv(
        [
            ("Data path", data_path),
            (
                "Format",
                predict_cfg.get(
                    "source_data_format", predict_cfg.get("data_format", "auto")
                ),
            ),
            ("Batch size", effective_batch_size),
            ("Chunk size", chunk_size),
            ("Streaming", streaming),
        ]
    )

    with profiler.track("create_dataloader"):
        pred_loader = rec_dataloader.create_dataloader(
            data=str(data_path),
            batch_size=1 if streaming else batch_size,
            shuffle=False,
            streaming=streaming,
            chunk_size=chunk_size,
            prefetch_factor=predict_cfg.get("prefetch_factor"),
        )

    save_format = predict_cfg.get(
        "save_data_format", predict_cfg.get("save_format", "csv")
    )
    pred_name = predict_cfg.get("name", "pred")
    pred_name_path = Path(pred_name)
    if pred_name_path.is_absolute():
        save_path = pred_name_path
        if save_path.suffix == "":
            save_path = save_path.with_suffix(f".{save_format}")
    else:
        save_path = checkpoint_base / "predictions" / f"{pred_name}.{save_format}"

    if use_onnx:
        log_section("Warning")
        logger.info(
            "ONNX prediction runs inside model.predict_onnx; per-step profiling is limited to total time."
        )

    max_batches = args.max_batches
    warmup_batches = max(0, args.warmup_batches)
    log_every = max(1, args.log_every)

    model.eval()
    total_rows = 0
    total_batches = 0

    if use_onnx:
        with profiler.track("predict_onnx_total"):
            model.predict_onnx(
                onnx_path=onnx_path,
                data=pred_loader,
                batch_size=effective_batch_size,
                include_ids=bool(id_columns),
                return_dataframe=False,
                save_path=None if args.skip_save else str(save_path),
                save_format=save_format,
                num_workers=predict_cfg.get("num_workers", 0),
            )
    else:
        iterator = iter(pred_loader)
        header_written = False
        parquet_writer = None
        if not args.skip_save:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            header_written = save_path.exists() and save_path.stat().st_size > 0

        batch_index = 0
        start_total = None
        while True:
            if max_batches > 0 and batch_index >= max_batches:
                break
            start_wait = time.perf_counter()
            try:
                batch_data = next(iterator)
            except StopIteration:
                break
            data_wait = time.perf_counter() - start_wait

            batch_index += 1
            if batch_index <= warmup_batches:
                continue
            if start_total is None:
                start_total = time.perf_counter()
            profiler.add("data_wait", data_wait)

            with profiler.track("batch_to_dict"):
                batch_dict = batch_to_dict(batch_data, include_ids=bool(id_columns))

            batch_size_current = get_batch_size(batch_dict)
            total_rows += batch_size_current
            total_batches += 1

            if args.skip_model:
                if batch_index % log_every == 0:
                    logger.info(f"Profiled {batch_index} batches (data-only)...")
                continue

            with profiler.track("get_input"):
                X_input, _ = model.get_input(batch_dict, require_labels=False)

            maybe_sync(device)
            with profiler.track("model_forward"):
                y_pred = model(X_input)
            maybe_sync(device)

            with profiler.track("postprocess"):
                if y_pred is None or not isinstance(y_pred, torch.Tensor):
                    continue
                y_pred_np = y_pred.detach().cpu().numpy()
                if y_pred_np.ndim == 1:
                    y_pred_np = y_pred_np.reshape(-1, 1)

            if args.skip_save:
                if batch_index % log_every == 0:
                    logger.info(f"Profiled {batch_index} batches...")
                continue

            with profiler.track("save_output"):
                pred_columns = [f"pred_{i}" for i in range(y_pred_np.shape[1])]
                df_batch = pd.DataFrame(y_pred_np, columns=pred_columns)
                if save_format == "csv":
                    df_batch.to_csv(
                        save_path, mode="a", header=not header_written, index=False
                    )
                    header_written = True
                elif save_format == "parquet":
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    table = pa.Table.from_pandas(df_batch, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(save_path, table.schema)
                    parquet_writer.write_table(table)
                else:
                    df_batch.to_csv(
                        save_path, mode="a", header=not header_written, index=False
                    )
                    header_written = True

            if batch_index % log_every == 0:
                logger.info(f"Profiled {batch_index} batches...")

        if start_total is not None:
            profiler.add("predict_total", time.perf_counter() - start_total)
        if parquet_writer is not None:
            parquet_writer.close()

    # Restore monkeypatches
    from nextrec.utils import data as data_utils
    from nextrec.data import dataloader as dl

    data_utils.iter_file_chunks = original_iter_file_chunks  # type: ignore[assignment]
    DataProcessor.transform = original_transform  # type: ignore[assignment]
    dl.build_tensors_from_data = original_build_tensors  # type: ignore[assignment]

    log_section("Profile Summary")
    logger.info(
        f"Processed rows: {total_rows} | batches: {total_batches} | warmup excluded: {warmup_batches}"
    )
    total_key = None
    if "predict_total" in profiler.times:
        total_key = "predict_total"
    elif "predict_onnx_total" in profiler.times:
        total_key = "predict_onnx_total"
    if total_key and total_rows > 0 and profiler.times.get(total_key, 0) > 0:
        throughput = total_rows / profiler.times[total_key]
        logger.info(f"Throughput: {throughput:.2f} rows/sec")

    rows = profiler.summary(total_key=total_key)
    for name, duration, count, pct in rows:
        logger.info(f"{name:<24} {duration:>9.3f}s  count={count:<6}  {pct:>6.2f}%")


if __name__ == "__main__":
    main()
