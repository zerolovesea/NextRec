"""
Command-line interface for NextRec training and prediction.


NextRec supports a flexible training and prediction pipeline driven by configuration files.
After preparing the configuration YAML files for training and prediction, users can run the
following script to execute the desired operations.

Examples:
    # Train a model
    nextrec --mode=train --train_config=nextrec_cli_preset/train_config.yaml

    # Run prediction
    nextrec --mode=predict --predict_config=nextrec_cli_preset/predict_config.yaml

    # Run evaluation
    nextrec --mode=evaluate --evaluate_config=nextrec_cli_preset/evaluate_config.yaml

Date: create on 06/12/2025
Checkpoint: edit on 07/02/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

import os
import argparse
import logging
import math
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.loggers import colorize, format_kv, setup_logger
from nextrec.basic.metrics import configure_metrics
from nextrec.data.data_utils import split_dict_random
from nextrec.data.dataloader import RecDataLoader
from nextrec.data.preprocessor import DataProcessor
from nextrec.utils.config import (
    build_feature_objects,
    build_model_instance,
    register_processor_features,
    resolve_path,
    select_features,
)
from nextrec.utils.console import get_nextrec_version
from nextrec.utils.data import (
    count_rows,
    iter_file_chunks,
    read_table,
    read_yaml,
    resolve_file_paths,
)
from nextrec.utils.timing import StageTimer
from nextrec.utils.torch_utils import to_list

logger = logging.getLogger(__name__)


def log_cli_section(title: str) -> None:
    logger.info("")
    logger.info(colorize(f"[{title}]", color="bright_blue", bold=True))
    logger.info(colorize("-" * 80, color="bright_blue"))


def log_kv_lines(items: list[tuple[str, Any]]) -> None:
    for label, value in items:
        logger.info(format_kv(label, value))


def train_model(train_config_path: str) -> None:
    """
    Train a NextRec model using the provided configuration file.

    configuration file must specify the below sections:
        - session: Session settings including id and artifact root
        - data: Data settings including path, format, target, validation split
        - dataloader: DataLoader settings including batch sizes and shuffling
        - model_config: Path to the model configuration YAML file
        - feature_config: Path to the feature configuration YAML file
        - train: Training settings including optimizer, loss, metrics, epochs, etc.
    """
    config_file = Path(train_config_path)
    config_dir = config_file.resolve().parent  # Directory of the config file
    cfg = read_yaml(config_file)

    # session / logger
    session_cfg = cfg.get("session", {}) or {}
    session_id = session_cfg.get("id", "nextrec_cli_session")
    artifact_root = Path(session_cfg.get("artifact_root", "nextrec_logs"))
    session_dir = artifact_root / session_id
    setup_logger(session_id=session_dir.resolve())

    log_cli_section("CLI")
    log_kv_lines(
        [
            ("Mode", "train"),
            ("Version", get_nextrec_version()),
            ("Session ID", session_id),
            ("Artifacts", session_dir.resolve()),
            ("Config", config_file.resolve()),
            ("Command", " ".join(sys.argv)),
        ]
    )

    processor_path = Path(session_dir / "processor.pkl")
    processor_path.parent.mkdir(parents=True, exist_ok=True)

    # config sections
    data_cfg = cfg.get("data", {}) or {}
    dataloader_cfg = cfg.get("dataloader", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    feature_cfg_path = resolve_path(cfg.get("feature_config", "feature_config.yaml"), config_dir)
    model_cfg_path = resolve_path(cfg.get("model_config", "model_config.yaml"), config_dir)

    log_cli_section("Config")
    log_kv_lines(
        [
            ("Train config", config_file.resolve()),
            ("Feature config", feature_cfg_path),
            ("Model config", model_cfg_path),
        ]
    )

    # dataloader settings
    streaming = bool(data_cfg.get("streaming", False))
    dataloader_chunk_size = dataloader_cfg.get("chunk_size", 20000)
    batch_size = dataloader_cfg.get("batch_size")
    if batch_size is None:
        batch_size = dataloader_cfg.get("train_batch_size", 512)
    batch_size = int(batch_size)
    shuffle = dataloader_cfg.get("shuffle")
    if shuffle is None:
        shuffle = dataloader_cfg.get("train_shuffle", True)
    shuffle = bool(shuffle)

    # train data basics
    data_path = resolve_path(data_cfg["path"], config_dir)
    target = to_list(data_cfg["target"])
    val_data_path = data_cfg.get("valid_path")

    feature_cfg = read_yaml(feature_cfg_path)
    model_cfg = read_yaml(model_cfg_path)

    # Extract id_column from data config for GAUC metrics
    id_column = data_cfg.get("id_column")
    id_columns = [id_column] if id_column else []

    log_cli_section("Data")
    log_kv_lines(
        [
            ("Data path", data_path),
            ("Format", data_cfg.get("format", "auto")),
            ("Streaming", streaming),
            ("Target", target),
            ("ID column", id_column or "(not set)"),
        ]
    )
    if data_cfg.get("valid_ratio") is not None:
        logger.info(format_kv("Valid ratio", data_cfg.get("valid_ratio")))
    if val_data_path:
        logger.info(
            format_kv(
                "Validation path",
                resolve_path(val_data_path, config_dir),
            )
        )

    file_paths = []
    file_type = None
    streaming_train_files = None
    streaming_valid_files = None

    if streaming:
        file_paths, file_type = resolve_file_paths(str(data_path))
        log_kv_lines(
            [
                ("File type", file_type),
                ("Files", len(file_paths)),
                ("Chunk size", dataloader_chunk_size),
            ]
        )
        first_file = file_paths[0]
        first_chunk_size = max(1, min(dataloader_chunk_size, 1000))
        chunk_iter = iter_file_chunks(first_file, file_type, first_chunk_size)
        try:
            first_chunk = next(chunk_iter)
        except StopIteration as exc:
            raise ValueError(f"Data file is empty: {first_file}") from exc
        df_columns = list(first_chunk.columns)

        # Decide training/validation file lists before fitting processor, to avoid
        # leaking validation statistics into preprocessing (scalers/encoders).
        streaming_train_files = file_paths
        streaming_valid_ratio = data_cfg.get("valid_ratio")
        if val_data_path:
            streaming_valid_files = None
        elif streaming_valid_ratio is not None:
            ratio = float(streaming_valid_ratio)
            if not (0 < ratio < 1):
                raise ValueError(
                    f"[NextRec CLI Error] Valid_ratio must be between 0 and 1, current value is {streaming_valid_ratio}"
                )
            total_files = len(file_paths)
            if total_files < 2:
                raise ValueError(
                    "[NextRec CLI Error] Must provide valid_path or increase the number of data files. At least 2 files are required for streaming validation split."
                )
            val_count = max(1, int(round(total_files * ratio)))
            if val_count >= total_files:
                val_count = total_files - 1
            streaming_valid_files = file_paths[-val_count:]
            streaming_train_files = file_paths[:-val_count]
            logger.info(
                f"Split files for streaming training and validation using valid_ratio={ratio:.3f}: training {len(streaming_train_files)} files, validation {len(streaming_valid_files)} files"
            )
        else:
            streaming_valid_files = None
    else:
        df = read_table(data_path, data_cfg.get("format"))
        logger.info(format_kv("Rows", len(df)))
        logger.info(format_kv("Columns", len(df.columns)))
        df_columns = list(df.columns)

    dense_names, sparse_names, sequence_names = select_features(feature_cfg, df_columns)

    used_columns = dense_names + sparse_names + sequence_names + target + id_columns

    # keep order but drop duplicates
    seen = set()
    unique_used_columns = []
    for col in used_columns:
        if col not in seen:
            unique_used_columns.append(col)
            seen.add(col)

    processor = DataProcessor()
    register_processor_features(processor, feature_cfg, dense_names, sparse_names, sequence_names)

    log_cli_section("Features")
    log_kv_lines(
        [
            ("Dense features", len(dense_names)),
            ("Sparse features", len(sparse_names)),
            ("Sequence features", len(sequence_names)),
            ("Targets", len(target)),
            ("Used columns", len(unique_used_columns)),
        ]
    )
    logger.info("")

    if streaming:
        if file_type is None:
            raise ValueError("[NextRec CLI Error] Streaming mode requires a valid file_type")
        processor.fit_from_files(
            file_paths=streaming_train_files or file_paths,
            file_type=file_type,
        )
        processed = None
        df = None  # type: ignore[assignment]
    else:
        df = df[unique_used_columns]
        processor.fit(df)
        processed = processor.transform(df, return_dict=True)

    processor.save(processor_path)
    dense_features, sparse_features, sequence_features = build_feature_objects(
        processor,
        feature_cfg,
        dense_names,
        sparse_names,
        sequence_names,
    )

    train_data: Dict[str, Any]
    valid_data: Dict[str, Any] | None

    if val_data_path and not streaming:
        # Use specified validation dataset path
        logger.info(f"Validation using specified validation dataset path: {val_data_path}")
        val_data_resolved = resolve_path(val_data_path, config_dir)
        val_df = read_table(val_data_resolved, data_cfg.get("format"))
        val_df = val_df[unique_used_columns]
        if not isinstance(processed, dict):
            raise TypeError("Processed data must be a dictionary")
        train_data = processed
        valid_data_result = processor.transform(val_df, return_dict=True)
        if not isinstance(valid_data_result, dict):
            raise TypeError("Validation data must be a dictionary")
        valid_data = valid_data_result
        train_size = len(list(train_data.values())[0])
        valid_size = len(list(valid_data.values())[0])
        logger.info(f"Sample count - Training set: {train_size}, Validation set: {valid_size}")
    elif streaming:
        train_data = None  # type: ignore[assignment]
        valid_data = None
        if not val_data_path and not streaming_valid_files:
            logger.info(
                "Streaming training mode: No validation dataset path specified and valid_ratio not configured, skipping validation dataset creation"
            )
    else:
        # Split data using valid_ratio
        logger.info(f"Splitting data using valid_ratio: {data_cfg.get('valid_ratio', 0.2)}")
        if not isinstance(processed, dict):
            raise TypeError("Processed data must be a dictionary for splitting")
        train_data, valid_data = split_dict_random(
            processed,
            test_size=data_cfg.get("valid_ratio", 0.2),
            random_state=data_cfg.get("random_state", 2024),
        )

    dataloader = RecDataLoader(
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        target=target,
        id_columns=id_columns,
        processor=processor if streaming else None,
    )
    loader_base_kwargs = {
        "batch_size": batch_size,
        "num_workers": dataloader_cfg.get("num_workers", 0),
        "prefetch_factor": dataloader_cfg.get("prefetch_factor"),
    }
    if streaming:
        train_stream_source = streaming_train_files or file_paths
        train_loader = dataloader.create_dataloader(
            data=train_stream_source,
            shuffle=shuffle,
            streaming=True,
            chunk_size=dataloader_chunk_size,
            **loader_base_kwargs,
        )
        valid_loader = None
        if val_data_path:
            val_data_resolved = resolve_path(val_data_path, config_dir)
            valid_loader = dataloader.create_dataloader(
                data=str(val_data_resolved),
                shuffle=False,
                streaming=True,
                chunk_size=dataloader_chunk_size,
                **loader_base_kwargs,
            )
        elif streaming_valid_files:
            valid_loader = dataloader.create_dataloader(
                data=streaming_valid_files,
                shuffle=False,
                streaming=True,
                chunk_size=dataloader_chunk_size,
                **loader_base_kwargs,
            )
    else:
        train_loader = dataloader.create_dataloader(
            data=train_data,
            shuffle=shuffle,
            **loader_base_kwargs,
        )
        valid_loader = dataloader.create_dataloader(
            data=valid_data,
            shuffle=False,
            **loader_base_kwargs,
        )

    model_cfg.setdefault("session_id", session_id)
    device = train_cfg.get("device", model_cfg.get("device", "cpu"))
    model = build_model_instance(
        model_cfg,
        model_cfg_path,
        dense_features,
        sparse_features,
        sequence_features,
        target,
        id_columns,
        device,
    )

    log_cli_section("Model")
    log_kv_lines(
        [
            ("Model", model.__class__.__name__),
            ("Device", device),
            ("Session ID", session_id),
        ]
    )

    model.compile(
        optimizer=train_cfg.get("optimizer", "adam"),
        optimizer_params=train_cfg.get("optimizer_params", {}),
        loss=train_cfg.get("loss", "focal"),
        loss_params=train_cfg.get("loss_params", {}),
        loss_weights=train_cfg.get("loss_weights"),
        ignore_label=train_cfg.get("ignore_label", -1),
    )

    model.fit(
        train_data=train_loader,
        valid_data=valid_loader,
        metrics=train_cfg.get("metrics"),
        epochs=train_cfg.get("epochs", 1),
        batch_size=batch_size,
        shuffle=train_cfg.get("shuffle", True),
        num_workers=dataloader_cfg.get("num_workers", 0),
        user_id_column=id_column,
        use_tensorboard=False,
        use_wandb=train_cfg.get("use_wandb", False),
        use_swanlab=train_cfg.get("use_swanlab", False),
        wandb_api=train_cfg.get("wandb_api"),
        swanlab_api=train_cfg.get("swanlab_api"),
        wandb_kwargs=train_cfg.get("wandb_kwargs"),
        swanlab_kwargs=train_cfg.get("swanlab_kwargs"),
        log_interval=train_cfg.get("log_interval", 1),
        note=train_cfg.get("note"),
    )

    export_cfg = cfg.get("export_onnx", {})
    export_enabled = bool(export_cfg.get("enable", False))

    if export_enabled:
        log_cli_section("ONNX Export")

        onnx_best_path = Path(model.best_path).with_suffix(".onnx")
        onnx_ckpt_path = Path(model.checkpoint_path).with_suffix(".onnx")
        onnx_batch_size = export_cfg.get("batch_size", 1)
        log_kv_lines(
            [
                ("ONNX best path", onnx_best_path),
                ("ONNX checkpoint path", onnx_ckpt_path),
                ("Batch size", onnx_batch_size),
            ]
        )
        model.export_onnx(
            save_path=onnx_best_path,
            batch_size=onnx_batch_size,
        )
        model.export_onnx(
            save_path=onnx_ckpt_path,
            batch_size=onnx_batch_size,
        )


def predict_model(predict_config_path: str) -> None:
    """
    Run prediction using a trained model and configuration file.
    """
    config_file = Path(predict_config_path)
    config_dir = config_file.resolve().parent
    cfg = read_yaml(config_file)

    # Session / logging
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
    log_cli_section("CLI")
    log_kv_lines(
        [
            ("Mode", "predict"),
            ("Version", get_nextrec_version()),
            ("Session ID", session_id),
            ("Checkpoint", session_dir.resolve()),
            ("Config", config_file.resolve()),
            ("Command", " ".join(sys.argv)),
        ]
    )

    processor_path = Path(session_dir / "processor.pkl")

    predict_cfg = cfg.get("predict", {})
    device = predict_cfg.get("device", "cpu")

    # Model config
    model_cfg_path = resolve_path(cfg["model_config"], config_dir)

    model_cfg = read_yaml(model_cfg_path)
    model_cfg.setdefault("session_id", session_id)
    model_cfg.setdefault("params", {})

    log_cli_section("Config")
    log_kv_lines(
        [
            ("Predict config", config_file.resolve()),
            ("Model config", model_cfg_path),
            ("Processor", processor_path),
        ]
    )

    processor = DataProcessor.load(processor_path)

    # Checkpoint & features
    checkpoint_base = Path(session_dir)
    if checkpoint_base.is_dir():
        best_candidates = sorted(checkpoint_base.glob("*_best.pt"))
        candidates = sorted(checkpoint_base.glob("*.pt"))
        if best_candidates:
            model_file = best_candidates[-1]
        elif candidates:
            model_file = candidates[-1]
        else:
            raise FileNotFoundError(f"[NextRec CLI Error]: Unable to find model checkpoint: {checkpoint_base}")
        config_dir_for_features = checkpoint_base
    else:
        model_file = checkpoint_base.with_suffix(".pt") if checkpoint_base.suffix == "" else checkpoint_base
        config_dir_for_features = model_file.parent

    features_config_path = config_dir_for_features / "features_config.pkl"
    if not features_config_path.exists():
        raise FileNotFoundError(f"[NextRec CLI Error]: Unable to find features_config.pkl: {features_config_path}")
    with open(features_config_path, "rb") as f:
        features_config = pickle.load(f)

    all_features = features_config.get("all_features", [])
    target_cols = features_config.get("target", [])
    # Read id_columns from saved config.
    id_columns = features_config.get("id_columns", [])

    dense_features = [f for f in all_features if isinstance(f, DenseFeature)]
    sparse_features = [f for f in all_features if isinstance(f, SparseFeature)]
    sequence_features = [f for f in all_features if isinstance(f, SequenceFeature)]

    # Build model
    model = build_model_instance(
        model_cfg=model_cfg,
        model_cfg_path=model_cfg_path,
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        target=target_cols,
        id_columns=id_columns,
        device=device,
    )

    model.load_model(model_file, map_location=device, verbose=True)

    # Load id_columns override from predict config
    input_id_columns = predict_cfg.get("id_column")
    effective_id_columns = to_list(input_id_columns) if input_id_columns is not None else (model.id_columns or [])
    if input_id_columns is not None:
        model.id_columns = effective_id_columns

    log_cli_section("Features")
    log_kv_lines(
        [
            ("Dense features", len(dense_features)),
            ("Sparse features", len(sparse_features)),
            ("Sequence features", len(sequence_features)),
            ("Targets", len(target_cols)),
            ("ID columns", len(effective_id_columns)),
        ]
    )

    # ONNX options
    log_cli_section("Model")
    use_onnx = bool(predict_cfg.get("use_onnx"))
    onnx_path = predict_cfg.get("onnx_path")
    if onnx_path:
        onnx_path = resolve_path(onnx_path, config_dir)
    if use_onnx and onnx_path is None:
        search_dir = checkpoint_base if checkpoint_base.is_dir() else checkpoint_base.parent
        best_candidates = sorted(search_dir.glob("*_best.onnx"))
        if best_candidates:
            onnx_path = best_candidates[-1]
        else:
            candidates = sorted(search_dir.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"[NextRec CLI Error]: Unable to find ONNX model in {search_dir}")
            onnx_path = candidates[-1]

    log_kv_lines(
        [
            ("Model", model.__class__.__name__),
            ("Checkpoint", model_file),
            ("Device", device),
            ("Use ONNX", use_onnx),
            ("ONNX path", onnx_path if use_onnx else "disabled"),
        ]
    )

    # Data & parallelism
    data_path = resolve_path(predict_cfg["data_path"], config_dir)
    streaming = bool(predict_cfg.get("streaming", True))
    chunk_size = int(predict_cfg.get("chunk_size", 20000))
    batch_size = int(predict_cfg.get("batch_size", 512))
    num_workers_cfg = int(predict_cfg.get("num_workers", 0))
    prefetch_factor = predict_cfg.get("prefetch_factor")
    data_format = predict_cfg.get("source_data_format", predict_cfg.get("data_format", "auto"))
    data_format_effective = data_format
    if data_format == "auto":
        _, data_format_effective = resolve_file_paths(str(data_path))
    num_processes_cfg = predict_cfg.get("num_processes")
    num_processes_auto = None
    if num_processes_cfg is None:
        cpu_count = os.cpu_count() or 1
        try:
            load_1m = os.getloadavg()[0]
        except (AttributeError, OSError):
            load_1m = 0.0
        free_cores = max(0.0, float(cpu_count) - float(load_1m))
        suggested = int(math.floor(free_cores))
        if suggested < 1:
            suggested = 1
        if suggested > 5:
            suggested = 5
        num_processes_auto = suggested
        num_processes = suggested
        if not streaming:
            num_processes = 1
    else:
        num_processes = int(num_processes_cfg)
    profile_enabled = bool(predict_cfg.get("profile", False))
    effective_batch_size = chunk_size if streaming else batch_size
    effective_num_workers = num_workers_cfg

    # Set default thread limits for various libraries to avoid oversubscription.
    # in case multiple users are sharing the same machine and running multiple processes in parallel.
    # feel bad for my shitbox, jeje
    if streaming and num_processes > 1:
        _THREAD_DEFAULTS = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
        }
        for _key, _value in _THREAD_DEFAULTS.items():
            os.environ.setdefault(_key, _value)

        try:
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception as exc:
            logger.warning(f"[NextRec CLI Warning] Failed to set torch thread limits: {exc}")

    log_cli_section("Data")
    log_kv_lines(
        [
            ("Data path", data_path),
            ("Format", data_format_effective),
            ("Batch size", effective_batch_size),
            ("Chunk size", chunk_size),
            ("Streaming", streaming),
            ("Num workers", effective_num_workers),
            (
                "Num processes",
                (
                    f"{num_processes} (auto)"
                    if num_processes_auto is not None and num_processes_cfg is None
                    else num_processes
                ),
            ),
            ("Profile", profile_enabled),
        ]
    )

    df = None
    if streaming:
        row_count = count_rows(data_path, data_format_effective)
        if row_count is not None:
            logger.info(format_kv("Row count", row_count))
    else:
        df = read_table(data_path, data_format=data_format_effective)
        logger.info(format_kv("Row count", len(df)))

    if num_processes_auto is not None and num_processes_cfg is None:
        logger.info(format_kv("CPU cores", os.cpu_count() or 1))
        logger.info(format_kv("Load avg (1m)", f"{load_1m:.2f}"))
    if num_processes > 1 and num_workers_cfg != 0:
        logger.info("")
        logger.info("[NextRec CLI Info] Multi-process streaming enforces num_workers=0 for each shard.")
        effective_num_workers = 0
    logger.info("")
    profiler = StageTimer(enabled=profile_enabled) if profile_enabled else None

    rec_dataloader = RecDataLoader(
        dense_features=model.dense_features,
        sparse_features=model.sparse_features,
        sequence_features=model.sequence_features,
        target=None,
        id_columns=effective_id_columns,
        processor=processor,
    )

    if num_processes > 1:
        if not streaming:
            raise ValueError("[NextRec CLI Error] num_processes > 1 requires streaming=true.")
        pred_data = str(data_path)
    else:
        pred_source = str(data_path) if streaming else df
        pred_data = rec_dataloader.create_dataloader(
            data=pred_source,
            batch_size=1 if streaming else batch_size,
            shuffle=False,
            streaming=streaming,
            chunk_size=chunk_size,
            num_workers=effective_num_workers,
            prefetch_factor=prefetch_factor,
            profiler=profiler,
        )

    # Output path
    save_format = predict_cfg.get("save_data_format", predict_cfg.get("save_format", "csv"))
    pred_name = predict_cfg.get("name", "pred")
    pred_name_path = Path(pred_name)
    if pred_name_path.is_absolute():
        save_path = pred_name_path
        if save_path.suffix == "":
            save_path = save_path.with_suffix(f".{save_format}")
    else:
        save_path = checkpoint_base / "predictions" / f"{pred_name}.{save_format}"

    # Predict
    start = time.time()
    logger.info("")
    if use_onnx:
        result = model.predict_onnx(
            onnx_path=onnx_path,
            data=pred_data,
            batch_size=effective_batch_size,
            include_ids=bool(effective_id_columns),
            return_dataframe=False,
            save_path=str(save_path),
            save_format=save_format,
            num_workers=effective_num_workers,
            num_processes=num_processes,
            profiler=profiler,
            processor=processor,
        )
    else:
        result = model.predict(
            data=pred_data,
            batch_size=effective_batch_size,
            return_dataframe=False,
            save_path=str(save_path),
            save_format=save_format,
            num_workers=effective_num_workers,
            num_processes=num_processes,
            processor=processor,
            profiler=profiler,
        )
    duration = time.time() - start
    # When return_dataframe=False, result is the actual file path
    if isinstance(result, (str, Path)):
        output_path = Path(result)
    else:
        output_path = save_path
    # logger.info(f"Prediction completed, results saved to: {output_path}")
    logger.info(f"Total time: {duration:.2f} seconds")
    logger.info("")
    if profiler is not None and profiler.stats:
        log_cli_section("Profile")
        logger.info(format_kv("Wall time", f"{duration:.2f}s"))
        prof_total = sum(stat.total for _, stat in profiler.summary_rows())
        logger.info(format_kv("Profiled sum", f"{prof_total:.2f}s"))
        for name, stat in profiler.summary_rows():
            avg_ms = stat.avg * 1000
            logger.info(
                format_kv(
                    name,
                    f"total={stat.total:.2f}s | avg={avg_ms:.2f}ms | n={stat.count}",
                )
            )

    preview_rows = predict_cfg.get("preview_rows", 0)
    if preview_rows > 0:
        try:
            if save_format == "parquet" or output_path.suffix.lower() == ".parquet":
                preview = pd.read_parquet(output_path)
                if preview_rows:
                    preview = preview.head(preview_rows)
            else:
                try:
                    preview = pd.read_csv(
                        output_path,
                        nrows=preview_rows,
                        low_memory=False,
                        encoding_errors="replace",
                    )
                except TypeError:
                    preview = pd.read_csv(
                        output_path,
                        nrows=preview_rows,
                        low_memory=False,
                        encoding="latin-1",
                    )
            logger.info(f"Output preview:\n{preview}")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to read output preview: {exc}")


def evaluate_model(evaluate_config_path: str) -> None:
    """
    Run evaluation using a trained model and configuration file.
    """
    config_file = Path(evaluate_config_path)
    config_dir = config_file.resolve().parent
    cfg = read_yaml(config_file)
    evaluate_cfg = cfg.get("evaluate", {})

    # Session / logging
    session_dir = Path(cfg["checkpoint_path"])
    session_id = session_dir.name
    evaluate_name = str(evaluate_cfg.get("name", "default"))
    evaluate_dir = session_dir / "evaluate" / evaluate_name
    evaluate_dir.mkdir(parents=True, exist_ok=True)
    evaluate_log_path = evaluate_dir / "evaluate_log.txt"

    setup_logger(session_id=evaluate_dir.resolve(), log_name="evaluate_log.txt")
    log_cli_section("CLI")
    log_kv_lines(
        [
            ("Mode", "evaluate"),
            ("Version", get_nextrec_version()),
            ("Session ID", session_id),
            ("Checkpoint", session_dir.resolve()),
            ("Evaluate name", evaluate_name),
            ("Evaluate log", evaluate_log_path.resolve()),
            ("Config", config_file.resolve()),
            ("Command", " ".join(sys.argv)),
        ]
    )

    processor_path = Path(session_dir / "processor.pkl")
    device = evaluate_cfg.get("device", "cpu")

    # Model config
    if "model_config" in cfg:
        model_cfg_path = resolve_path(cfg["model_config"], config_dir)
    else:
        auto_model_cfg = session_dir / "model_config.yaml"
        if auto_model_cfg.exists():
            model_cfg_path = auto_model_cfg
        else:
            model_cfg_path = resolve_path("model_config.yaml", config_dir)

    model_cfg = read_yaml(model_cfg_path)
    model_cfg.setdefault("session_id", session_id)
    model_cfg.setdefault("params", {})

    log_cli_section("Config")
    log_kv_lines(
        [
            ("Evaluate config", config_file.resolve()),
            ("Model config", model_cfg_path),
            ("Processor", processor_path),
        ]
    )

    processor = DataProcessor.load(processor_path)

    # Checkpoint & features
    checkpoint_base = Path(session_dir)
    if checkpoint_base.is_dir():
        best_candidates = sorted(checkpoint_base.glob("*_best.pt"))
        candidates = sorted(checkpoint_base.glob("*.pt"))
        if best_candidates:
            model_file = best_candidates[-1]
        elif candidates:
            model_file = candidates[-1]
        else:
            raise FileNotFoundError(f"[NextRec CLI Error]: Unable to find model checkpoint: {checkpoint_base}")
        config_dir_for_features = checkpoint_base
    else:
        model_file = checkpoint_base.with_suffix(".pt") if checkpoint_base.suffix == "" else checkpoint_base
        config_dir_for_features = model_file.parent

    features_config_path = config_dir_for_features / "features_config.pkl"
    if not features_config_path.exists():
        raise FileNotFoundError(f"[NextRec CLI Error]: Unable to find features_config.pkl: {features_config_path}")
    with open(features_config_path, "rb") as f:
        features_config = pickle.load(f)

    all_features = features_config.get("all_features", [])
    target_cols = features_config.get("target", [])
    id_columns = features_config.get("id_columns", [])

    dense_features = [f for f in all_features if isinstance(f, DenseFeature)]
    sparse_features = [f for f in all_features if isinstance(f, SparseFeature)]
    sequence_features = [f for f in all_features if isinstance(f, SequenceFeature)]

    input_targets = evaluate_cfg.get("target")
    if input_targets is not None:
        input_targets = to_list(input_targets)
        if list(input_targets) != list(target_cols):
            logger.warning(
                "[NextRec CLI Warning] evaluate.target does not match trained targets; "
                "using targets from features_config.pkl."
            )

    model = build_model_instance(
        model_cfg=model_cfg,
        model_cfg_path=model_cfg_path,
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        target=target_cols,
        id_columns=id_columns,
        device=device,
    )

    model.load_model(model_file, map_location=device, verbose=True)

    input_id_columns = evaluate_cfg.get("id_column")
    effective_id_columns = to_list(input_id_columns) if input_id_columns is not None else (model.id_columns or [])
    if input_id_columns is not None:
        model.id_columns = effective_id_columns

    log_cli_section("Features")
    log_kv_lines(
        [
            ("Dense features", len(dense_features)),
            ("Sparse features", len(sparse_features)),
            ("Sequence features", len(sequence_features)),
            ("Targets", len(target_cols)),
            ("ID columns", len(effective_id_columns)),
        ]
    )

    log_cli_section("Model")
    log_kv_lines(
        [
            ("Model", model.__class__.__name__),
            ("Checkpoint", model_file),
            ("Device", device),
        ]
    )

    # Data
    data_path = resolve_path(evaluate_cfg["data_path"], config_dir)
    streaming = bool(evaluate_cfg.get("streaming", True))
    chunk_size = int(evaluate_cfg.get("chunk_size", 20000))
    batch_size = int(evaluate_cfg.get("batch_size", 512))
    num_workers = int(evaluate_cfg.get("num_workers", 0))
    prefetch_factor = evaluate_cfg.get("prefetch_factor")
    data_format = evaluate_cfg.get("source_data_format", evaluate_cfg.get("data_format", "auto"))
    data_format_effective = data_format
    if data_format == "auto":
        _, data_format_effective = resolve_file_paths(str(data_path))
    effective_batch_size = chunk_size if streaming else batch_size

    log_cli_section("Data")
    log_kv_lines(
        [
            ("Data path", data_path),
            ("Format", data_format_effective),
            ("Batch size", effective_batch_size),
            ("Chunk size", chunk_size),
            ("Streaming", streaming),
            ("Num workers", num_workers),
        ]
    )

    rec_dataloader = RecDataLoader(
        dense_features=model.dense_features,
        sparse_features=model.sparse_features,
        sequence_features=model.sequence_features,
        target=target_cols,
        id_columns=effective_id_columns,
        processor=processor,
    )
    data_loader = rec_dataloader.create_dataloader(
        data=str(data_path),
        batch_size=1 if streaming else batch_size,
        shuffle=False,
        streaming=streaming,
        chunk_size=chunk_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    eval_metrics_cfg = evaluate_cfg.get("metrics")
    if eval_metrics_cfg is None:
        raise ValueError("[NextRec CLI Error] evaluate.metrics must be specified in evaluate_config.yaml")

    metrics_list, task_specific_metrics, _ = configure_metrics(
        task=model.task,
        metrics=eval_metrics_cfg,
        target_names=model.target_columns,
    )
    model.metrics = metrics_list
    model.task_specific_metrics = task_specific_metrics

    thresholds_cfg = None
    confusion_cfg = evaluate_cfg.get("confusion_matrix", {}) or {}
    confusion_enabled = bool(confusion_cfg.get("enable", False))
    if "thresholds" in confusion_cfg:
        thresholds_cfg = confusion_cfg.get("thresholds")

    metrics_dict = model.evaluate(
        data_loader,
        metrics=None,
        num_workers=num_workers,
        thresholds=thresholds_cfg,
        show_data_summary=True,
        show_confusion_matrix=confusion_enabled,
    )
    if not metrics_dict:
        raise ValueError("[NextRec CLI Error] Not enough evaluation data to compute metrics.")


def main() -> None:
    """Parse CLI arguments and dispatch to train or predict mode."""

    # Increase file descriptor limit to avoid "Too many open files" error
    # when using DataLoader with multiple workers
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target_limit = 65535
        if soft < target_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(target_limit, hard), hard))
    except (ValueError, OSError):
        # If we can't set the limit, continue anyway
        pass

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)

    parser = argparse.ArgumentParser(
        description="NextRec: Training, Prediction, and Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  nextrec --mode=train --train_config=configs/train_config.yaml

  # Run prediction
  nextrec --mode=predict --predict_config=configs/predict_config.yaml

  # Run evaluation
  nextrec --mode=evaluate --evaluate_config=configs/evaluate_config.yaml
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "predict", "evaluate"],
        help="Running mode: train, predict, or evaluate",
    )
    parser.add_argument("--train_config", help="Training configuration file path")
    parser.add_argument("--predict_config", help="Prediction configuration file path")
    parser.add_argument("--evaluate_config", help="Evaluation configuration file path")
    args = parser.parse_args()

    if not args.mode:
        parser.error("[NextRec CLI Error] --mode is required (train|predict)")

    try:
        if args.mode == "train":
            config_path = args.train_config
            if not config_path:
                parser.error("[NextRec CLI Error] train mode requires --train_config")
            train_model(config_path)
        elif args.mode == "predict":
            config_path = args.predict_config
            if not config_path:
                parser.error("[NextRec CLI Error] predict mode requires --predict_config")
            predict_model(config_path)
        else:
            config_path = args.evaluate_config
            if not config_path:
                parser.error("[NextRec CLI Error] evaluate mode requires --evaluate_config")
            evaluate_model(config_path)
    except Exception:
        logging.getLogger(__name__).exception("[NextRec CLI Error] Unhandled exception")
        raise


if __name__ == "__main__":
    main()
