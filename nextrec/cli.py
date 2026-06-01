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
Checkpoint: edit on 19/04/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

import os
import argparse
import json
import logging
import math
import pickle
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.loggers import colorize, format_kv, setup_logger
from nextrec.basic.metrics import configure_metrics
from nextrec.data.dataloader import RecDataLoader
from nextrec.data.preprocessor import DataProcessor
from nextrec.utils.config import (
    build_feature_objects,
    build_model_instance,
    get_path,
    register_processor_features,
    select_feature_names,
)
from nextrec.utils.console import get_nextrec_version
from nextrec.utils.data import (
    count_rows,
    get_expand_factor,
    get_expand_columns,
    get_file_paths,
    iter_file_chunks,
    read_table,
    read_yaml,
    split_path_files,
)
from nextrec.utils.timing import StageTimer
from nextrec.utils.torch_utils import to_list

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STUDIO_DEFAULT_PORT = 15123
DOCS_DEFAULT_PORT = 15124


def get_workspace_app_dir(app_name: str) -> Path:
    app_dirs = {
        "studio": PROJECT_ROOT / "nextrec_studio",
        "docs": PROJECT_ROOT / "docs",
    }
    try:
        app_dir = app_dirs[app_name]
    except KeyError as exc:
        raise ValueError(f"[NextRec CLI Error] Unsupported app: {app_name}") from exc

    if not app_dir.exists():
        raise FileNotFoundError(
            f"[NextRec CLI Error] Cannot find '{app_name}' app directory under project root: {app_dir}"
        )
    return app_dir


def run_frontend_app(app_name: str, script_name: str, host: str | None = None, port: int | None = None) -> None:
    app_dir = get_workspace_app_dir(app_name)
    package_json = app_dir / "package.json"
    if not package_json.exists():
        raise FileNotFoundError(f"[NextRec CLI Error] package.json not found for '{app_name}' app: {package_json}")

    npm_path = shutil.which("npm")
    if npm_path is None:
        raise FileNotFoundError("[NextRec CLI Error] npm is not installed or not available in PATH.")

    node_modules_dir = app_dir / "node_modules"
    if not node_modules_dir.exists():
        logger.info(f"Installing frontend dependencies for '{app_name}'...")
        subprocess.run([npm_path, "install"], cwd=app_dir, check=True)

    command = [npm_path, "run", script_name]
    extra_args = []
    if host:
        extra_args.extend(["--host", host])
    if port is not None:
        extra_args.extend(["--port", str(port)])
    if extra_args:
        command.extend(["--", *extra_args])

    logger.info(f"Starting {app_name} app from: {app_dir}")
    subprocess.run(command, cwd=app_dir, check=True)


def run_studio_app(host: str | None = None, port: int | None = None) -> None:
    run_frontend_app("studio", "dev", host=host, port=port or STUDIO_DEFAULT_PORT)


def run_docs_app(host: str | None = None, port: int | None = None) -> None:
    run_frontend_app("docs", "docs:dev", host=host, port=port or DOCS_DEFAULT_PORT)


def build_frontend_parser(command_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"nextrec {command_name}",
        description=f"Start the NextRec {command_name} development server.",
    )
    parser.add_argument("--host", help="Dev server host")
    parser.add_argument("--port", type=int, help="Dev server port")
    return parser


def log_cli_section(title: str) -> None:
    logger.info("")
    logger.info(colorize(f"[{title}]", color="bright_blue", bold=True))
    logger.info(colorize("-" * 80, color="bright_blue"))


def log_kv_lines(items: list[tuple[str, Any]]) -> None:
    for label, value in items:
        logger.info(format_kv(label, value))


def to_builtin_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            normalized_metrics[key] = value.item()
        else:
            normalized_metrics[key] = value
    return normalized_metrics


def train_model(train_config_path: str) -> None:
    """
    Train a NextRec model using the provided configuration file.

    configuration file must specify the below sections:
        - session: Session settings including id and artifact root
        - data: Data settings including path, format, target, validation split,
          optional split_stratify_by and split_group_by controls
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

    feature_cfg_path = get_path(cfg.get("feature_config", "feature_config.yaml"), config_dir)
    model_cfg_path = get_path(cfg.get("model_config", "model_config.yaml"), config_dir)

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
    batch_size = int(
        dataloader_cfg.get(
            "batch_size",
            dataloader_cfg.get("train_batch_size", 512),
        )
    )

    # train data basics
    data_path = get_path(data_cfg["path"], config_dir)
    target = to_list(data_cfg["target"])
    val_data_path = data_cfg.get("valid_path")
    split_stratify_by = data_cfg.get("split_stratify_by")
    split_group_by = data_cfg.get("split_group_by")
    valid_group_by = to_list(train_cfg.get("valid_group_by"))

    feature_cfg = read_yaml(feature_cfg_path)
    model_cfg = read_yaml(model_cfg_path)

    # Extract group_id from data config for grouped metrics such as GAUC and ranking@K.
    group_id = data_cfg.get("group_id")
    key_columns = [group_id] if group_id else []
    loader_key_columns = list(dict.fromkeys([*key_columns, *valid_group_by]))

    log_cli_section("Data")
    log_kv_lines(
        [
            ("Data path", data_path),
            ("Format", data_cfg.get("format", "auto")),
            ("Streaming", streaming),
            ("Target", target),
            ("Group ID", group_id or "(not set)"),
            ("Valid group by", ", ".join(valid_group_by) if valid_group_by else "disabled"),
            ("Split stratify by", split_stratify_by or "(disabled)"),
            ("Split group by", split_group_by or "(disabled)"),
        ]
    )
    if data_cfg.get("valid_ratio") is not None:
        logger.info(format_kv("Valid ratio", data_cfg.get("valid_ratio")))
    if val_data_path:
        logger.info(
            format_kv(
                "Validation path",
                get_path(val_data_path, config_dir),
            )
        )

    file_paths = []
    file_type = None
    streaming_train_files = None
    streaming_valid_files = None

    if streaming:
        file_paths, file_type = get_file_paths(str(data_path))
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

        streaming_train_files = file_paths
        if val_data_path:
            streaming_valid_files = None
        elif data_cfg.get("valid_ratio") is not None:
            ratio = float(data_cfg["valid_ratio"])
            streaming_train_files, streaming_valid_files = split_path_files(file_paths, ratio)
            logger.info(
                f"Split files for streaming training and validation using valid_ratio={ratio:.3f}: "
                f"training {len(streaming_train_files)} files, validation {len(streaming_valid_files)} files"
            )
        else:
            streaming_valid_files = None
    else:
        df = read_table(data_path, data_cfg.get("format"))
        logger.info(format_kv("Rows", len(df)))
        logger.info(format_kv("Columns", len(df.columns)))
        df_columns = list(df.columns)

    if streaming and (split_stratify_by or split_group_by):
        raise ValueError(
            "[NextRec CLI Error] split_stratify_by and split_group_by are not supported in streaming mode. "
            "Streaming validation currently splits by files only."
        )

    dense_names, sparse_names, sequence_names = select_feature_names(feature_cfg, df_columns)

    split_columns = [col for col in [split_stratify_by, split_group_by] if col]
    active_split_columns = split_columns if (not streaming and not val_data_path) else []
    used_columns = dense_names + sparse_names + sequence_names + target + loader_key_columns + active_split_columns

    # keep order but drop duplicates
    unique_used_columns = list(dict.fromkeys(used_columns))

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
        df = None  # type: ignore[assignment]
    else:
        df = df[unique_used_columns]
        processor.fit(df)

    processor.save(processor_path)
    dense_features, sparse_features, sequence_features = build_feature_objects(
        processor,
        feature_cfg,
        dense_names,
        sparse_names,
        sequence_names,
    )

    fit_train_data: Any
    fit_valid_data: Any
    fit_valid_split = None

    if val_data_path and not streaming:
        logger.info(f"Validation using specified validation dataset path: {val_data_path}")
        val_data_resolved = get_path(val_data_path, config_dir)
        val_df = read_table(val_data_resolved, data_cfg.get("format"))
        fit_train_data = df
        fit_valid_data = val_df[unique_used_columns]
        train_size = len(df)
        valid_size = len(fit_valid_data)
        logger.info(f"Sample count - Training set: {train_size}, Validation set: {valid_size}")
    elif streaming:
        if not val_data_path and not streaming_valid_files:
            logger.info(
                "Streaming training mode: No validation dataset path specified and valid_ratio not configured, skipping validation dataset creation"
            )
        fit_train_data = streaming_train_files or file_paths
        fit_valid_data = str(get_path(val_data_path, config_dir)) if val_data_path else streaming_valid_files
    else:
        fit_train_data = df
        fit_valid_data = None
        fit_valid_split = data_cfg.get("valid_ratio", 0.2)
        logger.info(
            f"Validation will be split inside model.fit using valid_ratio={fit_valid_split}, "
            f"split_stratify_by={split_stratify_by}, split_group_by={split_group_by}"
        )

    model_cfg.setdefault("session_id", str(session_dir.resolve()))
    device = train_cfg.get("device", model_cfg.get("device", "cpu"))
    model = build_model_instance(
        model_cfg,
        model_cfg_path,
        dense_features,
        sparse_features,
        sequence_features,
        target,
        loader_key_columns,
        device,
    )

    log_cli_section("Model")
    log_kv_lines(
        [
            ("Model", model.__class__.__name__),
            ("Device", device),
            ("Runtime device", model.device),
            ("Session ID", session_id),
        ]
    )

    model.compile(
        optimizer=train_cfg.get("optimizer", "adam"),
        optimizer_params=train_cfg.get("optimizer_params", {}),
        scheduler=train_cfg.get("scheduler"),
        scheduler_params=train_cfg.get("scheduler_params", {}),
        warmup=train_cfg.get("warmup"),
        loss=train_cfg.get("loss"),
        loss_params=train_cfg.get("loss_params", {}),
        loss_weights=train_cfg.get("loss_weights"),
        ignore_label=train_cfg.get("ignore_label", -1),
    )

    model.fit(
        train_data=fit_train_data,
        valid_data=fit_valid_data,
        metrics=train_cfg.get("metrics"),
        epochs=train_cfg.get("epochs", 1),
        batch_size=batch_size,
        shuffle=train_cfg.get("shuffle", True),
        streaming=streaming,
        chunk_size=dataloader_chunk_size,
        num_workers=dataloader_cfg.get("num_workers", 0),
        prefetch_factor=dataloader_cfg.get("prefetch_factor"),
        processor=processor,
        group_id=group_id,
        valid_split=fit_valid_split,
        split_stratify_by=split_stratify_by,
        split_group_by=split_group_by,
        early_stop_patience=train_cfg.get("early_stop_patience", 20),
        early_stop_monitor_task=train_cfg.get("early_stop_monitor_task"),
        valid_group_by=valid_group_by or None,
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
        model.export(
            format="onnx",
            save_path=onnx_best_path,
            batch_size=onnx_batch_size,
        )
        model.export(
            format="onnx",
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
    model_cfg_path = get_path(cfg["model_config"], config_dir)

    model_cfg = read_yaml(model_cfg_path)
    model_cfg.setdefault("session_id", str(session_dir.resolve()))
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
        model_file = (best_candidates or candidates)[-1] if (best_candidates or candidates) else None
        if model_file is None:
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
    key_columns = features_config.get("key_columns", [])

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
        key_columns=key_columns,
        device=device,
    )

    model.load_model(model_file, map_location=device)

    input_key_columns = predict_cfg.get("key_column")
    effective_key_columns = to_list(input_key_columns) if input_key_columns is not None else (model.key_columns or [])
    expand = get_expand_columns(predict_cfg.get("expand"))
    output_key_columns = list(dict.fromkeys([*effective_key_columns, *expand.keys()]))

    log_cli_section("Features")
    log_kv_lines(
        [
            ("Dense features", len(dense_features)),
            ("Sparse features", len(sparse_features)),
            ("Sequence features", len(sequence_features)),
            ("Targets", len(target_cols)),
            ("Key columns", len(output_key_columns)),
        ]
    )

    # ONNX options
    log_cli_section("Model")
    use_onnx = bool(predict_cfg.get("use_onnx"))
    onnx_path = predict_cfg.get("onnx_path")
    if onnx_path:
        onnx_path = get_path(onnx_path, config_dir)
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
    model.set_inference_backend(onnx_path if use_onnx else None)

    log_kv_lines(
        [
            ("Model", model.__class__.__name__),
            ("Checkpoint", model_file),
            ("Device", device),
            ("Use ONNX", use_onnx),
            ("Inference model", onnx_path if use_onnx else "current PyTorch model"),
        ]
    )

    # Data & parallelism
    data_path = get_path(predict_cfg["data_path"], config_dir)
    streaming = bool(predict_cfg.get("streaming", True))
    chunk_size = int(predict_cfg.get("chunk_size", 20000))
    batch_size = int(predict_cfg.get("batch_size", 512))
    num_workers_cfg = int(predict_cfg.get("num_workers", 0))
    prefetch_factor = predict_cfg.get("prefetch_factor")
    data_format = predict_cfg.get("source_data_format", predict_cfg.get("data_format", "auto"))
    data_format_effective = get_file_paths(str(data_path))[1] if data_format == "auto" else data_format
    num_processes_cfg = predict_cfg.get("num_processes")
    num_processes_auto = None
    if num_processes_cfg is None:
        cpu_count = os.cpu_count() or 1
        try:
            load_1m = os.getloadavg()[0]
        except (AttributeError, OSError):
            load_1m = 0.0
        free_cores = max(0.0, float(cpu_count) - float(load_1m))
        suggested = min(5, max(1, int(math.floor(free_cores))))
        num_processes_auto = suggested
        num_processes = suggested if streaming else 1
    else:
        num_processes = int(num_processes_cfg)
    effective_batch_size = chunk_size if streaming else batch_size
    effective_num_workers = num_workers_cfg

    # Set default thread limits for libraries to avoid oversubscription in multi-process inference.
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
            ("Profile", "enabled"),
            ("Expand columns", ", ".join(expand.keys()) if expand else "disabled"),
            ("Expand factor", get_expand_factor(expand)),
        ]
    )

    df = None
    if streaming:
        row_count = count_rows(data_path, data_format_effective)
        if row_count is not None:
            logger.info(format_kv("Row count", row_count))
            if expand:
                logger.info(format_kv("Expanded rows (est.)", row_count * get_expand_factor(expand)))
    else:
        df = read_table(data_path, data_format=data_format_effective)
        logger.info(format_kv("Row count", len(df)))
        if expand:
            logger.info(format_kv("Expanded rows (est.)", len(df) * get_expand_factor(expand)))

    if num_processes_auto is not None and num_processes_cfg is None:
        logger.info(format_kv("CPU cores", os.cpu_count() or 1))
        logger.info(format_kv("Load avg (1m)", f"{load_1m:.2f}"))
    if num_processes > 1 and num_workers_cfg != 0:
        logger.info("")
        logger.info("[NextRec CLI Info] Multi-process streaming enforces num_workers=0 for each shard.")
        effective_num_workers = 0
    logger.info("")
    profiler = StageTimer(enabled=True)
    model.profiler = profiler

    rec_dataloader = RecDataLoader(
        dense_features=model.dense_features,
        sparse_features=model.sparse_features,
        sequence_features=model.sequence_features,
        target=None,
        key_columns=output_key_columns,
        processor=processor,
        expand=expand,
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

    save_format = predict_cfg.get("save_data_format", predict_cfg.get("save_format", "csv"))
    pred_name = predict_cfg.get("name", "pred")
    pred_name_path = Path(pred_name)
    if pred_name_path.is_absolute():
        save_path = pred_name_path.with_suffix(f".{save_format}") if pred_name_path.suffix == "" else pred_name_path
    else:
        save_path = checkpoint_base / "predictions" / f"{pred_name}.{save_format}"

    start = time.time()
    logger.info("")
    result = model.predict(
        data=pred_data,
        batch_size=effective_batch_size,
        return_dataframe=False,
        save_path=str(save_path),
        save_format=save_format,
        num_workers=effective_num_workers,
        num_processes=num_processes,
        processor=processor,
        key_columns=output_key_columns,
        expand=expand,
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
    if profiler.stats:
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
        model_cfg_path = get_path(cfg["model_config"], config_dir)
    else:
        auto_model_cfg = session_dir / "model_config.yaml"
        if auto_model_cfg.exists():
            model_cfg_path = auto_model_cfg
        else:
            model_cfg_path = get_path("model_config.yaml", config_dir)

    model_cfg = read_yaml(model_cfg_path)
    model_cfg.setdefault("session_id", str(session_dir.resolve()))
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
        model_file = (best_candidates or candidates)[-1] if (best_candidates or candidates) else None
        if model_file is None:
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
    key_columns = features_config.get("key_columns", [])

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
        key_columns=key_columns,
        device=device,
    )

    model.load_model(model_file, map_location=device)

    input_key_columns = evaluate_cfg.get("key_column")
    effective_key_columns = to_list(input_key_columns) if input_key_columns is not None else (model.key_columns or [])
    group_id = evaluate_cfg.get("group_id")
    by_columns = to_list(evaluate_cfg.get("group_by"))
    loader_key_columns = list(dict.fromkeys([*effective_key_columns, *([group_id] if group_id else []), *by_columns]))

    log_cli_section("Features")
    log_kv_lines(
        [
            ("Dense features", len(dense_features)),
            ("Sparse features", len(sparse_features)),
            ("Sequence features", len(sequence_features)),
            ("Targets", len(target_cols)),
            ("Key columns", len(effective_key_columns)),
            ("Group ID", group_id or "(not set)"),
            ("Group by", ", ".join(by_columns) if by_columns else "disabled"),
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
    data_path = get_path(evaluate_cfg["data_path"], config_dir)
    streaming = bool(evaluate_cfg.get("streaming", True))
    chunk_size = int(evaluate_cfg.get("chunk_size", 20000))
    batch_size = int(evaluate_cfg.get("batch_size", 512))
    num_workers = int(evaluate_cfg.get("num_workers", 0))
    prefetch_factor = evaluate_cfg.get("prefetch_factor")
    data_format = evaluate_cfg.get("source_data_format", evaluate_cfg.get("data_format", "auto"))
    data_format_effective = get_file_paths(str(data_path))[1] if data_format == "auto" else data_format
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
            ("Group by", ", ".join(by_columns) if by_columns else "disabled"),
        ]
    )

    rec_dataloader = RecDataLoader(
        dense_features=model.dense_features,
        sparse_features=model.sparse_features,
        sequence_features=model.sequence_features,
        target=target_cols,
        key_columns=loader_key_columns,
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
        model_family=model.model_family,
        metrics=eval_metrics_cfg,
        target_names=model.target_columns,
    )
    model.metrics = metrics_list
    model.task_specific_metrics = task_specific_metrics

    confusion_cfg = evaluate_cfg.get("confusion_matrix", {}) or {}
    confusion_enabled = bool(confusion_cfg.get("enable", False))
    thresholds_cfg = confusion_cfg.get("thresholds")

    metrics_result = model.evaluate(
        data_loader,
        metrics=None,
        group_id=group_id,
        group_by=by_columns or None,
        num_workers=num_workers,
        thresholds=thresholds_cfg,
        show_data_summary=not by_columns,
        show_confusion_matrix=confusion_enabled and not by_columns,
    )
    metrics_dict = to_builtin_metrics(metrics_result.get("overall", {}))
    if by_columns:
        grouped_rows = [
            {key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items()}
            for row in metrics_result.get("grouped", [])
        ]
        (evaluate_dir / "metrics.json").write_text(
            json.dumps(metrics_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(grouped_rows).to_csv(evaluate_dir / "metrics_by.csv", index=False)
        (evaluate_dir / "metrics_by.json").write_text(
            json.dumps(grouped_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(format_kv("Grouped metrics", evaluate_dir / "metrics_by.csv"))
    if not metrics_dict:
        raise ValueError("[NextRec CLI Error] Not enough evaluation data to compute metrics.")

    success_flag_path = evaluate_dir / ".SUCCESS"
    success_flag_path.write_text("", encoding="utf-8")


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

    if len(sys.argv) > 1 and sys.argv[1] in {"studio", "docs"}:
        command_name = sys.argv[1]
        frontend_args = build_frontend_parser(command_name).parse_args(sys.argv[2:])
        try:
            if command_name == "studio":
                run_studio_app(host=frontend_args.host, port=frontend_args.port)
            else:
                run_docs_app(host=frontend_args.host, port=frontend_args.port)
        except Exception:
            logging.getLogger(__name__).exception("[NextRec CLI Error] Failed to start frontend app")
            raise
        return

    parser = argparse.ArgumentParser(
        description="NextRec: Training, Prediction, and Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  nextrec --mode=train --train_config=configs/train_config.yaml

  # Start NextRec Studio
  nextrec studio

  # Start docs site
  nextrec docs

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
        parser.error("[NextRec CLI Error] --mode is required (train|predict|evaluate)")

    try:
        config_path = (
            args.train_config
            if args.mode == "train"
            else args.predict_config if args.mode == "predict" else args.evaluate_config
        )
        if not config_path:
            parser.error(f"[NextRec CLI Error] {args.mode} mode requires --{args.mode}_config")
        if args.mode == "train":
            train_model(config_path)
        elif args.mode == "predict":
            predict_model(config_path)
        else:
            evaluate_model(config_path)
    except Exception:
        logging.getLogger(__name__).exception("[NextRec CLI Error] Unhandled exception")
        raise


if __name__ == "__main__":
    main()
