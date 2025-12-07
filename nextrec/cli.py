"""
Command-line interface for NextRec training and prediction.


NextRec supports a flexible training and prediction pipeline driven by configuration files.
After preparing the configuration YAML files for training and prediction, users can run the
following script to execute the desired operations.

Examples:
    # Train a model
    nextrec --mode=train --train_config=tutorials/iflytek/scripts/masknet/train_config.yaml

    # Run prediction
    nextrec --mode=predict --predict_config=tutorials/iflytek/scripts/masknet/predict_config.yaml

Date: create on 06/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.data.data_utils import split_dict_random
from nextrec.data.dataloader import RecDataLoader
from nextrec.data.preprocessor import DataProcessor
from nextrec.utils.config import (
    build_feature_objects,
    build_model_instance,
    extract_feature_groups,
    register_processor_features,
    resolve_path,
    select_features,
)
from nextrec.utils.feature import normalize_to_list
from nextrec.utils.file import (
    iter_file_chunks,
    read_table,
    read_yaml,
    resolve_file_paths,
)
from nextrec.basic.loggers import setup_logger

logger = logging.getLogger(__name__)


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
    config_dir = config_file.resolve().parent
    cfg = read_yaml(config_file)

    # read session configuration
    session_cfg = cfg.get("session", {}) or {}
    session_id = session_cfg.get("id", "nextrec_cli_session")
    artifact_root = Path(session_cfg.get("artifact_root", "nextrec_logs"))
    session_dir = artifact_root / session_id
    setup_logger(session_id=session_id)

    processor_path = session_dir / "processor.pkl"
    processor_path = Path(processor_path)
    processor_path.parent.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {}) or {}
    dataloader_cfg = cfg.get("dataloader", {}) or {}
    streaming = bool(data_cfg.get("streaming", False))
    dataloader_chunk_size = dataloader_cfg.get("chunk_size", 20000)

    # train data
    data_path = resolve_path(data_cfg["path"], config_dir)
    target = normalize_to_list(data_cfg["target"])
    file_paths: List[str] = []
    file_type: str | None = None
    streaming_train_files: List[str] | None = None
    streaming_valid_files: List[str] | None = None

    feature_cfg_path = resolve_path(
        cfg.get("feature_config", "feature_config.yaml"), config_dir
    )
    model_cfg_path = resolve_path(
        cfg.get("model_config", "model_config.yaml"), config_dir
    )

    feature_cfg = read_yaml(feature_cfg_path)
    model_cfg = read_yaml(model_cfg_path)

    if streaming:
        file_paths, file_type = resolve_file_paths(str(data_path))
        first_file = file_paths[0]
        first_chunk_size = max(1, min(dataloader_chunk_size, 1000))
        chunk_iter = iter_file_chunks(first_file, file_type, first_chunk_size)
        try:
            first_chunk = next(chunk_iter)
        except StopIteration as exc:
            raise ValueError(f"Data file is empty: {first_file}") from exc
        df_columns = list(first_chunk.columns)

    else:
        df = read_table(data_path, data_cfg.get("format"))
        df_columns = list(df.columns)

    # for some models have independent feature groups, we need to extract them here
    feature_groups, grouped_columns = extract_feature_groups(feature_cfg, df_columns)
    if feature_groups:
        model_cfg.setdefault("params", {})
        model_cfg["params"].setdefault("feature_groups", feature_groups)

    dense_names, sparse_names, sequence_names = select_features(feature_cfg, df_columns)
    used_columns = (
        dense_names + sparse_names + sequence_names + grouped_columns + target
    )

    # keep order but drop duplicates
    seen = set()
    unique_used_columns = []
    for col in used_columns:
        if col not in seen:
            unique_used_columns.append(col)
            seen.add(col)

    processor = DataProcessor()
    register_processor_features(
        processor, feature_cfg, dense_names, sparse_names, sequence_names
    )

    if streaming:
        processor.fit(str(data_path), chunk_size=dataloader_chunk_size)
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

    # Check if validation dataset path is specified
    val_data_path = data_cfg.get("val_path") or data_cfg.get("valid_path")
    if streaming:
        if not file_paths:
            file_paths, file_type = resolve_file_paths(str(data_path))
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
                    "[NextRec CLI Error] Must provide val_path or increase the number of data files. At least 2 files are required for streaming validation split."
                )
            val_count = max(1, int(round(total_files * ratio)))
            if val_count >= total_files:
                val_count = total_files - 1
            streaming_valid_files = file_paths[-val_count:]
            streaming_train_files = file_paths[:-val_count]
            logger.info(
                "使用 valid_ratio=%.3f 切分文件: 训练 %d 个文件, 验证 %d 个文件",
                ratio,
                len(streaming_train_files),
                len(streaming_valid_files),
            )
    train_data: Dict[str, Any]
    valid_data: Dict[str, Any] | None

    if val_data_path and not streaming:
        # Use specified validation dataset path
        logger.info("使用指定的验证集路径: %s", val_data_path)
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
        logger.info("训练集样本数: %s, 验证集样本数: %s", train_size, valid_size)
    elif streaming:
        train_data = None  # type: ignore[assignment]
        valid_data = None
        if not val_data_path and not streaming_valid_files:
            logger.info(
                "流式训练模式，未指定验证集路径且未配置 valid_ratio，跳过验证集创建"
            )
    else:
        # Split data using valid_ratio
        logger.info("使用 valid_ratio 切分数据: %s", data_cfg.get("valid_ratio", 0.2))
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
        processor=processor if streaming else None,
    )
    if streaming:
        train_stream_source = streaming_train_files or file_paths
        train_loader = dataloader.create_dataloader(
            data=train_stream_source,
            batch_size=dataloader_cfg.get("train_batch_size", 512),
            shuffle=dataloader_cfg.get("train_shuffle", True),
            load_full=False,
            chunk_size=dataloader_chunk_size,
        )
        valid_loader = None
        if val_data_path:
            val_data_resolved = resolve_path(val_data_path, config_dir)
            valid_loader = dataloader.create_dataloader(
                data=str(val_data_resolved),
                batch_size=dataloader_cfg.get("valid_batch_size", 512),
                shuffle=dataloader_cfg.get("valid_shuffle", False),
                load_full=False,
                chunk_size=dataloader_chunk_size,
            )
        elif streaming_valid_files:
            valid_loader = dataloader.create_dataloader(
                data=streaming_valid_files,
                batch_size=dataloader_cfg.get("valid_batch_size", 512),
                shuffle=dataloader_cfg.get("valid_shuffle", False),
                load_full=False,
                chunk_size=dataloader_chunk_size,
            )
    else:
        train_loader = dataloader.create_dataloader(
            data=train_data,
            batch_size=dataloader_cfg.get("train_batch_size", 512),
            shuffle=dataloader_cfg.get("train_shuffle", True),
        )
        valid_loader = dataloader.create_dataloader(
            data=valid_data,
            batch_size=dataloader_cfg.get("valid_batch_size", 512),
            shuffle=dataloader_cfg.get("valid_shuffle", False),
        )

    model_cfg.setdefault("session_id", session_id)
    train_cfg = cfg.get("train", {}) or {}
    device = train_cfg.get("device", model_cfg.get("device", "cpu"))
    model = build_model_instance(
        model_cfg,
        model_cfg_path,
        dense_features,
        sparse_features,
        sequence_features,
        target,
        device,
    )

    model.compile(
        optimizer=train_cfg.get("optimizer", "adam"),
        optimizer_params=train_cfg.get("optimizer_params", {}),
        loss=train_cfg.get("loss", "focal"),
        loss_params=train_cfg.get("loss_params", {}),
    )

    model.fit(
        train_data=train_loader,
        valid_data=valid_loader,
        metrics=train_cfg.get("metrics", ["auc", "recall", "precision"]),
        epochs=train_cfg.get("epochs", 1),
        batch_size=train_cfg.get(
            "batch_size", dataloader_cfg.get("train_batch_size", 512)
        ),
        shuffle=train_cfg.get("shuffle", True),
    )


def predict_model(predict_config_path: str) -> None:
    """
    Run prediction using a trained model and configuration file.
    """
    config_file = Path(predict_config_path)
    config_dir = config_file.resolve().parent
    cfg = read_yaml(config_file)

    session_cfg = cfg.get("session", {}) or {}
    session_id = session_cfg.get("id", "masknet_tutorial")
    artifact_root = Path(session_cfg.get("artifact_root", "nextrec_logs"))
    session_dir = Path(cfg.get("checkpoint_path") or (artifact_root / session_id))
    setup_logger(session_id=session_id)

    processor_path = Path(session_dir / "processor.pkl")
    if not processor_path.exists():
        processor_path = session_dir / "processor" / "processor.pkl"

    predict_cfg = cfg.get("predict", {}) or {}
    model_cfg_path = resolve_path(
        cfg.get("model_config", "model_config.yaml"), config_dir
    )
    feature_cfg_path = resolve_path(
        cfg.get("feature_config", "feature_config.yaml"), config_dir
    )

    model_cfg = read_yaml(model_cfg_path)
    feature_cfg = read_yaml(feature_cfg_path)
    model_cfg.setdefault("session_id", session_id)
    feature_groups_raw = feature_cfg.get("feature_groups") or {}
    model_cfg.setdefault("params", {})

    # attach feature_groups in predict phase to avoid missing bindings
    model_cfg["params"]["feature_groups"] = feature_groups_raw

    processor = DataProcessor.load(processor_path)

    # Load checkpoint and ensure required parameters are passed
    checkpoint_base = Path(session_dir)
    if checkpoint_base.is_dir():
        candidates = sorted(checkpoint_base.glob("*.model"))
        if not candidates:
            raise FileNotFoundError(
                f"[NextRec CLI Error]: Unable to find model checkpoint: {checkpoint_base}"
            )
        model_file = candidates[-1]
        config_dir_for_features = checkpoint_base
    else:
        model_file = (
            checkpoint_base.with_suffix(".model")
            if checkpoint_base.suffix == ""
            else checkpoint_base
        )
        config_dir_for_features = model_file.parent

    features_config_path = config_dir_for_features / "features_config.pkl"
    if not features_config_path.exists():
        raise FileNotFoundError(
            f"[NextRec CLI Error]: Unable to find features_config.pkl: {features_config_path}"
        )
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
        target_cols = normalize_to_list(target_override)

    # Recompute feature_groups with available feature names to drive bindings
    feature_group_names = [f.name for f in all_features if hasattr(f, "name")]
    parsed_feature_groups, _ = extract_feature_groups(feature_cfg, feature_group_names)
    if parsed_feature_groups:
        model_cfg.setdefault("params", {})
        model_cfg["params"]["feature_groups"] = parsed_feature_groups

    model = build_model_instance(
        model_cfg=model_cfg,
        model_cfg_path=model_cfg_path,
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        target=target_cols,
        device=predict_cfg.get("device", "cpu"),
    )
    model.id_columns = id_columns
    model.load_model(
        model_file, map_location=predict_cfg.get("device", "cpu"), verbose=True
    )

    id_columns = []
    if predict_cfg.get("id_column"):
        id_columns = [predict_cfg["id_column"]]
        model.id_columns = id_columns

    rec_dataloader = RecDataLoader(
        dense_features=model.dense_features,
        sparse_features=model.sparse_features,
        sequence_features=model.sequence_features,
        target=None,
        id_columns=id_columns or model.id_columns,
        processor=processor,
    )

    data_path = resolve_path(predict_cfg["data_path"], config_dir)
    batch_size = predict_cfg.get("batch_size", 512)

    pred_loader = rec_dataloader.create_dataloader(
        data=str(data_path),
        batch_size=batch_size,
        shuffle=False,
        load_full=predict_cfg.get("load_full", False),
        chunk_size=predict_cfg.get("chunk_size", 20000),
    )

    output_path = resolve_path(predict_cfg["output_path"], config_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    model.predict(
        data=pred_loader,
        batch_size=batch_size,
        include_ids=bool(id_columns),
        return_dataframe=False,
        save_path=output_path,
        save_format=predict_cfg.get("save_format", "csv"),
    )
    duration = time.time() - start
    logger.info(f"Prediction completed, results saved to: {output_path}")
    logger.info(f"Total time: {duration:.2f} seconds")

    preview_rows = predict_cfg.get("preview_rows", 0)
    if preview_rows > 0:
        try:
            preview = pd.read_csv(output_path, nrows=preview_rows)
            logger.info(f"Output preview:\n{preview}")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to read output preview: {exc}")


def main() -> None:
    """Parse CLI arguments and dispatch to train or predict mode."""
    parser = argparse.ArgumentParser(
        description="NextRec: Training and Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  nextrec --mode=train --train_config=configs/train_config.yaml

  # Run prediction
  nextrec --mode=predict --predict_config=configs/predict_config.yaml
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="运行模式：train 或 predict",
    )
    parser.add_argument("--train_config", help="训练配置文件路径")
    parser.add_argument("--predict_config", help="预测配置文件路径")
    parser.add_argument(
        "--config",
        help="通用配置文件路径（已废弃，建议使用 --train_config 或 --predict_config）",
    )
    args = parser.parse_args()

    if args.mode == "train":
        config_path = args.train_config or args.config
        if not config_path:
            parser.error("train 模式需要提供 --train_config")
        train_model(config_path)
    else:
        config_path = args.predict_config or args.config
        if not config_path:
            parser.error("predict 模式需要提供 --predict_config")
        predict_model(config_path)


if __name__ == "__main__":
    main()
