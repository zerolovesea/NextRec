"""
Configuration utilities for NextRec

This module provides utilities for loading and processing configuration files,
including feature configuration, model configuration, and training configuration.

Date: create on 27/10/2025
Checkpoint: edit on 13/03/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import pandas as pd
import torch

if TYPE_CHECKING:
    from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
    from nextrec.data.preprocessor import DataProcessor


_EMBEDDING_CFG_KEYS = {
    "vocab_size",
    "embedding_name",
    "embedding_dim",
    "padding_idx",
    "init_type",
    "init_params",
    "l1_reg",
    "l2_reg",
    "trainable",
    "combiner",
    "max_len",
    "proj_dim",
    "input_dim",
    "use_projection",
    "projection_type",
    "mlp_hidden_dims",
    "mlp_activation",
    "mlp_dropout",
}


def _normalize_processor_config(entry: Dict[str, Any], feature_type: str) -> List[Dict[str, Any]]:
    raw_cfg = entry.get("processor_config")

    if isinstance(raw_cfg, list):
        cfg_list = [cfg for cfg in raw_cfg if isinstance(cfg, dict)]
    elif isinstance(raw_cfg, dict):
        cfg_list = [raw_cfg]
    else:
        cfg_list = []

    if cfg_list:
        return cfg_list

    if feature_type == "dense":
        return [{"scaler": "standard"}]
    if feature_type == "sparse":
        return [{"encode_method": "label"}]
    if feature_type == "sequence":
        return [{"encode_method": "label"}]
    return [{}]


def _processor_method_name(feature_type: str, proc_cfg: Dict[str, Any]) -> str:
    if feature_type == "dense":
        scaler = proc_cfg.get("scaler")
        if isinstance(scaler, list):
            names = [str(v).strip().lower() for v in scaler if str(v).strip()]
            return "_".join(names) if names else "none"
        return str(scaler or "none")
    return str(proc_cfg.get("encode_method") or "hash")


def _processor_output_name(source_name: str, feature_type: str, proc_cfg: Dict[str, Any]) -> str:
    method_name = _processor_method_name(feature_type, proc_cfg)
    return f"{source_name}_{method_name}"


def _resolve_embedding_cfg(
    entry: Dict[str, Any],
    idx: int,
    output_name: str,
    feature_type: str,
    proc_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    raw_cfg = entry.get("embedding_config")
    if raw_cfg is None:
        raw_cfg = entry.get("embedding_config")

    if isinstance(raw_cfg, list):
        if not raw_cfg:
            return {}
        # Strict index-based pairing:
        # embedding_config[i] applies to processor_config[i].
        if idx >= len(raw_cfg):
            return {}
        item = raw_cfg[idx]
        return item if isinstance(item, dict) else {}

    if isinstance(raw_cfg, dict):
        # Plain embedding config (old format)
        if any(k in raw_cfg for k in _EMBEDDING_CFG_KEYS):
            return raw_cfg
        # Mapping format, e.g. by output_name / method / default
        method_name = _processor_method_name(feature_type, proc_cfg)
        for key in (output_name, method_name, str(idx), "default"):
            val = raw_cfg.get(key)
            if isinstance(val, dict):
                return val
        return {}

    return {}


def resolve_path(path_str: str | Path | None = None, base_dir: Path | None = None) -> Path:
    if path_str is None:
        return Path.cwd()
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    # Prefer resolving relative to current working directory when the path (or its parent)
    # already exists there; otherwise fall back to the config file's directory.
    candidates = (
        (Path.cwd() / path).resolve(),
        ((base_dir or Path.cwd()) / path).resolve(),
    )
    return next(
        (candidate for candidate in candidates if candidate.exists() or candidate.parent.exists()),
        candidates[0],
    )


def safe_value(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value]
    return str(value)


def select_features(feature_cfg: Dict[str, Any], df_columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    columns = set(df_columns)

    def pick(group: str) -> List[str]:
        cfg = feature_cfg.get(group, {}) or {}
        names = [name for name in cfg.keys() if name in columns]
        missing = [name for name in cfg.keys() if name not in columns]
        if missing:
            print(f"[Feature Config] skipped missing {group} columns: {missing}")
        return names

    dense_names = pick("dense")
    sparse_names = pick("sparse")
    sequence_names = pick("sequence")
    return dense_names, sparse_names, sequence_names


def register_processor_features(
    processor: "DataProcessor",
    feature_cfg: Dict[str, Any],
    dense_names: List[str],
    sparse_names: List[str],
    sequence_names: List[str],
) -> None:
    """
    Register features to DataProcessor based on feature configuration.

    Args:
        processor: DataProcessor instance
        feature_cfg: Feature configuration dictionary
        dense_names: List of dense feature names
        sparse_names: List of sparse feature names
        sequence_names: List of sequence feature names
    """
    dense_cfg = feature_cfg.get("dense", {}) or {}
    sparse_cfg = feature_cfg.get("sparse", {}) or {}
    sequence_cfg = feature_cfg.get("sequence", {}) or {}

    for name in dense_names:
        entry = dense_cfg.get(name, {}) or {}
        for proc_cfg in _normalize_processor_config(entry, "dense"):
            processor.add_numeric_feature(
                name,
                scaler=proc_cfg.get("scaler", "standard"),
                fill_na=proc_cfg.get("fill_na"),
            )

    for name in sparse_names:
        entry = sparse_cfg.get(name, {}) or {}
        for proc_cfg in _normalize_processor_config(entry, "sparse"):
            processor.add_sparse_feature(
                name,
                encode_method=proc_cfg.get("encode_method", "hash"),
                hash_size=proc_cfg.get("hash_size") or proc_cfg.get("vocab_size"),
                min_freq=proc_cfg.get("min_freq"),
                fill_na=proc_cfg.get("fill_na", "<UNK>"),
                filter_value=proc_cfg.get("filter_value", proc_cfg.get("filter_contains")),
                keep_value=proc_cfg.get("keep_value", proc_cfg.get("keep_contains")),
                match_mode=proc_cfg.get("match_mode", "exact"),
            )

    for name in sequence_names:
        entry = sequence_cfg.get(name, {}) or {}
        for proc_cfg in _normalize_processor_config(entry, "sequence"):
            processor.add_sequence_feature(
                name,
                encode_method=proc_cfg.get("encode_method", "hash"),
                hash_size=proc_cfg.get("hash_size") or proc_cfg.get("vocab_size"),
                min_freq=proc_cfg.get("min_freq"),
                max_len=proc_cfg.get("max_len", 50),
                pad_value=proc_cfg.get("pad_value", 0),
                truncate=proc_cfg.get("truncate", "post"),
                separator=proc_cfg.get("separator", ","),
                filter_value=proc_cfg.get("filter_value", proc_cfg.get("filter_contains")),
                keep_value=proc_cfg.get("keep_value", proc_cfg.get("keep_contains")),
                match_mode=proc_cfg.get("match_mode", "exact"),
            )


def build_feature_objects(
    processor: "DataProcessor",
    feature_cfg: Dict[str, Any],
    dense_names: List[str],
    sparse_names: List[str],
    sequence_names: List[str],
) -> Tuple[List["DenseFeature"], List["SparseFeature"], List["SequenceFeature"]]:
    """
    Build feature objects from processor and feature configuration.

    Args:
        processor: Fitted DataProcessor instance
        feature_cfg: Feature configuration dictionary
        dense_names: List of dense feature names
        sparse_names: List of sparse feature names
        sequence_names: List of sequence feature names
    """
    from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature

    dense_cfg = feature_cfg.get("dense", {}) or {}
    sparse_cfg = feature_cfg.get("sparse", {}) or {}
    sequence_cfg = feature_cfg.get("sequence", {}) or {}
    vocab_sizes = processor.get_vocab_sizes()

    dense_features: List[DenseFeature] = []
    for name in dense_names:
        entry = dense_cfg.get(name, {}) or {}
        proc_cfgs = _normalize_processor_config(entry, "dense")
        for idx, proc_cfg in enumerate(proc_cfgs):
            output_name = _processor_output_name(name, "dense", proc_cfg)
            feature_name = output_name
            embed_cfg = _resolve_embedding_cfg(entry, idx, output_name, "dense", proc_cfg)
            dense_features.append(
                DenseFeature(
                    name=feature_name,
                    proj_dim=embed_cfg.get("proj_dim"),
                    input_dim=embed_cfg.get("input_dim", 1),
                    use_projection=embed_cfg.get("use_projection", False),
                    projection_type=embed_cfg.get("projection_type", "linear"),
                    mlp_hidden_dims=embed_cfg.get("mlp_hidden_dims"),
                    mlp_activation=embed_cfg.get("mlp_activation", "relu"),
                    mlp_dropout=embed_cfg.get("mlp_dropout", 0.0),
                )
            )

    sparse_features: List[SparseFeature] = []
    for name in sparse_names:
        entry = sparse_cfg.get(name, {}) or {}
        proc_cfgs = _normalize_processor_config(entry, "sparse")
        for idx, proc_cfg in enumerate(proc_cfgs):
            output_name = _processor_output_name(name, "sparse", proc_cfg)
            feature_name = output_name
            embed_cfg = _resolve_embedding_cfg(entry, idx, output_name, "sparse", proc_cfg)
            vocab_size = (
                embed_cfg.get("vocab_size")
                or proc_cfg.get("hash_size")
                or vocab_sizes.get(feature_name)
                or vocab_sizes.get(output_name)
                or vocab_sizes.get(name, 0)
                or 1
            )
            sparse_features.append(
                SparseFeature(
                    name=feature_name,
                    vocab_size=int(vocab_size),
                    embedding_name=embed_cfg.get("embedding_name", feature_name),
                    embedding_dim=embed_cfg.get("embedding_dim"),
                    padding_idx=embed_cfg.get("padding_idx"),
                    init_type=embed_cfg.get("init_type", "xavier_uniform"),
                    init_params=embed_cfg.get("init_params"),
                    l1_reg=embed_cfg.get("l1_reg", 0.0),
                    l2_reg=embed_cfg.get("l2_reg", 1e-5),
                    trainable=embed_cfg.get("trainable", True),
                )
            )

    sequence_features: List[SequenceFeature] = []
    for name in sequence_names:
        entry = sequence_cfg.get(name, {}) or {}
        proc_cfgs = _normalize_processor_config(entry, "sequence")
        for idx, proc_cfg in enumerate(proc_cfgs):
            output_name = _processor_output_name(name, "sequence", proc_cfg)
            feature_name = output_name
            embed_cfg = _resolve_embedding_cfg(entry, idx, output_name, "sequence", proc_cfg)
            vocab_size = (
                embed_cfg.get("vocab_size")
                or proc_cfg.get("hash_size")
                or vocab_sizes.get(feature_name)
                or vocab_sizes.get(output_name)
                or vocab_sizes.get(name, 0)
                or 1
            )
            sequence_features.append(
                SequenceFeature(
                    name=feature_name,
                    vocab_size=int(vocab_size),
                    max_len=embed_cfg.get("max_len") or proc_cfg.get("max_len", 50),
                    embedding_name=embed_cfg.get("embedding_name", feature_name),
                    embedding_dim=embed_cfg.get("embedding_dim"),
                    padding_idx=embed_cfg.get("padding_idx"),
                    combiner=embed_cfg.get("combiner", "mean"),
                    init_type=embed_cfg.get("init_type", "xavier_uniform"),
                    init_params=embed_cfg.get("init_params"),
                    l1_reg=embed_cfg.get("l1_reg", 0.0),
                    l2_reg=embed_cfg.get("l2_reg", 1e-5),
                    trainable=embed_cfg.get("trainable", True),
                )
            )

    return dense_features, sparse_features, sequence_features


def load_model_class(model_cfg: Dict[str, Any], base_dir: Path) -> type:
    """
    Load model class from configuration.

    Args:
        model_cfg: Model configuration dictionary
        base_dir: Base directory for resolving relative paths
    """

    def camelize(name: str) -> str:
        """Convert snake_case or kebab-case to CamelCase."""
        return "".join(part.capitalize() for part in name.replace("_", " ").replace("-", " ").split())

    module_path = model_cfg.get("module_path")
    name = model_cfg.get("model") or model_cfg.get("name")
    module_name = model_cfg.get("module") or model_cfg.get("module_name")
    class_name = model_cfg.get("class_name")

    # Case 1: Custom file path
    if module_path:
        resolved = resolve_path(module_path, base_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"Custom model file not found: {resolved}")

        spec = importlib.util.spec_from_file_location(resolved.stem, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load custom model file: {resolved}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if class_name and hasattr(module, class_name):
            return getattr(module, class_name)

        # Auto-pick first BaseModel subclass
        from nextrec.basic.model import BaseModel

        for attr in module.__dict__.values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseModel)
                and attr is not BaseModel
                and attr.__module__ == module.__name__
            ):
                return attr

        raise AttributeError(f"No BaseModel subclass found in {resolved}, please provide class_name")

    # Case 2: Builtin model by short name
    if name and not module_name:
        from nextrec.basic.model import BaseModel

        candidates = [
            f"nextrec.models.{name.lower()}",
            f"nextrec.models.ranking.{name.lower()}",
            f"nextrec.models.match.{name.lower()}",
            f"nextrec.models.multi_task.{name.lower()}",
            f"nextrec.models.generative.{name.lower()}",
            f"nextrec.models.tree_base.{name.lower()}",
        ]
        errors = []

        for mod in candidates:
            try:
                module = importlib.import_module(mod)
                cls_name = class_name or camelize(name)

                if hasattr(module, cls_name):
                    return getattr(module, cls_name)

                # Fallback: first BaseModel subclass
                for attr in module.__dict__.values():
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseModel)
                        and attr is not BaseModel
                        and attr.__module__ == module.__name__
                    ):
                        return attr

                errors.append(f"{mod} missing class {cls_name}")
            except Exception as exc:
                errors.append(f"{mod}: {exc}")

        raise ImportError(f"Unable to find model for model='{name}'. Tried: {errors}")

    # Case 3: Explicit module + class
    if module_name and class_name:
        module = importlib.import_module(module_name)
        if not hasattr(module, class_name):
            raise AttributeError(f"Class {class_name} not found in {module_name}")
        return getattr(module, class_name)

    raise ValueError(
        "model configuration must provide 'model' (builtin name), 'module_path' (custom path), or 'module'+'class_name'"
    )


def build_model_instance(
    model_cfg: Dict[str, Any],
    model_cfg_path: Path,
    dense_features: List["DenseFeature"],
    sparse_features: List["SparseFeature"],
    sequence_features: List["SequenceFeature"],
    target: List[str],
    id_columns: list[str] | str | None,
    device: str,
) -> Any:
    """
    Build model instance from configuration and feature objects.

    Args:
        model_cfg: Model configuration dictionary
        model_cfg_path: Path to model config file (for resolving relative paths)
        dense_features: List of dense feature objects
        sparse_features: List of sparse feature objects
        sequence_features: List of sequence feature objects
        target: List of target column names
        id_columns: Identifier column name(s) for GAUC or ID passthrough
        device: Device string (e.g., 'cpu', 'cuda:0')
    """
    model_cls = load_model_class(model_cfg, model_cfg_path.parent)
    params_cfg = deepcopy(model_cfg.get("params") or {})
    # Deprecated: remove auto feature binding configs from init kwargs.
    params_cfg.pop("feature_groups", None)
    params_cfg.pop("feature_bindings", None)
    sig_params = inspect.signature(model_cls.__init__).parameters

    def accepts(name: str) -> bool:
        """Check if parameter name is accepted by model __init__."""
        return name in sig_params

    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig_params.values())

    init_kwargs = dict(params_cfg)
    # Feature wiring is explicit only: pass full feature lists when the model accepts them.
    if accepts("dense_features"):
        init_kwargs.setdefault("dense_features", dense_features)
    if accepts("sparse_features"):
        init_kwargs.setdefault("sparse_features", sparse_features)
    if accepts("sequence_features"):
        init_kwargs.setdefault("sequence_features", sequence_features)

    if accepts("target"):
        init_kwargs.setdefault("target", target)
    if accepts("id_columns") or accepts_var_kwargs:
        init_kwargs.setdefault("id_columns", id_columns)
    if accepts("device"):
        init_kwargs.setdefault("device", device)

    # Pass session_id if model accepts it
    if "session_id" not in init_kwargs and model_cfg.get("session_id") is not None:
        if accepts("session_id") or accepts_var_kwargs:
            init_kwargs["session_id"] = model_cfg.get("session_id")

    return model_cls(**init_kwargs)


def build_user_histories(
    df: pd.DataFrame,
    semantic_ids: torch.Tensor,
    codebook_size: int,
    log_time_column: str = "log_time",
    seq_len: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build autoregressive generative-retrieval training pairs per user using
    log_time ordering.
    """
    df_with_ids = df.copy().reset_index(drop=True)
    df_with_ids["log_time"] = pd.to_datetime(df_with_ids[log_time_column])
    df_with_ids["semantic_token"] = semantic_ids[:, 0].long() + 1  # 0 reserved for pad

    histories = []
    labels = []
    for _, user_df in df_with_ids.sort_values(["user_id", "log_time"]).groupby("user_id"):
        tokens = torch.tensor(user_df["semantic_token"].tolist(), dtype=torch.long)
        for i in range(1, len(tokens)):
            start = max(0, i - seq_len)
            hist = tokens[start:i]
            if len(hist) < seq_len:
                pad = torch.zeros(seq_len - len(hist), dtype=torch.long)
                hist = torch.cat([pad, hist], dim=0)
            histories.append(hist)
            labels.append(tokens[i])

    history_tensor = torch.stack(histories) if histories else torch.zeros(0, seq_len)
    label_tensor = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0)
    vocab_size = codebook_size + 1  # +1 for padding token
    return history_tensor, label_tensor, vocab_size
