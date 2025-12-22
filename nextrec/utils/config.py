"""
Configuration utilities for NextRec

This module provides utilities for loading and processing configuration files,
including feature configuration, model configuration, and training configuration.

Date: create on 27/10/2025
Checkpoint: edit on 19/12/2025
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

from nextrec.utils.feature import normalize_to_list

if TYPE_CHECKING:
    from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
    from nextrec.data.preprocessor import DataProcessor


def resolve_path(
    path_str: str | Path | None = None, base_dir: Path | None = None
) -> Path:
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
        (
            candidate
            for candidate in candidates
            if candidate.exists() or candidate.parent.exists()
        ),
        candidates[0],
    )


def select_features(
    feature_cfg: Dict[str, Any], df_columns: List[str]
) -> Tuple[List[str], List[str], List[str]]:
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
        proc_cfg = dense_cfg.get(name, {}).get("processor_config", {}) or {}
        processor.add_numeric_feature(
            name,
            scaler=proc_cfg.get("scaler", "standard"),
            fill_na=proc_cfg.get("fill_na"),
        )

    for name in sparse_names:
        proc_cfg = sparse_cfg.get(name, {}).get("processor_config", {}) or {}
        processor.add_sparse_feature(
            name,
            encode_method=proc_cfg.get("encode_method", "hash"),
            hash_size=proc_cfg.get("hash_size") or proc_cfg.get("vocab_size"),
            fill_na=proc_cfg.get("fill_na", "<UNK>"),
        )

    for name in sequence_names:
        proc_cfg = sequence_cfg.get(name, {}).get("processor_config", {}) or {}
        processor.add_sequence_feature(
            name,
            encode_method=proc_cfg.get("encode_method", "hash"),
            hash_size=proc_cfg.get("hash_size") or proc_cfg.get("vocab_size"),
            max_len=proc_cfg.get("max_len", 50),
            pad_value=proc_cfg.get("pad_value", 0),
            truncate=proc_cfg.get("truncate", "post"),
            separator=proc_cfg.get("separator", ","),
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
        embed_cfg = dense_cfg.get(name, {}).get("embedding_config", {}) or {}
        dense_features.append(
            DenseFeature(
                name=name,
                embedding_dim=embed_cfg.get("embedding_dim"),
                input_dim=embed_cfg.get("input_dim", 1),
                use_embedding=embed_cfg.get("use_embedding", False),
            )
        )

    sparse_features: List[SparseFeature] = []
    for name in sparse_names:
        entry = sparse_cfg.get(name, {}) or {}
        proc_cfg = entry.get("processor_config", {}) or {}
        embed_cfg = entry.get("embedding_config", {}) or {}
        vocab_size = (
            embed_cfg.get("vocab_size")
            or proc_cfg.get("hash_size")
            or vocab_sizes.get(name, 0)
            or 1
        )
        sparse_features.append(
            SparseFeature(
                name=name,
                vocab_size=int(vocab_size),
                embedding_name=embed_cfg.get("embedding_name", name),
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
        proc_cfg = entry.get("processor_config", {}) or {}
        embed_cfg = entry.get("embedding_config", {}) or {}
        vocab_size = (
            embed_cfg.get("vocab_size")
            or proc_cfg.get("hash_size")
            or vocab_sizes.get(name, 0)
            or 1
        )
        sequence_features.append(
            SequenceFeature(
                name=name,
                vocab_size=int(vocab_size),
                max_len=embed_cfg.get("max_len") or proc_cfg.get("max_len", 50),
                embedding_name=embed_cfg.get("embedding_name", name),
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


def extract_feature_groups(
    feature_cfg: Dict[str, Any], df_columns: List[str]
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Extract and validate feature groups from feature configuration.

    Args:
        feature_cfg: Feature configuration dictionary
        df_columns: Available dataframe columns
    """
    feature_groups = feature_cfg.get("feature_groups") or {}
    if not feature_groups:
        return {}, []

    defined = (
        set((feature_cfg.get("dense") or {}).keys())
        | set((feature_cfg.get("sparse") or {}).keys())
        | set((feature_cfg.get("sequence") or {}).keys())
    )
    available_cols = set(df_columns)
    resolved: Dict[str, List[str]] = {}
    collected: List[str] = []

    for group_name, names in feature_groups.items():
        name_list = normalize_to_list(names)
        filtered = []
        missing_defined = [n for n in name_list if n not in defined]
        missing_cols = [n for n in name_list if n not in available_cols]

        if missing_defined:
            print(
                f"[Feature Config] feature_groups.{group_name} contains features not defined in dense/sparse/sequence: {missing_defined}"
            )

        for n in name_list:
            if n in available_cols:
                if n not in filtered:
                    filtered.append(n)
            else:
                if n not in missing_cols:
                    missing_cols.append(n)

        if missing_cols:
            print(
                f"[Feature Config] feature_groups.{group_name} missing data columns: {missing_cols}"
            )

        resolved[group_name] = filtered
        collected.extend(filtered)

    return resolved, collected


def load_model_class(model_cfg: Dict[str, Any], base_dir: Path) -> type:
    """
    Load model class from configuration.

    Args:
        model_cfg: Model configuration dictionary
        base_dir: Base directory for resolving relative paths
    """

    def camelize(name: str) -> str:
        """Convert snake_case or kebab-case to CamelCase."""
        return "".join(
            part.capitalize()
            for part in name.replace("_", " ").replace("-", " ").split()
        )

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
            ):
                return attr

        raise AttributeError(
            f"No BaseModel subclass found in {resolved}, please provide class_name"
        )

    # Case 2: Builtin model by short name
    if name and not module_name:
        from nextrec.basic.model import BaseModel

        candidates = [
            f"nextrec.models.{name.lower()}",
            f"nextrec.models.ranking.{name.lower()}",
            f"nextrec.models.match.{name.lower()}",
            f"nextrec.models.multi_task.{name.lower()}",
            f"nextrec.models.generative.{name.lower()}",
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
        device: Device string (e.g., 'cpu', 'cuda:0')
    """
    dense_map = {f.name: f for f in dense_features}
    sparse_map = {f.name: f for f in sparse_features}
    sequence_map = {f.name: f for f in sequence_features}
    feature_pool: Dict[str, Any] = {**dense_map, **sparse_map, **sequence_map}

    model_cls = load_model_class(model_cfg, model_cfg_path.parent)
    params_cfg = deepcopy(model_cfg.get("params") or {})
    feature_groups = params_cfg.pop("feature_groups", {}) or {}
    feature_bindings_cfg = (
        model_cfg.get("feature_bindings")
        or params_cfg.pop("feature_bindings", {})
        or {}
    )
    sig_params = inspect.signature(model_cls.__init__).parameters

    def _select(names: List[str] | None, pool: Dict[str, Any], desc: str) -> List[Any]:
        """Select features from pool by names."""
        if names is None:
            return list(pool.values())
        missing = [n for n in names if n not in feature_pool]
        if missing:
            raise ValueError(
                f"feature_groups.{desc} contains unknown features: {missing}"
            )
        return [feature_pool[n] for n in names]

    def accepts(name: str) -> bool:
        """Check if parameter name is accepted by model __init__."""
        return name in sig_params

    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig_params.values()
    )

    init_kwargs: Dict[str, Any] = dict(params_cfg)

    # Explicit bindings (model_config.feature_bindings) take priority
    for param_name, binding in feature_bindings_cfg.items():
        if param_name in init_kwargs:
            continue

        if isinstance(binding, (list, tuple, set)):
            if accepts(param_name) or accepts_var_kwargs:
                init_kwargs[param_name] = _select(
                    list(binding), feature_pool, f"feature_bindings.{param_name}"
                )
            continue

        if isinstance(binding, dict):
            direct_features = binding.get("features") or binding.get("feature_names")
            if direct_features and (accepts(param_name) or accepts_var_kwargs):
                init_kwargs[param_name] = _select(
                    normalize_to_list(direct_features),
                    feature_pool,
                    f"feature_bindings.{param_name}",
                )
                continue
            group_key = binding.get("group") or binding.get("group_key")
        else:
            group_key = binding

        if group_key not in feature_groups:
            print(
                f"[Feature Config] feature_bindings refers to unknown group '{group_key}', skipped"
            )
            continue

        if accepts(param_name) or accepts_var_kwargs:
            init_kwargs[param_name] = _select(
                feature_groups[group_key], feature_pool, str(group_key)
            )

    # Dynamic feature groups: any key in feature_groups that matches __init__ will be filled
    for group_key, names in feature_groups.items():
        if accepts(str(group_key)):
            init_kwargs.setdefault(
                str(group_key), _select(names, feature_pool, str(group_key))
            )

    # Generalized mapping: match params to feature_groups by normalized names
    def _normalize_group_key(key: str) -> str:
        """Normalize group key by removing common suffixes."""
        key = key.lower()
        for suffix in ("_features", "_feature", "_feats", "_feat", "_list", "_group"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key

    normalized_groups = {}
    for gk in feature_groups:
        norm = _normalize_group_key(gk)
        normalized_groups.setdefault(norm, gk)

    for param_name in sig_params:
        if param_name in ("self",) or param_name in init_kwargs:
            continue
        norm_param = _normalize_group_key(param_name)
        if norm_param in normalized_groups and (
            accepts(param_name) or accepts_var_kwargs
        ):
            group_key = normalized_groups[norm_param]
            init_kwargs[param_name] = _select(
                feature_groups[group_key], feature_pool, str(group_key)
            )

    # Feature wiring: prefer explicit groups when provided
    if accepts("dense_features"):
        init_kwargs.setdefault("dense_features", dense_features)
    if accepts("sparse_features"):
        init_kwargs.setdefault("sparse_features", sparse_features)
    if accepts("sequence_features"):
        init_kwargs.setdefault("sequence_features", sequence_features)

    if accepts("target"):
        init_kwargs.setdefault("target", target)
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
    Build (history, next_item) pairs per user using log_time ordering.
    """
    df_with_ids = df.copy().reset_index(drop=True)
    df_with_ids["log_time"] = pd.to_datetime(df_with_ids[log_time_column])
    df_with_ids["semantic_token"] = semantic_ids[:, 0].long() + 1  # 0 reserved for pad

    histories = []
    labels = []
    for _, user_df in df_with_ids.sort_values(["user_id", "log_time"]).groupby(
        "user_id"
    ):
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
