"""
ONNX utilities for NextRec.

Date: create on 25/01/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import numpy as np
import onnxruntime as ort

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.utils.torch_utils import to_numpy


class OnnxModelWrapper(torch.nn.Module):
    """Wrap a NextRec model to accept positional ONNX inputs."""

    def __init__(self, model: torch.nn.Module, feature_names: Sequence[str]):
        super().__init__()
        self.model = model
        self.feature_names = list(feature_names)

    def forward(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.feature_names):
            raise ValueError(
                "[OnnxWrapper Error] Number of inputs does not match feature names."
            )
        x = {name: tensor for name, tensor in zip(self.feature_names, inputs)}
        output = self.model(x)
        if isinstance(output, list):
            return tuple(output)
        return output


def create_dummy_inputs(
    features: Iterable[object],
    batch_size: int,
    device: torch.device,
    default_seq_len: int = 10,
    seq_len_map: dict[str, int] | None = None,
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for feature in features:
        if isinstance(feature, DenseFeature):
            input_dim = max(int(feature.input_dim), 1)
            tensors.append(
                torch.zeros((batch_size, input_dim), dtype=torch.float32, device=device)
            )
        elif isinstance(feature, SequenceFeature):
            seq_len = None
            if seq_len_map:
                seq_len = seq_len_map.get(feature.name)
            if seq_len is None:
                seq_len = (
                    feature.max_len if feature.max_len is not None else default_seq_len
                )
            seq_len = max(int(seq_len), 1)
            tensors.append(
                torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            )
        else:
            tensors.append(torch.zeros((batch_size,), dtype=torch.long, device=device))
    return tensors


def normalize_dense(feature: DenseFeature, array: object) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    expected = max(int(feature.input_dim), 1)
    if arr.shape[1] != expected:
        raise ValueError(
            f"[ONNX Input Error] Dense feature '{feature.name}' expects {expected} dims but got {arr.shape[1]}."
        )
    return arr


def normalize_sparse(feature: SparseFeature, array: object) -> np.ndarray:
    arr = np.asarray(array, dtype=np.int64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] != 1:
            raise ValueError(
                f"[ONNX Input Error] Sparse feature '{feature.name}' expects 1 dim but got {arr.shape}."
            )
        arr = arr.reshape(-1)
    return arr


def normalize_sequence(feature: SequenceFeature, array: object) -> np.ndarray:
    arr = np.asarray(array, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    max_len = feature.max_len if feature.max_len is not None else arr.shape[1]
    max_len = max(int(max_len), 1)
    if arr.shape[1] > max_len:
        arr = arr[:, :max_len]
    elif arr.shape[1] < max_len:
        pad_value = feature.padding_idx if feature.padding_idx is not None else 0
        pad_width = max_len - arr.shape[1]
        arr = np.pad(arr, ((0, 0), (0, pad_width)), constant_values=pad_value)
    return arr


def build_onnx_input_feed(
    features: Iterable[object],
    feature_batch: dict[str, object],
    input_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for feature in features:
        if input_names is not None and feature.name not in input_names:
            continue
        if feature.name not in feature_batch:
            raise KeyError(
                f"[ONNX Input Error] Feature '{feature.name}' missing from batch data."
            )
        value = to_numpy(feature_batch[feature.name])
        if isinstance(feature, DenseFeature):
            value = normalize_dense(feature, value)
        elif isinstance(feature, SequenceFeature):
            value = normalize_sequence(feature, value)
        else:
            value = normalize_sparse(feature, value)
        feed[feature.name] = value
    return feed


def pad_tensor(
    value: torch.Tensor, pad_rows: int, pad_value: int | float
) -> torch.Tensor:
    if pad_rows <= 0:
        return value
    pad_shape = (pad_rows, *value.shape[1:])
    pad = torch.full(pad_shape, pad_value, dtype=value.dtype, device=value.device)
    return torch.cat([value, pad], dim=0)


def pad_array(value: np.ndarray, pad_rows: int, pad_value: int | float) -> np.ndarray:
    if pad_rows <= 0:
        return value
    pad_shape = (pad_rows, *value.shape[1:])
    pad = np.full(pad_shape, pad_value, dtype=value.dtype)
    return np.concatenate([value, pad], axis=0)


def pad_onnx_inputs(
    features: Iterable[object],
    feature_batch: dict[str, object],
    target_batch: int,
) -> tuple[dict[str, object], int]:
    if target_batch <= 0:
        return feature_batch, 0
    padded: dict[str, object] = {}
    orig_batch = None
    for feature in features:
        if feature.name not in feature_batch:
            continue
        value = feature_batch[feature.name]
        if isinstance(value, torch.Tensor):
            batch = value.shape[0] if value.dim() > 0 else 1
        else:
            arr = np.asarray(value)
            batch = arr.shape[0] if arr.ndim > 0 else 1
        if orig_batch is None:
            orig_batch = int(batch)
        pad_rows = max(int(target_batch) - int(batch), 0)
        if isinstance(feature, DenseFeature):
            pad_value: int | float = 0.0
        elif isinstance(feature, SequenceFeature):
            pad_value = feature.padding_idx if feature.padding_idx is not None else 0
        else:
            pad_value = 0
        if isinstance(value, torch.Tensor):
            padded[feature.name] = pad_tensor(value, pad_rows, pad_value)
        else:
            padded[feature.name] = pad_array(np.asarray(value), pad_rows, pad_value)
    if orig_batch is None:
        orig_batch = 0
    return padded, orig_batch


def pad_id_batch(
    id_batch: dict[str, object],
    target_batch: int,
    pad_value: int | float = 0,
) -> tuple[dict[str, object], int]:
    if target_batch <= 0:
        return id_batch, 0
    padded: dict[str, object] = {}
    orig_batch = None
    for name, value in id_batch.items():
        if isinstance(value, torch.Tensor):
            batch = value.shape[0] if value.dim() > 0 else 1
        else:
            arr = np.asarray(value)
            batch = arr.shape[0] if arr.ndim > 0 else 1
        if orig_batch is None:
            orig_batch = int(batch)
        pad_rows = max(int(target_batch) - int(batch), 0)
        if isinstance(value, torch.Tensor):
            padded[name] = pad_tensor(value, pad_rows, pad_value)
        else:
            padded[name] = pad_array(np.asarray(value), pad_rows, pad_value)
    if orig_batch is None:
        orig_batch = 0
    return padded, orig_batch


def merge_onnx_outputs(outputs: Sequence[np.ndarray]) -> np.ndarray:
    if not outputs:
        raise ValueError("[ONNX Output Error] Empty ONNX output list.")
    if len(outputs) == 1:
        return outputs[0]
    normalized: list[np.ndarray] = []
    batch = outputs[0].shape[0] if outputs[0].ndim > 0 else None
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if batch is not None and arr.shape[0] != batch:
            raise ValueError(
                "[ONNX Output Error] Output batch size mismatch across ONNX outputs."
            )
        normalized.append(arr)
    return np.concatenate(normalized, axis=1)


def load_onnx_session(
    onnx_path: str | Path,
):
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    if not selected:
        selected = available

    return ort.InferenceSession(str(onnx_path), providers=selected)
