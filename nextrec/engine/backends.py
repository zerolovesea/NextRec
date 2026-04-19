"""
Inference backend module for NextRec models, providing Runtime inference.

Date: create on 16/04/2025
Checkpoint: edit on 18/04/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from nextrec.utils.onnx_utils import build_onnx_input_feed, load_onnx_session, merge_onnx_outputs, pad_onnx_inputs
from nextrec.utils.torch_utils import to_numpy


@dataclass(frozen=True)
class PredictionBatch:
    values: np.ndarray
    # start and end is for onnx slicing
    start: int
    end: int


class InferenceBackend:
    """Common inference backend interface used by BasePredictor."""

    name = "base"
    progress_description = "Predicting"

    def prepare_batch_size(self, data: Any, batch_size: int) -> int:
        return batch_size

    def prepare_chunk_size(self, batch_size: int, stream_chunk_size: int) -> int:
        return stream_chunk_size

    def iter_predict(self, feature_batch: dict[str, Any]) -> Iterator[PredictionBatch]:
        raise NotImplementedError


class TorchInferenceBackend(InferenceBackend):
    name = "torch"
    progress_description = "Predicting"

    def __init__(self, model: Any):
        self.model = model

    def iter_predict(self, feature_batch: dict[str, Any]) -> Iterator[PredictionBatch]:
        y_pred = self.model.training_adapter.forward(self.model, feature_batch)
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError(
                f"[BaseModel-predict Error] Expected torch.Tensor predictions, got {type(y_pred).__name__}."
            )
        y_pred_np = to_numpy(y_pred, as_2d=True)
        yield PredictionBatch(values=y_pred_np, start=0, end=y_pred_np.shape[0])


class OnnxInferenceBackend(InferenceBackend):
    name = "onnx"
    progress_description = "Predicting with ONNX"

    def __init__(
        self,
        model: Any,
        onnx_path: str | Path,
    ):
        self.model = model
        self.session = load_onnx_session(Path(onnx_path))
        session_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in session_inputs]
        batch_dim = session_inputs[0].shape[0] if session_inputs else None
        self.fixed_batch = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None

    def prepare_batch_size(self, data: Any, batch_size: int) -> int:
        # File streaming yields one source chunk at a time; keep the existing behavior for streaming data.
        # NextRec behavior where batch_size controls the source chunk size.
        return 1 if isinstance(data, (str, os.PathLike)) else batch_size

    def prepare_chunk_size(self, batch_size: int, stream_chunk_size: int) -> int:
        return batch_size

    def iter_predict(self, feature_batch: dict[str, Any]) -> Iterator[PredictionBatch]:
        first_value = next(iter(feature_batch.values()), None)
        if first_value is None:
            return
        # current batch size
        current_batch = int(first_value.shape[0])
        if current_batch <= 0:
            return

        # if current batch size larger than the fixed batch size, split it into smaller slices
        if self.fixed_batch is not None and current_batch > self.fixed_batch:
            slice_ranges = [
                (start, min(start + self.fixed_batch, current_batch))
                for start in range(0, current_batch, self.fixed_batch)
            ]
        else:
            slice_ranges = [(0, current_batch)]

        for start_idx, end_idx in slice_ranges:
            X_sub = (
                {name: value[start_idx:end_idx] for name, value in feature_batch.items()}
                if len(slice_ranges) > 1
                else feature_batch
            )
            orig_batch = None
            if self.fixed_batch is not None:
                X_sub, orig_batch = pad_onnx_inputs(self.model.all_features, X_sub, target_batch=self.fixed_batch)
            feed = build_onnx_input_feed(self.model.all_features, X_sub, input_names=self.input_names)
            outputs = self.session.run(None, feed)
            y_pred_np = to_numpy(merge_onnx_outputs(outputs), as_2d=True)
            if orig_batch is not None and orig_batch > 0:
                y_pred_np = y_pred_np[:orig_batch]
            yield PredictionBatch(values=y_pred_np, start=start_idx, end=start_idx + y_pred_np.shape[0])
