"""
Assert function definitions for NextRec models.

Date: create on 01/01/2026
Checkpoint: edit on 17/03/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import os
from typing import Any

from nextrec.utils.types import TaskTypeName


def assert_task(
    task: list[TaskTypeName] | TaskTypeName | None,
    nums_task: int,
    *,
    model_name: str,
) -> None:
    if task is None:
        raise ValueError(f"{model_name} requires task to be specified.")

    # case 1: task is str
    if isinstance(task, str):
        if nums_task != 1:
            raise ValueError(
                f"{model_name} received task='{task}' but nums_task={nums_task}. "
                "String task is only allowed for single-task models."
            )
        return  # single-task, valid

    # case 2: task is list
    if not isinstance(task, list):
        raise TypeError(f"{model_name} requires task to be a string or a list of strings.")

    # list but length == 1
    if len(task) == 1:
        if nums_task != 1:
            raise ValueError(
                f"{model_name} received task list of length 1 but nums_task={nums_task}. "
                "Length-1 task list is only allowed for single-task models."
            )
        return  # single-task, valid

    # multi-task: length must match nums_task
    if len(task) != nums_task:
        raise ValueError(f"{model_name} requires task length {nums_task}, got {len(task)}.")


def assert_save_format(save_format: str, *, model_name: str) -> None:
    if save_format not in {"csv", "parquet"}:
        raise ValueError(f"[{model_name} Error] Unsupported save format: {save_format}. Supported: csv, parquet")


def assert_streaming_data_is_filepath(data: Any, *, model_name: str) -> None:
    if not isinstance(data, (str, os.PathLike)):
        raise ValueError(f"[{model_name} Error] Multi-process streaming requires data to be a file path.")


def assert_onnx_session_mp_compat(onnx_session: Any | None, num_processes: int, *, model_name: str) -> None:
    if num_processes > 1 and onnx_session is not None:
        raise ValueError(
            f"[{model_name} Error] onnx_session is not supported when num_processes > 1. "
            "Please pass onnx_path and let each worker create its own session."
        )


def assert_loss_weights(
    loss_weights: int | float | list[int | float] | tuple[int | float, ...] | None,
    nums_task: int,
    *,
    model_name: str,
) -> list[float] | None:
    if loss_weights is None:
        return None

    if nums_task == 1:
        if isinstance(loss_weights, (list, tuple)):
            if len(loss_weights) != 1:
                raise ValueError(
                    f"[{model_name} Error] loss_weights list must have exactly one element for single-task setup."
                )
            loss_weights = loss_weights[0]
        return [float(loss_weights)]

    if isinstance(loss_weights, (int, float)):
        return [float(loss_weights)] * nums_task

    if isinstance(loss_weights, (list, tuple)):
        weights = [float(w) for w in loss_weights]
        if len(weights) != nums_task:
            raise ValueError(
                f"[{model_name} Error] Number of loss_weights ({len(weights)}) must match number of tasks ({nums_task})."
            )
        return weights

    raise TypeError(f"[{model_name} Error] loss_weights must be int, float, list or tuple, got {type(loss_weights)}")
