"""
Assert function definitions for NextRec models.

Date: create on 01/01/2026
Checkpoint: edit on 15/04/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import os
from typing import Any

from nextrec.utils.torch_utils import to_list
from nextrec.utils.types import (
    ModelFamilyName,
    SamplingModeName,
    TaskTypeName,
    TrainingModeName,
)

SUPPORTED_TASKS = ["binary", "regression", "generative"]
SUPPORTED_MODEL_FAMILIES = ["ranking", "matching", "sequential", "multitask", "pretrain"]
SUPPORTED_TRAINING_MODES = ["pointwise", "pairwise", "listwise"]
SUPPORTED_SAMPLING_MODES = ["explicit", "inbatch"]


# assert num_task matches task length and all task types are supported
def assert_task(
    task: list[TaskTypeName] | TaskTypeName | None,
    nums_task: int,
    *,
    model_name: str,
) -> None:
    if task is None:
        raise ValueError(f"{model_name} requires task to be specified.")

    if isinstance(task, str):
        if nums_task != 1:
            raise ValueError(
                f"{model_name} received task='{task}' but nums_task={nums_task}. "
                "String task is only allowed for single-task models."
            )
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"{model_name} received unsupported task='{task}'. " f"Supported tasks: {sorted(SUPPORTED_TASKS)}."
            )

    elif len(task) == 1:
        if nums_task != 1:
            raise ValueError(
                f"{model_name} received task list of length 1 but nums_task={nums_task}. "
                "Length-1 task list is only allowed for single-task models."
            )
        if task[0] not in SUPPORTED_TASKS:
            raise ValueError(
                f"{model_name} received unsupported task='{task[0]}'. " f"Supported tasks: {sorted(SUPPORTED_TASKS)}."
            )

    else:
        if len(task) != nums_task:
            raise ValueError(f"{model_name} requires task length {nums_task}, got {len(task)}.")
        invalid_types = [task_name for task_name in task if task_name not in SUPPORTED_TASKS]
        if invalid_types:
            raise ValueError(
                f"{model_name} received unsupported tasks={invalid_types}. "
                f"Supported tasks: {sorted(SUPPORTED_TASKS)}."
            )


# assert model_family is supported
def assert_model_family(model_family: ModelFamilyName, *, model_name: str) -> None:
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"{model_name} received unsupported model_family='{model_family}'. "
            f"Supported model families: {sorted(SUPPORTED_MODEL_FAMILIES)}."
        )


# assert training_mode is supported
def assert_training_mode(training_mode: TrainingModeName, *, model_name: str) -> None:
    if training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            f"{model_name} received unsupported training_mode='{training_mode}'. "
            f"Supported training modes: {sorted(SUPPORTED_TRAINING_MODES)}."
        )


# assert sampling_mode is supported
def assert_sampling_mode(sampling_mode: SamplingModeName, *, model_name: str) -> None:
    if sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(
            f"{model_name} received unsupported sampling_mode='{sampling_mode}'. "
            f"Supported sampling modes: {sorted(SUPPORTED_SAMPLING_MODES)}."
        )


# assert if task family is compatible with model family, training mode and sampling mode
def assert_task_family_compatibility(
    task: list[TaskTypeName] | TaskTypeName,
    model_family: ModelFamilyName,
    training_mode: TrainingModeName,
    sampling_mode: SamplingModeName,
    *,
    model_name: str,
) -> None:
    tasks = to_list(task)

    # sequential models only support generative tasks: next item prediction
    if model_family == "sequential":
        unsupported = [item for item in tasks if item != "generative"]
        if unsupported:
            raise ValueError(f"{model_name} sequential models support only task='generative', " f"got {unsupported}.")

    # matching/two towers models only support binary tasks
    if model_family == "matching":
        unsupported = [item for item in tasks if item != "binary"]
        if unsupported:
            raise ValueError(f"{model_name} matching models support only task='binary', " f"got {unsupported}.")

    # only matching models support inbatch sampling for pairwise/listwise training
    if training_mode in {"pairwise", "listwise"} and sampling_mode == "inbatch" and model_family != "matching":
        raise ValueError(
            f"{model_name} only matching models support sampling_mode='inbatch' for pairwise/listwise training."
        )

    # regression tasks only support pointwise training
    if any(item == "regression" for item in tasks) and training_mode != "pointwise":
        raise ValueError(f"{model_name} regression tasks support only training_mode='pointwise'.")


def assert_save_format(save_format: str, *, model_name: str) -> None:
    if save_format not in {"csv", "parquet"}:
        raise ValueError(f"[{model_name} Error] Unsupported save format: {save_format}. Supported: csv, parquet")


def assert_streaming_data_is_filepath(data: Any, *, model_name: str) -> None:
    if not isinstance(data, (str, os.PathLike)):
        raise ValueError(f"[{model_name} Error] Multi-process streaming requires data to be a file path.")


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
