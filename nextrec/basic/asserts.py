"""
Assert function definitions for NextRec models.

Date: create on 01/01/2026
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

from nextrec.utils.types import TaskTypeName, TrainingModeName


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
        raise TypeError(
            f"{model_name} requires task to be a string or a list of strings."
        )

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
        raise ValueError(
            f"{model_name} requires task length {nums_task}, got {len(task)}."
        )


def assert_training_mode(
    training_mode: TrainingModeName | list[TrainingModeName],
    nums_task: int,
    *,
    model_name: str,
) -> None:
    valid_modes = {"pointwise", "pairwise", "listwise"}
    if not isinstance(training_mode, list):
        raise TypeError(
            f"[{model_name}-init Error] training_mode must be a list with length {nums_task}."
        )
    if len(training_mode) != nums_task:
        raise ValueError(
            f"[{model_name}-init Error] training_mode list length must match number of tasks."
        )
    if any(mode not in valid_modes for mode in training_mode):
        raise ValueError(
            f"[{model_name}-init Error] training_mode must be one of {'pointwise', 'pairwise', 'listwise'}."
        )
