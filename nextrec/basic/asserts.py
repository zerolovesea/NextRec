"""
Assert function definitions for NextRec models.

Date: create on 01/01/2026
Checkpoint: edit on 07/02/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

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
