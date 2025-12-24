"""
Console and CLI utilities for NextRec.

This module centralizes CLI logging helpers, progress display, and metric tables.

Date: create on 19/12/2025
Checkpoint: edit on 20/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import io
import logging
import numbers
import os
import platform
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, TypeVar

import numpy as np
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from nextrec.utils.feature import as_float, normalize_to_list

T = TypeVar("T")


def get_nextrec_version() -> str:
    """
    Best-effort version resolver for NextRec.

    Prefer in-repo `nextrec.__version__`, fall back to installed package metadata.
    """
    try:
        from nextrec import __version__  # type: ignore

        if __version__:
            return str(__version__)
    except Exception:
        pass

    try:
        from importlib.metadata import version

        return version("nextrec")
    except Exception:
        return "unknown"


def log_startup_info(
    logger: logging.Logger, *, mode: str, config_path: str | None
) -> None:
    """Log a short, user-friendly startup banner."""
    version = get_nextrec_version()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "NextRec CLI",
        f"- Version: {version}",
        f"- Time: {now}",
        f"- Mode: {mode}",
        f"- Config: {config_path or '(not set)'}",
        f"- Python: {platform.python_version()} ({sys.executable})",
        f"- Platform: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- Workdir: {os.getcwd()}",
        f"- Command: {' '.join(sys.argv)}",
    ]
    for line in lines:
        logger.info(line)


class BlackTimeElapsedColumn(TimeElapsedColumn):
    def render(self, task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="black")
        delta = timedelta(seconds=max(0, int(elapsed)))
        return Text(str(delta), style="black")


class BlackTimeRemainingColumn(TimeRemainingColumn):
    def render(self, task) -> Text:
        if self.elapsed_when_finished and task.finished:
            task_time = task.finished_time
        else:
            task_time = task.time_remaining

        if task.total is None:
            return Text("", style="black")

        if task_time is None:
            return Text("--:--" if self.compact else "-:--:--", style="black")

        minutes, seconds = divmod(int(task_time), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        return Text(formatted, style="black")


class BlackMofNCompleteColumn(MofNCompleteColumn):
    def render(self, task) -> Text:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return Text(
            f"{completed:{total_width}d}{self.separator}{total}",
            style="black",
        )


def progress(iterable, *, description=None, total=None, disable=False):
    if disable:
        yield from iterable
        return

    resolved_total = total
    if resolved_total is None:
        try:
            resolved_total = len(iterable)
        except TypeError:
            resolved_total = None

    stream = sys.stderr

    if not stream.isatty():
        start_time = time.monotonic()
        last_tick = start_time
        min_interval_seconds = 10.0
        max_interval_seconds = 300.0
        target_steps = (
            max(1, resolved_total // 20) if resolved_total is not None else 500
        )
        interval_seconds = min_interval_seconds
        completed = 0

        def emit(now: float):
            elapsed = max(0.0, now - start_time)
            speed = completed / elapsed if elapsed > 0 else 0.0
            if resolved_total is not None and speed > 0:
                remaining = max(0.0, resolved_total - completed)
                eta_seconds = remaining / speed
                eta_text = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_text = "--:--:--"
            total_text = str(resolved_total) if resolved_total is not None else "?"
            stream.write(
                f"{description or 'Working'}: {completed}/{total_text} "
                f"elapsed={timedelta(seconds=int(elapsed))} "
                f"speed={speed:.2f}/s ETA={eta_text}\n"
            )
            stream.flush()
            return speed

        for item in iterable:
            yield item
            completed += 1
            now = time.monotonic()
            if now - last_tick >= interval_seconds:
                speed = emit(now)
                last_tick = now
                if speed > 0:
                    interval_seconds = min(
                        max_interval_seconds,
                        max(min_interval_seconds, target_steps / speed),
                    )
        end_now = time.monotonic()
        if end_now - last_tick >= 1e-6:
            emit(end_now)
        return

    # TTY: rich
    console = Console(file=stream, force_terminal=True)
    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=36),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=12,
        console=console,
    )

    if hasattr(progress_bar, "__enter__"):
        with progress_bar:
            task_id = progress_bar.add_task(
                description or "Working", total=resolved_total
            )
            for item in iterable:
                yield item
                progress_bar.advance(task_id, 1)
    else:
        progress_bar.start()
        try:
            task_id = progress_bar.add_task(
                description or "Working", total=resolved_total
            )
            for item in iterable:
                yield item
                progress_bar.advance(task_id, 1)
        finally:
            progress_bar.stop()


def group_metrics_by_task(
    metrics: Mapping[str, Any] | None,
    target_names: list[str] | str | None,
    default_task_name: str = "overall",
):
    if not metrics:
        return [], {}

    if isinstance(target_names, str):
        target_names = [target_names]
    if not isinstance(target_names, list) or not target_names:
        target_names = [default_task_name]

    targets_by_len = sorted(target_names, key=len, reverse=True)
    grouped: dict[str, dict[str, float]] = {}
    for key, raw_value in metrics.items():
        value = as_float(raw_value)
        if value is None:
            continue

        matched_target: str | None = None
        metric_name = key
        for target in targets_by_len:
            suffix = f"_{target}"
            if key.endswith(suffix):
                metric_name = key[: -len(suffix)]
                matched_target = target
                break

        if matched_target is None:
            matched_target = (
                target_names[0] if len(target_names) == 1 else default_task_name
            )
        grouped.setdefault(matched_target, {})[metric_name] = value

    task_order: list[str] = []
    for target in target_names:
        if target in grouped:
            task_order.append(target)
    for task_name in grouped:
        if task_name not in task_order:
            task_order.append(task_name)
    return task_order, grouped


def display_metrics_table(
    epoch: int,
    epochs: int,
    split: str,
    loss: float | np.ndarray | None,
    metrics: Mapping[str, Any] | None,
    target_names: list[str] | str | None,
    base_metrics: list[str] | None = None,
    is_main_process: bool = True,
    colorize: Callable[[str], str] | None = None,
) -> None:
    if not is_main_process:
        return

    target_list = normalize_to_list(target_names)
    task_order, grouped = group_metrics_by_task(metrics, target_names=target_names)

    if isinstance(loss, np.ndarray) and target_list:
        # Ensure tasks with losses are shown even when metrics are missing for some targets.
        normalized_order: list[str] = []
        for name in target_list:
            if name not in normalized_order:
                normalized_order.append(name)
        for name in task_order:
            if name not in normalized_order:
                normalized_order.append(name)
        task_order = normalized_order

    if not task_order and not grouped and not metrics:
        if isinstance(loss, numbers.Number):
            msg = f"Epoch {epoch}/{epochs} - {split} (loss={float(loss):.4f})"
            if colorize is not None:
                msg = colorize(msg)
            logging.info(msg)
        return

    if Console is None or Table is None or box is None:
        prefix = f"Epoch {epoch}/{epochs} - {split}:"
        segments: list[str] = []
        if isinstance(loss, numbers.Number):
            segments.append(f"loss={float(loss):.4f}")
        if task_order and grouped:
            task_strs: list[str] = []
            for task_name in task_order:
                metric_items = grouped.get(task_name, {})
                if not metric_items:
                    continue
                metric_str = ", ".join(
                    f"{k}={float(v):.4f}" for k, v in metric_items.items()
                )
                task_strs.append(f"{task_name}[{metric_str}]")
            if task_strs:
                segments.append(", ".join(task_strs))
        elif metrics:
            metric_str = ", ".join(
                f"{k}={float(v):.4f}"
                for k, v in metrics.items()
                if as_float(v) is not None
            )
            if metric_str:
                segments.append(metric_str)
        if not segments:
            return
        msg = f"{prefix} " + ", ".join(segments)
        if colorize is not None:
            msg = colorize(msg)
        logging.info(msg)
        return

    title = f"Epoch {epoch}/{epochs} - {split}"
    if isinstance(loss, numbers.Number):
        title += f" (loss={float(loss):.4f})"

    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold",
        title_style="bold",
    )
    table.add_column("Task", style="bold")

    include_loss = isinstance(loss, np.ndarray)
    if include_loss:
        table.add_column("loss", justify="right")

    metric_names: list[str] = []
    for task_name in task_order:
        for metric_name in grouped.get(task_name, {}):
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    preferred_order: list[str] = []
    if isinstance(base_metrics, list):
        preferred_order = [m for m in base_metrics if m in metric_names]
    remaining = [m for m in metric_names if m not in preferred_order]
    metric_names = preferred_order + sorted(remaining)

    for metric_name in metric_names:
        table.add_column(metric_name, justify="right")

    def fmt(value: float | None) -> str:
        if value is None:
            return "-"
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.4f}"

    loss_by_task: dict[str, float] = {}
    if isinstance(loss, np.ndarray):
        if target_list:
            for i, task_name in enumerate(target_list):
                if i < loss.shape[0]:
                    loss_by_task[task_name] = float(loss[i])
            if "overall" in task_order and "overall" not in loss_by_task:
                loss_by_task["overall"] = float(np.sum(loss))
        elif task_order:
            for i, task_name in enumerate(task_order):
                if i < loss.shape[0]:
                    loss_by_task[task_name] = float(loss[i])
        else:
            task_order = ["overall"]
            loss_by_task["overall"] = float(np.sum(loss))

    if not task_order:
        task_order = ["__overall__"]

    for task_name in task_order:
        row: list[str] = [str(task_name)]
        if include_loss:
            row.append(fmt(loss_by_task.get(task_name)))
        for metric_name in metric_names:
            row.append(fmt(grouped.get(task_name, {}).get(metric_name)))
        table.add_row(*row)

    Console().print(table)

    record_console = Console(file=io.StringIO(), record=True, width=120)
    record_console.print(table)
    table_text = record_console.export_text(styles=False).rstrip()

    root_logger = logging.getLogger()
    record = root_logger.makeRecord(
        root_logger.name,
        logging.INFO,
        __file__,
        0,
        "[MetricsTable]\n" + table_text,
        args=(),
        exc_info=None,
        extra=None,
    )

    emitted = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.emit(record)
            emitted = True

    if not emitted:
        # Fallback: no file handlers configured, use standard logging.
        root_logger.log(logging.INFO, "[MetricsTable]\n" + table_text)
