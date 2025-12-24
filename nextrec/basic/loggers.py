"""
NextRec Basic Loggers

Date: create on 27/10/2025
Checkpoint: edit on 24/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import copy
import json
import logging
import numbers
import os
import re
import sys
from typing import Any, Mapping

from nextrec.basic.session import Session, create_session

ANSI_CODES = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}

ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"
ANSI_ESCAPE_PATTERN = re.compile(r"\033\[[0-9;]*m")

DEFAULT_LEVEL_COLORS = {
    "DEBUG": "cyan",
    "INFO": None,
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bright_red",
}


class AnsiFormatter(logging.Formatter):
    def __init__(
        self,
        *args,
        strip_ansi: bool = False,
        auto_color_level: bool = False,
        level_colors: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.strip_ansi = strip_ansi
        self.auto_color_level = auto_color_level
        self.level_colors = level_colors or DEFAULT_LEVEL_COLORS

    def format(self, record: logging.LogRecord) -> str:
        record_copy = copy.copy(record)
        formatted = super().format(record_copy)

        if self.auto_color_level and "\033[" not in formatted:
            color = self.level_colors.get(record.levelname)
            if color:
                formatted = colorize(formatted, color=color)

        if self.strip_ansi:
            return ANSI_ESCAPE_PATTERN.sub("", formatted)

        return formatted


def colorize(text: str, color: str | None = None, bold: bool = False) -> str:
    """Apply ANSI color and bold formatting to the given text."""
    if not color and not bold:
        return text
    result = ""
    if bold:
        result += ANSI_BOLD
    if color and color in ANSI_CODES:
        result += ANSI_CODES[color]
    result += text + ANSI_RESET
    return result


def format_kv(label: str, value: Any, width: int = 34, indent: int = 0) -> str:
    """Format key-value lines with consistent alignment."""
    label_text = label if label.endswith(":") else f"{label}:"
    prefix = " " * indent
    return f"{prefix}{label_text:<{width}} {value}"


def setup_logger(session_id: str | os.PathLike | None = None):
    """Set up a logger that logs to both console and a file with ANSI formatting.
    Only console output has colors; file output is stripped of ANSI codes.
    Logs are stored under ``log/<experiment_id>/logs`` by default. A stable
    log file is used per experiment so multiple components (e.g. data
    processor and model training) append to the same file instead of creating
    separate timestamped files.
    """

    session = create_session(str(session_id) if session_id is not None else None)
    log_dir = session.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "runs.log"

    console_format = "%(message)s"
    file_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        AnsiFormatter(file_format, datefmt=date_format, strip_ansi=True)
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        AnsiFormatter(
            console_format,
            datefmt=date_format,
            auto_color_level=True,
        )
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class TrainingLogger:
    def __init__(
        self,
        session: Session,
        use_tensorboard: bool,
        log_name: str = "training_metrics.jsonl",
    ) -> None:
        self.session = session
        self.use_tensorboard = use_tensorboard
        self.log_path = session.metrics_dir / log_name
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.tb_writer = None
        self.tb_dir = None

        if self.use_tensorboard:
            self._init_tensorboard()

    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except ImportError:
            logging.warning(
                "[TrainingLogger] tensorboard not installed, disable tensorboard logging."
            )
            self.use_tensorboard = False
            return
        tb_dir = self.session.logs_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = tb_dir
        self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

    @property
    def tensorboard_logdir(self):
        return self.tb_dir

    def format_metrics(
        self, metrics: Mapping[str, Any], split: str
    ) -> dict[str, float]:
        formatted: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, numbers.Real):
                formatted[f"{split}/{key}"] = float(value)
            elif hasattr(value, "item"):
                try:
                    formatted[f"{split}/{key}"] = float(value.item())
                except Exception:
                    continue
        return formatted

    def log_metrics(
        self, metrics: Mapping[str, Any], step: int, split: str = "train"
    ) -> None:
        payload = self.format_metrics(metrics, split)
        payload["step"] = int(step)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if not self.tb_writer:
            return
        step = int(payload.get("step", 0))
        for key, value in payload.items():
            if key == "step":
                continue
            self.tb_writer.add_scalar(key, value, global_step=step)

    def close(self) -> None:
        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None
