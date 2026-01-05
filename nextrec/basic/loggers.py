"""
NextRec Basic Loggers

Date: create on 27/10/2025
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

import copy
import json
import logging
import numbers
import os
import re
import sys
from typing import Any

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
    log_file = log_dir / "runs_log.txt"

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


class MetricsLoggerBackend:
    def log_payload(self, payload: dict[str, float]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class BasicLogger:
    def __init__(
        self,
        session: Session,
        log_name: str = "training_metrics.jsonl",
        backends: list[MetricsLoggerBackend] | None = None,
    ) -> None:
        self.session = session
        self.log_path = session.metrics_dir / log_name
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.backends = backends or []

    def format_metrics(self, metrics: dict[str, Any], split: str) -> dict[str, float]:
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
        self, metrics: dict[str, Any], step: int, split: str = "train"
    ) -> None:
        payload = self.format_metrics(metrics, split)
        payload["step"] = int(step)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        for backend in self.backends:
            backend.log_payload(payload)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()
        for backend in self.backends:
            if isinstance(backend, SwanLabLogger):
                swanlab = backend.swanlab
                if not backend.enabled or swanlab is None:
                    continue
                finish_fn = getattr(swanlab, "finish", None)
                if finish_fn is None:
                    continue
                try:
                    finish_fn()
                except TypeError:
                    finish_fn()
                break


class TensorBoardLogger(MetricsLoggerBackend):
    def __init__(
        self,
        session: Session,
        enabled: bool = True,
        log_dir_name: str = "tensorboard",
    ) -> None:
        self.enabled = enabled
        self.writer = None
        self.log_dir = None
        if self.enabled:
            self._init_writer(session, log_dir_name)

    def _init_writer(self, session: Session, log_dir_name: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except ImportError:
            logging.warning(
                "[TrainingLogger] tensorboard not installed, disable tensorboard logging."
            )
            self.enabled = False
            return
        log_dir = session.logs_dir / log_dir_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_payload(self, payload: dict[str, float]) -> None:
        if not self.writer:
            return
        step = int(payload.get("step", 0))
        for key, value in payload.items():
            if key == "step":
                continue
            self.writer.add_scalar(key, value, global_step=step)

    def close(self) -> None:
        if self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None


class WandbLogger(MetricsLoggerBackend):
    def __init__(
        self,
        session: Session,
        enabled: bool = True,
        project: str | None = None,
        run_name: str | None = None,
        init_run: bool = True,
        **init_kwargs: Any,
    ) -> None:
        self.enabled = enabled
        self.wandb = None
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            logging.warning("[WandbLogger] wandb not installed, disable wandb logging.")
            self.enabled = False
            return
        self.wandb = wandb
        if init_run and getattr(wandb, "run", None) is None:
            kwargs = dict(init_kwargs)
            if project is not None:
                kwargs.setdefault("project", project)
            if run_name is None:
                run_name = session.experiment_id
            if run_name is not None:
                kwargs.setdefault("name", run_name)
            try:
                wandb.init(**kwargs)
            except TypeError:
                wandb.init()

    def log_payload(self, payload: dict[str, float]) -> None:
        if not self.enabled or self.wandb is None:
            return
        step = int(payload.get("step", 0))
        log_payload = {k: v for k, v in payload.items() if k != "step"}
        if not log_payload:
            return
        try:
            self.wandb.log(log_payload, step=step)
        except TypeError:
            self.wandb.log(log_payload)


class SwanLabLogger(MetricsLoggerBackend):
    def __init__(
        self,
        session: Session,
        enabled: bool = True,
        project: str | None = None,
        run_name: str | None = None,
        init_run: bool = True,
        **init_kwargs: Any,
    ) -> None:
        self.enabled = enabled
        self.swanlab = None
        self._warned_missing_log = False
        if not self.enabled:
            return
        try:
            import swanlab  # type: ignore
        except ImportError:
            logging.warning(
                "[SwanLabLogger] swanlab not installed, disable swanlab logging."
            )
            self.enabled = False
            return
        self.swanlab = swanlab
        if init_run and hasattr(swanlab, "init"):
            kwargs = dict(init_kwargs)
            kwargs.setdefault("logdir", str(session.logs_dir) + "/swanlog")
            if project is not None:
                kwargs.setdefault("project", project)
            if run_name is None:
                run_name = session.experiment_id
            if run_name is not None:
                kwargs.setdefault("name", run_name)
            try:
                swanlab.init(**kwargs)
            except TypeError:
                swanlab.init()

    def log_payload(self, payload: dict[str, float]) -> None:
        if not self.enabled or self.swanlab is None:
            return
        log_fn = getattr(self.swanlab, "log", None)
        if log_fn is None:
            if not self._warned_missing_log:
                logging.warning(
                    "[SwanLabLogger] swanlab.log not found, disable swanlab logging."
                )
                self._warned_missing_log = True
            return
        step = int(payload.get("step", 0))
        log_payload = {k: v for k, v in payload.items() if k != "step"}
        if not log_payload:
            return
        try:
            log_fn(log_payload, step=step)
        except TypeError:
            log_fn(log_payload)


class TrainingLogger(BasicLogger):
    def __init__(
        self,
        session: Session,
        use_tensorboard: bool,
        log_name: str = "training_metrics.jsonl",
        use_wandb: bool = False,
        use_swanlab: bool = False,
        config: dict[str, Any] = {},
        wandb_kwargs: dict[str, Any] | None = None,
        swanlab_kwargs: dict[str, Any] | None = None,
    ):
        self.session = session
        self.use_tensorboard = use_tensorboard
        self.tensorboard_logger = TensorBoardLogger(
            session=session, enabled=use_tensorboard
        )
        self.use_tensorboard = self.tensorboard_logger.enabled
        self.tb_writer = self.tensorboard_logger.writer
        self.tb_dir = self.tensorboard_logger.log_dir

        backends = []
        if self.tensorboard_logger.enabled:
            backends.append(self.tensorboard_logger)

        wandb_kwargs = dict(wandb_kwargs or {})
        wandb_kwargs.setdefault("config", {})
        wandb_kwargs["config"].update(config)
        if "notes" in wandb_kwargs:
            wandb_kwargs["config"].pop("note", None)

        swanlab_kwargs = dict(swanlab_kwargs or {})
        swanlab_kwargs.setdefault("config", {})
        swanlab_kwargs["config"].update(config)
        if "description" in swanlab_kwargs:
            swanlab_kwargs["config"].pop("note", None)

        self.wandb_logger = None
        if use_wandb:
            self.wandb_logger = WandbLogger(
                session=session, enabled=use_wandb, **wandb_kwargs
            )
            if self.wandb_logger.enabled:
                backends.append(self.wandb_logger)

        self.swanlab_logger = None
        if use_swanlab:
            self.swanlab_logger = SwanLabLogger(
                session=session, enabled=use_swanlab, **swanlab_kwargs
            )
            if self.swanlab_logger.enabled:
                backends.append(self.swanlab_logger)

        super().__init__(session=session, log_name=log_name, backends=backends)

    def init_tensorboard(self) -> None:
        if self.tensorboard_logger and self.tensorboard_logger.enabled:
            return
        self.tensorboard_logger = TensorBoardLogger(session=self.session, enabled=True)
        self.use_tensorboard = self.tensorboard_logger.enabled
        self.tb_writer = self.tensorboard_logger.writer
        self.tb_dir = self.tensorboard_logger.log_dir
        if (
            self.tensorboard_logger.enabled
            and self.tensorboard_logger not in self.backends
        ):
            self.backends.append(self.tensorboard_logger)

    @property
    def tensorboard_logdir(self):
        return self.tb_dir

    def close(self) -> None:
        super().close()
        self.tb_writer = None
