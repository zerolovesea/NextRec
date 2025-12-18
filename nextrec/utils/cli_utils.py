"""
CLI utilities for NextRec.

This module provides small helpers used by command-line entrypoints, such as
printing startup banners and resolving the installed package version.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from datetime import datetime


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
