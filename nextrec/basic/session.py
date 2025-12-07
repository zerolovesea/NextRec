"""Session and experiment utilities.

Date: create on 23/11/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = [
    "Session",
    "resolve_save_path",
    "create_session",
]


@dataclass(frozen=True)
class Session:
    """Encapsulate standard folders for a NextRec experiment."""

    experiment_id: str
    root: Path
    log_basename: str  # The base name for log files, without path separators

    @property
    def logs_dir(self) -> Path:
        return self._ensure_dir(self.root)

    @property
    def checkpoints_dir(self) -> Path:
        return self._ensure_dir(self.root)

    @property
    def predictions_dir(self) -> Path:
        return self._ensure_dir(self.root / "predictions")

    @property
    def processor_dir(self) -> Path:
        return self._ensure_dir(self.root / "processor")

    @property
    def params_dir(self) -> Path:
        return self._ensure_dir(self.root)

    @property
    def metrics_dir(self) -> Path:
        return self._ensure_dir(self.root)

    def save_text(self, name: str, content: str) -> Path:
        """Convenience helper: write a text file under logs_dir."""
        path = self.logs_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


def create_session(experiment_id: str | Path | None = None) -> Session:

    if experiment_id is not None and str(experiment_id).strip():
        exp_id = str(experiment_id).strip()
    else:
        # Use local time for session naming
        exp_id = "nextrec_session_" + datetime.now().strftime("%Y%m%d")

    log_basename = Path(exp_id).name if exp_id else exp_id

    if (
        os.getenv("PYTEST_CURRENT_TEST")
        or os.getenv("PYTEST_RUNNING")
        or os.getenv("NEXTREC_TEST_MODE") == "1"
    ):
        session_path = Path(tempfile.gettempdir()) / "nextrec_logs" / exp_id
    else:
        # export NEXTREC_LOG_DIR=/data/nextrec/logs
        base_dir = Path(os.getenv("NEXTREC_LOG_DIR", Path.cwd() / "nextrec_logs"))
        session_path = base_dir / exp_id

    session_path.mkdir(parents=True, exist_ok=True)
    root = session_path.resolve()

    return Session(experiment_id=exp_id, root=root, log_basename=log_basename)


def resolve_save_path(
    path: str | os.PathLike | Path | None,
    default_dir: str | Path,
    default_name: str,
    suffix: str,
    add_timestamp: bool = False,
) -> Path:
    """
    Normalize and create a save path.

    - If ``path`` is ``None`` or has no suffix, place the file under
      ``default_dir``.
    - If ``path`` has no suffix, its stem is used as the file name; otherwise
      ``default_name``.
    - Relative paths with a suffix are also anchored under ``default_dir``.
    - Enforces ``suffix`` (with leading dot) and optionally appends a
      timestamp.
    - Parent directories are created.
    """
    # Use local time for file timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None

    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"

    if path is not None and Path(path).suffix:
        target = Path(path)
        if not target.is_absolute():
            target = Path(default_dir) / target
        if target.suffix != normalized_suffix:
            target = target.with_suffix(normalized_suffix)
        if timestamp:
            target = target.with_name(f"{target.stem}_{timestamp}{normalized_suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target.resolve()

    base_dir = Path(default_dir)
    candidate = Path(path) if path is not None else None

    if candidate is not None:
        if candidate.exists() and candidate.is_dir():
            base_dir = candidate
            file_stem = default_name
        else:
            base_dir = (
                candidate.parent
                if candidate.parent not in (Path("."), Path(""))
                else base_dir
            )
            file_stem = candidate.name or default_name
    else:
        file_stem = default_name

    base_dir.mkdir(parents=True, exist_ok=True)
    if timestamp:
        file_stem = f"{file_stem}_{timestamp}"

    return (base_dir / f"{file_stem}{normalized_suffix}").resolve()
