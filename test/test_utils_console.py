import importlib
import logging

import numpy as np

from nextrec.utils import console as console_utils


def test_get_nextrec_version_prefers_repo_version(monkeypatch):
    import nextrec

    monkeypatch.setattr(nextrec, "__version__", "9.9.9", raising=False)
    assert console_utils.get_nextrec_version() == "9.9.9"


def test_get_nextrec_version_fallback_metadata(monkeypatch):
    import nextrec

    monkeypatch.setattr(nextrec, "__version__", "", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    assert console_utils.get_nextrec_version() == "1.2.3"


def test_log_startup_info_emits_expected_lines(monkeypatch, caplog):
    logger = logging.getLogger("nextrec.test")
    caplog.set_level(logging.INFO, logger="nextrec.test")

    monkeypatch.setattr(console_utils, "get_nextrec_version", lambda: "0.0.0")
    monkeypatch.setattr("platform.python_version", lambda: "3.11.0")
    monkeypatch.setattr("platform.system", lambda: "TestOS")
    monkeypatch.setattr("platform.release", lambda: "1.0")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("os.getcwd", lambda: "/tmp/workdir")
    monkeypatch.setattr("sys.executable", "/usr/bin/python")
    monkeypatch.setattr("sys.argv", ["nextrec", "--mode", "train"])

    console_utils.log_startup_info(logger, mode="train", config_path="conf.yaml")

    message_text = "\n".join(record.message for record in caplog.records)
    assert "NextRec CLI" in message_text
    assert "- Version: 0.0.0" in message_text
    assert "- Mode: train" in message_text
    assert "- Config: conf.yaml" in message_text
    assert "- Python: 3.11.0 (/usr/bin/python)" in message_text
    assert "- Platform: TestOS 1.0 (x86_64)" in message_text
    assert "- Workdir: /tmp/workdir" in message_text
    assert "- Command: nextrec --mode train" in message_text


def test_group_metrics_by_task():
    metrics = {
        "auc_taskA": 0.9,
        "logloss_taskB": 0.3,
        "acc": 0.8,
    }
    task_order, grouped = console_utils.group_metrics_by_task(
        metrics, target_names=["taskA", "taskB"]
    )
    assert task_order == ["taskA", "taskB", "overall"]
    assert grouped["taskA"]["auc"] == 0.9
    assert grouped["taskB"]["logloss"] == 0.3
    assert grouped["overall"]["acc"] == 0.8


def test_display_metrics_table_fallback_logs(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(console_utils, "Console", None)
    monkeypatch.setattr(console_utils, "Table", None)
    monkeypatch.setattr(console_utils, "box", None)

    console_utils.display_metrics_table(
        epoch=1,
        epochs=2,
        split="train",
        loss=0.5,
        metrics={"auc": 0.9},
        target_names=None,
    )

    message_text = "\n".join(record.message for record in caplog.records)
    assert "Epoch 1/2 - train:" in message_text
    assert "loss=0.5000" in message_text
    assert "overall[auc=0.9000]" in message_text


def test_display_metrics_table_rich_path():
    console_utils.display_metrics_table(
        epoch=1,
        epochs=2,
        split="valid",
        loss=np.array([0.1, 0.2]),
        metrics={"auc_task1": 0.9, "auc_task2": 0.8},
        target_names=["task1", "task2"],
    )


def test_progress_iterable(monkeypatch):
    class DummyProgress:
        def __init__(self, *args, **kwargs):
            self._tasks = {}
            self._next_id = 0

        def add_task(self, description, total=None):
            task_id = self._next_id
            self._next_id += 1
            self._tasks[task_id] = {"description": description, "total": total}
            return task_id

        def start(self):
            return None

        def advance(self, task_id, advance=1):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(console_utils, "Progress", DummyProgress)

    items = [1, 2, 3]
    assert list(console_utils.progress(items, disable=True)) == items
    assert (
        list(console_utils.progress(items, description="test", disable=False)) == items
    )
