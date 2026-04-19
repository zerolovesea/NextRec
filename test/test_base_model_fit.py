import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nextrec.basic.callback import CheckpointSaver, EarlyStopper
from nextrec.engine.model import Model as BaseModel


class _DummyMultiTaskFitModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummyMultiTaskFit"

    @property
    def model_family(self) -> str:
        return "multitask"

    @property
    def default_task(self) -> list[str]:
        return ["binary", "binary"]

    def forward(self, X_input):
        return torch.zeros(1, 2)

    def __init__(self):
        super().__init__(
            dense_features=[],
            sparse_features=[],
            sequence_features=[],
            target=["y1", "y2"],
            task=["binary", "binary"],
        )
        self.dummy = nn.Parameter(torch.zeros(1))


def test_fit_without_validation_monitors_loss_for_early_stop(monkeypatch, caplog):
    model = _DummyMultiTaskFitModel()
    model.compile(loss="bce")

    monkeypatch.setattr(model, "prepare_data_loader", lambda *args, **kwargs: DataLoader([0], batch_size=1))
    monkeypatch.setattr(model, "train_epoch", lambda *args, **kwargs: (0.25, None))
    monkeypatch.setattr(model, "summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "build_train_data_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "load_model", lambda *args, **kwargs: None)

    caplog.set_level(logging.WARNING)

    model.fit(
        train_data={},
        valid_data=None,
        metrics=["auc"],
        epochs=1,
        batch_size=1,
        early_stop_patience=2,
        use_tensorboard=False,
    )

    early_stopper = next(cb for cb in model.callbacks.callbacks if isinstance(cb, EarlyStopper))
    checkpoint_saver = next(cb for cb in model.callbacks.callbacks if isinstance(cb, CheckpointSaver))

    assert early_stopper.monitor == "loss"
    assert early_stopper.mode == "min"
    assert checkpoint_saver.monitor == "loss"
    assert checkpoint_saver.mode == "min"
    assert "Early stopping conditioned on metric `val_auc_y1`" not in caplog.text


def test_fit_with_valid_group_by_uses_overall_metrics_for_callbacks(monkeypatch, caplog):
    model = _DummyMultiTaskFitModel()
    model.compile(loss="bce")

    monkeypatch.setattr(model, "prepare_data_loader", lambda *args, **kwargs: DataLoader([0], batch_size=1))
    monkeypatch.setattr(model, "train_epoch", lambda *args, **kwargs: (0.25, None))
    monkeypatch.setattr(
        model,
        "evaluate",
        lambda *args, **kwargs: {
            "overall": {"auc_y1": 0.8, "auc_y2": 0.7},
            "grouped": [{"product": "p1", "samples": 1, "auc_y1": 0.8, "auc_y2": 0.7}],
        },
    )
    monkeypatch.setattr(model, "summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "build_train_data_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "build_valid_data_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "load_model", lambda *args, **kwargs: None)

    caplog.set_level(logging.WARNING)

    model.fit(
        train_data={},
        valid_data={},
        metrics=["auc"],
        epochs=1,
        batch_size=1,
        early_stop_patience=2,
        valid_group_by="product",
        use_tensorboard=False,
    )

    assert "which is not available" not in caplog.text
