"""
Callback System for Training Process

Date: create on 27/10/2025
Checkpoint: edit on 19/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import copy
import logging
import pickle
from pathlib import Path
from typing import Optional

import torch

from nextrec import __version__
from nextrec.basic.loggers import colorize, format_kv


class Callback:
    """
    Base callback.

    Notes for DDP training:
    In distributed training, the training loop runs on every rank.
    For callbacks with side effects (saving, logging, etc.), set
    ``run_on_main_process_only=True`` to avoid multi-rank duplication.
    """

    run_on_main_process_only: bool = False

    def on_train_begin(self, logs: Optional[dict] = None):
        pass

    def on_train_end(self, logs: Optional[dict] = None):
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        pass

    def on_batch_begin(self, batch: int, logs: Optional[dict] = None):
        pass

    def on_batch_end(self, batch: int, logs: Optional[dict] = None):
        pass

    def on_validation_begin(self, logs: Optional[dict] = None):
        pass

    def on_validation_end(self, logs: Optional[dict] = None):
        pass

    def set_model(self, model):
        self.model = model

    def set_params(self, params: dict):
        self.params = params

    def should_run(self) -> bool:
        if not getattr(self, "run_on_main_process_only", False):
            return True
        model = getattr(self, "model", None)
        if model is None:
            return True
        return bool(getattr(model, "is_main_process", True))


class CallbackList:
    """Generates a list of callbacks"""

    def __init__(self, callbacks: Optional[list[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def call(self, fn_name: str, *args, **kwargs):
        for callback in self.callbacks:
            if not callback.should_run():
                continue
            getattr(callback, fn_name)(*args, **kwargs)

    def set_model(self, model):
        self.call("set_model", model)

    def set_params(self, params: dict):
        self.call("set_params", params)

    def on_train_begin(self, logs: Optional[dict] = None):
        self.call("on_train_begin", logs)

    def on_train_end(self, logs: Optional[dict] = None):
        self.call("on_train_end", logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        self.call("on_epoch_begin", epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        self.call("on_epoch_end", epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[dict] = None):
        self.call("on_batch_begin", batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[dict] = None):
        self.call("on_batch_end", batch, logs)

    def on_validation_begin(self, logs: Optional[dict] = None):
        self.call("on_validation_begin", logs)

    def on_validation_end(self, logs: Optional[dict] = None):
        self.call("on_validation_end", logs)


class EarlyStopper(Callback):

    def __init__(
        self,
        monitor: str = "val_auc",
        patience: int = 20,
        mode: str = "max",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        """
        Callback to stop training early if no improvement.

        Args:
            monitor: Metric name to monitor.
            patience: Number of epochs with no improvement after which training will be stopped.
            mode: One of {'min', 'max'}. In 'min' mode, training will stop when the
                monitored metric has stopped decreasing; in 'max' mode it will stop
                when the monitored metric has stopped increasing.
            min_delta: Minimum change in the monitored metric to qualify as an improvement.
            restore_best_weights: Whether to restore model weights from the epoch with the best value
                of the monitored metric.
            verbose: Verbosity mode. 1: messages will be printed. 0: silent.
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

        if mode == "min":
            self.best_value = float("inf")
            self.monitor_op = lambda current, best: current < (best - self.min_delta)
        elif mode == "max":
            self.best_value = float("-inf")
            self.monitor_op = lambda current, best: current > (best + self.min_delta)
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_train_begin(self, logs: Optional[dict] = None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            if self.verbose > 0:
                logging.warning(
                    f"Early stopping conditioned on metric `{self.monitor}` "
                    f"which is not available. Available metrics are: {','.join(list(logs.keys()))}"
                )
            return

        if self.monitor_op(current, self.best_value):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(self.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if hasattr(self.model, "stop_training"):
                    self.model.stop_training = True
                if self.verbose > 0:
                    logging.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch + 1}"
                    )

    def on_train_end(self, logs: Optional[dict] = None):
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                logging.info(
                    colorize(
                        format_kv(
                            "Restoring model weights from epoch",
                            f"{self.best_epoch + 1} with best {self.monitor}: {self.best_value:.6f}",
                        ),
                        color="bright_blue",
                    )
                )
            self.model.load_state_dict(self.best_weights)


class CheckpointSaver(Callback):
    """Callback to save model checkpoints during training.

    Args:
        save_path: Path to save checkpoints.
        monitor: Metric name to monitor for saving best model.
        mode: One of {'min', 'max'}.
        save_best_only: If True, only save when the model is considered the "best".
        save_freq: Frequency of checkpoint saving ('epoch' or integer for every N epochs).
        verbose: Verbosity mode.
        run_on_main_process_only: Whether to run this callback only on the main process in DDP.
    """

    def __init__(
        self,
        best_path: str | Path,
        checkpoint_path: str | Path,
        monitor: str = "val_auc",
        mode: str = "max",
        save_best_only: bool = False,
        save_freq: str | int = "epoch",
        verbose: int = 1,
        run_on_main_process_only: bool = True,
    ):
        super().__init__()
        self.run_on_main_process_only = run_on_main_process_only
        self.best_path = Path(best_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose

        if mode == "min":
            self.best_value = float("inf")
            self.monitor_op = lambda current, best: current < best
        elif mode == "max":
            self.best_value = float("-inf")
            self.monitor_op = lambda current, best: current > best
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_train_begin(self, logs: Optional[dict] = None):
        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}

        should_save = False
        if self.save_freq == "epoch":
            should_save = True
        elif isinstance(self.save_freq, int) and (epoch + 1) % self.save_freq == 0:
            should_save = True

        # Check if this is the best model
        current = logs.get(self.monitor)
        is_best = False

        if current is not None and self.monitor_op(current, self.best_value):
            self.best_value = current
            is_best = True
            should_save = True

        if should_save:
            if not self.save_best_only or is_best:
                self.save_checkpoint(self.checkpoint_path, epoch, logs)

                if is_best:
                    # Use save_path directly without adding _best suffix since it may already contain it
                    self.save_checkpoint(self.best_path, epoch, logs)
                    if self.verbose > 0:
                        logging.info(
                            colorize(
                                format_kv(
                                    "Saved best model to",
                                    f"{self.best_path} with {self.monitor}: {current:.6f}",
                                ),
                                color="bright_blue",
                            )
                        )

    def save_checkpoint(self, path: Path, epoch: int, logs: dict):

        # Get the actual model (unwrap DDP if needed)
        model_to_save = (
            self.model.ddp_model.module
            if getattr(self.model, "ddp_model", None) is not None
            else self.model
        )

        # Save only state_dict to match BaseModel.save_model() format
        torch.save(model_to_save.state_dict(), path)

        # Also save features_config.pkl if it doesn't exist
        config_path = path.parent / "features_config.pkl"
        if not config_path.exists():
            features_config = {
                "all_features": self.model.all_features,
                "target": self.model.target_columns,
                "id_columns": self.model.id_columns,
                "version": __version__,
            }
            with open(config_path, "wb") as f:
                pickle.dump(features_config, f)

        if self.verbose > 1:
            logging.info(f"Saved checkpoint to {path}")


class LearningRateScheduler(Callback):
    """Callback for learning rate scheduling.

    Args:
        scheduler: Learning rate scheduler instance or name.
        verbose: Verbosity mode.
    """

    def __init__(self, scheduler=None, verbose: int = 0):
        super().__init__()
        self.scheduler = scheduler
        self.verbose = verbose

    def on_train_begin(self, logs: Optional[dict] = None):
        if self.scheduler is None and hasattr(self.model, "scheduler_fn"):
            self.scheduler = self.model.scheduler_fn

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        if self.scheduler is not None:
            # Get current lr before step
            if hasattr(self.model, "optimizer_fn"):
                old_lr = self.model.optimizer_fn.param_groups[0]["lr"]

            # Step the scheduler
            if hasattr(self.scheduler, "step"):
                # Some schedulers need metrics
                if logs is None:
                    logs = {}
                if "val_loss" in logs and hasattr(self.scheduler, "mode"):
                    self.scheduler.step(logs["val_loss"])
                else:
                    self.scheduler.step()

            # Log new lr
            if self.verbose > 0 and hasattr(self.model, "optimizer_fn"):
                if getattr(self.model, "is_main_process", True):
                    new_lr = self.model.optimizer_fn.param_groups[0]["lr"]
                    if new_lr != old_lr:
                        logging.info(
                            f"Learning rate changed from {old_lr:.6e} to {new_lr:.6e}"
                        )


class MetricsLogger(Callback):
    """Callback for logging training metrics.

    Args:
        log_freq: Frequency of logging ('epoch', 'batch', or integer for every N epochs/batches).
        verbose: Verbosity mode.
    """

    def __init__(self, log_freq: str | int = "epoch", verbose: int = 1):
        super().__init__()
        self.run_on_main_process_only = True
        self.log_freq = log_freq
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        if self.verbose > 0 and (
            self.log_freq == "epoch"
            or (isinstance(self.log_freq, int) and (epoch + 1) % self.log_freq == 0)
        ):
            logs = logs or {}
            metrics_str = " - ".join(
                [
                    f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in logs.items()
                ]
            )
            logging.info(f"Epoch {epoch + 1}: {metrics_str}")

    def on_batch_end(self, batch: int, logs: Optional[dict] = None):
        if self.verbose > 1 and (
            self.log_freq == "batch"
            or (isinstance(self.log_freq, int) and (batch + 1) % self.log_freq == 0)
        ):
            logs = logs or {}
            metrics_str = " - ".join(
                [
                    f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in logs.items()
                ]
            )
            logging.info(f"Batch {batch}: {metrics_str}")
