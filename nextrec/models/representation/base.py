"""
Base class for representation models.

Representation models do not use the supervised task heads or ranking adapters
from the general recommendation stack. They reuse the shared infrastructure
from BaseModel but always route outputs through RepresentationAdapter.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from nextrec.basic.adapters import RepresentationAdapter
from nextrec.basic.loggers import colorize, setup_logger
from nextrec.basic.model import BaseModel
from nextrec.data.batch_utils import batch_to_dict
from nextrec.utils.console import progress


class BaseRepresentationModel(BaseModel):
    @property
    def default_task(self):  # type: ignore[override]
        return "regression"

    def set_adapter(self):
        self.training_adapter = RepresentationAdapter()
        self.prediction_layer = None

    def prepare_loader(
        self,
        data: DataLoader | dict | list | tuple,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        dataloader = self.prepare_data_loader(
            data=data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        if isinstance(dataloader, tuple):
            loader = dataloader[0]
        else:
            loader = dataloader
        return cast(DataLoader, loader)

    def extract_embeddings(self, batch_data: Any) -> torch.Tensor:
        batch_dict = batch_to_dict(batch_data)
        X_input, _ = self.get_input(batch_dict, require_labels=False)

        if not self.all_features:
            raise ValueError("[BaseRepresentationModel] dense_features are required to use fit/predict helpers.")

        tensors: list[torch.Tensor] = []
        for name in self.feature_names:
            if name not in X_input:
                raise KeyError(
                    f"[BaseRepresentationModel] Feature '{name}' not found in input batch. Available keys: {list(X_input.keys())}"
                )
            tensors.append(X_input[name].to(self.device).float())
        if not tensors:
            raise ValueError("[BaseRepresentationModel] No feature tensors found in batch.")

        init_embedding = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=-1)
        if hasattr(self, "input_dim") and init_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"[BaseRepresentationModel] Input dim mismatch: expected {self.input_dim}, got {init_embedding.shape[-1]}."
            )

        return init_embedding

    def init_codebook(self, train_loader: DataLoader, init_batches: int) -> None:
        cached: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(train_loader):
            cached.append(self.extract_embeddings(batch))
            if batch_idx >= init_batches - 1:
                break
        if not cached:
            raise ValueError("[BaseRepresentationModel] No data available for codebook initialization.")

        init_data = torch.cat(cached, dim=0)

        with torch.no_grad():
            z_e = self.encode(init_data)
            r = z_e
            for level in range(self.num_codebooks):  # type: ignore[attr-defined]
                vq = self.rq.vq(level)  # type: ignore[attr-defined]
                if not vq.codebook_initialized:
                    vq.create_codebook(r)
                q, _ = vq(r)
                r = r - q

    def get_semantic_ids(self, x_gt: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)  # type: ignore[attr-defined]
        return semantic_id_list

    def fit(
        self,
        train_data: DataLoader | dict | list | tuple,
        valid_data: DataLoader | dict | list | tuple | None = None,
        epochs: int = 1,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 0,
        lr: float = 1e-3,
        init_batches: int = 3,
    ):
        train_loader = self.prepare_loader(
            data=train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        valid_loader = (
            self.prepare_loader(
                data=valid_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            if valid_data is not None
            else None
        )

        if not self.logger_initialized and self.is_main_process:
            setup_logger(session_id=self.session_id)
            self.logger_initialized = True

        if not hasattr(self, "metrics"):
            self.metrics = ["loss"]
        if not hasattr(self, "task_specific_metrics"):
            self.task_specific_metrics = {}
        if not hasattr(self, "best_metrics_mode"):
            self.best_metrics_mode = "min"

        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        try:
            steps_per_epoch = len(train_loader)
            is_streaming = False
        except TypeError:
            steps_per_epoch = None
            is_streaming = True

        self.init_codebook(train_loader, init_batches=init_batches)

        if self.is_main_process:
            self.summary()
            logging.info("")
            logging.info(colorize("=" * 80, bold=True))
            logging.info(
                colorize(
                    "Start streaming training" if is_streaming else "Start training",
                    bold=True,
                )
            )
            logging.info(colorize("=" * 80, bold=True))
            logging.info("")
            logging.info(colorize(f"Model device: {self.device}", bold=True))

        for epoch in range(epochs):
            total_loss = 0.0
            step_count = 0
            if is_streaming and self.is_main_process:
                logging.info("")
                logging.info(colorize(f"Epoch {epoch + 1}/{epochs}", bold=True))
            if is_streaming:
                batch_iter = enumerate(train_loader)
            else:
                tqdm_disable = not self.is_main_process
                batch_iter = enumerate(
                    progress(
                        train_loader,
                        description=f"Epoch {epoch + 1}/{epochs}",
                        total=steps_per_epoch,
                        disable=tqdm_disable,
                    )
                )
            for _, batch in batch_iter:
                embeddings = self.extract_embeddings(batch)
                _, _, recon_loss, rqvae_loss, total_batch_loss = self(embeddings)  # type: ignore[misc]

                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                step_count += 1

            denom = steps_per_epoch if steps_per_epoch is not None else step_count
            avg_loss = total_loss / max(1, denom)
            train_log = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}"

            if valid_loader is not None:
                val_total = 0.0
                val_steps = 0
                with torch.no_grad():
                    for batch in valid_loader:
                        embeddings = self.extract_embeddings(batch)
                        _, _, _, _, val_loss = self(embeddings)  # type: ignore[misc]
                        val_total += val_loss.item()
                        val_steps += 1
                try:
                    val_denom = len(valid_loader)
                except TypeError:
                    val_denom = val_steps
                val_avg = val_total / max(1, val_denom)
                if self.is_main_process:
                    logging.info(colorize(train_log))
                    logging.info(
                        colorize(
                            f"  Epoch {epoch + 1}/{epochs} - Valid Loss: {val_avg:.4f}",
                            color="cyan",
                        )
                    )
            elif self.is_main_process:
                logging.info(colorize(train_log))

        if self.is_main_process:
            logging.info("")
            logging.info(colorize("Training finished.", bold=True))
            logging.info("")
        return self

    def predict(
        self,
        data: DataLoader | dict | list | tuple,
        batch_size: int = 256,
        num_workers: int = 0,
        return_reconstruction: bool = False,
        as_numpy: bool = True,
    ) -> torch.Tensor:
        data_loader = self.prepare_loader(
            data=data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        outputs: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                embeddings = self.extract_embeddings(batch)
                if return_reconstruction:
                    x_hat, _, _, _, _ = self(embeddings)  # type: ignore[misc]
                    outputs.append(x_hat.detach().cpu())
                else:
                    semantic_ids = self.get_semantic_ids(embeddings)
                    outputs.append(semantic_ids.detach().cpu())

        if outputs:
            result = torch.cat(outputs, dim=0)
        else:
            out_dim = self.input_dim if return_reconstruction else self.num_codebooks  # type: ignore[attr-defined]
            result = torch.empty((0, out_dim))
        return result.numpy() if as_numpy else result
