"""
Shared base class for sequential recommendation models.

This module extracts the sequence-paradigm concerns shared by sequential models:
- sequence mode selection: autoregressive vs masked
- padding mask utilities
- causal attention mask utilities
- last-valid-state selection
- sequence label construction for autoregressive training
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.adapters import SequentialAdapter
from nextrec.basic.heads import GenerativeMatchingHead
from nextrec.engine.model import Model as BaseModel
from nextrec.basic.loggers import colorize, format_kv
from nextrec.basic.metrics import (
    compute_sequential_metric_batch,
    flatten_metric_names,
    is_sequential_ranking_metric,
)
from nextrec.utils.types import LossName, SequenceModeName, TaskTypeInput, TrainingModeName


class BaseSequentialModel(BaseModel):
    supported_sampling_modes = ["explicit", "inbatch"]

    @property
    def default_task(self) -> str:
        return "generative"

    @property
    def model_family(self) -> str:
        return "sequential"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: list[str] | str | None = None,
        key_columns: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        training_mode: TrainingModeName | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        sequence_mode: SequenceModeName = "autoregressive",
        device: str = "cpu",
        session_id: str | None = None,
        sampling_mode: Literal["explicit", "inbatch"] = "explicit",
        **kwargs,
    ):
        self.sequence_mode = sequence_mode
        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            key_columns=key_columns,
            task=task,
            training_mode=training_mode,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            device=device,
            session_id=session_id,
            sampling_mode=sampling_mode,
            **kwargs,
        )

    def compile(
        self,
        *args,
        loss: LossName | nn.Module | None = None,
        sampled_softmax_num_negatives: int | None = None,
        **kwargs,
    ):
        """
        Configure sequential-model training loss.

        Default behavior:
            - full-vocab next-item training with CrossEntropyLoss(ignore_index=padding)

        Optional behavior:
            - if loss='sampled_softmax' or sampled_softmax_num_negatives is set,
              approximate the vocab softmax with sampled negatives.
        """
        self.sequence_loss_mode = "ce"
        self.sampled_softmax_num_negatives = sampled_softmax_num_negatives

        if loss is None:
            loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        elif isinstance(loss, str):
            if loss in {"sampled_softmax", "softmax"}:
                self.sequence_loss_mode = "sampled_softmax"
                if self.sampled_softmax_num_negatives is None:
                    self.sampled_softmax_num_negatives = min(1024, max(int(self.vocab_size) - 1, 1))
                loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
            elif loss in {"crossentropy", "ce"}:
                loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
            else:
                loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        else:
            loss_name = loss.__class__.__name__.lower()
            if "sampledsoftmax" in loss_name:
                self.sequence_loss_mode = "sampled_softmax"
                if self.sampled_softmax_num_negatives is None:
                    self.sampled_softmax_num_negatives = min(1024, max(int(self.vocab_size) - 1, 1))
            elif isinstance(loss, nn.CrossEntropyLoss):
                pass

        return super().compile(*args, loss=loss, **kwargs)

    def set_adapter(self):
        self.training_adapter = SequentialAdapter()

    def set_head(self):
        self.head = GenerativeMatchingHead(vocab_size=int(self.vocab_size), return_logits=True)

    def validate_sequence_mode(self) -> None:
        if self.sequence_mode not in {"autoregressive", "masked"}:
            raise ValueError(
                f"[BaseSequentialModel Error] Unsupported sequence_mode='{self.sequence_mode}'. "
                "Supported modes: 'autoregressive', 'masked'."
            )

    def build_sequence_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Build the sequence-level attention mask.

        - autoregressive: upper-triangular causal mask, True means masked.
        - masked: no causal mask, return None.
        """
        if self.sequence_mode not in {"autoregressive", "masked"}:
            raise ValueError(
                f"[BaseSequentialModel Error] Unsupported sequence_mode='{self.sequence_mode}'. "
                "Supported modes: 'autoregressive', 'masked'."
            )
        if self.sequence_mode == "masked":
            return None
        return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)

    def build_sequence_padding_mask(
        self,
        seq: torch.Tensor,
        padding_idx: int,
    ) -> torch.Tensor:
        """Return [B, L] mask where True means padding."""
        return seq.eq(padding_idx)

    def build_sequence_valid_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return [B, L, 1] float mask where 1 means valid token."""
        return (~padding_mask).unsqueeze(-1).float()

    def prepare_sequence_batch(
        self,
        x: dict[str, torch.Tensor],
        sequence_name: str,
        max_seq_len: int,
        padding_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare the common sequence tensors used by autoregressive and masked models.

        Returns:
            seq: [B, L]
            padding_mask: [B, L], True means padding
            valid_mask: [B, L, 1]
            attention_mask: [L, L] or None
        """
        seq = x[sequence_name].long()[:, -max_seq_len:]
        padding_mask = self.build_sequence_padding_mask(seq, padding_idx=padding_idx)
        valid_mask = self.build_sequence_valid_mask(padding_mask)
        attention_mask = self.build_sequence_attention_mask(seq_len=seq.size(1), device=seq.device)
        return seq, padding_mask, valid_mask, attention_mask

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor | None) -> torch.Tensor:
        """
        Generic next-item loss for sequential models.

        Expected shapes:
            y_pred: [B, L, V] or [B, V]
            y_true: [B, L] or [B]
        """
        if y_true is None:
            raise ValueError(
                f"[{self.__class__.__name__}-compute_loss Error] Ground truth labels (y_true) are required."
            )
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError(
                f"[{self.__class__.__name__}-compute_loss Error] Expected y_pred to be a torch.Tensor, got {type(y_pred)}."
            )

        labels = y_true
        if labels.dim() == y_pred.dim() and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        if y_pred.dim() == 3 and labels.dim() == 1:
            labels = labels.unsqueeze(1)
        if y_pred.dim() == 2 and labels.dim() == 2 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)

        if y_pred.dim() not in {2, 3}:
            raise ValueError(
                f"[{self.__class__.__name__}-compute_loss Error] Expected y_pred to have 2 or 3 dims, got shape {tuple(y_pred.shape)}."
            )
        if labels.dim() != y_pred.dim() - 1:
            raise ValueError(
                f"[{self.__class__.__name__}-compute_loss Error] Sequential labels must have one fewer dimension than logits. "
                f"Got y_pred shape {tuple(y_pred.shape)} and y_true shape {tuple(labels.shape)}."
            )
        if y_pred.shape[-1] != self.vocab_size:
            raise ValueError(
                f"[{self.__class__.__name__}-compute_loss Error] Logit dimension ({y_pred.shape[-1]}) must match vocab_size ({self.vocab_size})."
            )

        labels = labels.long()
        ignore_label = getattr(self, "ignore_label", None)
        flat_logits = y_pred.reshape(-1, y_pred.size(-1))
        flat_labels = labels.reshape(-1)

        valid_mask = flat_labels.ne(self.padding_idx)
        if ignore_label is not None:
            valid_mask = valid_mask & flat_labels.ne(int(ignore_label))

        if not torch.any(valid_mask):
            return flat_logits.sum() * 0.0

        flat_logits = flat_logits[valid_mask]
        flat_labels = flat_labels[valid_mask]

        if getattr(self, "sequence_loss_mode", "ce") == "sampled_softmax":
            return self._sampled_softmax_loss(flat_logits, flat_labels)

        return F.cross_entropy(flat_logits, flat_labels, reduction="mean")

    def evaluate(
        self,
        data: dict | pd.DataFrame | DataLoader,
        metrics: list[str] | dict[str, list[str]] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        show_data_summary: bool = False,
    ) -> dict:
        """
        Evaluate sequential models with next-item ranking metrics.

        This override consumes full-vocab logits with shape [B, L, V] or [B, V],
        filters padding / ignore labels, and computes ranking-style metrics such as
        hitrate@k, recall@k, mrr@k, ndcg@k, and the top-k percent variants.
        """
        self.eval()

        eval_metrics = metrics if metrics is not None else self.metrics
        if eval_metrics is None:
            raise ValueError(
                "[BaseSequentialModel-evaluate Error] No metrics specified for evaluation. Please provide metrics parameter or call fit() first."
            )

        metric_names = flatten_metric_names(eval_metrics)
        metric_names = [metric for metric in metric_names if is_sequential_ranking_metric(metric)]
        unsupported_metrics = [metric for metric in flatten_metric_names(eval_metrics) if metric not in metric_names]
        if unsupported_metrics:
            logging.warning(
                "[BaseSequentialModel-evaluate Warning] Ignoring unsupported metrics for sequential evaluation: "
                + ", ".join(unsupported_metrics)
            )
        if not metric_names:
            raise ValueError(
                "[BaseSequentialModel-evaluate Error] No supported ranking metrics were provided. "
                "Sequential evaluation currently supports hitrate@k/recall@k/precision@k/mrr@k/ndcg@k/map@k "
                "and topk_*@percent metrics."
            )

        if isinstance(data, DataLoader):
            data_loader = data
        else:
            data_loader = self.prepare_data_loader(
                data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        metric_sums = {metric: 0.0 for metric in metric_names}
        valid_count = 0
        vocab_size: int | None = None

        with torch.no_grad():
            for batch_data in data_loader:
                X_input, y_true = self.get_input(batch_data, require_labels=True)
                y_pred = self.training_adapter.forward(self, X_input)

                if not isinstance(y_pred, torch.Tensor):
                    raise TypeError(
                        f"[BaseSequentialModel-evaluate Error] Expected sequential model output to be a torch.Tensor, got {type(y_pred)}."
                    )
                if y_true is None:
                    continue

                batch_metric_sums, batch_valid_count, batch_vocab_size = compute_sequential_metric_batch(
                    logits=y_pred,
                    labels=y_true,
                    metrics=metric_names,
                    padding_idx=self.padding_idx,
                    ignore_label=self.ignore_label,
                )
                if batch_valid_count == 0 or batch_vocab_size is None:
                    continue
                if vocab_size is None:
                    vocab_size = batch_vocab_size
                elif vocab_size != batch_vocab_size:
                    logging.warning(
                        "[BaseSequentialModel-evaluate Warning] Vocabulary size changed across batches; using the current batch size for percentage-based metrics."
                    )

                valid_count += batch_valid_count
                for metric_name, metric_value in batch_metric_sums.items():
                    metric_sums[metric_name] += metric_value

        if valid_count == 0:
            logging.info(
                colorize(
                    "  Warning: No valid sequential evaluation samples were found after filtering padding / ignore labels.",
                    color="yellow",
                )
            )
            return {"overall": {}, "grouped": []}

        metrics_dict = {metric: metric_sums[metric] / float(valid_count) for metric in metric_names}

        if show_data_summary:
            logging.info("")
            logging.info(colorize("[Data Summary]", color="cyan", bold=True))
            logging.info(colorize("-" * 80, color="cyan"))
            logging.info(format_kv("Valid next-item samples", valid_count))
            if vocab_size is not None:
                logging.info(format_kv("Vocabulary size", vocab_size))

        logging.info("")
        logging.info(colorize("[Metrics]", color="cyan", bold=True))
        logging.info(colorize("-" * 80, color="cyan"))
        for metric_name, metric_value in metrics_dict.items():
            logging.info(format_kv(metric_name, f"{metric_value:.6g}"))

        return {"overall": metrics_dict, "grouped": []}

    def _sampled_softmax_loss(self, flat_logits: torch.Tensor, flat_labels: torch.Tensor) -> torch.Tensor:
        """
        Approximate full-vocab softmax by sampling a subset of negative items.
        The positive label is kept for every example; shared negatives are sampled once per batch.
        """
        if flat_logits.numel() == 0:
            return flat_logits.sum() * 0.0

        num_negatives = int(getattr(self, "sampled_softmax_num_negatives", 0) or 0)
        if num_negatives <= 0 or num_negatives >= self.vocab_size - 1:
            return F.cross_entropy(flat_logits, flat_labels, reduction="mean")

        device = flat_logits.device
        positive_ids = torch.unique(flat_labels)

        candidate_ids = torch.arange(1, self.vocab_size, device=device)
        if positive_ids.numel() > 0:
            negative_pool = candidate_ids[~torch.isin(candidate_ids, positive_ids)]
        else:
            negative_pool = candidate_ids

        if negative_pool.numel() == 0:
            return F.cross_entropy(flat_logits, flat_labels, reduction="mean")

        if num_negatives < negative_pool.numel():
            perm = torch.randperm(negative_pool.numel(), device=device)[:num_negatives]
            sampled_negatives = negative_pool[perm]
        else:
            sampled_negatives = negative_pool

        subset = torch.cat([positive_ids, sampled_negatives], dim=0)
        subset = torch.unique(subset, sorted=True)
        subset_logits = flat_logits.index_select(dim=1, index=subset)
        remapped_labels = torch.searchsorted(subset, flat_labels)
        return F.cross_entropy(subset_logits, remapped_labels, reduction="mean")

    def select_last_valid_state(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select the hidden state at the last valid token for each sequence.

        Returns:
            last_hidden: [B, H]
            last_index: [B]
        """
        valid_lengths = (~padding_mask).sum(dim=1)
        last_index = (valid_lengths - 1).clamp(min=0)
        batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_index, last_index]
        return last_hidden, last_index

    def predict_last_logits(
        self,
        logits: torch.Tensor,
        x: dict[str, torch.Tensor],
        sequence_name: str,
        max_seq_len: int,
        padding_idx: int,
    ) -> torch.Tensor:
        """Select logits at the last valid position of each sequence."""
        _, padding_mask, _, _ = self.prepare_sequence_batch(
            x=x,
            sequence_name=sequence_name,
            max_seq_len=max_seq_len,
            padding_idx=padding_idx,
        )
        last_logits, _ = self.select_last_valid_state(logits, padding_mask)
        return last_logits

    def predict_last(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convenience wrapper for next-item inference at the last valid position.

        Sequential models conventionally produce full-sequence logits during
        forward(), and prediction usually consumes the last valid timestep.
        """
        logits = self.forward(x)
        return self.predict_last_logits(
            logits=logits,
            x=x,
            sequence_name=self.item_history_feature.name,
            max_seq_len=self.max_seq_len,
            padding_idx=self.padding_idx,
        )
