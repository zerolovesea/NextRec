"""
Base class for retrieval models.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nextrec.basic.adapters import CandidateListAdapter, RetrievalAdapter, TwoTowerAdapter
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.heads import RetrievalHead
from nextrec.basic.model import BaseModel
from nextrec.data.batch_utils import batch_to_dict, collate_fn
from nextrec.data.dataloader import RecDataLoader, TensorDictDataset, build_tensors_from_data
from nextrec.data.data_processing import get_column_data
from nextrec.utils.console import progress
from nextrec.utils.torch_utils import to_tensor
from nextrec.utils.types import LossName, OptimizerName, SchedulerName, TaskTypeInput, TaskTypeName, TrainingModeName


class BaseMatchModel(BaseModel):
    """
    Base class for retrieval models.

    - Pointwise: predicts a user-item match score/probability using labels (default target: 'label')
    - Pairwise/Listwise with explicit candidate lists: uses labels to separate positives and negatives
    - Pairwise/Listwise with in-batch negatives: labels are optional; diagonal pairs are treated as positives

    Sampling mode support:
    - 'explicit': supported by BaseModel and BaseMatchModel
    - 'inbatch': supported only by BaseMatchModel
    """

    @property
    def model_name(self) -> str:  # type: ignore[override]
        raise NotImplementedError

    supported_sampling_modes = ["explicit", "inbatch"]

    @property
    def default_task(self) -> TaskTypeName:
        return "binary"

    @property
    def supported_training_modes(self) -> list[str]:
        return ["pointwise", "pairwise", "listwise"]

    def set_adapter(self):
        if self.training_modes[0] in {"pairwise", "listwise"}:
            if self.sampling_mode == "explicit":
                self.training_adapter = CandidateListAdapter()
            elif self.sampling_mode == "inbatch":
                self.training_adapter = TwoTowerAdapter()
            else:
                self.training_adapter = RetrievalAdapter()
        else:
            self.training_adapter = RetrievalAdapter()

    def get_default_metrics_task(self) -> TaskTypeName:
        return "matching"

    def __init__(
        self,
        user_dense_features: list[DenseFeature] | None = None,
        user_sparse_features: list[SparseFeature] | None = None,
        user_sequence_features: list[SequenceFeature] | None = None,
        item_dense_features: list[DenseFeature] | None = None,
        item_sparse_features: list[SparseFeature] | None = None,
        item_sequence_features: list[SequenceFeature] | None = None,
        training_mode: TrainingModeName = "pointwise",
        num_negative_samples: int = 4,
        temperature: float = 1.0,
        similarity_metric: Literal["dot", "cosine", "euclidean"] = "dot",
        device: str = "cpu",
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        target: list[str] | str | None = "label",
        id_columns: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        session_id: str | None = None,
        distributed: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
        ddp_find_unused_parameters: bool = False,
        sampling_mode: Literal["explicit", "inbatch"] = "explicit",
        **kwargs,
    ):
        user_dense_features = list(user_dense_features or [])
        user_sparse_features = list(user_sparse_features or [])
        user_sequence_features = list(user_sequence_features or [])
        item_dense_features = list(item_dense_features or [])
        item_sparse_features = list(item_sparse_features or [])
        item_sequence_features = list(item_sequence_features or [])

        all_dense_features = user_dense_features + item_dense_features
        all_sparse_features = user_sparse_features + item_sparse_features
        all_sequence_features = user_sequence_features + item_sequence_features

        super(BaseMatchModel, self).__init__(
            dense_features=all_dense_features,
            sparse_features=all_sparse_features,
            sequence_features=all_sequence_features,
            target=target,
            id_columns=id_columns,
            task=task,
            training_mode=training_mode,
            sampling_mode=sampling_mode,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            device=device,
            session_id=session_id,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
        )

        self.user_dense_features = user_dense_features
        self.user_sparse_features = user_sparse_features
        self.user_sequence_features = user_sequence_features
        self.item_dense_features = item_dense_features
        self.item_sparse_features = item_sparse_features
        self.item_sequence_features = item_sequence_features

        self.num_negative_samples = num_negative_samples
        self.temperature = temperature
        self.similarity_metric = similarity_metric
        self.primary_mode = self.training_modes[0] if self.training_modes else "pointwise"
        if self.primary_mode not in self.supported_training_modes:
            raise ValueError(
                f"{self.model_name.upper()} does not support training_mode='{self.primary_mode}'. Supported modes: {self.supported_training_modes}"
            )
        self.user_features_all = self.user_dense_features + self.user_sparse_features + self.user_sequence_features
        self.item_features_all = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
        self.user_feature_names = {feature.name for feature in self.user_features_all}
        self.item_feature_names = {feature.name for feature in self.item_features_all}
        self.head = RetrievalHead(
            similarity_metric=self.similarity_metric,
            temperature=self.temperature,
            training_mode=self.primary_mode,
            sampling_mode=self.sampling_mode,
            apply_sigmoid=True,
        )

    def compile(
        self,
        optimizer: OptimizerName | torch.optim.Optimizer = "adam",
        optimizer_params: dict | None = None,
        scheduler: (
            SchedulerName
            | torch.optim.lr_scheduler._LRScheduler
            | torch.optim.lr_scheduler.LRScheduler
            | type[torch.optim.lr_scheduler._LRScheduler]
            | type[torch.optim.lr_scheduler.LRScheduler]
            | None
        ) = None,
        scheduler_params: dict | None = None,
        warmup: bool | dict | None = None,
        loss: LossName | nn.Module | list[LossName | nn.Module] | None = None,
        loss_params: dict | list[dict] | None = None,
        loss_weights: int | float | list[int | float] | dict | None = None,
        ignore_label: int | float | None = -1,
    ):
        return super().compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            warmup=warmup,
            loss=loss,
            loss_params=loss_params,
            loss_weights=loss_weights,
            ignore_label=ignore_label,
        )

    def inbatch_logits(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if self.similarity_metric == "dot":
            logits = torch.matmul(user_emb, item_emb.t())
        elif self.similarity_metric == "cosine":
            user_norm = F.normalize(user_emb, p=2, dim=-1)
            item_norm = F.normalize(item_emb, p=2, dim=-1)
            logits = torch.matmul(user_norm, item_norm.t())
        elif self.similarity_metric == "euclidean":
            user_sq = torch.sum(user_emb**2, dim=1, keepdim=True)
            item_sq = torch.sum(item_emb**2, dim=1, keepdim=True).t()
            logits = -(user_sq + item_sq - 2.0 * torch.matmul(user_emb, item_emb.t()))
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        return logits / self.temperature

    def compute_similarity(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if user_emb.dim() == 2 and item_emb.dim() == 3:
            user_emb = user_emb.unsqueeze(1)

        if self.similarity_metric == "dot":
            similarity = torch.sum(user_emb * item_emb, dim=-1)
        elif self.similarity_metric == "cosine":
            similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
        elif self.similarity_metric == "euclidean":
            similarity = -torch.sum((user_emb - item_emb) ** 2, dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        similarity = similarity / self.temperature
        return similarity

    def user_tower(self, user_input: dict) -> torch.Tensor:
        raise NotImplementedError

    def item_tower(self, item_input: dict) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, X_input: dict) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        user_input = {name: tensor for name, tensor in X_input.items() if name in self.user_feature_names}
        item_input = {name: tensor for name, tensor in X_input.items() if name in self.item_feature_names}

        user_emb = self.user_tower(user_input)
        item_emb = self.item_tower(item_input)

        return user_emb, item_emb

    def prepare_feature_data(
        self,
        data,
        features: list,
        batch_size: int,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, (str, os.PathLike)):
            dense_features = [f for f in features if isinstance(f, DenseFeature)]
            sparse_features = [f for f in features if isinstance(f, SparseFeature)]
            sequence_features = [f for f in features if isinstance(f, SequenceFeature)]
            rec_loader = RecDataLoader(
                dense_features=dense_features,
                sparse_features=sparse_features,
                sequence_features=sequence_features,
                target=[],
                id_columns=[],
            )
            return rec_loader.create_dataloader(
                data=data,
                batch_size=batch_size,
                shuffle=False,
                streaming=True,
                chunk_size=stream_chunk_size,
                num_workers=num_workers,
            )
        tensors = build_tensors_from_data(
            data=data,
            raw_data=data,
            features=features,
            target_columns=[],
            id_columns=[],
        )
        if tensors is None:
            raise ValueError("[BaseMatchModel-prepare_feature_data Error] No data available to create DataLoader.")
        dataset = TensorDictDataset(tensors)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def build_feature_tensors(self, feature_source: dict, features: list) -> dict:
        tensors = {}
        for feature in features:
            if feature.name not in feature_source:
                raise KeyError(f"[BaseMatchModel-feature Error] Feature '{feature.name}' not found in input data.")
            feature_data = get_column_data(feature_source, feature.name)
            tensors[feature.name] = to_tensor(
                feature_data,
                dtype=(torch.float32 if isinstance(feature, DenseFeature) else torch.long),
                device=self.device,
            )
        return tensors

    def encode_user(
        self,
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> np.ndarray:
        self.eval()
        data_loader = self.prepare_feature_data(
            data,
            self.user_features_all,
            batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        )

        embeddings_list = []
        with torch.no_grad():
            for batch_data in progress(data_loader, description="Encoding users"):
                batch_dict = batch_to_dict(batch_data, include_ids=False)
                user_input = self.build_feature_tensors(batch_dict["features"], self.user_features_all)
                user_emb = self.user_tower(user_input)
                embeddings_list.append(user_emb.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)

    def encode_item(
        self,
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> np.ndarray:
        self.eval()
        data_loader = self.prepare_feature_data(
            data,
            self.item_features_all,
            batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        )

        embeddings_list = []
        with torch.no_grad():
            for batch_data in progress(data_loader, description="Encoding items"):
                batch_dict = batch_to_dict(batch_data, include_ids=False)
                item_input = self.build_feature_tensors(batch_dict["features"], self.item_features_all)
                item_emb = self.item_tower(item_input)
                embeddings_list.append(item_emb.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)
