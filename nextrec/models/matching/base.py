"""
Base class for matching models.
"""

from __future__ import annotations

import os
import logging
import inspect
import io
import json
from contextlib import redirect_stdout
from typing import Literal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nextrec.basic.adapters import CandidateListAdapter, MatchingAdapter, TwoTowerAdapter
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.heads import MatchingHead
from nextrec.basic.loggers import colorize
from nextrec.basic.session import get_save_path
from nextrec.engine.model import Model as BaseModel
from nextrec.data.batch_utils import collate_fn
from nextrec.data.dataloader import RecDataLoader, TensorDictDataset, build_tensors_from_data
from nextrec.data.data_processing import get_column_data
from nextrec.utils.onnx_utils import create_dummy_inputs
from nextrec.utils.matching import (
    EmbeddingTable,
    FaissVectorIndex,
    compute_full_corpus_recall_metrics,
)
from nextrec.utils.console import progress
from nextrec.utils.torch_utils import to_tensor
from nextrec.utils.types import (
    LossName,
    OptimizerName,
    SchedulerName,
    TaskTypeInput,
    TaskTypeName,
    TrainingModeName,
)


class TowerOnnxWrapper(nn.Module):
    """Wrap one matching tower so it can be exported as a standalone ONNX graph."""

    def __init__(
        self,
        model,
        tower: Literal["user", "item"],
        feature_names: list[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.tower = tower
        self.feature_names = feature_names

    def forward(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.feature_names):
            raise ValueError("[TowerOnnxWrapper Error] Number of inputs does not match feature names.")
        x = {name: tensor for name, tensor in zip(self.feature_names, inputs)}
        if self.tower == "user":
            return self.model.user_tower(x)
        if self.tower == "item":
            return self.model.item_tower(x)
        raise ValueError("[TowerOnnxWrapper Error] tower must be either 'user' or 'item'.")


class BaseMatchModel(BaseModel):
    """
    Base class for matching models.

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
    def model_family(self) -> str:
        return "matching"

    @property
    def supported_training_modes(self) -> list[str]:
        return ["pointwise", "pairwise", "listwise"]

    def set_adapter(self):
        if self.training_mode in {"pairwise", "listwise"}:
            if self.sampling_mode == "explicit":
                self.training_adapter = CandidateListAdapter()
            elif self.sampling_mode == "inbatch":
                self.training_adapter = TwoTowerAdapter()
            else:
                self.training_adapter = MatchingAdapter()
        else:
            self.training_adapter = MatchingAdapter()

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
        key_columns: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        session_id: str | None = None,
        sampling_mode: Literal["explicit", "inbatch"] = "explicit",
        **kwargs,
    ):
        self.num_negative_samples = num_negative_samples
        self.temperature = temperature
        self.similarity_metric = similarity_metric

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
            key_columns=key_columns,
            task=task,
            training_mode=training_mode,
            sampling_mode=sampling_mode,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            device=device,
            session_id=session_id,
        )

        self.user_dense_features = user_dense_features
        self.user_sparse_features = user_sparse_features
        self.user_sequence_features = user_sequence_features
        self.item_dense_features = item_dense_features
        self.item_sparse_features = item_sparse_features
        self.item_sequence_features = item_sequence_features

        if self.training_mode not in self.supported_training_modes:
            raise ValueError(
                f"{self.model_name.upper()} does not support training_mode='{self.training_mode}'. Supported modes: {self.supported_training_modes}"
            )
        self.user_features_all = self.user_dense_features + self.user_sparse_features + self.user_sequence_features
        self.item_features_all = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
        if self.training_mode in {"pairwise", "listwise"} and self.sampling_mode == "explicit":
            self.feature_scopes = {feature.name: "shared" for feature in self.user_features_all}
            self.feature_scopes.update({feature.name: "candidate" for feature in self.item_features_all})
        else:
            self.feature_scopes = {feature.name: "shared" for feature in self.all_features}
        self.user_feature_names = {feature.name for feature in self.user_features_all}
        self.item_feature_names = {feature.name for feature in self.item_features_all}

    def search_user_vectors(
        self,
        index,
        user_embeddings: np.ndarray,
        top_k: int,
        exclude_ids: list[set[str]] | None = None,
        user_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for user vectors in the given index.

        Args:
            index: The FaissVectorIndex to search.
            user_embeddings: User embeddings to search for. [B, D] for single-vector or [B, N, D] for multi-vector per user.
            top_k (int): Number of top results to return.
            exclude_ids (list[set[str]] | None): item ids to exclude from the search.
            user_ids (np.ndarray | None): User ids corresponding to the embeddings.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Retrieved ids, scores, and user IDs.
        """
        raw_embeddings = np.asarray(user_embeddings, dtype=np.float32)
        expanded_exclude_ids = exclude_ids

        if raw_embeddings.ndim == 2:  # [B, D] single vector per user
            flat_embeddings = raw_embeddings
            group_keys = np.asarray(user_ids).astype(str)
        else:
            batch_size, num_vectors, dim = raw_embeddings.shape  # [B, N, D] multiple vectors per user
            flat_embeddings = raw_embeddings.reshape(batch_size * num_vectors, dim)
            group_keys = np.repeat(np.asarray(user_ids).astype(str), num_vectors)

        needs_grouping = len(pd.unique(group_keys)) != len(group_keys)
        if needs_grouping and exclude_ids is not None:
            grouped_user_ids = pd.unique(group_keys)
            if len(exclude_ids) != len(grouped_user_ids):
                raise ValueError("[BaseMatchModel-search Error] exclude_ids length must match grouped user count.")
            exclude_by_user = {user_id: blocked_ids for user_id, blocked_ids in zip(grouped_user_ids, exclude_ids)}
            expanded_exclude_ids = [exclude_by_user[user_id] for user_id in group_keys]

        item_ids, scores = index.search(
            flat_embeddings,
            top_k=top_k,
            exclude_ids=expanded_exclude_ids,
        )

        if needs_grouping:
            grouped_user_ids = pd.unique(group_keys)
            grouped_ids = np.full((len(grouped_user_ids), int(top_k)), "", dtype=object)
            grouped_scores = np.full((len(grouped_user_ids), int(top_k)), -np.inf, dtype=np.float32)

            for group_idx, group_key in enumerate(grouped_user_ids):
                row_positions = np.flatnonzero(group_keys == group_key)
                best_by_id: dict[str, float] = {}
                for row_pos in row_positions:
                    for item_id, score in zip(item_ids[row_pos], scores[row_pos]):
                        item_id = str(item_id)
                        score = float(score)
                        if item_id not in best_by_id or score > best_by_id[item_id]:
                            best_by_id[item_id] = score
                ranked = sorted(best_by_id.items(), key=lambda item: item[1], reverse=True)[: int(top_k)]
                for rank_idx, (item_id, score) in enumerate(ranked):
                    grouped_ids[group_idx, rank_idx] = item_id
                    grouped_scores[group_idx, rank_idx] = score

            return grouped_ids, grouped_scores, grouped_user_ids

        return item_ids, scores, user_ids

    def set_head(self):
        self.head = MatchingHead(
            similarity_metric=self.similarity_metric,
            temperature=self.temperature,
            training_mode=self.training_mode,
            sampling_mode=self.sampling_mode,
            apply_sigmoid=True,
        )

    def call_model(self, X_input: dict[str, torch.Tensor]):
        """
        Call a matching model and apply its matching head.
        """
        raw_output = self(X_input)
        if self.head is None:
            return raw_output
        if isinstance(raw_output, (tuple, list)) and len(raw_output) == 2:
            user_emb, item_emb = raw_output
            return self.head(user_emb, item_emb, similarity_fn=self.compute_similarity)
        raise ValueError(
            "[BaseMatchModel-call_model Error] Matching models must return "
            "(user_emb, item_emb) before head processing."
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
        user_input = user_emb
        item_input = item_emb

        if user_emb.dim() == 2 and item_emb.dim() == 3:
            user_emb = user_emb.unsqueeze(1)
        elif user_emb.dim() == 3 and item_emb.dim() == 2:
            item_emb = item_emb.unsqueeze(1)
        elif user_emb.dim() == 3 and item_emb.dim() == 3:
            user_emb = user_emb.unsqueeze(2)
            item_emb = item_emb.unsqueeze(1)

        if self.similarity_metric == "dot":
            similarity = torch.sum(user_emb * item_emb, dim=-1)
        elif self.similarity_metric == "cosine":
            similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
        elif self.similarity_metric == "euclidean":
            similarity = -torch.sum((user_emb - item_emb) ** 2, dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        if user_input.dim() == 3 and item_input.dim() == 2:
            similarity = similarity.max(dim=1).values
        elif user_input.dim() == 3 and item_input.dim() == 3 and similarity.dim() == 3:
            similarity = similarity.max(dim=1).values
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
        key_columns: str | list[str] | None = None,
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
                key_columns=key_columns,
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
            key_columns=key_columns,
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

    def encode_tower(
        self,
        tower: Literal["user", "item"],
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        id_column: str,
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> EmbeddingTable:
        if id_column is None:
            raise ValueError("[BaseMatchModel-encode Error] id_column is required for tower embedding tables.")

        ids_list = []
        embeddings_list = []
        for ids, embeddings in self.iter_tower_embeddings(
            tower=tower,
            data=data,
            id_column=id_column,
            batch_size=batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        ):
            ids_list.append(ids)
            embeddings_list.append(embeddings)
        if not embeddings_list:
            raise ValueError("[BaseMatchModel-encode Error] No embeddings were produced from input data.")
        ids = np.concatenate(ids_list, axis=0)
        embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)
        return EmbeddingTable(
            ids=ids,
            embeddings=embeddings,
            id_column=id_column,
            tower=tower,
            metric=self.similarity_metric,
        )

    def encode_user(
        self,
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        id_column: str,
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> np.ndarray:
        table = self.encode_tower(
            tower="user",
            data=data,
            id_column=id_column,
            batch_size=batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        )
        return table.embeddings

    def encode_item(
        self,
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        id_column: str,
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> np.ndarray:
        table = self.encode_tower(
            tower="item",
            data=data,
            id_column=id_column,
            batch_size=batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        )
        return table.embeddings

    def iter_tower_embeddings(
        self,
        tower: Literal["user", "item"],
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        id_column: str,
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ):
        if id_column is None:
            raise ValueError("[BaseMatchModel-encode Error] id_column is required for tower embedding iteration.")
        if tower == "user":
            features = self.user_features_all
            tower_fn = self.user_tower
        elif tower == "item":
            features = self.item_features_all
            tower_fn = self.item_tower
        else:
            raise ValueError("[BaseMatchModel-encode Error] tower must be either 'user' or 'item'.")

        self.eval()
        data_loader = self.prepare_feature_data(
            data,
            features,
            batch_size,
            key_columns=id_column,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        )
        with torch.no_grad():
            for batch_data in progress(data_loader, description=f"Encoding {tower}s"):
                tower_input = self.build_feature_tensors(batch_data["features"], features)
                tower_emb = tower_fn(tower_input).cpu().numpy()
                keys = batch_data.get("keys") or {}
                if id_column not in keys:
                    raise KeyError(f"[BaseMatchModel-encode Error] id_column '{id_column}' not found in batch keys.")
                ids = np.asarray(keys[id_column]).reshape(-1)
                tower_emb = np.asarray(tower_emb, dtype=np.float32)
                if tower_emb.ndim == 2:
                    tower_emb = tower_emb[:, None, :]
                elif tower_emb.ndim != 3:
                    raise ValueError(
                        "[BaseMatchModel-encode Error] tower embeddings must be 2D or 3D, "
                        f"got shape {tuple(tower_emb.shape)}."
                    )
                batch_size, num_vectors, dim = tower_emb.shape
                ids = np.repeat(ids, num_vectors)
                tower_emb = tower_emb.reshape(batch_size * num_vectors, dim)
                yield ids, tower_emb

    def export_embedding_table(
        self,
        tower: Literal["user", "item"],
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        id_column: str,
        save_dir: str | os.PathLike | None = None,
        save_format: Literal["parquet", "csv"] = "parquet",
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
        streaming: bool = False,
    ) -> Path:
        """
        Export EmbeddingTable, saved to disk in the specified format.
        """
        if save_format not in {"parquet", "csv"}:
            raise ValueError("[BaseMatchModel-export-embedding Error] save_format must be one of: parquet, csv.")

        target_dir = Path(save_dir) if save_dir is not None else self.session.root / "embeddings" / tower
        target_dir.mkdir(parents=True, exist_ok=True)

        if streaming:
            data_path = target_dir / f"{tower}_embeddings.{save_format}"
            if data_path.exists():
                data_path.unlink()
            parquet_writer = None
            wrote_csv_header = False
            row_count = 0
            dim = None

            for ids, embeddings in self.iter_tower_embeddings(
                tower=tower,
                data=data,
                id_column=id_column,
                batch_size=batch_size,
                num_workers=num_workers,
                stream_chunk_size=stream_chunk_size,
            ):
                table_batch = EmbeddingTable(
                    ids=ids,
                    embeddings=embeddings,
                    id_column=id_column,
                    tower=tower,
                    metric=self.similarity_metric,
                )
                df_batch = table_batch.to_dataframe()
                row_count += int(table_batch.embeddings.shape[0])
                if dim is None:
                    dim = table_batch.dim
                elif table_batch.dim != dim:
                    raise ValueError(
                        f"[BaseMatchModel-export-embedding Error] embedding dim mismatch: expected {dim}, got {table_batch.dim}."
                    )
                if save_format == "csv":
                    df_batch.to_csv(data_path, mode="a", header=not wrote_csv_header, index=False)
                    wrote_csv_header = True
                else:
                    arrow_table = pa.Table.from_pandas(df_batch, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(data_path, arrow_table.schema)
                    parquet_writer.write_table(arrow_table)

            if parquet_writer is not None:
                parquet_writer.close()
            if row_count == 0 or dim is None:
                raise ValueError("[BaseMatchModel-export-embedding Error] No embeddings were produced from input data.")

            metadata = {
                "tower": tower,
                "id_column": id_column,
                "metric": self.similarity_metric,
                "row_count": row_count,
                "dim": dim,
                "format": save_format,
                "model_name": self.model_name,
                "similarity_metric": self.similarity_metric,
            }
            (target_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logging.info(colorize(f"{tower.capitalize()} embedding table exported to: {data_path}", color="green"))
            return data_path

        table = self.encode_tower(
            tower=tower,
            data=data,
            id_column=id_column,
            batch_size=batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        )
        if not isinstance(table, EmbeddingTable):
            raise RuntimeError(
                "[BaseMatchModel-export-embedding Error] Expected encode_tower to return EmbeddingTable."
            )
        path = table.save(
            target_dir,
            save_format=save_format,
            metadata={"model_name": self.model_name, "similarity_metric": self.similarity_metric},
        )
        logging.info(colorize(f"{tower.capitalize()} embedding table exported to: {path}", color="green"))
        return path

    def build_item_index(
        self,
        item_data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike] | None = None,
        embedding_table: EmbeddingTable | str | os.PathLike | None = None,
        id_column: str | None = None,
        batch_size: int = 512,
        save_dir: str | os.PathLike | None = None,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ):
        if item_data is not None and embedding_table is not None:
            raise ValueError("[BaseMatchModel-index Error] Provide only one of item_data or embedding_table.")
        if item_data is None and embedding_table is None:
            raise ValueError("[BaseMatchModel-index Error] Provide embedding_table or item_data for indexing.")

        if isinstance(embedding_table, EmbeddingTable) and id_column is None:
            id_column = embedding_table.id_column
        if item_data is not None and id_column is None:
            raise ValueError("[BaseMatchModel-index Error] id_column is required for item indexing.")

        index = FaissVectorIndex(metric=self.similarity_metric)
        index.id_column = id_column or "item_id"

        if isinstance(embedding_table, EmbeddingTable):
            if embedding_table.metric != self.similarity_metric:
                raise ValueError(
                    "[BaseMatchModel-index Error] embedding_table metric "
                    f"'{embedding_table.metric}' does not match model similarity_metric '{self.similarity_metric}'."
                )
            index.add(embedding_table.ids, embedding_table.embeddings)
        elif embedding_table is not None:
            for table_batch in EmbeddingTable.iter_batches(
                embedding_table,
                id_column=id_column,
                tower="item",
                batch_size=batch_size,
            ):
                if table_batch.metric != self.similarity_metric:
                    raise ValueError(
                        "[BaseMatchModel-index Error] embedding_table metric "
                        f"'{table_batch.metric}' does not match model similarity_metric '{self.similarity_metric}'."
                    )
                index.id_column = table_batch.id_column
                index.add(table_batch.ids, table_batch.embeddings)
        else:
            for ids, embeddings in self.iter_tower_embeddings(
                "item",
                item_data,
                id_column=id_column,
                batch_size=batch_size,
                num_workers=num_workers,
                stream_chunk_size=stream_chunk_size,
            ):
                index.add(ids, embeddings)

        if save_dir is not None:
            index.save(save_dir)
        return index

    def search(
        self,
        index,
        user_data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike] | None = None,
        user_embeddings: np.ndarray | None = None,
        user_id_column: str | None = None,
        top_k: int = 10,
        batch_size: int = 512,
        exclude_ids: list[set[str]] | None = None,
        return_dataframe: bool = True,
        save_path: str | os.PathLike | None = None,
        save_format: Literal["csv", "parquet"] = "csv",
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ):
        if save_path is not None and not return_dataframe:
            if user_data is None or user_id_column is None:
                raise ValueError(
                    "[BaseMatchModel-search Error] user_data and user_id_column are required when saving retrieval results."
                )
            return self.search_streaming(
                index=index,
                user_data=user_data,
                user_id_column=user_id_column,
                save_path=save_path,
                top_k=top_k,
                batch_size=batch_size,
                save_format=save_format,
                num_workers=num_workers,
                stream_chunk_size=stream_chunk_size,
                exclude_ids=exclude_ids,
            )

        user_ids = None
        if user_embeddings is not None:
            if user_data is None or user_id_column is None:
                raise ValueError(
                    "[BaseMatchModel-search Error] user_data and user_id_column are required with user_embeddings."
                )
            user_ids = pd.DataFrame(user_data)[user_id_column].astype(str).to_numpy()
        else:
            if user_data is None or user_id_column is None:
                raise ValueError("[BaseMatchModel-search Error] user_data and user_id_column are required.")
            encoded = self.encode_tower(
                "user",
                user_data,
                id_column=user_id_column,
                batch_size=batch_size,
                num_workers=num_workers,
                stream_chunk_size=stream_chunk_size,
            )
            user_embeddings = encoded.embeddings
            user_ids = encoded.ids.astype(str)
        retrieved_ids, scores, user_ids = self.search_user_vectors(
            index=index,
            user_embeddings=user_embeddings,
            top_k=top_k,
            exclude_ids=exclude_ids,
            user_ids=user_ids,
        )
        if not return_dataframe:
            return retrieved_ids, scores

        rows = []
        item_id_column = getattr(index, "id_column", "item_id")
        for row_idx in range(retrieved_ids.shape[0]):
            for rank_idx in range(retrieved_ids.shape[1]):
                row = {
                    "query_index": row_idx,
                    user_id_column: str(user_ids[row_idx]),
                    item_id_column: str(retrieved_ids[row_idx, rank_idx]),
                    "score": float(scores[row_idx, rank_idx]),
                    "rank": rank_idx + 1,
                }
                rows.append(row)
        result = pd.DataFrame(rows)
        if save_path is not None:
            target_path = Path(save_path)
            if target_path.suffix == "":
                target_path = target_path.with_suffix(f".{save_format}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if save_format == "csv":
                result.to_csv(target_path, index=False)
            elif save_format == "parquet":
                result.to_parquet(target_path, index=False)
            else:
                raise ValueError("[BaseMatchModel-search Error] save_format must be either 'csv' or 'parquet'.")
            logging.info(colorize(f"Matching results saved to: {target_path}", color="green"))
        return result

    def search_streaming(
        self,
        index,
        user_data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        user_id_column: str,
        save_path: str | os.PathLike,
        top_k: int = 10,
        batch_size: int = 512,
        save_format: Literal["csv", "parquet"] = "csv",
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
        exclude_ids: list[set[str]] | None = None,
    ) -> Path:
        target_path = Path(save_path)
        if target_path.suffix == "":
            target_path = target_path.with_suffix(f".{save_format}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if save_format not in {"csv", "parquet"}:
            raise ValueError("[BaseMatchModel-search Error] save_format must be either 'csv' or 'parquet'.")

        item_id_column = getattr(index, "id_column", "item_id")
        parquet_writer = None
        wrote_csv_header = False

        query_offset = 0
        for ids, embeddings in self.iter_tower_embeddings(
            "user",
            user_data,
            id_column=user_id_column,
            batch_size=batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        ):
            batch_start = query_offset
            batch_user_count = len(pd.unique(np.asarray(ids).astype(str)))
            batch_end = batch_start + batch_user_count
            batch_exclude_ids = None
            if exclude_ids is not None:
                batch_exclude_ids = exclude_ids[batch_start:batch_end]
            query_offset = batch_end
            retrieved_ids, scores, ids = self.search_user_vectors(
                index=index,
                user_embeddings=embeddings,
                top_k=top_k,
                exclude_ids=batch_exclude_ids,
                user_ids=ids,
            )
            rows = []
            for row_idx in range(retrieved_ids.shape[0]):
                for rank_idx in range(retrieved_ids.shape[1]):
                    rows.append(
                        {
                            user_id_column: str(ids[row_idx]),
                            item_id_column: str(retrieved_ids[row_idx, rank_idx]),
                            "score": float(scores[row_idx, rank_idx]),
                            "rank": rank_idx + 1,
                        }
                    )
            df_batch = pd.DataFrame(rows)
            if save_format == "csv":
                df_batch.to_csv(target_path, mode="a", header=not wrote_csv_header, index=False)
                wrote_csv_header = True
            else:
                table = pa.Table.from_pandas(df_batch, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(target_path, table.schema)
                parquet_writer.write_table(table)

        if parquet_writer is not None:
            parquet_writer.close()
        logging.info(colorize(f"Retrieval results saved to: {target_path}", color="green"))
        return target_path

    def full_corpus_evaluate(
        self,
        eval_data: pd.DataFrame | dict,
        item_data: pd.DataFrame | dict | DataLoader | str | os.PathLike | list[str | os.PathLike],
        user_col: str,
        item_col: str,
        top_k: list[int] | tuple[int, ...] = (5, 10, 20),
        train_interactions: pd.DataFrame | dict | None = None,
        exclude_seen: bool = True,
        batch_size: int = 512,
    ) -> dict[str, float]:
        eval_df = pd.DataFrame(eval_data)
        if user_col not in eval_df.columns or item_col not in eval_df.columns:
            raise KeyError("[BaseMatchModel-full-corpus-eval Error] eval_data must contain user_col and item_col.")
        eval_df = eval_df.copy()
        eval_df[user_col] = eval_df[user_col].astype(str)
        eval_df[item_col] = eval_df[item_col].astype(str)
        user_df = eval_df.drop_duplicates(user_col)
        positive_by_user = (
            eval_df.groupby(user_col)[item_col].apply(lambda values: {str(value) for value in values}).to_dict()
        )
        user_table = self.encode_tower(
            "user",
            user_df,
            id_column=user_col,
            batch_size=batch_size,
        )

        item_index = self.build_item_index(
            item_data=item_data,
            id_column=item_col,
            batch_size=batch_size,
        )
        eval_user_ids = pd.unique(user_table.ids.astype(str))
        exclude_sets = None
        if exclude_seen and train_interactions is not None:
            train_df = pd.DataFrame(train_interactions).copy()
            train_df[user_col] = train_df[user_col].astype(str)
            train_df[item_col] = train_df[item_col].astype(str)
            seen_by_user = (
                train_df.groupby(user_col)[item_col].apply(lambda values: {str(value) for value in values}).to_dict()
            )
            exclude_sets = [seen_by_user.get(str(user_id), set()) for user_id in eval_user_ids]

        retrieved_ids, _, grouped_user_ids = self.search_user_vectors(
            index=item_index,
            user_embeddings=user_table.embeddings,
            top_k=max(top_k),
            exclude_ids=exclude_sets,
            user_ids=user_table.ids.astype(str),
        )
        metric_user_ids = grouped_user_ids.astype(str)
        positive_ids = [positive_by_user.get(user_id, set()) for user_id in metric_user_ids]
        return compute_full_corpus_recall_metrics(
            query_ids=metric_user_ids,
            positive_ids=positive_ids,
            retrieved_ids=retrieved_ids,
            top_k=list(top_k),
        )

    @torch.no_grad()
    def export_tower_onnx(
        self,
        tower: Literal["user", "item"],
        save_path: str | Path | None = None,
        batch_size: int = 1,
    ) -> Path:
        if tower == "user":
            features = self.user_features_all
        elif tower == "item":
            features = self.item_features_all
        else:
            raise ValueError("[BaseMatchModel-export-tower Error] tower must be either 'user' or 'item'.")

        self.to(self.device)
        self.eval()
        input_names = [feature.name for feature in features]
        dummy_inputs = create_dummy_inputs(features, batch_size=batch_size, device=self.device)
        wrapper = TowerOnnxWrapper(self, tower=tower, feature_names=input_names)
        with torch.no_grad():
            wrapper(*dummy_inputs)

        target_path = get_save_path(
            path=save_path,
            default_dir=self.session.root,
            default_name=f"{self.model_name}_{tower}_tower_onnx",
            suffix="onnx",
        )
        export_kwargs = {}
        export_sig = inspect.signature(torch.onnx.export)
        if "dynamo" in export_sig.parameters:
            export_kwargs["dynamo"] = True

        with redirect_stdout(io.StringIO()):
            torch.onnx.export(
                wrapper,
                tuple(dummy_inputs),
                target_path,
                input_names=input_names,
                output_names=[f"{tower}_embedding"],
                opset_version=18,
                do_constant_folding=True,
                **export_kwargs,
            )
        logging.info(colorize(f"{tower.capitalize()} tower ONNX exported to: {target_path}", color="green"))
        return Path(target_path)
