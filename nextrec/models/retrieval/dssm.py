"""
Date: create on 09/11/2025
Checkpoint: edit on 18/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Huang P S, He X, Gao J, et al. Learning deep structured semantic models for web search using clickthrough data[C]
//Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013: 2333-2338.
"""

from typing import Literal

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.model import BaseMatchModel


class DSSM(BaseMatchModel):
    """
    Deep Structured Semantic Model

    Dual-tower model that encodes user and item features separately and
    computes similarity via cosine or dot product.
    """

    @property
    def model_name(self) -> str:
        return "DSSM"

    @property
    def support_training_modes(self) -> list[str]:
        return ["pointwise", "pairwise", "listwise"]

    def __init__(
        self,
        user_dense_features: list[DenseFeature] | None = None,
        user_sparse_features: list[SparseFeature] | None = None,
        user_sequence_features: list[SequenceFeature] | None = None,
        item_dense_features: list[DenseFeature] | None = None,
        item_sparse_features: list[SparseFeature] | None = None,
        item_sequence_features: list[SequenceFeature] | None = None,
        user_dnn_hidden_units: list[int] = [256, 128, 64],
        item_dnn_hidden_units: list[int] = [256, 128, 64],
        embedding_dim: int = 64,
        dnn_activation: str = "relu",
        dnn_dropout: float = 0.0,
        training_mode: Literal["pointwise", "pairwise", "listwise"] = "pointwise",
        num_negative_samples: int = 4,
        temperature: float = 1.0,
        similarity_metric: Literal["dot", "cosine", "euclidean"] = "cosine",
        device: str = "cpu",
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        early_stop_patience: int = 20,
        optimizer: str | torch.optim.Optimizer = "adam",
        optimizer_params: dict | None = None,
        scheduler: (
            str
            | torch.optim.lr_scheduler._LRScheduler
            | type[torch.optim.lr_scheduler._LRScheduler]
            | None
        ) = None,
        scheduler_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        **kwargs,
    ):

        super(DSSM, self).__init__(
            user_dense_features=user_dense_features,
            user_sparse_features=user_sparse_features,
            user_sequence_features=user_sequence_features,
            item_dense_features=item_dense_features,
            item_sparse_features=item_sparse_features,
            item_sequence_features=item_sequence_features,
            training_mode=training_mode,
            num_negative_samples=num_negative_samples,
            temperature=temperature,
            similarity_metric=similarity_metric,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            early_stop_patience=early_stop_patience,
            **kwargs,
        )

        self.embedding_dim = embedding_dim
        self.user_dnn_hidden_units = user_dnn_hidden_units
        self.item_dnn_hidden_units = item_dnn_hidden_units

        # User tower embedding layer
        user_features = []
        if user_dense_features:
            user_features.extend(user_dense_features)
        if user_sparse_features:
            user_features.extend(user_sparse_features)
        if user_sequence_features:
            user_features.extend(user_sequence_features)

        if len(user_features) > 0:
            self.user_embedding = EmbeddingLayer(user_features)

            # Compute user tower input dimension
            user_input_dim = 0
            for feat in user_dense_features or []:
                user_input_dim += 1
            for feat in user_sparse_features or []:
                user_input_dim += feat.embedding_dim
            for feat in user_sequence_features or []:
                user_input_dim += feat.embedding_dim

            # User DNN
            user_dnn_units = user_dnn_hidden_units + [embedding_dim]
            self.user_dnn = MLP(
                input_dim=user_input_dim,
                dims=user_dnn_units,
                output_layer=False,
                dropout=dnn_dropout,
                activation=dnn_activation,
            )

        # Item tower embedding layer
        item_features = []
        if item_dense_features:
            item_features.extend(item_dense_features)
        if item_sparse_features:
            item_features.extend(item_sparse_features)
        if item_sequence_features:
            item_features.extend(item_sequence_features)

        if len(item_features) > 0:
            self.item_embedding = EmbeddingLayer(item_features)

            # Compute item tower input dimension
            item_input_dim = 0
            for feat in item_dense_features or []:
                item_input_dim += 1
            for feat in item_sparse_features or []:
                item_input_dim += feat.embedding_dim
            for feat in item_sequence_features or []:
                item_input_dim += feat.embedding_dim

            # Item DNN
            item_dnn_units = item_dnn_hidden_units + [embedding_dim]
            self.item_dnn = MLP(
                input_dim=item_input_dim,
                dims=item_dnn_units,
                output_layer=False,
                dropout=dnn_dropout,
                activation=dnn_activation,
            )

        self.register_regularization_weights(
            embedding_attr="user_embedding", include_modules=["user_dnn"]
        )
        self.register_regularization_weights(
            embedding_attr="item_embedding", include_modules=["item_dnn"]
        )

        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            loss=loss,
            loss_params=loss_params,
        )

        self.to(device)

    def user_tower(self, user_input: dict) -> torch.Tensor:
        """
        User tower encodes user features into embeddings.

        Args:
            user_input: user feature dict

        Returns:
            user_emb: [batch_size, embedding_dim]
        """
        all_user_features = (
            self.user_dense_features
            + self.user_sparse_features
            + self.user_sequence_features
        )
        user_emb = self.user_embedding(user_input, all_user_features, squeeze_dim=True)

        user_emb = self.user_dnn(user_emb)

        # L2 normalize for cosine similarity
        if self.similarity_metric == "cosine":
            user_emb = torch.nn.functional.normalize(user_emb, p=2, dim=1)

        return user_emb

    def item_tower(self, item_input: dict) -> torch.Tensor:
        """
        Item tower encodes item features into embeddings.

        Args:
            item_input: item feature dict

        Returns:
            item_emb: [batch_size, embedding_dim] or [batch_size, num_items, embedding_dim]
        """
        all_item_features = (
            self.item_dense_features
            + self.item_sparse_features
            + self.item_sequence_features
        )
        item_emb = self.item_embedding(item_input, all_item_features, squeeze_dim=True)

        item_emb = self.item_dnn(item_emb)

        # L2 normalize for cosine similarity
        if self.similarity_metric == "cosine":
            item_emb = torch.nn.functional.normalize(item_emb, p=2, dim=1)

        return item_emb
