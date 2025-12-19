"""
Date: create on 09/11/2025
Checkpoint: edit on 18/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Covington P, Adams J, Sargin E. Deep neural networks for youtube recommendations[C]
//Proceedings of the 10th ACM conference on recommender systems. 2016: 191-198.
"""

from typing import Literal

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.model import BaseMatchModel


class YoutubeDNN(BaseMatchModel):
    """
    YouTube Deep Neural Network for Recommendations.
    User tower: behavior sequence + user features -> user embedding.
    Item tower: item features -> item embedding.
    Training usually uses listwise / sampled softmax style objectives.
    """

    @property
    def model_name(self) -> str:
        return "YouTubeDNN"

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
        training_mode: Literal["pointwise", "pairwise", "listwise"] = "listwise",
        num_negative_samples: int = 100,
        temperature: float = 1.0,
        similarity_metric: Literal["dot", "cosine", "euclidean"] = "dot",
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

        super(YoutubeDNN, self).__init__(
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

        # User tower
        user_features = []
        if user_dense_features:
            user_features.extend(user_dense_features)
        if user_sparse_features:
            user_features.extend(user_sparse_features)
        if user_sequence_features:
            user_features.extend(user_sequence_features)

        if len(user_features) > 0:
            self.user_embedding = EmbeddingLayer(user_features)

            user_input_dim = 0
            for feat in user_dense_features or []:
                user_input_dim += 1
            for feat in user_sparse_features or []:
                user_input_dim += feat.embedding_dim
            for feat in user_sequence_features or []:
                # Sequence features are pooled before entering the DNN
                user_input_dim += feat.embedding_dim

            user_dnn_units = user_dnn_hidden_units + [embedding_dim]
            self.user_dnn = MLP(
                input_dim=user_input_dim,
                dims=user_dnn_units,
                output_layer=False,
                dropout=dnn_dropout,
                activation=dnn_activation,
            )

        # Item tower
        item_features = []
        if item_dense_features:
            item_features.extend(item_dense_features)
        if item_sparse_features:
            item_features.extend(item_sparse_features)
        if item_sequence_features:
            item_features.extend(item_sequence_features)

        if len(item_features) > 0:
            self.item_embedding = EmbeddingLayer(item_features)

            item_input_dim = 0
            for feat in item_dense_features or []:
                item_input_dim += 1
            for feat in item_sparse_features or []:
                item_input_dim += feat.embedding_dim
            for feat in item_sequence_features or []:
                item_input_dim += feat.embedding_dim

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
        User tower to encode historical behavior sequences and user features.
        """
        all_user_features = (
            self.user_dense_features
            + self.user_sparse_features
            + self.user_sequence_features
        )
        user_emb = self.user_embedding(user_input, all_user_features, squeeze_dim=True)
        user_emb = self.user_dnn(user_emb)

        # L2 normalization
        user_emb = torch.nn.functional.normalize(user_emb, p=2, dim=1)

        return user_emb

    def item_tower(self, item_input: dict) -> torch.Tensor:
        """Item tower"""
        all_item_features = (
            self.item_dense_features
            + self.item_sparse_features
            + self.item_sequence_features
        )
        item_emb = self.item_embedding(item_input, all_item_features, squeeze_dim=True)
        item_emb = self.item_dnn(item_emb)

        # L2 normalization
        item_emb = torch.nn.functional.normalize(item_emb, p=2, dim=1)

        return item_emb
