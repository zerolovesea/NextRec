"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//ICDM. 2016: 1149-1154.
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MLP, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class PNN(BaseModel):
    @property
    def model_name(self):
        return "PNN"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | list = [],
        sparse_features: list[SparseFeature] | list = [],
        sequence_features: list[SequenceFeature] | list = [],
        mlp_params: dict = {},
        product_type: str = "inner",
        outer_product_dim: int | None = None,
        target: list[str] | list = [],
        task: str | list[str] | None = None,
        optimizer: str = "adam",
        optimizer_params: dict = {},
        loss: str | nn.Module | None = "bce",
        loss_params: dict | list[dict] | None = None,
        device: str = "cpu",
        embedding_l1_reg=1e-6,
        dense_l1_reg=1e-5,
        embedding_l2_reg=1e-5,
        dense_l2_reg=1e-4,
        **kwargs,
    ):

        super(PNN, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task or self.default_task,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        self.loss = loss
        if self.loss is None:
            self.loss = "bce"

        self.field_features = sparse_features + sequence_features
        if len(self.field_features) < 2:
            raise ValueError("PNN requires at least two sparse/sequence features.")

        self.embedding = EmbeddingLayer(features=self.field_features)
        self.num_fields = len(self.field_features)
        self.embedding_dim = self.field_features[0].embedding_dim
        if any(f.embedding_dim != self.embedding_dim for f in self.field_features):
            raise ValueError(
                "All field features must share the same embedding_dim for PNN."
            )

        self.product_type = product_type.lower()
        if self.product_type not in {"inner", "outer"}:
            raise ValueError("product_type must be 'inner' or 'outer'.")

        self.num_pairs = self.num_fields * (self.num_fields - 1) // 2
        if self.product_type == "outer":
            self.outer_dim = outer_product_dim or self.embedding_dim
            self.kernel = nn.Linear(self.embedding_dim, self.outer_dim, bias=False)
            product_dim = self.num_pairs * self.outer_dim
        else:
            self.outer_dim = None
            product_dim = self.num_pairs

        linear_dim = self.num_fields * self.embedding_dim
        self.mlp = MLP(input_dim=linear_dim + product_dim, **mlp_params)
        self.prediction_layer = PredictionLayer(task_type=self.task)

        modules = ["mlp"]
        if self.product_type == "outer":
            modules.append("kernel")
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=modules
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        field_emb = self.embedding(x=x, features=self.field_features, squeeze_dim=False)
        linear_signal = field_emb.flatten(start_dim=1)

        if self.product_type == "inner":
            interactions = []
            for i in range(self.num_fields - 1):
                vi = field_emb[:, i, :]
                for j in range(i + 1, self.num_fields):
                    vj = field_emb[:, j, :]
                    interactions.append(torch.sum(vi * vj, dim=1, keepdim=True))
            product_signal = torch.cat(interactions, dim=1)
        else:
            transformed = self.kernel(field_emb)  # [B, F, outer_dim]
            interactions = []
            for i in range(self.num_fields - 1):
                vi = transformed[:, i, :]
                for j in range(i + 1, self.num_fields):
                    vj = transformed[:, j, :]
                    interactions.append(vi * vj)
            product_signal = torch.stack(interactions, dim=1).flatten(start_dim=1)

        deep_input = torch.cat([linear_signal, product_signal], dim=1)
        y = self.mlp(deep_input)
        return self.prediction_layer(y)
