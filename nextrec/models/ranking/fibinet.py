"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Huang T, Zhang Z, Zhang B, et al. FiBiNET: Combining feature importance and bilinear feature interaction
        for click-through rate prediction[C]//RecSys. 2019: 169-177.
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import (
    BiLinearInteractionLayer,
    EmbeddingLayer,
    LR,
    MLP,
    PredictionLayer,
    SENETLayer,
)
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class FiBiNET(BaseModel):
    @property
    def model_name(self):
        return "FiBiNET"

    @property
    def default_task(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature] | list = [],
                 sparse_features: list[SparseFeature] | list = [],
                 sequence_features: list[SequenceFeature] | list = [],
                 mlp_params: dict = {},
                 bilinear_type: str = "field_interaction",
                 senet_reduction: int = 3,
                 target: list[str] | list = [],
                 task: str | list[str] | None = None,
                 optimizer: str = "adam",
                 optimizer_params: dict = {},
                 loss: str | nn.Module | None = "bce",
                 loss_params: dict | list[dict] | None = None,
                 device: str = 'cpu',
                 embedding_l1_reg=1e-6,
                 dense_l1_reg=1e-5,
                 embedding_l2_reg=1e-5,
                 dense_l2_reg=1e-4,
                 **kwargs):
        
        super(FiBiNET, self).__init__(
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
            **kwargs
        )

        self.loss = loss
        if self.loss is None:
            self.loss = "bce"
            
        self.linear_features = sparse_features + sequence_features
        self.deep_features = dense_features + sparse_features + sequence_features
        self.interaction_features = sparse_features + sequence_features

        if len(self.interaction_features) < 2:
            raise ValueError("FiBiNET requires at least two sparse/sequence features for interactions.")

        self.embedding = EmbeddingLayer(features=self.deep_features)

        self.num_fields = len(self.interaction_features)
        self.embedding_dim = self.interaction_features[0].embedding_dim
        if any(f.embedding_dim != self.embedding_dim for f in self.interaction_features):
            raise ValueError("All interaction features must share the same embedding_dim in FiBiNET.")

        self.senet = SENETLayer(num_fields=self.num_fields, reduction_ratio=senet_reduction)
        self.bilinear_standard = BiLinearInteractionLayer(
            input_dim=self.embedding_dim,
            num_fields=self.num_fields,
            bilinear_type=bilinear_type,
        )
        self.bilinear_senet = BiLinearInteractionLayer(
            input_dim=self.embedding_dim,
            num_fields=self.num_fields,
            bilinear_type=bilinear_type,
        )

        linear_dim = sum([f.embedding_dim for f in self.linear_features])
        self.linear = LR(linear_dim)

        num_pairs = self.num_fields * (self.num_fields - 1) // 2
        interaction_dim = num_pairs * self.embedding_dim * 2
        self.mlp = MLP(input_dim=interaction_dim, **mlp_params)
        self.prediction_layer = PredictionLayer(task_type=self.default_task)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['linear', 'senet', 'bilinear_standard', 'bilinear_senet', 'mlp']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        input_linear = self.embedding(x=x, features=self.linear_features, squeeze_dim=True)
        y_linear = self.linear(input_linear)

        field_emb = self.embedding(x=x, features=self.interaction_features, squeeze_dim=False)
        senet_emb = self.senet(field_emb)

        bilinear_standard = self.bilinear_standard(field_emb).flatten(start_dim=1)
        bilinear_senet = self.bilinear_senet(senet_emb).flatten(start_dim=1)
        deep_input = torch.cat([bilinear_standard, bilinear_senet], dim=1)
        y_deep = self.mlp(deep_input)

        y = y_linear + y_deep
        return self.prediction_layer(y)
