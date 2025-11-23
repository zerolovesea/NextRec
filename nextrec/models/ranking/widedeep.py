"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]
        //Proceedings of the 1st workshop on deep learning for recommender systems. 2016: 7-10.
        (https://arxiv.org/abs/1606.07792)
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import LR, EmbeddingLayer, MLP, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class WideDeep(BaseModel):
    @property
    def model_name(self):
        return "WideDeep"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 mlp_params: dict,
                 target: list[str] = [],
                 optimizer: str = "adam",
                 optimizer_params: dict = {},
                 loss: str | nn.Module | None = "bce",
                 device: str = 'cpu',
                 embedding_l1_reg=1e-6,
                 dense_l1_reg=1e-5,
                 embedding_l2_reg=1e-5,
                 dense_l2_reg=1e-4,
                 **kwargs):
        
        super(WideDeep, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=self.task_type,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            early_stop_patience=20,
            **kwargs
        )

        self.loss = loss
        if self.loss is None:
            self.loss = "bce"
            
        # Wide part: use all features for linear model
        self.wide_features = sparse_features + sequence_features
        
        # Deep part: use all features
        self.deep_features = dense_features + sparse_features + sequence_features

        # Embedding layer for deep part
        self.embedding = EmbeddingLayer(features=self.deep_features)

        # Wide part: Linear layer
        wide_dim = sum([f.embedding_dim for f in self.wide_features])
        self.linear = LR(wide_dim)
        
        # Deep part: MLP
        deep_emb_dim_total = sum([f.embedding_dim for f in self.deep_features if not isinstance(f, DenseFeature)])
        dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        self.mlp = MLP(input_dim=deep_emb_dim_total + dense_input_dim, **mlp_params)
        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['linear', 'mlp']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss
        )

    def forward(self, x):
        # Deep part
        input_deep = self.embedding(x=x, features=self.deep_features, squeeze_dim=True)
        y_deep = self.mlp(input_deep)  # [B, 1]
        
        # Wide part
        input_wide = self.embedding(x=x, features=self.wide_features, squeeze_dim=True)
        y_wide = self.linear(input_wide)

        # Combine wide and deep
        y = y_wide + y_deep
        return self.prediction_layer(y)
