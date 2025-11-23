"""
Date: create on 27/10/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import FM, LR, EmbeddingLayer, MLP, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

class DeepFM(BaseModel):
    @property
    def model_name(self):
        return "DeepFM"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature]|list = [],
                 sparse_features: list[SparseFeature]|list = [],
                 sequence_features: list[SequenceFeature]|list = [],
                 mlp_params: dict = {},
                 target: list[str]|str = [],
                 optimizer: str = "adam",
                 optimizer_params: dict = {},
                 loss: str | nn.Module | None = "bce",
                 loss_params: dict | list[dict] | None = None,
                 device: str = 'cpu',
                 embedding_l1_reg=1e-6,
                 dense_l1_reg=1e-5,
                 embedding_l2_reg=1e-5,
                 dense_l2_reg=1e-4, **kwargs):
        
        super(DeepFM, self).__init__(
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
            
        self.fm_features = sparse_features + sequence_features
        self.deep_features = dense_features + sparse_features + sequence_features

        self.embedding = EmbeddingLayer(features=self.deep_features)

        fm_emb_dim_total = sum([f.embedding_dim for f in self.fm_features])
        deep_emb_dim_total = sum([f.embedding_dim for f in self.deep_features if not isinstance(f, DenseFeature)])
        dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])

        self.linear = LR(fm_emb_dim_total)
        self.fm = FM(reduce_sum=True)
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
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        input_deep = self.embedding(x=x, features=self.deep_features, squeeze_dim=True)
        input_fm = self.embedding(x=x, features=self.fm_features, squeeze_dim=False)

        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        y_deep = self.mlp(input_deep)  # [B, 1]

        y = y_linear + y_fm + y_deep
        return self.prediction_layer(y)
