"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Rendle S. Factorization machines[C]//ICDM. 2010: 995-1000.
"""

import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, FM as FMInteraction, LR, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class FM(BaseModel):
    @property
    def model_name(self):
        return "FM"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature] | list = [],
                 sparse_features: list[SparseFeature] | list = [],
                 sequence_features: list[SequenceFeature] | list = [],
                 target: list[str] | list = [],
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
        
        super(FM, self).__init__(
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
            **kwargs
        )

        self.loss = loss
        if self.loss is None:
            self.loss = "bce"
            
        self.fm_features = sparse_features + sequence_features
        if len(self.fm_features) == 0:
            raise ValueError("FM requires at least one sparse or sequence feature.")

        self.embedding = EmbeddingLayer(features=self.fm_features)

        fm_input_dim = sum([f.embedding_dim for f in self.fm_features])
        self.linear = LR(fm_input_dim)
        self.fm = FMInteraction(reduce_sum=True)
        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['linear']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        input_fm = self.embedding(x=x, features=self.fm_features, squeeze_dim=False)
        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        y = y_linear + y_fm
        return self.prediction_layer(y)
