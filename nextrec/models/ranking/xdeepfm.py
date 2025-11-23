"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Lian J, Zhou X, Zhang F, et al. xdeepfm: Combining explicit and implicit feature interactions 
        for recommender systems[C]//Proceedings of the 24th ACM SIGKDD international conference on 
        knowledge discovery & data mining. 2018: 1754-1763.
        (https://arxiv.org/abs/1803.05170)
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import LR, EmbeddingLayer, MLP, CIN, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class xDeepFM(BaseModel):
    @property
    def model_name(self):
        return "xDeepFM"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 mlp_params: dict,
                 cin_size: list[int] = [128, 128],
                 split_half: bool = True,
                 target: list[str] = [],
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
        
        super(xDeepFM, self).__init__(
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
            
        # Linear part and CIN part: use sparse and sequence features
        self.linear_features = sparse_features + sequence_features
        
        # Deep part: use all features
        self.deep_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.deep_features)

        # Linear part
        linear_dim = sum([f.embedding_dim for f in self.linear_features])
        self.linear = LR(linear_dim)
        
        # CIN part: Compressed Interaction Network
        num_fields = len(self.linear_features)
        self.cin = CIN(input_dim=num_fields, cin_size=cin_size, split_half=split_half)
        
        # Deep part: DNN
        deep_emb_dim_total = sum([f.embedding_dim for f in self.deep_features if not isinstance(f, DenseFeature)])
        dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        self.mlp = MLP(input_dim=deep_emb_dim_total + dense_input_dim, **mlp_params)
        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['linear', 'cin', 'mlp']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        # Get embeddings for linear and CIN (sparse features only)
        input_linear = self.embedding(x=x, features=self.linear_features, squeeze_dim=False)
        
        # Linear part
        y_linear = self.linear(input_linear.flatten(start_dim=1))
        
        # CIN part
        y_cin = self.cin(input_linear)  # [B, 1]
        
        # Deep part
        input_deep = self.embedding(x=x, features=self.deep_features, squeeze_dim=True)
        y_deep = self.mlp(input_deep)  # [B, 1]

        # Combine all parts
        y = y_linear + y_cin + y_deep
        return self.prediction_layer(y)
