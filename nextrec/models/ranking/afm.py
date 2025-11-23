"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of
        feature interactions via attention networks[C]//IJCAI. 2017: 3119-3125.
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, LR, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class AFM(BaseModel):
    @property
    def model_name(self):
        return "AFM"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature] | list = [],
                 sparse_features: list[SparseFeature] | list = [],
                 sequence_features: list[SequenceFeature] | list = [],
                 attention_dim: int = 32,
                 attention_dropout: float = 0.0,
                 target: list[str] | list = [],
                 optimizer: str = "adam",
                 optimizer_params: dict = {},
                 loss: str | nn.Module | None = "bce",
                 device: str = 'cpu',
                 embedding_l1_reg=1e-6,
                 dense_l1_reg=1e-5,
                 embedding_l2_reg=1e-5,
                 dense_l2_reg=1e-4, **kwargs):
        
        super(AFM, self).__init__(
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
        if len(self.fm_features) < 2:
            raise ValueError("AFM requires at least two sparse/sequence features to build pairwise interactions.")

        # Assume uniform embedding dimension across FM fields
        self.embedding_dim = self.fm_features[0].embedding_dim
        if any(f.embedding_dim != self.embedding_dim for f in self.fm_features):
            raise ValueError("All FM features must share the same embedding_dim for AFM.")

        self.embedding = EmbeddingLayer(features=self.fm_features)

        fm_input_dim = sum([f.embedding_dim for f in self.fm_features])
        self.linear = LR(fm_input_dim)

        self.attention_linear = nn.Linear(self.embedding_dim, attention_dim)
        self.attention_p = nn.Linear(attention_dim, 1, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_projection = nn.Linear(self.embedding_dim, 1, bias=False)
        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['linear', 'attention_linear', 'attention_p', 'output_projection']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss
        )

    def forward(self, x):
        field_emb = self.embedding(x=x, features=self.fm_features, squeeze_dim=False)  # [B, F, D]
        input_linear = field_emb.flatten(start_dim=1)
        y_linear = self.linear(input_linear)

        interactions = []
        num_fields = field_emb.shape[1]
        for i in range(num_fields - 1):
            vi = field_emb[:, i, :]
            for j in range(i + 1, num_fields):
                vj = field_emb[:, j, :]
                interactions.append(vi * vj)

        pair_tensor = torch.stack(interactions, dim=1)  # [B, num_pairs, D]
        attention_scores = torch.tanh(self.attention_linear(pair_tensor))
        attention_scores = self.attention_p(attention_scores)  # [B, num_pairs, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        weighted_sum = torch.sum(attention_weights * pair_tensor, dim=1)
        weighted_sum = self.attention_dropout(weighted_sum)
        y_afm = self.output_projection(weighted_sum)

        y = y_linear + y_afm
        return self.prediction_layer(y)
