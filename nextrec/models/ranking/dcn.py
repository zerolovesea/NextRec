"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]
        //Proceedings of the ADKDD'17. 2017: 1-7.
        (https://arxiv.org/abs/1708.05123)
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MLP, CrossNetwork, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class DCN(BaseModel):
    @property
    def model_name(self):
        return "DCN"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 cross_num: int = 3,
                 mlp_params: dict | None = None,
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
        
        super(DCN, self).__init__(
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
            
        # All features
        self.all_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Calculate input dimension
        emb_dim_total = sum([f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature)])
        dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        input_dim = emb_dim_total + dense_input_dim
        
        # Cross Network
        self.cross_network = CrossNetwork(input_dim=input_dim, num_layers=cross_num)
        
        # Deep Network (optional)
        if mlp_params is not None:
            self.use_dnn = True
            self.mlp = MLP(input_dim=input_dim, **mlp_params)
            # Final layer combines cross and deep
            self.final_layer = nn.Linear(input_dim + 1, 1)  # +1 for MLP output
        else:
            self.use_dnn = False
            # Final layer only uses cross network output
            self.final_layer = nn.Linear(input_dim, 1)

        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['cross_network', 'mlp', 'final_layer']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        # Get all embeddings and flatten
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        
        # Cross Network
        cross_output = self.cross_network(input_flat)  # [B, input_dim]
        
        if self.use_dnn:
            # Deep Network
            deep_output = self.mlp(input_flat)  # [B, 1]
            # Concatenate cross and deep
            combined = torch.cat([cross_output, deep_output], dim=-1)  # [B, input_dim + 1]
        else:
            combined = cross_output
        
        # Final prediction
        y = self.final_layer(combined)
        return self.prediction_layer(y)
