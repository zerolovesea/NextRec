"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Song W, Shi C, Xiao Z, et al. Autoint: Automatic feature interaction learning via 
        self-attentive neural networks[C]//Proceedings of the 28th ACM international conference 
        on information and knowledge management. 2019: 1161-1170.
        (https://arxiv.org/abs/1810.11921)
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MultiHeadSelfAttention, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class AutoInt(BaseModel):
    @property
    def model_name(self):
        return "AutoInt"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 att_layer_num: int = 3,
                 att_embedding_dim: int = 8,
                 att_head_num: int = 2,
                 att_dropout: float = 0.0,
                 att_use_residual: bool = True,
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
        
        super(AutoInt, self).__init__(
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
            
        self.att_layer_num = att_layer_num
        self.att_embedding_dim = att_embedding_dim
        
        # Use sparse and sequence features for interaction
        self.interaction_features = sparse_features + sequence_features
        
        # All features for embedding
        self.all_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)
        
        # Project embeddings to attention embedding dimension
        num_fields = len(self.interaction_features)
        total_embedding_dim = sum([f.embedding_dim for f in self.interaction_features])
        
        # If embeddings have different dimensions, project them to att_embedding_dim
        self.need_projection = not all(f.embedding_dim == att_embedding_dim for f in self.interaction_features)
        self.projection_layers = None
        if self.need_projection:
            self.projection_layers = nn.ModuleList([
                nn.Linear(f.embedding_dim, att_embedding_dim, bias=False) 
                for f in self.interaction_features
            ])
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(
                embedding_dim=att_embedding_dim,
                num_heads=att_head_num,
                dropout=att_dropout,
                use_residual=att_use_residual
            ) for _ in range(att_layer_num)
        ])
        
        # Final prediction layer
        self.fc = nn.Linear(num_fields * att_embedding_dim, 1)
        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['projection_layers', 'attention_layers', 'fc']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss
        )

    def forward(self, x):
        # Get embeddings field-by-field so mixed dimensions can be projected safely
        field_embeddings = []
        if len(self.interaction_features) == 0:
            raise ValueError("AutoInt requires at least one sparse or sequence feature for interactions.")
        for idx, feature in enumerate(self.interaction_features):
            feature_emb = self.embedding(x=x, features=[feature], squeeze_dim=False)
            feature_emb = feature_emb.squeeze(1)  # [B, embedding_dim]
            if self.need_projection and self.projection_layers is not None:
                feature_emb = self.projection_layers[idx](feature_emb)
            field_embeddings.append(feature_emb.unsqueeze(1))  # [B, 1, att_embedding_dim or original_dim]
        embeddings = torch.cat(field_embeddings, dim=1)
        
        # Apply multi-head self-attention layers
        attention_output = embeddings
        for att_layer in self.attention_layers:
            attention_output = att_layer(attention_output)  # [B, num_fields, att_embedding_dim]
        
        # Flatten and predict
        attention_output_flat = attention_output.flatten(start_dim=1)  # [B, num_fields * att_embedding_dim]
        y = self.fc(attention_output_flat)  # [B, 1]
        return self.prediction_layer(y)
