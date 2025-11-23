"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]
        //Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018: 1059-1068.
        (https://arxiv.org/abs/1706.06978)
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MLP, AttentionPoolingLayer, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class DIN(BaseModel):
    @property
    def model_name(self):
        return "DIN"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 mlp_params: dict,
                 attention_hidden_units: list[int] = [80, 40],
                 attention_activation: str = 'sigmoid',
                 attention_use_softmax: bool = True,
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
        
        super(DIN, self).__init__(
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
        
        # Features classification
        # DIN requires: candidate item + user behavior sequence + other features
        if len(sequence_features) == 0:
            raise ValueError("DIN requires at least one sequence feature for user behavior history")
        
        self.behavior_feature = sequence_features[0]  # User behavior sequence
        self.candidate_feature = sparse_features[-1] if sparse_features else None  # Candidate item
        
        # Other features (excluding behavior sequence in final concatenation)
        self.other_sparse_features = sparse_features[:-1] if self.candidate_feature else sparse_features
        self.dense_features_list = dense_features
        
        # All features for embedding
        self.all_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)
        
        # Attention layer for behavior sequence
        behavior_emb_dim = self.behavior_feature.embedding_dim
        self.candidate_attention_proj = None
        if self.candidate_feature is not None and self.candidate_feature.embedding_dim != behavior_emb_dim:
            self.candidate_attention_proj = nn.Linear(self.candidate_feature.embedding_dim, behavior_emb_dim)
        self.attention = AttentionPoolingLayer(
            embedding_dim=behavior_emb_dim,
            hidden_units=attention_hidden_units,
            activation=attention_activation,
            use_softmax=attention_use_softmax
        )
        
        # Calculate MLP input dimension
        # candidate + attention_pooled_behavior + other_sparse + dense
        mlp_input_dim = 0
        if self.candidate_feature:
            mlp_input_dim += self.candidate_feature.embedding_dim
        mlp_input_dim += behavior_emb_dim  # attention pooled
        mlp_input_dim += sum([f.embedding_dim for f in self.other_sparse_features])
        mlp_input_dim += sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        
        # MLP for final prediction
        self.mlp = MLP(input_dim=mlp_input_dim, **mlp_params)
        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['attention', 'mlp', 'candidate_attention_proj']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss
        )

    def forward(self, x):
        # Get candidate item embedding
        if self.candidate_feature:
            candidate_emb = self.embedding.embed_dict[self.candidate_feature.embedding_name](
                x[self.candidate_feature.name].long()
            )  # [B, emb_dim]
        else:
            candidate_emb = None
        
        # Get behavior sequence embedding
        behavior_seq = x[self.behavior_feature.name].long()  # [B, seq_len]
        behavior_emb = self.embedding.embed_dict[self.behavior_feature.embedding_name](
            behavior_seq
        )  # [B, seq_len, emb_dim]
        
        # Create mask for padding
        if self.behavior_feature.padding_idx is not None:
            mask = (behavior_seq != self.behavior_feature.padding_idx).unsqueeze(-1).float()
        else:
            mask = (behavior_seq != 0).unsqueeze(-1).float()
        
        # Apply attention pooling
        if candidate_emb is not None:
            candidate_query = candidate_emb
            if self.candidate_attention_proj is not None:
                candidate_query = self.candidate_attention_proj(candidate_query)
            pooled_behavior = self.attention(
                query=candidate_query,
                keys=behavior_emb,
                mask=mask
            )  # [B, emb_dim]
        else:
            # If no candidate, use mean pooling
            pooled_behavior = torch.sum(behavior_emb * mask, dim=1) / (mask.sum(dim=1) + 1e-9)
        
        # Get other features
        other_embeddings = []
        
        if candidate_emb is not None:
            other_embeddings.append(candidate_emb)
        
        other_embeddings.append(pooled_behavior)
        
        # Other sparse features
        for feat in self.other_sparse_features:
            feat_emb = self.embedding.embed_dict[feat.embedding_name](x[feat.name].long())
            other_embeddings.append(feat_emb)
        
        # Dense features
        for feat in self.dense_features_list:
            val = x[feat.name].float()
            if val.dim() == 1:
                val = val.unsqueeze(1)
            other_embeddings.append(val)
        
        # Concatenate all features
        concat_input = torch.cat(other_embeddings, dim=-1)  # [B, total_dim]
        
        # MLP prediction
        y = self.mlp(concat_input)  # [B, 1]
        return self.prediction_layer(y)
