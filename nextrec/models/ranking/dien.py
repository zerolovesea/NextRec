"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Zhou G, Mou N, Fan Y, et al. Deep interest evolution network for click-through rate prediction[C]
        //Proceedings of the AAAI conference on artificial intelligence. 2019, 33(01): 5941-5948.
        (https://arxiv.org/abs/1809.03672)
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MLP, AttentionPoolingLayer, DynamicGRU, AUGRU, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class DIEN(BaseModel):
    @property
    def model_name(self):
        return "DIEN"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 mlp_params: dict,
                 gru_hidden_size: int = 64,
                 attention_hidden_units: list[int] = [80, 40],
                 attention_activation: str = 'sigmoid',
                 use_negsampling: bool = False,
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
        
        super(DIEN, self).__init__(
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
        
        self.use_negsampling = use_negsampling
        
        # Features classification
        if len(sequence_features) == 0:
            raise ValueError("DIEN requires at least one sequence feature for user behavior history")
        
        self.behavior_feature = sequence_features[0]  # User behavior sequence
        self.candidate_feature = sparse_features[-1] if sparse_features else None  # Candidate item
        
        self.other_sparse_features = sparse_features[:-1] if self.candidate_feature else sparse_features
        self.dense_features_list = dense_features
        
        # All features for embedding
        self.all_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)
        
        behavior_emb_dim = self.behavior_feature.embedding_dim
        self.candidate_proj = None
        if self.candidate_feature is not None and self.candidate_feature.embedding_dim != gru_hidden_size:
            self.candidate_proj = nn.Linear(self.candidate_feature.embedding_dim, gru_hidden_size)
        
        # Interest Extractor Layer (GRU)
        self.interest_extractor = DynamicGRU(
            input_size=behavior_emb_dim,
            hidden_size=gru_hidden_size
        )
        
        # Attention layer for computing attention scores
        self.attention_layer = AttentionPoolingLayer(
            embedding_dim=gru_hidden_size,
            hidden_units=attention_hidden_units,
            activation=attention_activation,
            use_softmax=False  # We'll use scores directly for AUGRU
        )
        
        # Interest Evolution Layer (AUGRU)
        self.interest_evolution = AUGRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size
        )
        
        # Calculate MLP input dimension
        mlp_input_dim = 0
        if self.candidate_feature:
            mlp_input_dim += self.candidate_feature.embedding_dim
        mlp_input_dim += gru_hidden_size  # final interest state
        mlp_input_dim += sum([f.embedding_dim for f in self.other_sparse_features])
        mlp_input_dim += sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        
        # MLP for final prediction
        self.mlp = MLP(input_dim=mlp_input_dim, **mlp_params)
        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['interest_extractor', 'interest_evolution', 'attention_layer', 'mlp', 'candidate_proj']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        # Get candidate item embedding
        if self.candidate_feature:
            candidate_emb = self.embedding.embed_dict[self.candidate_feature.embedding_name](
                x[self.candidate_feature.name].long()
            )  # [B, emb_dim]
        else:
            raise ValueError("DIEN requires a candidate item feature")
        
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
        
        # Step 1: Interest Extractor (GRU)
        interest_states, _ = self.interest_extractor(behavior_emb)  # [B, seq_len, hidden_size]
        
        # Step 2: Compute attention scores for each time step
        batch_size, seq_len, hidden_size = interest_states.shape
        
        # Project candidate to hidden_size if necessary (defined in __init__)
        if self.candidate_proj is not None:
            candidate_for_attention = self.candidate_proj(candidate_emb)
        else:
            candidate_for_attention = candidate_emb
        
        # Compute attention scores for AUGRU
        attention_scores = []
        for t in range(seq_len):
            score = self.attention_layer.attention_net(
                torch.cat([
                    candidate_for_attention,
                    interest_states[:, t, :],
                    candidate_for_attention - interest_states[:, t, :],
                    candidate_for_attention * interest_states[:, t, :]
                ], dim=-1)
            )  # [B, 1]
            attention_scores.append(score)
        
        attention_scores = torch.cat(attention_scores, dim=1).unsqueeze(-1)  # [B, seq_len, 1]
        attention_scores = torch.sigmoid(attention_scores)  # Normalize to [0, 1]
        
        # Apply mask to attention scores
        attention_scores = attention_scores * mask
        
        # Step 3: Interest Evolution (AUGRU)
        final_states, final_interest = self.interest_evolution(
            interest_states,
            attention_scores
        )  # final_interest: [B, hidden_size]
        
        # Get other features
        other_embeddings = []
        other_embeddings.append(candidate_emb)
        other_embeddings.append(final_interest)
        
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
