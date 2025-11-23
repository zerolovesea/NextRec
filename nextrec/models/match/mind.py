"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Li C, Liu Z, Wu M, et al. Multi-interest network with dynamic routing for recommendation at Tmall[C]
        //Proceedings of the 28th ACM international conference on information and knowledge management. 2019: 2615-2623.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from nextrec.basic.model import BaseMatchModel
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature
from nextrec.basic.layers import MLP, EmbeddingLayer, CapsuleNetwork


class MIND(BaseMatchModel):
    @property
    def model_name(self) -> str:
        return "MIND"
    
    @property
    def support_training_modes(self) -> list[str]:
        """MIND only supports pointwise training mode"""
        return ['pointwise']
    
    def __init__(self,
                 user_dense_features: list[DenseFeature] | None = None,
                 user_sparse_features: list[SparseFeature] | None = None,
                 user_sequence_features: list[SequenceFeature] | None = None,
                 item_dense_features: list[DenseFeature] | None = None,
                 item_sparse_features: list[SparseFeature] | None = None,
                 item_sequence_features: list[SequenceFeature] | None = None,
                 embedding_dim: int = 64,
                 num_interests: int = 4,
                 capsule_bilinear_type: int = 2,
                 routing_times: int = 3,
                 relu_layer: bool = False,
                 item_dnn_hidden_units: list[int] = [256, 128],
                 dnn_activation: str = 'relu',
                 dnn_dropout: float = 0.0,
                 training_mode: Literal['pointwise', 'pairwise', 'listwise'] = 'pointwise',
                 num_negative_samples: int = 100,
                 temperature: float = 1.0,
                 similarity_metric: Literal['dot', 'cosine', 'euclidean'] = 'dot',
                 device: str = 'cpu',
                 embedding_l1_reg: float = 0.0,
                 dense_l1_reg: float = 0.0,
                 embedding_l2_reg: float = 0.0,
                 dense_l2_reg: float = 0.0,
                 early_stop_patience: int = 20,
                 optimizer: str | torch.optim.Optimizer = "adam",
                 optimizer_params: dict | None = None,
                 scheduler: str | torch.optim.lr_scheduler._LRScheduler | type[torch.optim.lr_scheduler._LRScheduler] | None = None,
                 scheduler_params: dict | None = None,
                 loss: str | nn.Module | list[str | nn.Module] | None = "bce",
                 loss_params: dict | list[dict] | None = None,
                 **kwargs):
        
        super(MIND, self).__init__(
            user_dense_features=user_dense_features,
            user_sparse_features=user_sparse_features,
            user_sequence_features=user_sequence_features,
            item_dense_features=item_dense_features,
            item_sparse_features=item_sparse_features,
            item_sequence_features=item_sequence_features,
            training_mode=training_mode,
            num_negative_samples=num_negative_samples,
            temperature=temperature,
            similarity_metric=similarity_metric,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            early_stop_patience=early_stop_patience,
            **kwargs
        )
        
        self.embedding_dim = embedding_dim
        self.num_interests = num_interests
        self.item_dnn_hidden_units = item_dnn_hidden_units

        user_features = []
        if user_dense_features:
            user_features.extend(user_dense_features)
        if user_sparse_features:
            user_features.extend(user_sparse_features)
        if user_sequence_features:
            user_features.extend(user_sequence_features)
        
        if len(user_features) > 0:
            self.user_embedding = EmbeddingLayer(user_features)
            
            if not user_sequence_features or len(user_sequence_features) == 0:
                raise ValueError("MIND requires at least one user sequence feature")
            
            seq_max_len = user_sequence_features[0].max_len if user_sequence_features[0].max_len else 50
            seq_embedding_dim = user_sequence_features[0].embedding_dim
            
            # Capsule Network for multi-interest extraction
            self.capsule_network = CapsuleNetwork(
                embedding_dim=seq_embedding_dim,
                seq_len=seq_max_len,
                bilinear_type=capsule_bilinear_type,
                interest_num=num_interests,
                routing_times=routing_times,
                relu_layer=relu_layer
            )
            
            if seq_embedding_dim != embedding_dim:
                self.interest_projection = nn.Linear(seq_embedding_dim, embedding_dim, bias=False)
                nn.init.xavier_uniform_(self.interest_projection.weight)
            else:
                self.interest_projection = None
        
        # Item tower
        item_features = []
        if item_dense_features:
            item_features.extend(item_dense_features)
        if item_sparse_features:
            item_features.extend(item_sparse_features)
        if item_sequence_features:
            item_features.extend(item_sequence_features)
        
        if len(item_features) > 0:
            self.item_embedding = EmbeddingLayer(item_features)
            
            item_input_dim = 0
            for feat in item_dense_features or []:
                item_input_dim += 1
            for feat in item_sparse_features or []:
                item_input_dim += feat.embedding_dim
            for feat in item_sequence_features or []:
                item_input_dim += feat.embedding_dim
            
            # Item DNN
            if len(item_dnn_hidden_units) > 0:
                item_dnn_units = item_dnn_hidden_units + [embedding_dim]
                self.item_dnn = MLP(
                    input_dim=item_input_dim,
                    dims=item_dnn_units,
                    output_layer=False,
                    dropout=dnn_dropout,
                    activation=dnn_activation
                )
            else:
                self.item_dnn = None
        
        self._register_regularization_weights(
            embedding_attr='user_embedding',
            include_modules=['capsule_network']
        )
        self._register_regularization_weights(
            embedding_attr='item_embedding',
            include_modules=['item_dnn'] if self.item_dnn else []
        )
        
        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            loss=loss,
            loss_params=loss_params,
        )

        self.to(device)
    
    def user_tower(self, user_input: dict) -> torch.Tensor:
        """
        User tower with multi-interest extraction
        
        Returns:
            user_interests: [batch_size, num_interests, embedding_dim]
        """
        seq_feature = self.user_sequence_features[0]
        seq_input = user_input[seq_feature.name]

        embed = self.user_embedding.embed_dict[seq_feature.embedding_name]
        seq_emb = embed(seq_input.long())  # [batch_size, seq_len, embedding_dim]

        mask = (seq_input != seq_feature.padding_idx).float()  # [batch_size, seq_len]
        
        multi_interests = self.capsule_network(seq_emb, mask)  # [batch_size, num_interests, seq_embedding_dim]
        
        if self.interest_projection is not None:
            multi_interests = self.interest_projection(multi_interests)  # [batch_size, num_interests, embedding_dim]
        
        # L2 normalization
        multi_interests = F.normalize(multi_interests, p=2, dim=-1)
        
        return multi_interests
    
    def item_tower(self, item_input: dict) -> torch.Tensor:
        """Item tower"""
        all_item_features = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
        item_emb = self.item_embedding(item_input, all_item_features, squeeze_dim=True)
        
        if self.item_dnn is not None:
            item_emb = self.item_dnn(item_emb)
        
        # L2 normalization
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        return item_emb
    
    def compute_similarity(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        item_emb_expanded = item_emb.unsqueeze(1)
        
        if self.similarity_metric == 'dot':
            similarities = torch.sum(user_emb * item_emb_expanded, dim=-1)
        elif self.similarity_metric == 'cosine':
            similarities = F.cosine_similarity(user_emb, item_emb_expanded, dim=-1)
        elif self.similarity_metric == 'euclidean':
            similarities = -torch.sum((user_emb - item_emb_expanded) ** 2, dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        max_similarity, _ = torch.max(similarities, dim=1)  # [batch_size]
        max_similarity = max_similarity / self.temperature
        
        return max_similarity
