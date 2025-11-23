"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Ma X, Zhao L, Huang G, et al. Entire space multi-task model: An effective approach for estimating post-click conversion rate[C]//SIGIR. 2018: 1137-1140.
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MLP, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class ESMM(BaseModel):
    """
    Entire Space Multi-Task Model
    
    ESMM is designed for CVR (Conversion Rate) prediction. It models two related tasks:
    - CTR task: P(click | impression)
    - CVR task: P(conversion | click)
    - CTCVR task (auxiliary): P(click & conversion | impression) = P(click) * P(conversion | click)
    
    This design addresses the sample selection bias and data sparsity issues in CVR modeling.
    """
    
    @property
    def model_name(self):
        return "ESMM"

    @property
    def task_type(self):
        # ESMM has fixed task types: CTR (binary) and CVR (binary)
        return ['binary', 'binary']
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 ctr_params: dict,
                 cvr_params: dict,
                 target: list[str] = ['ctr', 'ctcvr'],  # Note: ctcvr = ctr * cvr
                 task: str | list[str] = 'binary',
                 optimizer: str = "adam",
                 optimizer_params: dict = {},
                 loss: str | nn.Module | list[str | nn.Module] | None = "bce",
                 device: str = 'cpu',
                 embedding_l1_reg=1e-6,
                 dense_l1_reg=1e-5,
                 embedding_l2_reg=1e-5,
                 dense_l2_reg=1e-4,
                 **kwargs):
        
        # ESMM requires exactly 2 targets: ctr and ctcvr
        if len(target) != 2:
            raise ValueError(f"ESMM requires exactly 2 targets (ctr and ctcvr), got {len(target)}")
        
        super(ESMM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,  # Both CTR and CTCVR are binary classification
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

        # Shared embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Calculate input dimension
        emb_dim_total = sum([f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature)])
        dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        input_dim = emb_dim_total + dense_input_dim
        
        # CTR tower
        self.ctr_tower = MLP(input_dim=input_dim, output_layer=True, **ctr_params)
        
        # CVR tower
        self.cvr_tower = MLP(input_dim=input_dim, output_layer=True, **cvr_params)
        self.prediction_layer = PredictionLayer(
            task_type=self.task_type,
            task_dims=[1, 1]
        )

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['ctr_tower', 'cvr_tower']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss
        )

    def forward(self, x):
        # Get all embeddings and flatten
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        
        # CTR prediction: P(click | impression)
        ctr_logit = self.ctr_tower(input_flat)  # [B, 1]
        cvr_logit = self.cvr_tower(input_flat)  # [B, 1]
        logits = torch.cat([ctr_logit, cvr_logit], dim=1)
        preds = self.prediction_layer(logits)
        ctr, cvr = preds.chunk(2, dim=1)
        
        # CTCVR prediction: P(click & conversion | impression) = P(click) * P(conversion | click)
        ctcvr = ctr * cvr  # [B, 1]
        
        # Output: [CTR, CTCVR]
        # Note: We supervise CTR with click labels and CTCVR with conversion labels
        y = torch.cat([ctr, ctcvr], dim=1)  # [B, 2]
        return y  # [B, 2], where y[:, 0] is CTR and y[:, 1] is CTCVR
