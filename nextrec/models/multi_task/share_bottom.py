"""
Date: create on 09/11/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference: [1] Caruana R. Multitask learning[J]. Machine learning, 1997, 28: 41-75.
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MLP, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class ShareBottom(BaseModel):
    @property
    def model_name(self):
        return "ShareBottom"

    @property
    def task_type(self):
        # Multi-task model, return list of task types
        return self.task if isinstance(self.task, list) else [self.task]
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 bottom_params: dict,
                 tower_params_list: list[dict],
                 target: list[str],
                 task: str | list[str] = 'binary',
                 optimizer: str = "adam",
                 optimizer_params: dict = {},
                 loss: str | nn.Module | list[str | nn.Module] | None = "bce",
                 loss_params: dict | list[dict] | None = None,
                 device: str = 'cpu',
                 embedding_l1_reg=1e-6,
                 dense_l1_reg=1e-5,
                 embedding_l2_reg=1e-5,
                 dense_l2_reg=1e-4,
                 **kwargs):
        
        super(ShareBottom, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
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
        
        # Number of tasks
        self.num_tasks = len(target)
        if len(tower_params_list) != self.num_tasks:
            raise ValueError(f"Number of tower params ({len(tower_params_list)}) must match number of tasks ({self.num_tasks})")
            
        # All features
        self.all_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Calculate input dimension
        emb_dim_total = sum([f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature)])
        dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        input_dim = emb_dim_total + dense_input_dim
        
        # Shared bottom network
        self.bottom = MLP(input_dim=input_dim, output_layer=False, **bottom_params)
        
        # Get bottom output dimension
        if 'dims' in bottom_params and len(bottom_params['dims']) > 0:
            bottom_output_dim = bottom_params['dims'][-1]
        else:
            bottom_output_dim = input_dim
        
        # Task-specific towers
        self.towers = nn.ModuleList()
        for tower_params in tower_params_list:
            tower = MLP(input_dim=bottom_output_dim, output_layer=True, **tower_params)
            self.towers.append(tower)
        self.prediction_layer = PredictionLayer(
            task_type=self.task_type,
            task_dims=[1] * self.num_tasks
        )

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['bottom', 'towers']
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
        
        # Shared bottom
        bottom_output = self.bottom(input_flat)  # [B, bottom_dim]
        
        # Task-specific towers
        task_outputs = []
        for tower in self.towers:
            tower_output = tower(bottom_output)  # [B, 1]
            task_outputs.append(tower_output)
        
        # Stack outputs: [B, num_tasks]
        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
