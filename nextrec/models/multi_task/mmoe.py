"""
Date: create on 09/11/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference: [1] Ma J, Zhao Z, Yi X, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts[C]//KDD. 2018: 1930-1939.
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MLP, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class MMOE(BaseModel):
    """
    Multi-gate Mixture-of-Experts
    
    MMOE improves upon shared-bottom architecture by using multiple expert networks
    and task-specific gating networks. Each task has its own gate that learns to
    weight the contributions of different experts, allowing for both task-specific
    and shared representations.
    """
    
    @property
    def model_name(self):
        return "MMOE"

    @property
    def task_type(self):
        return self.task if isinstance(self.task, list) else [self.task]
    
    def __init__(self,
                 dense_features: list[DenseFeature]=[],
                 sparse_features: list[SparseFeature]=[],
                 sequence_features: list[SequenceFeature]=[],
                 expert_params: dict={},
                 num_experts: int=3,
                 tower_params_list: list[dict]=[],
                 target: list[str]=[],
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
        
        super(MMOE, self).__init__(
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
        
        # Number of tasks and experts
        self.num_tasks = len(target)
        self.num_experts = num_experts
        
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
        
        # Expert networks (shared by all tasks)
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = MLP(input_dim=input_dim, output_layer=False, **expert_params)
            self.experts.append(expert)
        
        # Get expert output dimension
        if 'dims' in expert_params and len(expert_params['dims']) > 0:
            expert_output_dim = expert_params['dims'][-1]
        else:
            expert_output_dim = input_dim
        
        # Task-specific gates
        self.gates = nn.ModuleList()
        for _ in range(self.num_tasks):
            gate = nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=1)
            )
            self.gates.append(gate)
        
        # Task-specific towers
        self.towers = nn.ModuleList()
        for tower_params in tower_params_list:
            tower = MLP(input_dim=expert_output_dim, output_layer=True, **tower_params)
            self.towers.append(tower)
        self.prediction_layer = PredictionLayer(
            task_type=self.task_type,
            task_dims=[1] * self.num_tasks
        )

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['experts', 'gates', 'towers']
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
        
        # Expert outputs: [num_experts, B, expert_dim]
        expert_outputs = [expert(input_flat) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, B, expert_dim]
        
        # Task-specific processing
        task_outputs = []
        for task_idx in range(self.num_tasks):
            # Gate weights for this task: [B, num_experts]
            gate_weights = self.gates[task_idx](input_flat)  # [B, num_experts]
            
            # Weighted sum of expert outputs
            # gate_weights: [B, num_experts, 1]
            # expert_outputs: [num_experts, B, expert_dim]
            gate_weights = gate_weights.unsqueeze(2)  # [B, num_experts, 1]
            expert_outputs_t = expert_outputs.permute(1, 0, 2)  # [B, num_experts, expert_dim]
            gated_output = torch.sum(gate_weights * expert_outputs_t, dim=1)  # [B, expert_dim]
            
            # Tower output
            tower_output = self.towers[task_idx](gated_output)  # [B, 1]
            task_outputs.append(tower_output)
        
        # Stack outputs: [B, num_tasks]
        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
