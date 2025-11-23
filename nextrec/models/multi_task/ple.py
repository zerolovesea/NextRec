"""
Date: create on 09/11/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference: [1] Tang H, Liu J, Zhao M, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations[C]//RecSys. 2020: 269-278.
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, MLP, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class PLE(BaseModel):
    """
    Progressive Layered Extraction
    
    PLE is an advanced multi-task learning model that extends MMOE by introducing
    both task-specific experts and shared experts at each level. It uses a progressive
    routing mechanism where experts from level k feed into gates at level k+1.
    This design better captures task-specific and shared information progressively.
    """
    
    @property
    def model_name(self):
        return "PLE"

    @property
    def task_type(self):
        return self.task if isinstance(self.task, list) else [self.task]
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 shared_expert_params: dict,
                 specific_expert_params: dict,
                 num_shared_experts: int,
                 num_specific_experts: int,
                 num_levels: int,
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
        
        super(PLE, self).__init__(
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
        
        # Number of tasks, experts, and levels
        self.num_tasks = len(target)
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_levels = num_levels
        if optimizer_params is None:
            optimizer_params = {}
            
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
        
        # Get expert output dimension
        if 'dims' in shared_expert_params and len(shared_expert_params['dims']) > 0:
            expert_output_dim = shared_expert_params['dims'][-1]
        else:
            expert_output_dim = input_dim
        
        # Build extraction layers (CGC layers)
        self.shared_experts_layers = nn.ModuleList()  # [num_levels]
        self.specific_experts_layers = nn.ModuleList()  # [num_levels, num_tasks]
        self.gates_layers = nn.ModuleList()  # [num_levels, num_tasks + 1] (+1 for shared gate)
        
        for level in range(num_levels):
            # Input dimension for this level
            level_input_dim = input_dim if level == 0 else expert_output_dim
            
            # Shared experts for this level
            shared_experts = nn.ModuleList()
            for _ in range(num_shared_experts):
                expert = MLP(input_dim=level_input_dim, output_layer=False, **shared_expert_params)
                shared_experts.append(expert)
            self.shared_experts_layers.append(shared_experts)
            
            # Task-specific experts for this level
            specific_experts_for_tasks = nn.ModuleList()
            for _ in range(self.num_tasks):
                task_experts = nn.ModuleList()
                for _ in range(num_specific_experts):
                    expert = MLP(input_dim=level_input_dim, output_layer=False, **specific_expert_params)
                    task_experts.append(expert)
                specific_experts_for_tasks.append(task_experts)
            self.specific_experts_layers.append(specific_experts_for_tasks)
            
            # Gates for this level (num_tasks task gates + 1 shared gate)
            gates = nn.ModuleList()
            # Task-specific gates
            num_experts_for_task_gate = num_shared_experts + num_specific_experts
            for _ in range(self.num_tasks):
                gate = nn.Sequential(
                    nn.Linear(level_input_dim, num_experts_for_task_gate),
                    nn.Softmax(dim=1)
                )
                gates.append(gate)
            # Shared gate: contains all tasks' specific experts + shared experts
            # expert counts = num_shared_experts + num_specific_experts * num_tasks
            num_experts_for_shared_gate = num_shared_experts + num_specific_experts * self.num_tasks
            shared_gate = nn.Sequential(
                nn.Linear(level_input_dim, num_experts_for_shared_gate),
                nn.Softmax(dim=1)
            )
            gates.append(shared_gate)
            self.gates_layers.append(gates)
        
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
            include_modules=['shared_experts_layers', 'specific_experts_layers', 'gates_layers', 'towers']
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

        # Initial features for each task and shared
        task_fea = [input_flat for _ in range(self.num_tasks)]
        shared_fea = input_flat

        # Progressive Layered Extraction: CGC
        for level in range(self.num_levels):
            shared_experts = self.shared_experts_layers[level]      # ModuleList[num_shared_experts]
            specific_experts = self.specific_experts_layers[level]  # ModuleList[num_tasks][num_specific_experts]
            gates = self.gates_layers[level]                        # ModuleList[num_tasks + 1]

            # Compute shared experts output for this level
            # shared_expert_list: List[Tensor[B, expert_dim]]
            shared_expert_list = [expert(shared_fea) for expert in shared_experts] # type: ignore[list-item]
            # [num_shared_experts, B, expert_dim]
            shared_expert_outputs = torch.stack(shared_expert_list, dim=0)

            all_specific_outputs_for_shared = []

            # Compute task's gated output and specific outputs
            new_task_fea = []
            for task_idx in range(self.num_tasks):
                # Current input for this task at this level
                current_task_in = task_fea[task_idx]

                # Specific task experts for this task
                task_expert_modules = specific_experts[task_idx] # type: ignore

                # Specific task expert output list List[Tensor[B, expert_dim]]
                task_specific_list = []
                for expert in task_expert_modules:
                    out = expert(current_task_in)
                    task_specific_list.append(out)
                    # All specific task experts are candidates for the shared gate
                    all_specific_outputs_for_shared.append(out)

                # [num_specific_taskexperts, B, expert_dim]
                task_specific_outputs = torch.stack(task_specific_list, dim=0)

                # Input for gate: shared_experts + own specific task experts
                # [num_shared + num_specific, B, expert_dim]
                all_expert_outputs = torch.cat(
                    [shared_expert_outputs, task_specific_outputs],
                    dim=0
                )
                # [B, num_experts, expert_dim]
                all_expert_outputs_t = all_expert_outputs.permute(1, 0, 2)

                # Gate for task (gates[task_idx])
                # Output shape: [B, num_shared + num_specific]
                gate_weights = gates[task_idx](current_task_in)
                # [B, num_experts, 1]
                gate_weights = gate_weights.unsqueeze(2)

                # Weighted sum to get this task's features at this level: [B, expert_dim]
                gated_output = torch.sum(gate_weights * all_expert_outputs_t, dim=1)
                new_task_fea.append(gated_output)

            # compute shared gate output
            # Input for shared gate: specific task experts + shared experts
            # all_specific_outputs_for_shared: List[Tensor[B, expert_dim]]
            # shared_expert_list: List[Tensor[B, expert_dim]]
            all_for_shared_list = all_specific_outputs_for_shared + shared_expert_list
            # [B, num_all_experts, expert_dim]
            all_for_shared = torch.stack(all_for_shared_list, dim=1)

            # [B, num_all_experts]
            shared_gate_weights = gates[self.num_tasks](shared_fea) # type: ignore
            # [B, 1, num_all_experts]
            shared_gate_weights = shared_gate_weights.unsqueeze(1)

            # weighted sum: [B, 1, expert_dim] → [B, expert_dim]
            new_shared_fea = torch.bmm(shared_gate_weights, all_for_shared).squeeze(1)

            task_fea = new_task_fea
            shared_fea = new_shared_fea

        # task tower
        task_outputs = []
        for task_idx in range(self.num_tasks):
            tower_output = self.towers[task_idx](task_fea[task_idx])  # [B, 1]
            task_outputs.append(tower_output)

        # [B, num_tasks]
        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)