"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Pan Z, Sun F, Liu J, et al. MaskNet: Introducing feature-wise gating blocks for high-dimensional
        sparse recommendation data (CCF-Tencent CTR competition solution, 2020).
"""

import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.basic.layers import EmbeddingLayer, LR, MLP, PredictionLayer
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature


class MaskNet(BaseModel):
    @property
    def model_name(self):
        return "MaskNet"

    @property
    def task_type(self):
        return "binary"
    
    def __init__(self,
                 dense_features: list[DenseFeature] | list = [],
                 sparse_features: list[SparseFeature] | list = [],
                 sequence_features: list[SequenceFeature] | list = [],
                 num_blocks: int = 3,
                 mask_hidden_dim: int = 64,
                 block_dropout: float = 0.1,
                 mlp_params: dict = {},
                 target: list[str] | list = [],
                 optimizer: str = "adam",
                 optimizer_params: dict = {},
                 loss: str | nn.Module | None = "bce",
                 device: str = 'cpu',
                 embedding_l1_reg=1e-6,
                 dense_l1_reg=1e-5,
                 embedding_l2_reg=1e-5,
                 dense_l2_reg=1e-4,
                 **kwargs):
        
        super(MaskNet, self).__init__(
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
            
        self.mask_features = sparse_features + sequence_features
        if len(self.mask_features) == 0:
            raise ValueError("MaskNet requires at least one sparse/sequence feature.")

        self.embedding = EmbeddingLayer(features=self.mask_features)
        self.num_fields = len(self.mask_features)
        self.embedding_dim = self.mask_features[0].embedding_dim
        if any(f.embedding_dim != self.embedding_dim for f in self.mask_features):
            raise ValueError("MaskNet expects identical embedding_dim across mask_features.")

        self.num_blocks = max(1, num_blocks)
        self.field_dim = self.num_fields * self.embedding_dim

        self.linear = LR(self.field_dim)
        self.mask_generators = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.mask_generators.append(
                nn.Sequential(
                    nn.Linear(self.field_dim, mask_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(mask_hidden_dim, self.num_fields)
                )
            )

        self.block_dropout = nn.Dropout(block_dropout)
        self.final_mlp = MLP(input_dim=self.field_dim * self.num_blocks, **mlp_params)
        self.prediction_layer = PredictionLayer(task_type=self.task_type)

        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['linear', 'mask_generators', 'final_mlp']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss
        )

    def forward(self, x):
        field_emb = self.embedding(x=x, features=self.mask_features, squeeze_dim=False)
        flat_input = field_emb.flatten(start_dim=1)
        y_linear = self.linear(flat_input)

        block_input = field_emb
        mask_input = flat_input
        block_outputs = []
        for mask_gen in self.mask_generators:
            mask_logits = mask_gen(mask_input)
            mask = torch.sigmoid(mask_logits).unsqueeze(-1)
            masked_emb = block_input * mask
            block_output = self.block_dropout(masked_emb.flatten(start_dim=1))
            block_outputs.append(block_output)
            mask_input = block_output
            block_input = masked_emb.view_as(field_emb)

        stacked = torch.cat(block_outputs, dim=1)
        y_deep = self.final_mlp(stacked)

        y = y_linear + y_deep
        return self.prediction_layer(y)
