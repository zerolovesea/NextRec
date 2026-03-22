"""
Date: create on 09/11/2025
Checkpoint: edit on 15/02/2026
Author: Yang Zhou,zyaztec@gmail.com

Shared-Bottom is the classic hard-parameter-sharing baseline for multi-task learning.
All tasks share a common bottom network to learn general representations, and each
task has its own tower head for task-specific refinement and prediction. This
architecture is simple, parameter-efficient, and helps regularize related tasks.

Workflow:
- Unified embeddings convert dense/sparse/sequence features
- A shared bottom MLP learns common representations
- Task-specific towers further transform the shared features
- Separate prediction heads output each task’s logits/probabilities

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding: all features -> flattened embedding -> input_flat: [Batch, Dim_embedding]
- Shared Bottom: input_flat: [Batch, Dim_embedding] -> bottom MLP -> bottom_output: [Batch, Dim_bottom]
- Task Towers: each tower_i(bottom_output) -> tower_output_i: [Batch, 1]
- Concatenate Towers: cat(tower_output_1...tower_output_T, dim=1) -> logits: [Batch, Task_num]
- Prediction Head: logits: [Batch, Task_num] -> task activations -> y: [Batch, Task_num]

Share-Bottom（硬共享底层）是多任务学习的经典基线：所有任务共享一个底层网络，
各任务拥有独立塔头进行细化与预测，简单高效且能通过共享正则化相关任务。

流程：
- 统一 embedding 处理稠密、稀疏与序列特征
- 共享底层 MLP 学习通用表示
- 任务塔在共享表示上做任务特定变换
- 各任务预测头输出对应结果

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：所有特征拼接展平后 -> input_flat: [Batch, Dim_embedding]
- 共享底层：input_flat: [Batch, Dim_embedding] -> 底层 MLP -> bottom_output: [Batch, Dim_bottom]
- 任务塔：每个 tower_i(bottom_output) -> tower_output_i: [Batch, 1]
- 拼接任务输出：cat(tower_output_1...tower_output_T, dim=1) -> logits: [Batch, Task_num]
- 预测头：logits: [Batch, Task_num] -> 各任务激活 -> y: [Batch, Task_num]

"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeInput


class ShareBottom(BaseModel):
    @property
    def model_name(self):
        return "ShareBottom"

    @property
    def default_task(self):
        nums_task = self.nums_task if hasattr(self, "nums_task") else None
        if nums_task is not None and nums_task > 0:
            return ["binary"] * nums_task
        return ["binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        bottom_mlp_params: dict,
        tower_mlp_params_list: list[dict],
        target: list[str],
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        **kwargs,
    ):
        """
        Initialize ShareBottom model.
        初始化 ShareBottom 模型。

        Args:
            bottom_mlp_params: Parameters for the shared bottom MLP, e.g.
                {"hidden_dims": [256, 128], "dropout": 0.2, "activation": "relu"}.
                共享底层 MLP 参数，例如 {"hidden_dims": [256, 128], "dropout": 0.2, "activation": "relu"}。
            tower_mlp_params_list: Per-task tower MLP parameter list. Its length must equal
                the number of tasks; each tower outputs one logit, e.g.
                [{"hidden_dims": [64], "dropout": 0.1, "activation": "relu"}, ...].
                每个任务对应一个 tower 的 MLP 参数列表，长度必须等于任务数；每个 tower 输出一个 logit，
                例如 [{"hidden_dims": [64], "dropout": 0.1, "activation": "relu"}, ...]。
        """

        self.nums_task = len(target)

        super(ShareBottom, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        # Number of tasks
        self.nums_task = len(target)
        if len(tower_mlp_params_list) != self.nums_task:
            raise ValueError(
                "Number of tower mlp params "
                f"({len(tower_mlp_params_list)}) must match number of tasks ({self.nums_task})"
            )
        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)
        # Calculate input dimension
        input_dim = self.embedding.input_dim
        # emb_dim_total = sum([f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature)])
        # dense_input_dim = sum([(f.embedding_dim or 1) for f in dense_features])
        # input_dim = emb_dim_total + dense_input_dim

        # Shared bottom network
        self.bottom = MLP(input_dim=input_dim, output_dim=None, **bottom_mlp_params)
        self.grad_norm_shared_modules = ["embedding", "bottom"]

        # Get bottom output dimension
        if "hidden_dims" in bottom_mlp_params and len(bottom_mlp_params["hidden_dims"]) > 0:
            bottom_output_dim = bottom_mlp_params["hidden_dims"][-1]
        else:
            bottom_output_dim = input_dim

        # Task-specific towers
        self.towers = nn.ModuleList()
        for tower_mlp_params in tower_mlp_params_list:
            tower = MLP(input_dim=bottom_output_dim, output_dim=1, **tower_mlp_params)
            self.towers.append(tower)
        # Register regularization weights
        self.register_regularization_weights(embedding_attr="embedding", include_modules=["bottom", "towers"])

    def forward(self, x):
        # Embedding flatten: [Batch, Dim_embedding]
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)  # [Batch, Dim_embedding]

        # Shared bottom MLP: [Batch, Dim_embedding] -> [Batch, Dim_bottom]
        bottom_output = self.bottom(input_flat)  # [Batch, Dim_bottom]

        # Task-specific towers: each tower output is [Batch, 1]
        task_outputs = []
        for tower in self.towers:
            tower_output = tower(bottom_output)  # [Batch, 1]
            task_outputs.append(tower_output)

        logits = torch.cat(task_outputs, dim=1)  # [Batch, Task_num]
        return logits
