"""
Date: create on 15/02/2026
Checkpoint: edit on 21/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Xi D, Chen Z, Yan P, Zhang Y, Zhu Y, Zhuang F, Chen Y. Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising. Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD ’21), 2021, pp. 3745–3755.
URL: https://arxiv.org/abs/2105.08489
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/
- [3] Torch Rechub AITM Implementation: https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/multi_task/aitm.py

Attentive Information Transfer Multi-Task Model (AITM) was proposed by Meituan, Zhejiang University,
and other institutions in KDD 2021 for multi-task conversion funnel scenarios with explicit sequential dependencies,
e.g. impression -> click -> add-to-cart -> purchase.
Compared to modeling each task independently, AITM is a framework with two main components:
AITMTransfer for transferring information between tasks, and a sequential monotonic calibrator
to constrain the predicted probabilities of later stages not to exceed those of earlier stages.
Behavior Expectation Calibrator (BEC), which is used to improve the accuracy of end-to-end conversion identification.

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding: all features -> input_flat: [Batch, Dim_embedding]
- Task bottoms: bottom_i(input_flat) -> task_feat_i: [Batch, Dim_bottom]
- Transfer i-1 -> i:
  prev_proj(task_feat_{i-1}) -> [Batch, Dim_bottom]
  stack([prev, curr], dim=1) -> [Batch, 2, Dim_bottom]
  attention(Q/K/V) over 2 slots -> transfer_feat_i: [Batch, Dim_bottom]
- Task towers: tower_i(task_feat_i) -> logit_i: [Batch, 1]
- Concatenate logits: cat(logit_1 ... logit_T, dim=1) -> [Batch, Task_num]
- Output head: task activation (binary sigmoid) -> y_pred: [Batch, Task_num]
- Calibrator loss: penalize relu(p_{i+1} - p_i) to enforce funnel monotonicity

AITM 由美团，浙江大学等机构发表在KDD2021，面向具有明确阶段顺序依赖的多任务转化漏斗场景，
例如曝光 -> 点击 -> 加购 -> 购买。
与各任务独立建模相比，AITM是一个框架，包含两个主要部分：AITMTransfer用于在任务之间传递信息，
以及一个顺序单调校准器用于约束后续阶段的预测概率不超过前序阶段。
行为期望校准器（BEC），它用于提高端到端转化识别的准确性。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：所有特征拼接展平 -> input_flat: [Batch, Dim_embedding]
- 各任务底层塔：bottom_i(input_flat) -> task_feat_i: [Batch, Dim_bottom]
- 迁移模块 i-1 -> i：
  prev_proj(task_feat_{i-1}) -> [Batch, Dim_bottom]
  stack([prev, curr], dim=1) -> [Batch, 2, Dim_bottom]
  在 2 个信息槽上做 Q/K/V 注意力 -> transfer_feat_i: [Batch, Dim_bottom]
- 任务塔：tower_i(task_feat_i) -> logit_i: [Batch, 1]
- 拼接 logits：cat(logit_1 ... logit_T, dim=1) -> [Batch, Task_num]
- 输出头：按任务类型做激活（binary 使用 sigmoid）-> y_pred: [Batch, Task_num]
- 校准约束：惩罚 relu(p_{i+1} - p_i)，鼓励后序阶段概率不超过前序阶段
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.models.multitask.base import BaseMultitaskModel
from nextrec.utils.model import get_mlp_output_dim
from nextrec.utils.types import TaskTypeInput


class AITMTransfer(nn.Module):
    """Attentive information transfer from previous task to current task."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.prev_proj = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.query = nn.Linear(input_dim, input_dim)

    def forward(self, prev_feat: torch.Tensor, curr_feat: torch.Tensor) -> torch.Tensor:
        prev = self.prev_proj(prev_feat).unsqueeze(1)
        curr = curr_feat.unsqueeze(1)
        stacked = torch.cat([prev, curr], dim=1)
        value = self.value(stacked)
        key = self.key(stacked)
        query = self.query(stacked)
        attn_scores = torch.sum(key * query, dim=2, keepdim=True) / math.sqrt(self.input_dim)
        attn = torch.softmax(attn_scores, dim=1)
        return torch.sum(attn * value, dim=1)


class AITM(BaseMultitaskModel):
    """
    Attentive Information Transfer Multi-Task model.

    AITM learns task-specific representations and transfers information from
    task i-1 to task i via attention, enabling sequential task dependency modeling.
    """

    @property
    def model_name(self):
        return "AITM"

    @property
    def default_task(self):
        nums_task = getattr(self, "nums_task", None)
        if nums_task is not None and nums_task > 0:
            return ["binary"] * nums_task
        return ["binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        bottom_mlp_params: dict | list[dict] | None = None,
        tower_mlp_params_list: list[dict] | None = None,
        calibrator_alpha: float = 0.1,
        target: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        **kwargs,
    ):
        """
        Initialize AITM model.
        初始化 AITM 模型。

        Args:
            bottom_mlp_params: Bottom tower MLP config, can be:
                - dict: shared config for all tasks
                - list[dict]: per-task configs (length must equal task number)
                底层 MLP 参数，可为共享 dict 或按任务配置的 list[dict]（长度必须等于任务数）。
            tower_mlp_params_list: Per-task tower MLP configs.
                Length must equal number of tasks.
                任务塔 MLP 参数列表，长度必须等于任务数。
            calibrator_alpha: Weight of sequential monotonic calibrator loss.
                0 means disable calibrator term.
                序关系校准损失权重，0 表示关闭该项。
        """
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        bottom_mlp_params = bottom_mlp_params or {}
        tower_mlp_params_list = tower_mlp_params_list or []
        self.calibrator_alpha = calibrator_alpha

        if target is None:
            raise ValueError("AITM requires target names for all tasks.")
        if isinstance(target, str):
            target = [target]
        if calibrator_alpha < 0:
            raise ValueError(f"calibrator_alpha must be >= 0, got {calibrator_alpha}")

        self.nums_task = len(target)
        if self.nums_task < 2:
            raise ValueError("AITM requires at least 2 tasks.")
        if isinstance(task, str):
            raise ValueError(
                "AITM requires multi-task `task` to be a list, e.g. ['binary', 'binary'], not a single string."
            )
        if isinstance(task, list) and len(task) != self.nums_task:
            raise ValueError(f"AITM requires task length {self.nums_task}, got {len(task)}.")

        super(AITM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )
        self.nums_task = len(target)
        task_types = [self.task] if isinstance(self.task, str) else list(self.task)
        for task_type in task_types:
            if task_type != "binary":
                raise ValueError(f"AITM task must be binary, got '{task_type}'.")

        if len(tower_mlp_params_list) != self.nums_task:
            raise ValueError(
                f"Number of tower mlp params ({len(tower_mlp_params_list)}) must match number of tasks ({self.nums_task})."
            )

        bottom_mlp_params_list: list[dict]
        if isinstance(bottom_mlp_params, list):
            if len(bottom_mlp_params) != self.nums_task:
                raise ValueError(
                    f"Number of bottom mlp params ({len(bottom_mlp_params)}) must match number of tasks ({self.nums_task})."
                )
            bottom_mlp_params_list = [params.copy() for params in bottom_mlp_params]
        else:
            bottom_mlp_params_list = [bottom_mlp_params.copy() for _ in range(self.nums_task)]

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        # Remove output_dim from bottom MLP params since AITMTransfer handles fixed-dim transfer
        bottom_runtime_params = []
        for params in bottom_mlp_params_list:
            p = params.copy()
            p.pop("output_dim", None)
            bottom_runtime_params.append(p)

        self.bottoms = nn.ModuleList(
            [MLP(input_dim=input_dim, output_dim=None, **params) for params in bottom_runtime_params]
        )
        bottom_dims = [get_mlp_output_dim(params, input_dim) for params in bottom_runtime_params]
        if len(set(bottom_dims)) != 1:
            raise ValueError(f"All bottom output dims must match, got {bottom_dims}.")
        bottom_output_dim = bottom_dims[0]

        self.transfers = nn.ModuleList([AITMTransfer(bottom_output_dim) for _ in range(self.nums_task - 1)])
        self.grad_norm_shared_modules = ["embedding", "transfers"]

        self.towers = nn.ModuleList(
            [MLP(input_dim=bottom_output_dim, output_dim=1, **params) for params in tower_mlp_params_list]
        )

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["bottoms", "transfers", "towers"],
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        # Shared embedding flatten:
        # input_flat: [Batch, Dim_embedding]
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Per-task bottom encoders:
        # task_feats[i]: [Batch, Dim_bottom]
        task_feats = [bottom(input_flat) for bottom in self.bottoms]

        # Sequential attentive transfer:
        # transfer input: prev/curr each [Batch, Dim_bottom]
        # transfer output overwrites current task feature: [Batch, Dim_bottom]
        for idx in range(1, self.nums_task):
            task_feats[idx] = self.transfers[idx - 1](task_feats[idx - 1], task_feats[idx])

        # Task tower logits:
        # task_outputs[i]: [Batch, 1]
        task_outputs = [tower(task_feats[idx]) for idx, tower in enumerate(self.towers)]

        # Concatenate all task logits:
        # logits: [Batch, Task_num]
        logits = torch.cat(task_outputs, dim=1)

        # Apply task head activations (binary -> sigmoid):
        # y_pred: [Batch, Task_num]
        return logits

    def compute_calibrator_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sequential monotonic constraint:
        enforce p(task_i) >= p(task_{i+1}) by penalizing relu(p_{i+1} - p_i).
        """
        if y_pred.dim() == 1:
            y_pred = y_pred.view(-1, 1)
        if y_pred.shape[1] != self.nums_task:
            raise ValueError(
                f"[AITM-calibrator] Prediction columns ({y_pred.shape[1]}) must equal nums_task ({self.nums_task})."
            )

        if y_true is not None and y_true.dim() == 1:
            y_true = y_true.view(-1, 1)

        penalties = []
        for idx in range(1, self.nums_task):
            prev_pred = y_pred[:, idx - 1 : idx]
            curr_pred = y_pred[:, idx : idx + 1]
            pair_penalty = torch.relu(curr_pred - prev_pred).view(-1)

            if self.ignore_label is not None and y_true is not None:
                if y_true.shape[1] < self.nums_task:
                    raise ValueError(
                        f"[AITM-calibrator] Label columns ({y_true.shape[1]}) must be >= nums_task ({self.nums_task})."
                    )
                valid_mask = (y_true[:, idx - 1 : idx] != self.ignore_label) & (
                    y_true[:, idx : idx + 1] != self.ignore_label
                )
                pair_penalty = pair_penalty[valid_mask.view(-1)]

            if pair_penalty.numel() > 0:
                penalties.append(pair_penalty.mean())

        if not penalties:
            return y_pred.sum() * 0.0
        return torch.stack(penalties).mean()

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        base_loss = super().compute_loss(y_pred, y_true)
        if self.calibrator_alpha <= 0:
            return base_loss
        calibrator_loss = self.compute_calibrator_loss(y_pred=y_pred, y_true=y_true)
        return base_loss + self.calibrator_alpha * calibrator_loss
