"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference:
[1] Ma X, Zhao L, Huang G, et al. Entire space multi-task model: An effective approach
for estimating post-click conversion rate[C]//SIGIR. 2018: 1137-1140.
(https://dl.acm.org/doi/10.1145/3209978.3210007)

Entire Space Multi-task Model (ESMM) targets CVR estimation by jointly optimizing
CTR and CTCVR on the full impression space, mitigating sample selection bias and
conversion sparsity. CTR predicts P(click | impression), CVR predicts P(conversion |
click), and their product forms CTCVR supervised on impression labels.

Workflow:
  (1) Shared embeddings encode all features from impressions
  (2) CTR tower outputs click probability conditioned on impression
  (3) CVR tower outputs conversion probability conditioned on click
  (4) CTCVR = CTR * CVR enables end-to-end training without filtering clicked data

Key Advantages:
- Trains on the entire impression space to remove selection bias
- Transfers rich click signals to sparse conversion prediction via shared embeddings
- Stable optimization by decomposing CTCVR into well-defined sub-tasks
- Simple architecture that can pair with other multi-task variants

ESMM（Entire Space Multi-task Model）用于 CVR 预估，通过在曝光全空间联合训练
CTR 与 CTCVR，缓解样本选择偏差和转化数据稀疏问题。CTR 预测 P(click|impression)，
CVR 预测 P(conversion|click)，二者相乘得到 CTCVR 并在曝光标签上直接监督。

流程：
  (1) 共享 embedding 统一处理曝光特征
  (2) CTR 塔输出曝光下的点击概率
  (3) CVR 塔输出点击后的转化概率
  (4) CTR 与 CVR 相乘得到 CTCVR，无需只在点击子集上训练

主要优点：
- 在曝光空间训练，避免样本选择偏差
- 通过共享表示将点击信号迁移到稀疏的转化任务
- 将 CTCVR 分解为子任务，优化稳定
- 结构简单，可与其它多任务方法组合使用
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


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
    def default_task(self):
        return ["binary", "binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        ctr_params: dict,
        cvr_params: dict,
        target: list[str] | None = None,  # Note: ctcvr = ctr * cvr
        task: list[str] | None = None,
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        device: str = "cpu",
        embedding_l1_reg=1e-6,
        dense_l1_reg=1e-5,
        embedding_l2_reg=1e-5,
        dense_l2_reg=1e-4,
        **kwargs,
    ):

        target = target or ["ctr", "ctcvr"]
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        if len(target) != 2:
            raise ValueError(
                f"ESMM requires exactly 2 targets (ctr and ctcvr), got {len(target)}"
            )

        self.nums_task = len(target)
        resolved_task = task
        if resolved_task is None:
            resolved_task = self.default_task
        elif isinstance(resolved_task, str):
            resolved_task = [resolved_task] * self.nums_task
        elif len(resolved_task) == 1 and self.nums_task > 1:
            resolved_task = resolved_task * self.nums_task
        elif len(resolved_task) != self.nums_task:
            raise ValueError(
                f"Length of task ({len(resolved_task)}) must match number of targets ({self.nums_task})."
            )
        # resolved_task is now guaranteed to be a list[str]

        super(ESMM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=resolved_task,  # Both CTR and CTCVR are binary classification
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        self.loss = loss

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        # CTR tower
        self.ctr_tower = MLP(input_dim=input_dim, output_layer=True, **ctr_params)

        # CVR tower
        self.cvr_tower = MLP(input_dim=input_dim, output_layer=True, **cvr_params)
        self.grad_norm_shared_modules = ["embedding"]
        self.prediction_layer = TaskHead(task_type=self.default_task, task_dims=[1, 1])
        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["ctr_tower", "cvr_tower"]
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

        # CTR prediction: P(click | impression)
        ctr_logit = self.ctr_tower(input_flat)  # [B, 1]
        cvr_logit = self.cvr_tower(input_flat)  # [B, 1]
        logits = torch.cat([ctr_logit, cvr_logit], dim=1)
        preds = self.prediction_layer(logits)
        ctr, cvr = preds.chunk(2, dim=1)
        ctcvr = ctr * cvr  # [B, 1]

        # Output: [CTR, CTCVR], We supervise CTR with click labels and CTCVR with conversion labels
        y = torch.cat([ctr, ctcvr], dim=1)  # [B, 2]
        return y  # [B, 2], where y[:, 0] is CTR and y[:, 1] is CTCVR
