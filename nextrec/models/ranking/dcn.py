"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]
//Proceedings of the ADKDD'17. 2017: 1-7.
(https://arxiv.org/abs/1708.05123)

Deep & Cross Network (DCN) mixes explicit polynomial feature crosses with a deep
MLP branch to capture both low-order and high-order interactions for CTR-style
tasks. Cross Layers repeatedly apply x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l,
which expands feature crosses with linear parameter growth, while the deep branch
learns nonlinear patterns on the same shared embeddings. The final prediction
concatenates (or solely uses) cross outputs before a linear head, offering a
balanced trade-off between interpretability and expressiveness.

Workflow:
  (1) Embed sparse/sequence features and concatenate with dense inputs
  (2) Cross Network builds explicit polynomial interactions via residual crosses
  (3) Optional MLP models implicit high-order nonlinear relationships
  (4) Cross output (and deep output if enabled) are fused for the final logit
  (5) Prediction layer maps logits to binary CTR scores

Key Advantages:
- Explicit, low-cost cross features with O(L * d) parameters
- Residual cross formulation stabilizes optimization
- Optional deep tower increases capacity without losing interpretability
- Shared embeddings reduce redundant parameters and preprocessing
- Strong, simple baseline for ad/recommendation ranking tasks

DCN（Deep & Cross Network）通过 Cross 层显式生成多项式特征交互，同时可选 Deep
分支学习高阶非线性关系，两者共享 embedding。Cross 层按
x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l 递推，参数线性增长且具解释性；
Deep 分支提升表达能力；最终将 Cross（及 Deep）结果送入线性层与预测层，形成兼具
效率与效果的 CTR/CVR 预估模型。

流程：
  (1) 对稀疏/序列特征做 embedding，并与稠密特征拼接
  (2) Cross 层以残差形式显式构造多阶交叉特征
  (3) 可选 MLP 学习隐式高阶非线性交互
  (4) 将 Cross（及 Deep）输出融合后接线性头得到 logit
  (5) 预测层输出二分类 CTR 分数

主要优点：
- 显式交叉特征、参数线性增长、易解释
- 残差式 Cross 提升训练稳定性
- Deep 分支可灵活增强模型容量
- 共享 embedding，减少冗余参数与预处理
- CTR/CVR 排序任务的简洁强基线
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class CrossNetwork(nn.Module):
    """Stacked Cross Layers from DCN (Wang et al., 2017)."""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)]
        )
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)]
        )

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x  # [batch_size, input_dim]


class DCN(BaseModel):
    @property
    def model_name(self):
        return "DCN"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        cross_num: int = 3,
        mlp_params: dict | None = None,
        target: list[str] | str | None = None,
        task: str | list[str] | None = None,
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss: str | nn.Module | None = "bce",
        loss_params: dict | list[dict] | None = None,
        device: str = "cpu",
        embedding_l1_reg=1e-6,
        dense_l1_reg=1e-5,
        embedding_l2_reg=1e-5,
        dense_l2_reg=1e-4,
        **kwargs,
    ):

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        super(DCN, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task or self.default_task,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Calculate input dimension
        emb_dim_total = sum(
            [
                f.embedding_dim
                for f in self.all_features
                if not isinstance(f, DenseFeature)
            ]
        )
        dense_input_dim = sum(
            [getattr(f, "embedding_dim", 1) or 1 for f in dense_features]
        )
        input_dim = emb_dim_total + dense_input_dim

        # Cross Network for explicit feature crosses
        self.cross_network = CrossNetwork(input_dim=input_dim, num_layers=cross_num)

        # Deep Network for implicit high-order interactions
        if mlp_params is not None:
            self.use_dnn = True
            self.mlp = MLP(input_dim=input_dim, **mlp_params)
            deep_dim = self.mlp.output_dim
            # Final layer combines cross and deep
            self.final_layer = nn.Linear(
                input_dim + deep_dim, 1
            )  # + deep_dim for MLP output
        else:
            self.use_dnn = False
            # Final layer only uses cross network output
            self.final_layer = nn.Linear(input_dim, 1)

        self.prediction_layer = TaskHead(task_type=self.task)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["cross_network", "mlp", "final_layer"],
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

        # Cross Network
        cross_output = self.cross_network(input_flat)  # [B, input_dim]

        if self.use_dnn:
            # Deep Network
            deep_output = self.mlp(input_flat)  # [B, 1]
            # Concatenate cross and deep
            combined = torch.cat(
                [cross_output, deep_output], dim=-1
            )  # [B, input_dim + 1]
        else:
            combined = cross_output

        # Final prediction
        y = self.final_layer(combined)
        return self.prediction_layer(y)
