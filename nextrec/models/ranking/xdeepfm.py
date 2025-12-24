"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Lian J, Zhou X, Zhang F, et al. xdeepfm: Combining explicit and implicit feature interactions
for recommender systems[C]//Proceedings of the 24th ACM SIGKDD international conference on
knowledge discovery & data mining. 2018: 1754-1763.
(https://arxiv.org/abs/1803.05170)

xDeepFM is a CTR prediction model that unifies explicit and implicit
feature interaction learning. It extends DeepFM by adding the
Compressed Interaction Network (CIN) to explicitly model high-order
interactions at the vector-wise level, while an MLP captures implicit
non-linear crosses. A linear term retains first-order signals, and all
three parts are learned jointly end-to-end.

In the forward pass:
  (1) Embedding Layer: transforms sparse/sequence fields into dense vectors
  (2) Linear Part: captures first-order contributions of sparse/sequence fields
  (3) CIN: explicitly builds higher-order feature crosses via convolution over
      outer products of field embeddings, with optional split-half connections
  (4) Deep Part (MLP): models implicit, non-linear interactions across all fields
  (5) Combination: sums outputs from linear, CIN, and deep branches before the
      task-specific prediction layer

Key Advantages:
- Jointly learns first-order, explicit high-order, and implicit interactions
- CIN offers interpretable vector-wise crosses with controlled complexity
- Deep branch enhances representation power for non-linear patterns
- End-to-end optimization eliminates heavy manual feature engineering
- Flexible design supports both sparse and sequence features

xDeepFM 是一个 CTR 预估模型，将显式与隐式的特征交互学习统一到同一框架。
在 DeepFM 的基础上，额外引入了 CIN（Compressed Interaction Network）
显式建模高阶向量级交互，同时 MLP 负责隐式非线性交互，线性部分保留一阶信号，
三者联合训练。

前向流程：
  (1) 嵌入层：将稀疏/序列特征映射为稠密向量
  (2) 线性部分：建模稀疏/序列特征的一阶贡献
  (3) CIN：通过对字段嵌入做外积并卷积，显式捕获高阶交叉，可选 split-half 以控参
  (4) 深层部分（MLP）：对所有特征进行隐式非线性交互建模
  (5) 融合：线性、CIN、MLP 输出求和后进入任务预测层

主要优点：
- 同时学习一阶、显式高阶、隐式交互
- CIN 提供可解释的向量级交叉并可控复杂度
- 深层分支提升非线性表达能力
- 端到端训练降低人工特征工程需求
- 兼容稀疏与序列特征的建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import LR, MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class CIN(nn.Module):
    """Compressed Interaction Network from xDeepFM (Lian et al., 2018)."""

    def __init__(self, input_dim, cin_size, split_half=True):
        super().__init__()
        self.num_layers = len(cin_size)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cin_size[i]
            self.conv_layers.append(
                torch.nn.Conv1d(
                    input_dim * prev_dim,
                    cross_layer_size,
                    1,
                    stride=1,
                    dilation=1,
                    bias=True,
                )
            )
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class xDeepFM(BaseModel):
    @property
    def model_name(self):
        return "xDeepFM"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        mlp_params: dict,
        cin_size: list[int] | None = None,
        split_half: bool = True,
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

        cin_size = cin_size or [128, 128]
        if target is None:
            target = []
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        super(xDeepFM, self).__init__(
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

        self.loss = loss

        # Linear part and CIN part: use sparse and sequence features
        self.linear_features = sparse_features + sequence_features

        # Deep part: use all features
        self.deep_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.deep_features)

        # Linear part
        linear_dim = sum([f.embedding_dim for f in self.linear_features])
        self.linear = LR(linear_dim)

        # CIN part: Compressed Interaction Network
        num_fields = len(self.linear_features)
        self.cin = CIN(input_dim=num_fields, cin_size=cin_size, split_half=split_half)

        # Deep part: DNN
        deep_emb_dim_total = sum(
            [
                f.embedding_dim
                for f in self.deep_features
                if not isinstance(f, DenseFeature)
            ]
        )
        dense_input_dim = sum(
            [getattr(f, "embedding_dim", 1) or 1 for f in dense_features]
        )
        self.mlp = MLP(input_dim=deep_emb_dim_total + dense_input_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.task)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["linear", "cin", "mlp"]
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        # Get embeddings for linear and CIN (sparse features only)
        input_linear = self.embedding(
            x=x, features=self.linear_features, squeeze_dim=False
        )

        # Linear part
        y_linear = self.linear(input_linear.flatten(start_dim=1))

        # CIN part
        y_cin = self.cin(input_linear)  # [B, 1]

        # Deep part
        input_deep = self.embedding(x=x, features=self.deep_features, squeeze_dim=True)
        y_deep = self.mlp(input_deep)  # [B, 1]

        # Combine all parts
        y = y_linear + y_cin + y_deep
        return self.prediction_layer(y)
