"""
Date: create on 09/11/2025
Checkpoint: edit on 21/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C] //Proceedings of the ADKDD'17. 2017: 1-7.
URL: https://arxiv.org/abs/1708.05123

Deep & Cross Network (DCN) mixes explicit polynomial feature crosses with a deep
MLP branch to capture both low-order and high-order interactions for CTR-style
tasks. Cross Layers repeatedly apply x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l,
which expands feature crosses with linear parameter growth, while the deep branch
learns nonlinear patterns on the same shared embeddings. The final prediction
concatenates (or solely uses) cross outputs before a linear head, offering a
balanced trade-off between interpretability and expressiveness.

Workflow:
- Embed sparse/sequence features and concatenate with dense features
- Cross layers explicitly construct multi-order cross features in a residual form
- Optional MLP learns implicit high-order nonlinear interactions
- Fuse Cross (and Deep) outputs and feed into linear head for logit
- Prediction layer outputs binary CTR score

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length] -> embedding layer + flatten -> [Batch, Dim_embedding]
- Cross tower: [Batch, Dim_embedding] -> stacked cross layers -> cross_out:[Batch, Dim_embedding]
- Deep tower (optional): [Batch, Dim_embedding] -> MLP -> deep_out[Batch, Hidden_dim]
- Fusion: concat(cross_out[Batch, Dim_embedding], deep_out[Batch, Hidden_dim]) -> [Batch, Dim_embedding + Hidden_dim]
  (or [Batch, Dim_embedding] if no deep tower)
- Output: linear -> [Batch, 1] -> prediction head

DCN（Deep & Cross Network）通过 Cross 层显式生成多项式特征交互，同时可选 Deep
分支学习高阶非线性关系，两者共享 embedding。Cross 层按
x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l 递推，参数线性增长且具解释性；
Deep 分支提升表达能力；最终将 Cross（及 Deep）结果送入线性层与预测层，形成兼具
效率与效果的 CTR/CVR 预估模型。

流程：
- 对稀疏/序列特征做 embedding，并与稠密特征拼接
- Cross 层以残差形式显式构造多阶交叉特征
- 可选 MLP 学习隐式高阶非线性交互
- 将 Cross（及 Deep）输出融合后接线性头得到 logit
- 预测层输出二分类 CTR 分数

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length] -> embedding层 + 展平 -> [Batch, Dim_embedding]
- Cross [Batch, Dim_embedding] -> 多层显式交叉 -> cross_out: [Batch, Dim_embedding]
- Deep 分支（可选）：[Batch, Dim_embedding] -> MLP -> deep_out: [Batch, Hidden_dim]
- 融合：concat(cross_out[Batch, Dim_embedding], deep_out[Batch, Hidden_dim]) -> [Batch, Dim_embedding + Hidden_dim]
  （无 Deep 时为 [Batch, Dim_embedding]）
- 输出：线性层 -> [Batch, 1] -> 预测层

"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class CrossNetwork(nn.Module):
    """Stacked Cross Layers from DCN (Wang et al., 2017)."""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        # x: [B, D]
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)  # [B, 1]
            x = x0 * xw + self.b[i] + x  # [B, D]
        return x  # [batch_size, input_dim]


class DCN(BaseRankingModel):
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
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        cross_num: int = 3,
        mlp_params: dict | None = None,
        **kwargs,
    ):
        """
        Initialize DCN model.
        初始化 DCN 模型。

        Args:
            cross_num: Number of cross layers in the explicit cross network.
                显式交叉网络中的 Cross 层数量。
            mlp_params: Parameters for optional deep tower MLP. e.g., {"hidden_units": [128, 64], "activation": "relu"}.
                If None, DCN runs in cross-only mode.
                可选深度塔 MLP 参数，例如 {"hidden_units": [128, 64], "activation": "relu"}。为 None 时仅使用 Cross 分支。
        """

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        use_dnn = mlp_params is not None
        dnn_params = dict(mlp_params) if mlp_params is not None else {}

        super(DCN, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        # Cross Network for explicit feature crosses
        self.cross_network = CrossNetwork(input_dim=input_dim, num_layers=cross_num)

        # Deep Network for implicit high-order interactions
        if use_dnn:
            self.use_dnn = True
            dnn_params.pop("input_dim", None)
            dnn_params.setdefault("output_dim", None)
            self.mlp = MLP(input_dim=input_dim, **dnn_params)
            deep_dim = self.mlp.output_dim
            # Final layer combines cross and deep
            self.final_layer = nn.Linear(input_dim + deep_dim, 1)
        else:
            self.use_dnn = False
            self.mlp = None
            # Final layer only uses cross network output
            self.final_layer = nn.Linear(input_dim, 1)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["cross_network", "mlp", "final_layer"],
        )

    def forward(self, x):
        # Get all embeddings and flatten: input_flat[Batch, Dim_embedding]
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Cross tower output: cross_out[Batch, Dim_embedding]
        cross_output = self.cross_network(input_flat)  # [Batch, input_dim]

        if self.use_dnn:
            # Deep tower output: deep_out[Batch, Hidden_dim]
            deep_output = self.mlp(input_flat)
            # Concatenate cross and deep: combined[Batch, Dim_embedding + Hidden_dim]
            combined = torch.cat([cross_output, deep_output], dim=-1)
        else:
            combined = cross_output  # [Batch, Dim_embedding]

        logits = self.final_layer(combined)  # [Batch, 1]
        return logits
