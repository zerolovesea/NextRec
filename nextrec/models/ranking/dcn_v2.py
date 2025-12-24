"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] R. Wang et al. DCN V2: Improved Deep & Cross Network and Practical Lessons for
Web-scale Learning to Rank Systems. KDD 2021.
(https://arxiv.org/abs/2008.13535)

DCN v2 enhances the original Deep & Cross Network by replacing the scalar cross
weights with vector-wise (matrix) parameters and a Mixture-of-Low-Rank-Experts
variant. The matrix cross (CrossNetV2) improves expressiveness with manageable
parameter growth, while CrossNetMix decomposes the matrix into low-rank factors and
gates across experts for stronger modeling at a similar cost. As in DCN, the cross
tower explicitly builds polynomial feature interactions and can be paired with a
deep MLP tower; their outputs are concatenated before a final linear head.

Workflow:
  (1) Embed sparse/sequence fields and concatenate with dense inputs
  (2) Cross tower: choose matrix CrossNetV2 or low-rank CrossNetMix for explicit crosses
  (3) Optional deep tower: MLP over the same flattened embeddings
  (4) Fuse cross and deep outputs, then predict via a linear + prediction layer

Key Advantages:
- Vector-wise/matrix cross weights capture richer interactions than DCN v1
- Low-rank MoE cross (CrossNetMix) boosts capacity without quadratic parameters
- Compatible with a deep tower for additional nonlinear modeling

DCN v2 在原始 DCN 基础上，将标量交叉权重升级为向量/矩阵参数，并引入低秩专家混合
的 CrossNetMix。矩阵交叉（CrossNetV2）在参数可控的前提下提升表达力，CrossNetMix
通过低秩分解和 gating 进一步增强建模能力且保持参数效率。和 DCN 一样，交叉塔显式
构造多项式交互，可并行或串联一个 MLP 深塔，最终拼接/输出到线性头做预测。

流程：
  (1) 对稀疏/序列特征做 embedding，并与稠密特征拼接
  (2) 交叉塔：可选矩阵 CrossNetV2 或低秩混合 CrossNetMix 显式构造交互
  (3) 可选深塔：MLP 处理同一展平后的输入或交叉输出
  (4) 融合交叉与深塔输出，经线性层和预测层得到最终得分

主要优点：
  - 矩阵交叉相较 DCN v1 捕获更丰富的交互
  - 低秩专家混合在相近参数量下带来更强建模能力
  - 兼容并行/串行深塔，灵活扩展非线性表示
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class CrossNetV2(nn.Module):
    """Vector-wise cross network proposed in DCN V2 (Wang et al., 2021)."""

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_dim, input_dim, bias=False)
                for _ in range(num_layers)
            ]
        )
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for i in range(self.num_layers):
            x = x0 * self.w[i](x) + self.b[i] + x
        return x


class CrossNetMix(nn.Module):
    """Mixture of low-rank cross experts from DCN V2 (Wang et al., 2021)."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        low_rank: int = 32,
        num_experts: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        self.u_list = nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(
                        torch.empty(num_experts, input_dim, low_rank)
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.v_list = nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(
                        torch.empty(num_experts, input_dim, low_rank)
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.c_list = nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(num_experts, low_rank, low_rank))
                )
                for _ in range(num_layers)
            ]
        )

        self.gating = nn.ModuleList(
            [nn.Linear(input_dim, 1, bias=False) for _ in range(num_experts)]
        )

        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, in_features)
        x_0 = x.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0

        for i in range(self.num_layers):
            output_of_experts = []
            gating_score_experts = []

            gating_input = x_l.squeeze(2)  # (bs, in_features)

            for expert_id in range(self.num_experts):
                # Gating
                gating_score_experts.append(
                    self.gating[expert_id](gating_input)
                )  # (bs, 1)

                # Low-rank cross: U C V^T x_l
                V = self.v_list[i][expert_id]  # (in_features, low_rank)
                C = self.c_list[i][expert_id]  # (low_rank, low_rank)
                U = self.u_list[i][expert_id]  # (in_features, low_rank)

                # (bs, 1, low_rank)
                v_x = x_l.transpose(1, 2).matmul(V)  # x_l^T V
                v_x = v_x.matmul(C)  # · C
                v_x = torch.tanh(v_x)

                # (bs, in_features, 1)
                uv_x = U.matmul(v_x.transpose(1, 2))

                # x_0 ⊙ (uv_x + b)
                dot_ = x_0 * (uv_x + self.bias[i])  # (bs, in_features, 1)

                output_of_experts.append(dot_.squeeze(2))  # (bs, in_features)

            # (3) Mixture of experts
            output_of_experts = torch.stack(
                output_of_experts, dim=2
            )  # (bs, in_features, num_experts)
            gating_score_experts = torch.stack(
                gating_score_experts, dim=1
            )  # (bs, num_experts, 1)
            gating_score_experts = gating_score_experts.softmax(dim=1)

            moe_out = torch.matmul(
                output_of_experts, gating_score_experts
            )  # (bs, in_features, 1)
            x_l = moe_out + x_l  # residual

        return x_l.squeeze(-1)  # (bs, in_features)


class DCNv2(BaseModel):
    @property
    def model_name(self) -> str:
        return "DCNv2"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        cross_num: int = 3,
        cross_type: str = "matrix",
        architecture: str = "parallel",
        low_rank: int = 32,
        num_experts: int = 4,
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

        super(DCNv2, self).__init__(
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

        self.all_features = dense_features + sparse_features + sequence_features
        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        architecture = architecture.lower()
        if architecture not in {"parallel", "stacked"}:
            raise ValueError("architecture must be 'parallel' or 'stacked'.")
        self.architecture = architecture

        cross_type = cross_type.lower()
        if cross_type == "matrix":
            self.cross_network = CrossNetV2(input_dim=input_dim, num_layers=cross_num)
        elif cross_type in {"mix", "low_rank"}:
            self.cross_network = CrossNetMix(
                input_dim=input_dim,
                num_layers=cross_num,
                low_rank=low_rank,
                num_experts=num_experts,
            )
        else:
            raise ValueError("Unsupported cross_type for DCNv2. Use 'matrix' or 'mix'.")

        if mlp_params is not None:
            self.use_dnn = True
            dnn_params = dict(mlp_params)
            dnn_params.setdefault("output_layer", False)
            self.mlp = MLP(input_dim=input_dim, **dnn_params)
            deep_dim = self.mlp.output_dim
            final_input_dim = (
                input_dim + deep_dim if architecture == "parallel" else deep_dim
            )
        else:
            if architecture == "stacked":
                raise ValueError(
                    "Stacked architecture requires mlp_params (deep tower)."
                )
            self.use_dnn = False
            self.mlp = None
            final_input_dim = input_dim

        self.final_layer = nn.Linear(final_input_dim, 1)
        self.prediction_layer = TaskHead(task_type=self.default_task)

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

    def forward(self, x) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        cross_out = self.cross_network(input_flat)

        if self.use_dnn and self.mlp is not None:
            if self.architecture == "parallel":
                deep_out = self.mlp(input_flat)
                combined = torch.cat([cross_out, deep_out], dim=-1)
            else:  # stacked
                deep_out = self.mlp(cross_out)
                combined = deep_out
        else:
            combined = cross_out

        logit = self.final_layer(combined)
        return self.prediction_layer(logit)
