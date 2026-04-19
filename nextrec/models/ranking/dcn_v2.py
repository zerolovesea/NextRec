"""
Date: create on 09/11/2025
Checkpoint: edit on 14/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] R. Wang et al. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. KDD 2021.
URL: https://arxiv.org/abs/2008.13535

DCN v2 enhances the original Deep & Cross Network by replacing the scalar cross
weights with vector-wise (matrix) parameters and a Mixture-of-Low-Rank-Experts
variant. The matrix cross (CrossNetV2) improves expressiveness with manageable
parameter growth, while CrossNetMix decomposes the matrix into low-rank factors and
gates across experts for stronger modeling at a similar cost. As in DCN, the cross
tower explicitly builds polynomial feature interactions and can be paired with a
deep MLP tower; their outputs are concatenated before a final linear head.

When parallel, it approximates the original DCN.

Workflow:
- Embed sparse/sequence fields and concatenate with dense inputs
- Cross tower: choose matrix CrossNetV2 or low-rank CrossNetMix for explicit crosses
- Optional deep tower: MLP over the same flattened embeddings
- Fuse cross and deep outputs, then predict via a linear + prediction layer

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length] -> embedding layer + flatten -> input_flat: [Batch, Dim_embedding]
- Cross tower (matrix/mix): input_flat: [Batch, Dim_embedding] -> cross_out: [Batch, Dim_embedding]
- Parallel deep tower (optional): input_flat: [Batch, Dim_embedding] -> parallel_mlp -> parallel_out: [Batch, Hidden_parallel]
- Stacked deep tower (optional): cross_out: [Batch, Dim_embedding] -> stacked_mlp -> stacked_out: [Batch, Hidden_stacked]
- Fusion (crossnet_only): cross_out -> [Batch, Dim_embedding]
- Fusion (parallel): concat(cross_out, parallel_out) -> [Batch, Dim_embedding + Hidden_parallel]
- Fusion (stacked): stacked_out -> [Batch, Hidden_stacked]
- Fusion (stacked_parallel): concat(stacked_out, parallel_out) -> [Batch, Hidden_stacked + Hidden_parallel]
- Output: linear -> [Batch, 1] -> prediction head

DCN v2 在原始 DCN 基础上，将标量交叉权重升级为向量/矩阵参数，并引入低秩专家混合
的 CrossNetMix。矩阵交叉（CrossNetV2）在参数可控的前提下提升表达力，CrossNetMix
通过低秩分解和 gating 进一步增强建模能力且保持参数效率。和 DCN 一样，交叉塔显式
构造多项式交互，可并行或串联一个 MLP 深塔，最终拼接/输出到线性头做预测。

当并行时，近似等同于原始 DCN。

流程：
- 对稀疏/序列特征做 embedding，并与稠密特征拼接
- 交叉塔：可选矩阵 CrossNetV2 或低秩混合 CrossNetMix 显式构造交互
- 可选深塔：MLP 处理同一展平后的输入或交叉输出
- 融合交叉与深塔输出，经线性层和预测层得到最终得分

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length] -> embedding层 + 展平 -> input_flat: [Batch, Dim_embedding]
- Cross 分支（matrix/mix）：input_flat: [Batch, Dim_embedding] -> cross_out: [Batch, Dim_embedding]
- Parallel 深塔（可选）：input_flat: [Batch, Dim_embedding] -> parallel_mlp -> parallel_out: [Batch, Hidden_parallel]
- Stacked 深塔（可选）：cross_out: [Batch, Dim_embedding] -> stacked_mlp -> stacked_out: [Batch, Hidden_stacked]
- 融合（crossnet_only）：cross_out -> [Batch, Dim_embedding]
- 融合（parallel）：concat(cross_out, parallel_out) -> [Batch, Dim_embedding + Hidden_parallel]
- 融合（stacked）：stacked_out -> [Batch, Hidden_stacked]
- 融合（stacked_parallel）：concat(stacked_out, parallel_out) -> [Batch, Hidden_stacked + Hidden_parallel]
- 输出：线性层 -> [Batch, 1] -> 预测层

"""

import torch
import torch.nn as nn
from typing import Literal
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class CrossNetV2(nn.Module):
    """Vector-wise cross network proposed in DCN V2 (Wang et al., 2021)."""

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

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
                nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, input_dim, low_rank)))
                for _ in range(num_layers)
            ]
        )
        self.v_list = nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, input_dim, low_rank)))
                for _ in range(num_layers)
            ]
        )
        self.c_list = nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, low_rank, low_rank)))
                for _ in range(num_layers)
            ]
        )

        self.gating = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_experts)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, in_features)
        x_0 = x.unsqueeze(2)  # (Batch, in_features, 1)
        x_l = x_0

        for i in range(self.num_layers):
            output_of_experts = []
            gating_score_experts = []

            gating_input = x_l.squeeze(2)  # (Batch, in_features)

            for expert_id in range(self.num_experts):
                # Gating
                gating_score_experts.append(self.gating[expert_id](gating_input))  # (Batch, 1)

                # Low-rank cross: U C V^T x_l
                V = self.v_list[i][expert_id]  # (in_features, low_rank)
                C = self.c_list[i][expert_id]  # (low_rank, low_rank)
                U = self.u_list[i][expert_id]  # (in_features, low_rank)

                # (Batch, 1, low_rank): V -> tanh -> C -> tanh (align with torch-rechub/FuxiCTR)
                v_x = x_l.transpose(1, 2).matmul(V)  # x_l^T V
                v_x = torch.tanh(v_x)
                v_x = v_x.matmul(C)  # · C
                v_x = torch.tanh(v_x)

                # (Batch, in_features, 1)
                uv_x = U.matmul(v_x.transpose(1, 2))

                # x_0 ⊙ (uv_x + b)
                dot_ = x_0 * (uv_x + self.bias[i])  # (Batch, in_features, 1)

                output_of_experts.append(dot_.squeeze(2))  # (Batch, in_features)

            # (3) Mixture of experts
            output_of_experts = torch.stack(output_of_experts, dim=2)  # (Batch, in_features, num_experts)
            gating_score_experts = torch.stack(gating_score_experts, dim=1)  # (Batch, num_experts, 1)
            gating_score_experts = gating_score_experts.softmax(dim=1)

            moe_out = torch.matmul(output_of_experts, gating_score_experts)  # (Batch, in_features, 1)
            x_l = moe_out + x_l  # residual

        return x_l.squeeze(-1)  # (Batch, in_features)


class DCNv2(BaseRankingModel):
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
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        cross_num: int = 3,
        cross_type: Literal["matrix", "mix", "low_rank"] = "matrix",
        architecture: Literal["crossnet_only", "parallel", "stacked", "stacked_parallel"] = "parallel",
        low_rank: int = 32,
        num_experts: int = 4,
        mlp_params: dict | None = None,
        **kwargs,
    ):
        """
        Initialize DCNv2 model.
        初始化 DCNv2 模型。

        Args:
            cross_num: Number of layers in cross network.
                交叉网络层数。
            cross_type: Type of cross network.
                `"matrix"` uses standard CrossNetV2, `"mix"`/`"low_rank"`
                use low-rank mixture-of-experts cross.
                交叉网络类型：`"matrix"` 为标准 CrossNetV2，
                `"mix"`/`"low_rank"` 为低秩专家混合交叉。
            architecture: Fusion architecture between cross and deep branches.
                Supported: `"crossnet_only"`, `"parallel"`, `"stacked"`,
                `"stacked_parallel"`.
                Cross 与 Deep 的融合结构，可选：
                `"crossnet_only"`、`"parallel"`、`"stacked"`、`"stacked_parallel"`。
            low_rank: Low-rank bottleneck dimension for mix/low_rank cross network.
                mix/low_rank 交叉网络中的低秩瓶颈维度。
            num_experts: Number of experts for mix/low_rank cross network.
                mix/low_rank 交叉网络中的专家数量。
            mlp_params: Parameters for deep tower MLP(s). e.g. {"hidden_dims": [64, 32], "dropout": 0.2}.
                Required by `"stacked"` and `"stacked_parallel"` architectures.
                深度塔 MLP 参数，例如 {"hidden_dims": [64, 32], "dropout": 0.2}。
                在 `"stacked"` 和 `"stacked_parallel"` 架构下必填。
        """
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []

        super(DCNv2, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        if architecture not in {"crossnet_only", "parallel", "stacked", "stacked_parallel"}:
            raise ValueError("architecture must be 'crossnet_only', 'parallel', 'stacked', or 'stacked_parallel'.")
        self.architecture = architecture

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
            raise ValueError("Unsupported cross_type for DCNv2. Use 'matrix', 'mix', or 'low_rank'.")

        self.use_dnn = False
        self.mlp = None
        self.stacked_mlp = None
        self.parallel_mlp = None

        if architecture == "crossnet_only":
            final_input_dim = input_dim
        elif architecture == "parallel":
            if mlp_params is not None:
                self.use_dnn = True
                dnn_params = dict(mlp_params)
                dnn_params.pop("input_dim", None)
                dnn_params.setdefault("output_dim", None)
                self.parallel_mlp = MLP(input_dim=input_dim, **dnn_params)
                self.mlp = self.parallel_mlp
                final_input_dim = input_dim + self.parallel_mlp.output_dim
            else:
                final_input_dim = input_dim
        elif architecture == "stacked":
            if mlp_params is None:
                raise ValueError("Stacked architecture requires mlp_params (deep tower).")
            self.use_dnn = True
            dnn_params = dict(mlp_params)
            dnn_params.pop("input_dim", None)
            dnn_params.setdefault("output_dim", None)
            self.stacked_mlp = MLP(input_dim=input_dim, **dnn_params)
            self.mlp = self.stacked_mlp
            final_input_dim = self.stacked_mlp.output_dim
        else:  # stacked_parallel
            if mlp_params is None:
                raise ValueError("Stacked_parallel architecture requires mlp_params (deep tower).")
            self.use_dnn = True
            stacked_params = dict(mlp_params)
            stacked_params.pop("input_dim", None)
            stacked_params.setdefault("output_dim", None)
            parallel_params = dict(mlp_params)
            parallel_params.pop("input_dim", None)
            parallel_params.setdefault("output_dim", None)
            self.stacked_mlp = MLP(input_dim=input_dim, **stacked_params)
            self.parallel_mlp = MLP(input_dim=input_dim, **parallel_params)
            final_input_dim = self.stacked_mlp.output_dim + self.parallel_mlp.output_dim

        self.final_layer = nn.Linear(final_input_dim, 1)

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["cross_network", "mlp", "final_layer"],
        )

    def forward(self, x) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        cross_out = self.cross_network(input_flat)

        if self.architecture == "crossnet_only":
            combined = cross_out
        elif self.architecture == "parallel":
            if self.parallel_mlp is None:
                combined = cross_out
            else:
                combined = torch.cat([cross_out, self.parallel_mlp(input_flat)], dim=-1)
        elif self.architecture == "stacked":
            if self.stacked_mlp is None:
                raise RuntimeError("Internal error: stacked_mlp is not initialized for 'stacked' architecture.")
            combined = self.stacked_mlp(cross_out)
        elif self.architecture == "stacked_parallel":
            if self.stacked_mlp is None or self.parallel_mlp is None:
                raise RuntimeError(
                    "Internal error: stacked_mlp/parallel_mlp are not initialized for 'stacked_parallel' architecture."
                )
            stacked_out = self.stacked_mlp(cross_out)
            parallel_out = self.parallel_mlp(input_flat)
            combined = torch.cat([stacked_out, parallel_out], dim=-1)
        else:
            raise RuntimeError(f"Unsupported architecture at forward: {self.architecture}")

        logit = self.final_layer(combined)
        return logit
