"""
Date: create on 09/11/2025
Checkpoint: edit on 15/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//ICDM. 2016: 1149-1154. (https://arxiv.org/abs/1611.00144)

Product-based Neural Networks (PNN)  explicitly encode feature interactions by combining:
- A linear signal from concatenated field embeddings
- A product signal capturing pairwise feature interactions (inner or outer)
The product layer augments the linear input to an MLP, enabling the network to
model both first-order and high-order feature interactions in a structured way.

Workflow:
- Embed each categorical/sequence field with a shared embedding dimension
- Linear signal: flatten and concatenate all field embeddings
- Product signal:
    * Inner product: dot products over all field pairs
    * Outer product: project embeddings then compute element-wise products
- Concatenate linear and product signals; feed into MLP for prediction

Dimension Flow:
- Input: sparse[Batch] + sequence[Batch, Length] -> embedding -> field_emb: [Batch, Field_num, Dim_emb]
- Linear sparse/sequence signal: flatten(field_emb) -> linear_sparse: [Batch, Field_num * Dim_emb]
- Optional dense signal: dense features -> dense_signal: [Batch, Dim_dense]
- Linear signal fusion: linear_sparse (+ dense_signal) -> linear_signal: [Batch, Field_num * Dim_emb + Dim_dense]
- Pair construction: all field pairs -> Pair_num = Field_num * (Field_num - 1) / 2
- Inner product branch: pairwise dot(field_i, field_j) -> inner_product: [Batch, Pair_num]
- Outer product branch: kernelized pairwise interaction (mat/vec/num) -> outer_product: [Batch, Pair_num]
- Product signal:
    * inner  -> [Batch, Pair_num]
    * outer  -> [Batch, Pair_num]
    * both   -> concat(inner, outer) -> [Batch, 2 * Pair_num]
- MLP input: concat(linear_signal, product_signal) -> deep_input: [Batch, Linear_dim + Product_dim]
- Output: MLP(deep_input) -> logits: [Batch, 1] -> prediction layer -> [Batch, 1]

PNN 通过将不同域的特征嵌入后，通过两种乘积方式显式建模特征交互：
  - 线性信号：将各字段的 embedding 拼接，用于保留一阶信息
  - 乘积信号：对所有字段对做内积或外积，捕捉二阶及更高阶交互
随后将两类信号拼接送入 MLP，实现对用户响应的预测。内积版本计算量更低，
外积版本表达力更强，可根据场景取舍。

流程：
- 对每个类别/序列字段进行 embedding，统一维度
- 线性信号：将所有字段 embedding 展平拼接
- 乘积信号：
    * 内积：对所有字段对计算点积
    * 外积：先投影 embedding 再计算元素级乘积
- 将线性信号与乘积信号拼接，输入 MLP进行预测

维度变化：
- 输入：sparse[Batch] + sequence[Batch, Length] -> embedding -> field_emb: [Batch, Field_num, Dim_emb]
- 线性稀疏/序列信号：flatten(field_emb) -> linear_sparse: [Batch, Field_num * Dim_emb]
- 可选 dense 信号：dense features -> dense_signal: [Batch, Dim_dense]
- 线性信号融合：linear_sparse (+ dense_signal) -> linear_signal: [Batch, Field_num * Dim_emb + Dim_dense]
- 两两组合：字段对数量 Pair_num = Field_num * (Field_num - 1) / 2
- 内积分支：pairwise dot(field_i, field_j) -> inner_product: [Batch, Pair_num]
- 外积分支：kernel 化两两交互（mat/vec/num）-> outer_product: [Batch, Pair_num]
- 乘积信号：
    * inner -> [Batch, Pair_num]
    * outer -> [Batch, Pair_num]
    * both  -> concat(inner, outer) -> [Batch, 2 * Pair_num]
- MLP 输入：concat(linear_signal, product_signal) -> deep_input: [Batch, Linear_dim + Product_dim]
- 输出：MLP(deep_input) -> logits: [Batch, 1] -> prediction layer -> [Batch, 1]
"""

import torch
import torch.nn as nn
from typing import Literal
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeInput


class PNN(BaseModel):

    @property
    def model_name(self):
        return "PNN"

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
        mlp_params: dict | None = None,
        product_type: Literal["inner", "outer", "both"] = "inner",  # "inner" (IPNN), "outer" (OPNN), "both" (PNN*)
        outer_kernel_type: Literal["mat", "vec", "num"] = "mat",
        product_embedding_dim: int | None = None,
        **kwargs,
    ):
        """
        Initialize PNN model.
        初始化 PNN 模型。

        Args:
            mlp_params: Parameters for MLP after signal fusion, e.g.
                {"hidden_dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}.
                融合线性信号与乘积信号后的 MLP 参数。
            product_type: Product branch type: "inner", "outer", or "both".
                乘积分支类型："inner"、"outer" 或 "both"。
            outer_kernel_type: Kernel form for outer-product branch: "mat", "vec", or "num".
                外积分支使用的核形式："mat"、"vec" 或 "num"。
            product_embedding_dim: Shared embedding dimension used by product branch.
                If None, it is inferred from field effective dims (max dim when dims differ).
                乘积分支使用的统一维度；为 None 时自动从字段有效维度推断（维度不一致时取最大值）。
        """

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}

        super(PNN, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        self.dense_features = dense_features
        self.field_features = sparse_features + sequence_features
        if len(self.field_features) < 2:
            raise ValueError("PNN requires at least two sparse/sequence features.")

        self.embedding = EmbeddingLayer(features=self.dense_features + self.field_features)
        self.num_fields = len(self.field_features)

        for feature in self.field_features:
            if isinstance(feature, SequenceFeature) and feature.combiner == "concat" and feature.max_len is None:
                raise ValueError(
                    f"PNN requires SequenceFeature('{feature.name}') to set max_len when combiner='concat'."
                )
        self.field_dims = [self.embedding.compute_output_dim([feature]) for feature in self.field_features]
        inferred_product_dim = self.field_dims[0] if len(set(self.field_dims)) == 1 else max(self.field_dims)
        self.product_embedding_dim = (
            inferred_product_dim if product_embedding_dim is None else int(product_embedding_dim)
        )
        if self.product_embedding_dim <= 0:
            raise ValueError("product_embedding_dim must be a positive integer.")
        self.product_projections = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if field_dim == self.product_embedding_dim
                    else nn.Linear(field_dim, self.product_embedding_dim, bias=False)
                )
                for field_dim in self.field_dims
            ]
        )

        self.product_type = product_type.lower()
        if self.product_type not in {"inner", "outer", "both"}:
            raise ValueError("product_type must be 'inner', 'outer', or 'both'.")
        self.outer_kernel_type = outer_kernel_type.lower()
        if self.outer_kernel_type not in {"mat", "vec", "num"}:
            raise ValueError("outer_kernel_type must be 'mat', 'vec', or 'num'.")

        self.num_pairs = self.num_fields * (self.num_fields - 1) // 2

        row = []
        col = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row.append(i)
                col.append(j)
        self.register_buffer("row_idx", torch.tensor(row, dtype=torch.long))
        self.register_buffer("col_idx", torch.tensor(col, dtype=torch.long))

        if self.product_type in {"outer", "both"}:
            if self.outer_kernel_type == "mat":
                self.kernel = nn.Parameter(
                    torch.empty(self.product_embedding_dim, self.num_pairs, self.product_embedding_dim)
                )
            elif self.outer_kernel_type == "vec":
                self.kernel = nn.Parameter(torch.empty(self.num_pairs, self.product_embedding_dim))
            else:
                self.kernel = nn.Parameter(torch.empty(self.num_pairs, 1))
            nn.init.xavier_uniform_(self.kernel)
        else:
            self.kernel = None

        dense_dim = sum(feature.embedding_dim for feature in self.dense_features)
        linear_dim = sum(self.field_dims) + dense_dim

        if self.product_type == "inner":
            product_dim = self.num_pairs
        elif self.product_type == "outer":
            product_dim = self.num_pairs
        else:
            product_dim = 2 * self.num_pairs

        self.mlp = MLP(input_dim=linear_dim + product_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.task)

        self.register_regularization_weights(embedding_attr="embedding", include_modules=["mlp", "product_projections"])
        if self.kernel is not None and all(id(self.kernel) != id(param) for param in self.regularization_weights):
            self.regularization_weights.append(self.kernel)

    def compute_inner_products(self, field_emb: torch.Tensor) -> torch.Tensor:
        p = field_emb[:, self.row_idx, :]  # [B, num_pairs, D]
        q = field_emb[:, self.col_idx, :]  # [B, num_pairs, D]
        return torch.sum(p * q, dim=-1)  # [B, num_pairs]

    def compute_outer_kernel_products(self, field_emb: torch.Tensor) -> torch.Tensor:
        if self.kernel is None:
            raise RuntimeError("kernel is not initialized for outer product.")
        p = field_emb[:, self.row_idx, :]  # [B, num_pairs, D]
        q = field_emb[:, self.col_idx, :]  # [B, num_pairs, D]

        if self.outer_kernel_type == "mat":
            # deepctr-compatible "mat" kernel: [D, num_pairs, D]
            # kp[b,p] = q[b,p,:]^T * K[:,p,:] * p[b,p,:]
            weighted_p = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).transpose(1, 2)  # [B, num_pairs, D]
            return torch.sum(weighted_p * q, dim=-1)  # [B, num_pairs]

        # deepctr-compatible "vec"/"num" kernel forms
        return torch.sum(p * q * self.kernel.unsqueeze(0), dim=-1)  # [B, num_pairs]

    def forward(self, x):
        field_vectors = [self.embedding(x=x, features=[feature], squeeze_dim=True) for feature in self.field_features]
        linear_signal = torch.cat(field_vectors, dim=1)
        product_vectors = [projection(vector) for projection, vector in zip(self.product_projections, field_vectors)]
        product_field_emb = torch.stack(product_vectors, dim=1)
        if self.dense_features:
            # dense_signal: [Batch, Dim_dense]
            dense_signal = self.embedding(x=x, features=self.dense_features, squeeze_dim=True)
            # linear_signal = concat(linear_sparse, dense_signal): [Batch, Linear_dim]
            linear_signal = torch.cat([linear_signal, dense_signal], dim=1)

        if self.product_type == "inner":
            # inner_product: [Batch, Pair_num]
            product_signal = self.compute_inner_products(product_field_emb)
        elif self.product_type == "outer":
            # outer_product: [Batch, Pair_num]
            product_signal = self.compute_outer_kernel_products(product_field_emb)
        else:
            # inner_product: [Batch, Pair_num]
            inner_p = self.compute_inner_products(product_field_emb)
            # outer_product: [Batch, Pair_num]
            outer_p = self.compute_outer_kernel_products(product_field_emb)
            # product_signal (both): [Batch, 2 * Pair_num]
            product_signal = torch.cat([inner_p, outer_p], dim=1)

        # deep_input = concat(linear_signal, product_signal): [Batch, Linear_dim + Product_dim]
        deep_input = torch.cat([linear_signal, product_signal], dim=1)
        # logits: [Batch, 1]
        y = self.mlp(deep_input)
        return self.prediction_layer(y)
