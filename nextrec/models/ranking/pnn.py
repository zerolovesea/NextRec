"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response
prediction[C]//ICDM. 2016: 1149-1154. (https://arxiv.org/abs/1611.00144)

Product-based Neural Networks (PNN) are CTR prediction models that explicitly
encode feature interactions by combining:
  (1) A linear signal from concatenated field embeddings
  (2) A product signal capturing pairwise feature interactions (inner or outer)
The product layer augments the linear input to an MLP, enabling the network to
model both first-order and high-order feature interactions in a structured way.

Computation workflow:
  - Embed each categorical/sequence field with a shared embedding dimension
  - Linear signal: flatten and concatenate all field embeddings
  - Product signal:
      * Inner product: dot products over all field pairs
      * Outer product: project embeddings then compute element-wise products
  - Concatenate linear and product signals; feed into MLP for prediction

Key Advantages:
- Explicit pairwise interaction modeling without heavy feature engineering
- Flexible choice between inner/outer products to trade off capacity vs. cost
- Combines linear context with interaction signal for stronger expressiveness
- Simple architecture that integrates cleanly with standard MLP pipelines

PNN 是一种 CTR 预估模型，通过将线性信号与乘积信号结合，显式建模特征交互：
  - 线性信号：将各字段的 embedding 拼接，用于保留一阶信息
  - 乘积信号：对所有字段对做内积或外积，捕捉二阶及更高阶交互
随后将两类信号拼接送入 MLP，实现对用户响应的预测。内积版本计算量更低，
外积版本表达力更强，可根据场景取舍。
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


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
        mlp_params: dict | None = None,
        product_type: str = "inner",  # "inner" (IPNN), "outer" (OPNN), "both" (PNN*)
        outer_product_dim: int | None = None,
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
        mlp_params = mlp_params or {}
        if outer_product_dim is not None and outer_product_dim <= 0:
            raise ValueError("outer_product_dim must be a positive integer.")
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        super(PNN, self).__init__(
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

        self.field_features = dense_features + sparse_features + sequence_features
        if len(self.field_features) < 2:
            raise ValueError("PNN requires at least two sparse/sequence features.")

        self.embedding = EmbeddingLayer(features=self.field_features)
        self.num_fields = len(self.field_features)

        self.embedding_dim = self.field_features[0].embedding_dim
        if any(f.embedding_dim != self.embedding_dim for f in self.field_features):
            raise ValueError(
                "All field features must share the same embedding_dim for PNN."
            )

        self.product_type = product_type.lower()
        if self.product_type not in {"inner", "outer", "both"}:
            raise ValueError("product_type must be 'inner', 'outer', or 'both'.")

        self.num_pairs = self.num_fields * (self.num_fields - 1) // 2
        self.outer_product_dim = outer_product_dim or self.embedding_dim

        if self.product_type in {"outer", "both"}:
            self.kernel = nn.Parameter(
                torch.randn(self.embedding_dim, self.outer_product_dim)
            )
            nn.init.xavier_uniform_(self.kernel)
        else:
            self.kernel = None

        linear_dim = self.num_fields * self.embedding_dim

        if self.product_type == "inner":
            product_dim = self.num_pairs
        elif self.product_type == "outer":
            product_dim = self.num_pairs
        else:
            product_dim = 2 * self.num_pairs

        self.mlp = MLP(input_dim=linear_dim + product_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.task)

        modules = ["mlp"]
        if self.kernel is not None:
            modules.append("kernel")
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=modules
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def compute_inner_products(self, field_emb: torch.Tensor) -> torch.Tensor:
        interactions = []
        for i in range(self.num_fields - 1):
            vi = field_emb[:, i, :]  # [B, D]
            for j in range(i + 1, self.num_fields):
                vj = field_emb[:, j, :]  # [B, D]
                # <v_i, v_j> = sum_k v_i,k * v_j,k
                pij = torch.sum(vi * vj, dim=1, keepdim=True)  # [B, 1]
                interactions.append(pij)
        return torch.cat(interactions, dim=1)  # [B, num_pairs]

    def compute_outer_kernel_products(self, field_emb: torch.Tensor) -> torch.Tensor:
        if self.kernel is None:
            raise RuntimeError("kernel is not initialized for outer product.")

        interactions = []
        for i in range(self.num_fields - 1):
            vi = field_emb[:, i, :]  # [B, D]
            # Project vi with kernel -> [B, K]
            vi_proj = torch.matmul(vi, self.kernel)  # [B, K]
            for j in range(i + 1, self.num_fields):
                vj = field_emb[:, j, :]  # [B, D]
                vj_proj = torch.matmul(vj, self.kernel)  # [B, K]
                # g(vi, vj) = (v_i^T W) * (v_j^T W) summed over projection dim
                pij = torch.sum(vi_proj * vj_proj, dim=1, keepdim=True)  # [B, 1]
                interactions.append(pij)
        return torch.cat(interactions, dim=1)  # [B, num_pairs]

    def forward(self, x):
        # field_emb: [B, F, D]
        field_emb = self.embedding(x=x, features=self.field_features, squeeze_dim=False)
        # Z = [v_1; v_2; ...; v_F]
        linear_signal = field_emb.flatten(start_dim=1)  # [B, F*D]

        if self.product_type == "inner":
            product_signal = self.compute_inner_products(field_emb)
        elif self.product_type == "outer":
            product_signal = self.compute_outer_kernel_products(field_emb)
        else:
            inner_p = self.compute_inner_products(field_emb)
            outer_p = self.compute_outer_kernel_products(field_emb)
            product_signal = torch.cat([inner_p, outer_p], dim=1)

        deep_input = torch.cat([linear_signal, product_signal], dim=1)
        y = self.mlp(deep_input)
        return self.prediction_layer(y)
