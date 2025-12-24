"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Zhao Z, Zhang H, Tang H, et al. EulerNet: Efficient and Effective Feature
Interaction Modeling with Euler's Formula. (SIGIR 2021)

EulerNet models feature interactions in the complex domain using Euler's
formula. Each field embedding is transformed into amplitude and phase,
then mapped to a complex vector. Feature interactions are captured by
multiplying complex vectors across fields, which corresponds to multiplying
amplitudes and summing phases. The resulting complex representation is
converted back to real-valued features for a linear readout, optionally
paired with a linear term for first-order signals.

Pipeline:
  (1) Embed sparse/sequence features with a shared embedding dimension
  (2) Map embeddings to complex vectors via amplitude/phase transforms
  (3) Multiply complex vectors across fields (Euler interaction)
  (4) Concatenate real & imaginary parts and apply a linear regression head
  (5) Optionally add a linear term and apply the prediction layer

Key Advantages:
- Efficient higher-order interaction modeling via complex multiplication
- Compact representation without explicit cross-feature enumeration
- Works well on sparse high-dimensional feature spaces

EulerNet 使用欧拉公式将特征嵌入映射到复数域，通过复数相乘实现高效的
特征交互建模，再将复数表示转回实数向量做线性回归，并可选叠加线性项
以保留一阶信号。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import LR, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class EulerInteractionLayer(nn.Module):
    """
    Paper-aligned Euler Interaction Layer.

    Input:  r, p  (rectangular form) as tensors with shape [B, m, d]
            where each field j is complex feature: r_j + i p_j.

    Output: r_out, p_out as tensors with shape [B, n, d]
            representing {o_k}_{k=1..n} (Eq.15) which can be stacked.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_fields: int,
        num_orders: int,
        use_implicit: bool = True,
        norm: str | None = "ln",  # None | "bn" | "ln"
        eps: float = 1e-9,
    ):
        super().__init__()
        self.d = embedding_dim
        self.m = num_fields
        self.n = num_orders
        self.use_implicit = use_implicit
        self.eps = eps

        # Explicit part parameters
        # alpha_{k,j} : shape [n, m, d] (vector-wise coefficients)
        self.alpha = nn.Parameter(torch.empty(self.n, self.m, self.d))
        # delta_k, delta'_k : shape [n, d]
        self.delta_phase = nn.Parameter(torch.zeros(self.n, self.d))
        self.delta_logmod = nn.Parameter(torch.zeros(self.n, self.d))
        nn.init.xavier_uniform_(self.alpha)

        # Implicit part parameters
        if self.use_implicit:
            # W_k in R^{d x (m*d)} and bias b_k in R^d
            self.W_r = nn.Parameter(torch.empty(self.n, self.d, self.m * self.d))
            self.b_r = nn.Parameter(torch.zeros(self.n, self.d))
            self.W_p = nn.Parameter(torch.empty(self.n, self.d, self.m * self.d))
            self.b_p = nn.Parameter(torch.zeros(self.n, self.d))
            nn.init.xavier_uniform_(self.W_r)
            nn.init.xavier_uniform_(self.W_p)
        else:
            self.W, self.b = None, None

        # Normalization
        # Apply on concatenated [r_k, p_k] per k.
        self.norm = norm
        if norm == "bn":
            self.bn = nn.BatchNorm1d(self.n * self.d * 2)
            self.ln = None
        elif norm == "ln":
            self.ln = nn.LayerNorm(self.d * 2)
            self.bn = None
        else:
            self.bn = None
            self.ln = None

    def forward(
        self, r: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        r, p: [B, m, d]
        return r_out, p_out: [B, n, d]
        """
        B, m, d = r.shape
        assert (
            m == self.m and d == self.d
        ), f"Expected [B,{self.m},{self.d}] got {r.shape}"

        # Euler Transformation: rectangular -> polar
        lam = torch.sqrt(r * r + p * p + self.eps)  # [B,m,d]
        theta = torch.atan2(p, r)  # [B,m,d]
        log_lam = torch.log(lam + self.eps)  # [B,m,d]

        # Generalized Multi-order Transformation
        # psi_k = sum_j alpha_{k,j} * theta_j + delta_k
        # l_k   = exp(sum_j alpha_{k,j} * log(lam_j) + delta'_k)
        psi = (
            torch.einsum("bmd,nmd->bnd", theta, self.alpha) + self.delta_phase
        )  # [B,n,d]
        log_l = (
            torch.einsum("bmd,nmd->bnd", log_lam, self.alpha) + self.delta_logmod
        )  # [B,n,d]
        l = torch.exp(log_l)  # [B,n,d]

        # Inverse Euler Transformation
        r_hat = l * torch.cos(psi)  # [B,n,d]
        p_hat = l * torch.sin(psi)  # [B,n,d]

        # Implicit interactions + fusion
        if self.use_implicit:
            r_cat = r.reshape(B, self.m * self.d)  # [B, m*d]
            p_cat = p.reshape(B, self.m * self.d)  # [B, m*d]
            # For each k: W_k @ r_cat + b_k -> [B,d]
            r_imp = torch.einsum("bq,ndq->bnd", r_cat, self.W_r) + self.b_r
            p_imp = torch.einsum("bq,ndq->bnd", p_cat, self.W_p) + self.b_p
            r_imp = F.relu(r_imp)
            p_imp = F.relu(p_imp)
            r_out = r_hat + r_imp
            p_out = p_hat + p_imp
        else:
            r_out, p_out = r_hat, p_hat

        # Optional normalization (paper says BN/LN can be used between layers)
        if self.bn is not None:
            x = torch.cat([r_out, p_out], dim=-1).reshape(B, self.n * self.d * 2)
            x = self.bn(x).reshape(B, self.n, self.d * 2)
            r_out, p_out = x[..., : self.d], x[..., self.d :]
        elif self.ln is not None:
            x = torch.cat([r_out, p_out], dim=-1)  # [B,n,2d]
            x = self.ln(x)
            r_out, p_out = x[..., : self.d], x[..., self.d :]

        return r_out, p_out


class ComplexSpaceMapping(nn.Module):
    """
    Map real embeddings e_j to complex features via Euler's formula (Eq.6-7).
    For each field j:
        r_j = mu_j * cos(e_j)
        p_j = mu_j * sin(e_j)
    mu_j is field-specific learnable vector (positive via exp).
    """

    def __init__(self, embedding_dim: int, num_fields: int):
        super().__init__()
        self.d = embedding_dim
        self.m = num_fields
        self.log_mu = nn.Parameter(torch.zeros(self.m, self.d))  # mu = exp(log_mu)

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # e: [B, m, d]
        mu = torch.exp(self.log_mu).unsqueeze(0)  # [1,m,d]
        r = mu * torch.cos(e)
        p = mu * torch.sin(e)
        return r, p


class EulerNet(BaseModel):
    @property
    def model_name(self):
        return "EulerNet"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        num_layers: int = 2,
        num_orders: int = 8,
        use_implicit: bool = True,
        norm: str | None = "ln",
        use_linear: bool = False,
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

        super(EulerNet, self).__init__(
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
        self.use_linear = use_linear

        self.linear_features = dense_features + sparse_features + sequence_features
        self.interaction_features = (
            [f for f in dense_features if getattr(f, "use_embedding", False)]
            + sparse_features
            + sequence_features
        )

        if len(self.interaction_features) < 2:
            raise ValueError(
                "EulerNet requires at least two embedded features for interactions."
            )

        self.embedding = EmbeddingLayer(features=self.all_features)

        self.num_fields = len(self.interaction_features)
        self.embedding_dim = self.interaction_features[0].embedding_dim
        if any(
            f.embedding_dim != self.embedding_dim for f in self.interaction_features
        ):
            raise ValueError(
                "All interaction features must share the same embedding_dim in EulerNet."
            )

        self.num_layers = num_layers
        self.num_orders = num_orders
        self.mapping = ComplexSpaceMapping(self.embedding_dim, self.num_fields)
        self.layers = nn.ModuleList(
            [
                EulerInteractionLayer(
                    embedding_dim=self.embedding_dim,
                    num_fields=(self.num_fields if i == 0 else self.num_orders),
                    num_orders=self.num_orders,
                    use_implicit=use_implicit,
                    norm=norm,
                )
                for i in range(self.num_layers)
            ]
        )
        self.w = nn.Linear(self.num_orders * self.embedding_dim, 1, bias=False)
        self.w_im = nn.Linear(self.num_orders * self.embedding_dim, 1, bias=False)

        if self.use_linear:
            if len(self.linear_features) == 0:
                raise ValueError(
                    "EulerNet linear term requires at least one input feature."
                )
            linear_dim = self.embedding.get_input_dim(self.linear_features)
            if linear_dim <= 0:
                raise ValueError("EulerNet linear input_dim must be positive.")
            self.linear = LR(linear_dim)
        else:
            self.linear = None

        self.prediction_layer = TaskHead(task_type=self.task)

        modules = ["mapping", "layers", "w", "w_im"]
        if self.use_linear:
            modules.append("linear")
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=modules
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        field_emb = self.embedding(
            x=x, features=self.interaction_features, squeeze_dim=False
        )
        y_euler = self.euler_forward(field_emb)

        if self.use_linear and self.linear is not None:
            linear_input = self.embedding(
                x=x, features=self.linear_features, squeeze_dim=True
            )
            y_euler = y_euler + self.linear(linear_input)

        return self.prediction_layer(y_euler)

    def euler_forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        r, p = self.mapping(field_emb)
        for layer in self.layers:
            r, p = layer(r, p)
        r_flat = r.reshape(r.size(0), self.num_orders * self.embedding_dim)
        p_flat = p.reshape(p.size(0), self.num_orders * self.embedding_dim)
        return self.w(r_flat) + self.w_im(p_flat)
