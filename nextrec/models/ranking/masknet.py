"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Wang Z, She Q, Zhang J. MaskNet: Introducing Feature-Wise
Multiplication to CTR Ranking Models by Instance-Guided Mask.

MaskNet is a CTR prediction model that introduces instance-guided,
feature-wise multiplicative interactions into deep ranking networks.
Instead of relying solely on additive feature interactions from MLPs,
MaskNet generates a personalized “mask” vector for each instance based
on its embedding representation. This mask selectively scales hidden
features through element-wise multiplication, enabling the network to
emphasize informative dimensions and suppress irrelevant noise.

Each MaskBlock consists of:
  (1) Instance-Guided Mask Generation (two-layer MLP)
  (2) Feature-wise Multiplication with hidden representations
  (3) Layer Normalization and nonlinear transformation

By stacking (SerialMaskNet) or parallelizing (ParallelMaskNet) multiple
MaskBlocks, MaskNet enhances expressive power while remaining efficient,
improving CTR performance without heavy feature engineering.

Key Advantages:
- Learns higher-order interactions via multiplicative gating
- Instance-adaptive feature importance modulation
- Better discrimination of informative vs. noisy dimensions
- Flexible architecture for both sequential and parallel design

MaskNet 是一种用于 CTR 预估的模型，它在深度排序网络中引入了
基于实例（Instance-Guided）的特征级逐元素（Feature-wise）
乘法交互机制。

与传统仅依赖 MLP 的加性特征交互不同，MaskNet 会根据每个样本的
embedding 表示生成一个个性化的 “mask” 向量，通过逐元素的乘法
选择性地放大有效特征维度、抑制无关或噪声特征。

每个 MaskBlock 包含以下关键步骤：
  (1) 基于当前样本 embedding 的双层 MLP Mask 生成
  (2) Mask 与隐藏表示之间的逐元素乘法交互
  (3) Layer Normalization 与非线性变换

通过串联（SerialMaskNet）或并联（ParallelMaskNet）
多个 MaskBlock，MaskNet 在保持高效的同时显著增强了特征表达能力，
在无需大量特征工程的情况下提升 CTR 模型性能。

核心优势：
- 通过乘法门控学习高阶特征交互关系
- 针对每个样本自适应调整特征重要性
- 有效区分信息特征与噪声特征
- 支持灵活的串行与并行网络结构设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class InstanceGuidedMask(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, v_emb_flat: torch.Tensor) -> torch.Tensor:
        # v_emb_flat: [batch, features count * embedding_dim]
        x = self.fc1(v_emb_flat)
        x = F.relu(x)
        v_mask = self.fc2(x)
        return v_mask


class MaskBlockOnEmbedding(nn.Module):
    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        mask_hidden_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.input_dim = (
            num_fields * embedding_dim
        )  # input_dim = features count * embedding_dim
        self.ln_emb = nn.LayerNorm(embedding_dim)
        self.mask_gen = InstanceGuidedMask(
            input_dim=self.input_dim,
            hidden_dim=mask_hidden_dim,
            output_dim=self.input_dim,
        )
        self.ffn = nn.Linear(self.input_dim, hidden_dim)
        self.ln_hid = nn.LayerNorm(hidden_dim)

    # different from MaskBlockOnHidden: input is field embeddings
    def forward(
        self, field_emb: torch.Tensor, v_emb_flat: torch.Tensor
    ) -> torch.Tensor:
        B = field_emb.size(0)
        norm_emb = self.ln_emb(field_emb)  # [B, features count, embedding_dim]
        norm_emb_flat = norm_emb.view(B, -1)  # [B, features count * embedding_dim]
        v_mask = self.mask_gen(v_emb_flat)  # [B, features count * embedding_dim]
        v_masked_emb = v_mask * norm_emb_flat  # [B, features count * embedding_dim]
        hidden = self.ffn(v_masked_emb)  # [B, hidden_dim]
        hidden = self.ln_hid(hidden)
        hidden = F.relu(hidden)

        return hidden


class MaskBlockOnHidden(nn.Module):
    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        mask_hidden_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.v_emb_dim = num_fields * embedding_dim

        self.ln_input = nn.LayerNorm(hidden_dim)
        self.ln_output = nn.LayerNorm(hidden_dim)

        self.mask_gen = InstanceGuidedMask(
            input_dim=self.v_emb_dim,
            hidden_dim=mask_hidden_dim,
            output_dim=hidden_dim,
        )
        self.ffn = nn.Linear(hidden_dim, hidden_dim)

    # different from MaskBlockOnEmbedding: input is hidden representation
    def forward(
        self, hidden_in: torch.Tensor, v_emb_flat: torch.Tensor
    ) -> torch.Tensor:
        norm_hidden = self.ln_input(hidden_in)
        v_mask = self.mask_gen(v_emb_flat)
        v_masked_hid = v_mask * norm_hidden
        out = self.ffn(v_masked_hid)
        out = self.ln_output(out)
        out = F.relu(out)
        return out


class MaskNet(BaseModel):
    @property
    def model_name(self):
        return "MaskNet"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        architecture: str = "parallel",  # "serial" or "parallel"
        num_blocks: int = 3,
        mask_hidden_dim: int = 64,
        block_hidden_dim: int = 256,
        block_dropout: float = 0.0,
        mlp_params: dict | None = None,
        target: list[str] | str | None = None,
        task: str | list[str] | None = None,
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss: str | nn.Module | None = "bce",
        loss_params: dict | list[dict] | None = None,
        device: str = "cpu",
        embedding_l1_reg: float = 1e-6,
        dense_l1_reg: float = 1e-5,
        embedding_l2_reg: float = 1e-5,
        dense_l2_reg: float = 1e-4,
        **kwargs,
    ):
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        target = target or []
        mlp_params = mlp_params or {}
        optimizer_params = optimizer_params or {}

        super().__init__(
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

        if loss is None:
            loss = "bce"
        self.loss = loss

        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.sequence_features = sequence_features
        self.mask_features = self.all_features  # use all features for masking
        assert (
            len(self.mask_features) > 0
        ), "MaskNet requires at least one feature for masking."
        self.embedding = EmbeddingLayer(features=self.mask_features)
        self.num_fields = len(self.mask_features)
        self.embedding_dim = getattr(self.mask_features[0], "embedding_dim", None)
        assert (
            self.embedding_dim is not None
        ), "MaskNet requires mask_features to have 'embedding_dim' defined."

        for f in self.mask_features:
            edim = getattr(f, "embedding_dim", None)
            if edim is None or edim != self.embedding_dim:
                raise ValueError(
                    f"MaskNet expects identical embedding_dim across all mask_features, but got {edim} for feature {getattr(f, 'name', type(f))}."
                )

        self.v_emb_dim = self.num_fields * self.embedding_dim
        self.architecture = architecture.lower()
        assert self.architecture in (
            "serial",
            "parallel",
        ), "architecture must be either 'serial' or 'parallel'."

        self.num_blocks = max(1, num_blocks)
        self.block_hidden_dim = block_hidden_dim
        self.block_dropout = (
            nn.Dropout(block_dropout) if block_dropout > 0 else nn.Identity()
        )

        if self.architecture == "serial":
            self.first_block = MaskBlockOnEmbedding(
                num_fields=self.num_fields,
                embedding_dim=self.embedding_dim,
                mask_hidden_dim=mask_hidden_dim,
                hidden_dim=block_hidden_dim,
            )
            self.hidden_blocks = nn.ModuleList(
                [
                    MaskBlockOnHidden(
                        num_fields=self.num_fields,
                        embedding_dim=self.embedding_dim,
                        mask_hidden_dim=mask_hidden_dim,
                        hidden_dim=block_hidden_dim,
                    )
                    for _ in range(self.num_blocks - 1)
                ]
            )
            self.mask_blocks = nn.ModuleList([self.first_block, *self.hidden_blocks])
            self.output_layer = nn.Linear(block_hidden_dim, 1)
            self.final_mlp = None

        else:  # parallel
            self.mask_blocks = nn.ModuleList(
                [
                    MaskBlockOnEmbedding(
                        num_fields=self.num_fields,
                        embedding_dim=self.embedding_dim,
                        mask_hidden_dim=mask_hidden_dim,
                        hidden_dim=block_hidden_dim,
                    )
                    for _ in range(self.num_blocks)
                ]
            )
            self.final_mlp = MLP(
                input_dim=self.num_blocks * block_hidden_dim, **mlp_params
            )
            self.output_layer = None
        self.prediction_layer = TaskHead(task_type=self.task)

        if self.architecture == "serial":
            self.register_regularization_weights(
                embedding_attr="embedding",
                include_modules=["mask_blocks", "output_layer"],
            )
        else:
            self.register_regularization_weights(
                embedding_attr="embedding", include_modules=["mask_blocks", "final_mlp"]
            )
        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        field_emb = self.embedding(x=x, features=self.mask_features, squeeze_dim=False)
        B = field_emb.size(0)
        v_emb_flat = field_emb.view(B, -1)  # flattened embeddings

        if self.architecture == "parallel":
            block_outputs = []
            for block in self.mask_blocks:
                h = block(field_emb, v_emb_flat)  # [B, block_hidden_dim]
                h = self.block_dropout(h)
                block_outputs.append(h)
            concat_hidden = torch.cat(block_outputs, dim=-1)
            logit = self.final_mlp(concat_hidden)  # [B, 1]
        else:
            hidden = self.first_block(field_emb, v_emb_flat)
            hidden = self.block_dropout(hidden)
            for block in self.hidden_blocks:
                hidden = block(hidden, v_emb_flat)
                hidden = self.block_dropout(hidden)
            logit = self.output_layer(hidden)  # [B, 1]
        y = self.prediction_layer(logit)
        return y
