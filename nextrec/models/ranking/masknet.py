"""
Date: create on 09/11/2025
Checkpoint: edit on 15/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Wang Z, She Q, Zhang J. MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask.

MaskNet was proposed by the Sina machine learning team and applied in the open-source code of the Twitter team.
The paper proposes a gating mechanism to perform feature selection on the input raw features,
improving feature mining capabilities.

Workflow:
- Build field-wise embeddings from dense/sparse/sequence features selected by `mask_features`.
- Flatten embeddings as mask generator input (`v_emb_flat`).
- Serial architecture:
  first block consumes field embeddings, subsequent blocks consume hidden states.
- Parallel architecture:
  each block consumes the same field embeddings, then concatenate block outputs and feed MLP.
- Apply output layer and task head for final prediction.

Dimension Flow:
- Input: dense[Batch], sparse[Batch], sequence[Batch, Length]
- Embedding: `field_emb`: [Batch, Field_num, Dim_field]
- Flatten for mask generation: `v_emb_flat`: [Batch, Field_num * Dim_field]
- Parallel:
  each block -> [Batch, Dim_block], concat -> [Batch, Num_blocks * Dim_block], final MLP -> `logit`: [Batch, 1]
- Serial:
  first block -> [Batch, Dim_block], hidden blocks -> [Batch, Dim_block], output layer -> `logit`: [Batch, 1]
- Output: task head(`logit`) -> [Batch, Task_total_dim] (binary single-task usually [Batch, 1])

By stacking (SerialMaskNet) or parallelizing (ParallelMaskNet) multiple
MaskBlocks, MaskNet enhances expressive power while remaining efficient,
improving CTR performance without heavy feature engineering.

MaskNet 由新浪机器学习团队提出，并在Twitter团队的开源代码里得到了应用。论文提出了类似门控机制
的方式，对输入的原始特征进行特征选择，以提高特征挖掘能力。

与传统仅依赖 MLP 的加性特征交互不同，MaskNet 会根据每个样本的
embedding 表示生成一个个性化的 “mask” 向量，通过逐元素的乘法
选择性地放大有效特征维度、抑制无关或噪声特征。

流程：
- 由 `mask_features`（dense/sparse/sequence）构建 field 级 embedding。
- 将 embedding 展平为 `v_emb_flat`，作为 Mask 生成器输入。
- 串行结构分支：
  首个 block 输入 field embedding，后续 block 输入 hidden 表示逐层细化。
- 并行结构分支：
  多个 block 并行建模后拼接，再送入 MLP。
- 最后经过输出层和任务头得到预测值。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：`field_emb`：[Batch, Field_num, Dim_field]
- Mask输入展平：`v_emb_flat`：[Batch, Field_num * Dim_field]
- 并行结构分支：
  每个 block 输出 [Batch, Dim_block]，拼接为 [Batch, Num_blocks * Dim_block]，经 MLP 得 `logit`：[Batch, 1]
- 串行结构分支：
  首层 block 输出 [Batch, Dim_block]，后续 hidden block 保持 [Batch, Dim_block]，输出层得 `logit`：[Batch, 1]
- 输出：任务头(`logit`) -> [Batch, Task_total_dim]（单任务二分类通常为 [Batch, 1]）

通过串联（SerialMaskNet）或并联（ParallelMaskNet）
多个 MaskBlock，MaskNet 在保持高效的同时显著增强了特征表达能力，
在无需大量特征工程的情况下提升 CTR 模型性能。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class InstanceGuidedMask(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, v_emb_flat: torch.Tensor) -> torch.Tensor:
        # v_emb_flat: [Batch, features count * embedding_dim]
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
        self.input_dim = num_fields * embedding_dim  # input_dim = features count * embedding_dim
        self.ln_emb = nn.LayerNorm(embedding_dim)
        self.mask_gen = InstanceGuidedMask(
            input_dim=self.input_dim,
            hidden_dim=mask_hidden_dim,
            output_dim=self.input_dim,
        )
        self.ffn = nn.Linear(self.input_dim, hidden_dim)
        self.ln_hid = nn.LayerNorm(hidden_dim)

    # different from MaskBlockOnHidden: input is field embeddings
    def forward(self, field_emb: torch.Tensor, v_emb_flat: torch.Tensor) -> torch.Tensor:
        B = field_emb.size(0)
        norm_emb = self.ln_emb(field_emb)  # [Batch, features count, embedding_dim]
        norm_emb_flat = norm_emb.view(B, -1)  # [Batch, features count * embedding_dim]
        v_mask = self.mask_gen(v_emb_flat)  # [Batch, features count * embedding_dim]
        v_masked_emb = v_mask * norm_emb_flat  # [Batch, features count * embedding_dim]
        hidden = self.ffn(v_masked_emb)  # [Batch, hidden_dim]
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
    def forward(self, hidden_in: torch.Tensor, v_emb_flat: torch.Tensor) -> torch.Tensor:
        norm_hidden = self.ln_input(hidden_in)  # [Batch, hidden_dim]
        v_mask = self.mask_gen(v_emb_flat)
        v_masked_hid = v_mask * norm_hidden
        out = self.ffn(v_masked_hid)
        out = self.ln_output(out)
        out = F.relu(out)
        return out


class MaskNet(BaseRankingModel):
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
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        architecture: Literal["serial", "parallel"] = "parallel",  # "serial" or "parallel"
        num_blocks: int = 3,
        mask_hidden_dim: int = 64,
        block_hidden_dim: int = 256,
        block_dropout: float = 0.0,
        mlp_params: dict | None = None,
        **kwargs,
    ):
        """
        Initialize MaskNet model.
        初始化 MaskNet 模型。

        Args:
            architecture: Mask block topology, supports "serial" and "parallel".
                "serial" 为串行堆叠，"parallel" 为并行多分支。
            num_blocks: Number of mask blocks. In serial mode, total depth is this value.
                在串行模式下表示总 block 数；并行模式下表示并行分支数。
            mask_hidden_dim: Hidden dimension of the two-layer instance-guided mask generator.
                两层 Mask 生成器的隐藏维度。
            block_hidden_dim: Output hidden dimension of each mask block.
                每个 MaskBlock 的输出维度。
            block_dropout: Dropout rate applied after each block output.
                每个 block 输出后的 dropout 比例。
            mlp_params: Parameters for the final MLP in parallel mode,
                e.g. {"hidden_dims": [256, 128], "dropout": 0.2}.
                并行结构末端 MLP 参数；例如 {"hidden_dims": [256, 128], "dropout": 0.2}。
        """
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = dict(mlp_params or {})

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.sequence_features = sequence_features
        self.mask_features = self.all_features  # use all features for masking
        assert len(self.mask_features) > 0, "MaskNet requires at least one feature for masking."
        self.num_fields = len(self.mask_features)
        first_feature = self.mask_features[0]
        if isinstance(first_feature, SequenceFeature):
            if first_feature.combiner == "concat":
                if first_feature.max_len is None:
                    raise ValueError(
                        f"MaskNet requires SequenceFeature('{first_feature.name}') to set max_len when combiner='concat'."
                    )
                self.embedding_dim = first_feature.embedding_dim * first_feature.max_len
            else:
                self.embedding_dim = first_feature.embedding_dim
        elif isinstance(first_feature, SparseFeature):
            self.embedding_dim = first_feature.embedding_dim
        elif isinstance(first_feature, DenseFeature):
            self.embedding_dim = (
                first_feature.embedding_dim if first_feature.use_projection else first_feature.input_dim
            )
        else:
            raise TypeError(f"Unsupported feature type for MaskNet: {type(first_feature)}")

        for f in self.mask_features:
            if isinstance(f, SequenceFeature):
                if f.combiner == "concat":
                    if f.max_len is None:
                        raise ValueError(
                            f"MaskNet requires SequenceFeature('{f.name}') to set max_len when combiner='concat'."
                        )
                    edim = f.embedding_dim * f.max_len
                else:
                    edim = f.embedding_dim
            elif isinstance(f, SparseFeature):
                edim = f.embedding_dim
            elif isinstance(f, DenseFeature):
                edim = f.embedding_dim if f.use_projection else f.input_dim
            else:
                raise TypeError(f"Unsupported feature type for MaskNet: {type(f)}")
            if edim != self.embedding_dim:
                feat_name = f.name
                raise ValueError(
                    "MaskNet expects identical effective field dimensions across all mask_features, "
                    f"but got {edim} for feature {feat_name} (expected {self.embedding_dim}). "
                    "For SequenceFeature(combiner='concat'), effective dim is embedding_dim * max_len."
                )

        self.embedding = EmbeddingLayer(features=self.mask_features)
        self.v_emb_dim = self.num_fields * self.embedding_dim
        self.architecture = architecture.lower()
        assert self.architecture in (
            "serial",
            "parallel",
        ), "architecture must be either 'serial' or 'parallel'."

        self.num_blocks = max(1, num_blocks)
        self.block_hidden_dim = block_hidden_dim
        self.block_dropout = nn.Dropout(block_dropout) if block_dropout > 0 else nn.Identity()

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
            mlp_output_dim = mlp_params.get("output_dim", 1)
            if mlp_output_dim != 1:
                raise ValueError(
                    f"MaskNet(parallel) expects mlp_params['output_dim']=1 to match TaskHead input, but got {mlp_output_dim}."
                )
            mlp_params["output_dim"] = 1
            self.final_mlp = MLP(input_dim=self.num_blocks * block_hidden_dim, **mlp_params)
            self.output_layer = None

        if self.architecture == "serial":
            self.register_regularization_weights(
                embedding_attr="embedding",
                include_modules=["first_block", "hidden_blocks", "output_layer"],
            )
        else:
            self.register_regularization_weights(
                embedding_attr="embedding", include_modules=["mask_blocks", "final_mlp"]
            )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        field_emb = self.embedding(x=x, features=self.mask_features, squeeze_dim=False)
        # field_emb: [Batch, Field_num, Dim_field]
        batch_size = field_emb.size(0)
        v_emb_flat = field_emb.view(batch_size, -1)
        # v_emb_flat: [Batch, Field_num * Dim_field]

        if self.architecture == "parallel":
            block_outputs = []
            for block in self.mask_blocks:
                h = block(field_emb, v_emb_flat)
                # h: [Batch, Dim_block]
                h = self.block_dropout(h)
                block_outputs.append(h)
            concat_hidden = torch.cat(block_outputs, dim=-1)  # [Batch, Num_blocks * Dim_block]
            logit = self.final_mlp(concat_hidden)  # [Batch, 1]
        else:
            hidden = self.first_block(field_emb, v_emb_flat)  # [Batch, Dim_block]
            hidden = self.block_dropout(hidden)
            for block in self.hidden_blocks:
                hidden = block(hidden, v_emb_flat)  # [Batch, Dim_block]
                hidden = self.block_dropout(hidden)
            logit = self.output_layer(hidden)  # [Batch, 1]
        return logit  # [Batch, Task_total_dim]
