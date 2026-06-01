"""
Date: create on 01/12/2025
Checkpoint: edit on 07/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Meta AI. Generative Recommenders (HSTU encoder) — https://github.com/meta-recsys/generative-recommenders
- [2] Ma W, Li P, Chen C, et al. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. arXiv:2402.17152.

Hierarchical Sequential Transduction Unit (HSTU) is the core encoder behind
Meta’s Generative Recommenders. It replaces softmax attention with lightweight
pointwise activations, enabling extremely deep stacks on long behavior sequences.

In each HSTU layer:
  (1) Tokens are projected into four streams U, V, Q, K via a shared feed-forward block
  (2) Softmax-free interactions combine QK^T with Relative Attention Bias (RAB) to encode distance
  (3) Aggregated context is modulated by U-gating and mapped back through an output projection

Stacking layers yields an efficient causal encoder for generative
matching. With a tied-embedding LM head, HSTU forms
a full generative recommendation model.

Key Advantages:
- Softmax-free attention scales better on deep/long sequences
- RAB captures temporal structure without extra attention heads
- Causal masking and padding-aware normalization fit real logs
- Weight tying reduces parameters and stabilizes training
- Serves as a drop-in backbone for generative recommendation

HSTU（层次化序列转导单元）是 Meta 生成式推荐的核心编码器，
用点式激活替代 softmax 注意力，可在长序列上轻松堆叠深层结构。

单层 HSTU 的主要步骤：
  (1) 将输入一次性映射到 U、V、Q、K 四条通路
  (2) 利用不含 softmax 的 QK^T 结合相对位置偏置（RAB）建模距离信息
  (3) 用 U 对聚合上下文进行门控，再映射回输出空间

多层堆叠后，可得到高效的因果编码器；与绑权 LM 头配合即可完成生成式召回目标建模。

主要优势：
- 摆脱 softmax，在长序列、深层模型上更易扩展
- 相对位置偏置稳健刻画时序结构
- 因果 mask 与 padding 感知归一化贴合真实日志
- 绑权输出头降低参数量并提升训练稳定性
- 直接作为生成式推荐的骨干网络
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import RMSNorm
from nextrec.models.sequential.base import BaseSequentialModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import SequenceModeName


def relative_position_bucket(
    relative_position: torch.Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """
    Map relative position (i-j) to a bucket in [0, num_buckets).
    Only the causal direction is needed for autoregressive attention.
    """
    n = torch.clamp(-relative_position, min=0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    large_val = (
        max_exact
        + (
            (torch.log(n.float() / max_exact + 1e-6) / math.log(max_distance / max_exact)) * (num_buckets - max_exact)
        ).long()
    )
    large_val = torch.clamp(large_val, max=num_buckets - 1)
    return torch.where(is_small, n.long(), large_val)


class RelativePositionBias(nn.Module):
    """
    Compute relative position bias (RAB) for HSTU attention.
    Output shape: [num_heads, seq_len, seq_len].
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.embedding = self.relative_attention_bias

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        ctx = torch.arange(seq_len, device=device)[:, None]
        mem = torch.arange(seq_len, device=device)[None, :]
        rel_pos = mem - ctx
        buckets = relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        return self.embedding(buckets).permute(2, 0, 1)


class TemporalBias(nn.Module):
    """
    Temporal attention bias using logarithmic buckets of pairwise timestamp differences.
    Output shape: [B, num_heads, seq_len, seq_len].
    """

    def __init__(self, num_buckets: int = 64, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.temporal_attention_bias = nn.Embedding(num_buckets, num_heads)

    def temporal_bucket(self, time_diff: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.clamp(torch.abs(time_diff), min=1).float()
        buckets = (torch.log(abs_diff) / 0.693).long()
        return torch.clamp(buckets, min=0, max=self.num_buckets - 1)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        buckets = self.temporal_bucket(timestamps.unsqueeze(2) - timestamps.unsqueeze(1))
        return self.temporal_attention_bias(buckets).permute(0, 3, 1, 2)


class HSTUPointwiseAttention(nn.Module):
    """
    Pointwise aggregation attention without softmax:
        [U, V, Q, K] = split(SiLU(f1(X)))
        AV = SiLU(QK^T + rab) V
        Y = Norm(AV) * U
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        alpha: float | None = None,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"[HSTUPointwiseAttention Error] hidden_dim({hidden_dim}) % num_heads({num_heads}) != 0")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.alpha = alpha
        self.in_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = RMSNorm(hidden_dim) if use_rms_norm else nn.LayerNorm(hidden_dim)

    def reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, H, T, head_dim]."""
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # [T, T] with 0 or -inf
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T], True = pad
        rab: Optional[torch.Tensor] = None,  # [H, T, T], [1, H, T, T], or [B, H, T, T]
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Eq.(1): one projection followed by SiLU, then split into the four HSTU streams.
        U, V, Q, K = F.silu(self.in_proj(x)).chunk(4, dim=-1)
        Qh, Kh, Vh, Uh = (self.reshape_heads(tensor) for tensor in (Q, K, V, U))

        logits = torch.matmul(Qh, Kh.transpose(-2, -1))
        if self.alpha is not None:
            logits = logits * self.alpha

        if rab is not None:
            if rab.dim() == 3:
                logits = logits + rab.unsqueeze(0)
            else:
                logits = logits + rab

        if attention_mask is not None:
            logits = logits + attention_mask.view(1, 1, T, T)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.view(B, 1, 1, T), float("-inf"))

        # F.silu(-inf) can produce NaN on some kernels, so replace masked -inf before activation.
        logits_safe = logits.masked_fill(torch.isinf(logits) & (logits < 0), -1e9)
        AV = torch.matmul(F.silu(logits_safe), Vh)
        AV = AV.transpose(1, 2).contiguous().view(B, T, D)
        U_flat = Uh.transpose(1, 2).contiguous().view(B, T, D)
        return self.attn_norm(AV) * U_flat


class HSTULayer(nn.Module):
    """
    HSTUPointwiseAttention with residual connection and a compact point-wise FFN.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rab_pos: bool = True,
        rab_num_buckets: int = 32,
        rab_max_distance: int = 128,
        use_rms_norm: bool = False,
        ff_hidden_dim: int | None = None,
        use_temporal_bias: bool = False,
        temporal_num_buckets: int = 64,
    ):
        super().__init__()
        self.attention = HSTUPointwiseAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rms_norm=use_rms_norm,
        )
        self.dropout = nn.Dropout(dropout)
        self.use_rab_pos = use_rab_pos
        self.use_temporal_bias = use_temporal_bias
        self.rel_pos_bias = (
            RelativePositionBias(
                num_heads=num_heads,
                num_buckets=rab_num_buckets,
                max_distance=rab_max_distance,
            )
            if use_rab_pos
            else None
        )
        self.temporal_bias = (
            TemporalBias(num_buckets=temporal_num_buckets, num_heads=num_heads) if use_temporal_bias else None
        )
        self.ffn_norm = RMSNorm(hidden_dim) if use_rms_norm else nn.LayerNorm(hidden_dim)
        ff_hidden_dim = ff_hidden_dim or (hidden_dim * 4)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rab = None
        if self.use_rab_pos:
            rab = self.rel_pos_bias(seq_len=x.size(1), device=x.device)
        if self.use_temporal_bias and timestamps is not None:
            time_bias = self.temporal_bias(timestamps)
            rab = time_bias if rab is None else rab + time_bias
        out = self.attention(
            x=x,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            rab=rab,
        )
        x = x + self.dropout(out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HSTU(BaseSequentialModel):
    """
    HSTU encoder for next-item prediction in a causal autoregressive setup.
    It returns full-vocabulary logits for every sequence position.
    """

    @property
    def model_name(self) -> str:
        return "HSTU"

    @property
    def default_task(self) -> str:
        return "generative"

    def __init__(
        self,
        sequence_features: list[SequenceFeature] | None = None,
        dense_features: Optional[list[DenseFeature]] = None,
        sparse_features: Optional[list[SparseFeature]] = None,
        item_history_name: str = "item_history",
        num_items: int | None = None,
        embed_dim: int | None = None,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 4,
        num_blocks: int | None = None,
        ff_hidden_dim: Optional[int] = None,
        max_seq_len: int = 200,
        dropout_rate: float = 0.1,
        dropout: float | None = None,
        # RAB settings
        use_rab_pos: bool = True,
        num_position_buckets: int = 32,
        max_position_distance: int = 128,
        use_temporal_bias: bool = False,
        num_time_buckets: int = 64,
        # Normalization settings
        use_rms_norm: bool = False,
        tie_embeddings: bool = True,
        sequence_mode: SequenceModeName = "autoregressive",
        target: Optional[list[str] | str] = None,
        task: str | list[str] | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        **kwargs,
    ):
        if not sequence_features:
            if num_items is None:
                raise ValueError("[HSTU Error] HSTU requires sequence_features or num_items.")
            hidden_for_feature = int(embed_dim or hidden_dim or 64)
            sequence_features = [
                SequenceFeature(
                    name=item_history_name,
                    vocab_size=int(num_items) + 1,
                    max_len=max_seq_len,
                    embedding_dim=hidden_for_feature,
                    padding_idx=0,
                )
            ]

        if num_blocks is not None:
            num_layers = num_blocks
        if dropout is not None:
            dropout_rate = dropout

        self.item_history_feature = select_feature_objects(sequence_features, [item_history_name], "item_history")[0]

        self.hidden_dim = int(embed_dim or hidden_dim or max(int(self.item_history_feature.embedding_dim or 0), 32))
        self.ff_hidden_dim = ff_hidden_dim or (self.hidden_dim * 4)
        # Make hidden_dim divisible by num_heads
        if self.hidden_dim % num_heads != 0:
            self.hidden_dim = num_heads * math.ceil(self.hidden_dim / num_heads)

        self.padding_idx = (
            self.item_history_feature.padding_idx if self.item_history_feature.padding_idx is not None else 0
        )
        self.num_items = int(num_items or (int(self.item_history_feature.vocab_size) - 1))
        self.vocab_size = self.num_items + 1
        self.max_seq_len = max_seq_len

        if sequence_mode != "autoregressive":
            raise ValueError("[HSTU Error] HSTU currently only supports sequence_mode='autoregressive'.")

        self.disable_default_head = True
        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task or self.default_task,
            sequence_mode=sequence_mode,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        self.item_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_dim,
            padding_idx=self.padding_idx,
        )
        self.emb_dropout = nn.Dropout(dropout_rate)

        # HSTU layers
        self.layers = nn.ModuleList(
            [
                HSTULayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    use_rab_pos=use_rab_pos,
                    rab_num_buckets=num_position_buckets,
                    rab_max_distance=max_position_distance,
                    use_rms_norm=use_rms_norm,
                    ff_hidden_dim=self.ff_hidden_dim,
                    use_temporal_bias=use_temporal_bias,
                    temporal_num_buckets=num_time_buckets,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(self.hidden_dim) if use_rms_norm else nn.LayerNorm(self.hidden_dim)

        self.register_regularization_weights(
            embedding_attr="item_embedding",
            include_modules=["layers"],
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_sequence(
        self,
        seq: torch.Tensor,
        padding_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, T = seq.shape
        device = seq.device
        hidden_states = self.emb_dropout(self.item_embedding(seq))

        if attention_mask is not None and attention_mask.dtype == torch.bool:
            additive_mask = torch.zeros((T, T), device=device, dtype=hidden_states.dtype)
            additive_mask = additive_mask.masked_fill(attention_mask, float("-inf"))
            attention_mask = additive_mask

        for layer in self.layers:
            hidden_states = layer(
                x=hidden_states,
                attention_mask=attention_mask,
                key_padding_mask=padding_mask,
                timestamps=timestamps,
            )
        hidden_states = self.final_norm(hidden_states)
        return hidden_states * valid_mask

    def forward(
        self,
        x: dict[str, torch.Tensor] | torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        called_with_dict = isinstance(x, dict)
        if called_with_dict:
            input_dict = x
            seq, padding_mask, valid_mask, attention_mask = self.prepare_sequence_batch(
                x=input_dict,
                sequence_name=self.item_history_feature.name,
                max_seq_len=self.max_seq_len,
                padding_idx=self.padding_idx,
            )
            timestamps = input_dict.get("timestamps", timestamps)
        else:
            seq = x.long()[:, -self.max_seq_len :]
            padding_mask = seq.eq(self.padding_idx)
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            attention_mask = torch.triu(
                torch.ones((seq.size(1), seq.size(1)), dtype=torch.bool, device=seq.device),
                diagonal=1,
            )
            if timestamps is not None:
                timestamps = timestamps[:, -seq.size(1) :]
        hidden_states = self._encode_sequence(
            seq=seq,
            padding_mask=padding_mask,
            valid_mask=valid_mask,
            attention_mask=attention_mask,
            timestamps=timestamps,
        )
        logits = hidden_states @ self.item_embedding.weight.T  # [B, T, num_items + 1]

        if called_with_dict:
            return logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.long().reshape(-1),
                ignore_index=self.padding_idx,
            )
        return logits, loss

    def encode(self, input_ids: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq = input_ids.long()[:, -self.max_seq_len :]
        padding_mask = seq.eq(self.padding_idx)
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        attention_mask = torch.triu(
            torch.ones((seq.size(1), seq.size(1)), dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        if timestamps is not None:
            timestamps = timestamps[:, -seq.size(1) :]
        return self._encode_sequence(
            seq=seq,
            padding_mask=padding_mask,
            valid_mask=valid_mask,
            attention_mask=attention_mask,
            timestamps=timestamps,
        )
