"""
[Info: this version is not released yet, i need to more research on source code and paper]
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
    map the relative position (i-j) to a bucket in [0, num_buckets).
    """
    # only need the negative part for causal attention
    n = -relative_position
    n = torch.clamp(n, min=0)

    # when the distance is small, keep it exact
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # when the distance is too far, do log scaling
    large_val = (
        max_exact
        + (
            (torch.log(n.float() / max_exact + 1e-6) / math.log(max_distance / max_exact)) * (num_buckets - max_exact)
        ).long()
    )
    large_val = torch.clamp(large_val, max=num_buckets - 1)

    buckets = torch.where(is_small, n.long(), large_val)
    return buckets


class RelativePositionBias(nn.Module):
    """
    Compute relative position bias (RAB) for HSTU attention.
    The input is the sequence length T, output is [1, num_heads, seq_len, seq_len].
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # positions: [T]
        ctx = torch.arange(seq_len, device=device)[:, None]
        mem = torch.arange(seq_len, device=device)[None, :]
        rel_pos = (
            mem - ctx
        )  # a matrix to describe all relative positions for each [i,j] pair, shape = [seq_len, seq_len]
        buckets = relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )  # map to buckets
        values = self.embedding(
            buckets
        )  # embedding vector for each [i,j] pair, shape = [seq_len, seq_len, embedding_dim=num_heads]
        return values.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, seq_len, seq_len]


class TemporalBias(nn.Module):
    """
    Temporal attention bias using logarithmic bucketing of time differences.

    Quantizes timestamp differences into log-spaced buckets, capturing both
    recent and long-term temporal patterns.
    """

    def __init__(self, num_buckets: int = 64, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.temporal_attention_bias = nn.Embedding(num_buckets, num_heads)

    def temporal_bucket(self, time_diff: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.clamp(torch.abs(time_diff), min=1).float()
        buckets = (torch.log(abs_diff) / 0.693).long()
        return torch.clamp(buckets, min=0, max=self.num_buckets - 1)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal bias matrix.

        Args:
            timestamps: [B, L] unix timestamps

        Returns:
            bias: [B, num_heads, L, L]
        """
        time_diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [B, L, L]
        buckets = self.temporal_bucket(time_diff)
        bias = self.temporal_attention_bias(buckets)  # [B, L, L, H]
        return bias.permute(0, 3, 1, 2)  # [B, H, L, L]


class HSTUPointwiseAttention(nn.Module):
    """
    Pointwise aggregation attention that implements HSTU without softmax:
        1) [U, V, Q, K] = split( φ1(f1(X)) ), U: gate, V: value, Q: query, K: key
        2) AV = φ2(QK^T + rab) V / N, av is attention-weighted value
        3) Y  = f2( Norm(AV) ⊙ U ), y is output
    φ1, φ2 use SiLU; Norm uses LayerNorm.
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
        self.alpha = alpha if alpha is not None else (self.head_dim**-0.5)
        # project input to 4 * hidden_dim for U, V, Q, K
        self.in_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        # project output back to hidden_dim
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = RMSNorm(hidden_dim) if use_rms_norm else nn.LayerNorm(hidden_dim)

    def reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, T, D] -> [B, H, T, head_dim]
        """
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # [T, T] with 0 or -inf
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T], True = pad
        rab: Optional[torch.Tensor] = None,  # [1, H, T, T] or None
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Eq.(1): [U, V, Q, K] = split( φ1(f1(X)) )
        h = F.silu(self.in_proj(x))  # [B, T, 4D]
        U, V, Q, K = h.chunk(4, dim=-1)  # each [B, T, D]

        Qh = self.reshape_heads(Q)  # [B, H, T, d_head]
        Kh = self.reshape_heads(K)  # [B, H, T, d_head]
        Vh = self.reshape_heads(V)  # [B, H, T, d_head]
        Uh = self.reshape_heads(U)  # [B, H, T, d_head]

        # attention logits: QK^T (without 1/sqrt(d) and softmax)
        logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.alpha  # [B, H, T, T]

        # add relative position bias (rab^p), and future extensible rab^t
        if rab is not None:
            # rab: [1, H, T, T] or [B, H, T, T]
            logits = logits + rab

        # causal mask: attention_mask is usually an upper triangular matrix of -inf with shape [T, T]
        if attention_mask is not None:
            logits = logits + attention_mask.view(1, 1, T, T)

        # padding mask: key_padding_mask is usually [B, T], True = pad
        if key_padding_mask is not None:
            # valid: 1 for non-pad, 0 for pad
            valid = (~key_padding_mask).float()  # [B, T]
            valid = valid.view(B, 1, 1, T)  # [B, 1, 1, T]
            logits = logits.masked_fill(valid == 0, float("-inf"))

        # Note: F.silu(-inf) = nan, so we need to handle -inf values carefully
        logits_safe = logits.masked_fill(torch.isinf(logits) & (logits < 0), -1e9)
        attention = F.silu(logits_safe)  # [B, H, T, T]
        AV = torch.matmul(attention, Vh)  # [B, H, T, head_dim]
        AV = AV.transpose(1, 2).contiguous().view(B, T, D)  # reshape back to [B, T, D]
        U_flat = Uh.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out_proj(self.dropout(self.attn_norm(AV) * U_flat))  # [B, T, D]
        return y


class HSTULayer(nn.Module):
    """
    HSTUPointwiseAttention + Residual Connection
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
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        device = x.device
        rab = None
        if self.use_rab_pos:
            rab = self.rel_pos_bias(seq_len=T, device=device)  # [1, H, T, T]
        if self.use_temporal_bias and timestamps is not None:
            time_bias = self.temporal_bias(timestamps)  # [B, H, T, T]
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
    Pipeline:
      1) Embed tokens from the behavior history
      2) Apply stacked HSTU layers with causal mask and optional temporal bias
      3) Produce vocabulary logits for every position via tied LM head
    """

    @property
    def model_name(self) -> str:
        return "HSTU"

    @property
    def default_task(self) -> str:
        return "sequential"

    def __init__(
        self,
        sequence_features: list[SequenceFeature],
        dense_features: Optional[list[DenseFeature]] = None,
        sparse_features: Optional[list[SparseFeature]] = None,
        item_history_name: str = "item_history",
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
            raise ValueError("[HSTU Error] HSTU requires at least one SequenceFeature (user behavior history).")

        if num_blocks is not None:
            num_layers = num_blocks
        if dropout is not None:
            dropout_rate = dropout

        self.item_history_feature = select_feature_objects(sequence_features, [item_history_name], "item_history")[0]

        self.hidden_dim = hidden_dim or max(int(self.item_history_feature.embedding_dim or 0), 32)
        self.ff_hidden_dim = ff_hidden_dim or (self.hidden_dim * 4)
        # Make hidden_dim divisible by num_heads
        if self.hidden_dim % num_heads != 0:
            self.hidden_dim = num_heads * math.ceil(self.hidden_dim / num_heads)

        self.padding_idx = (
            self.item_history_feature.padding_idx if self.item_history_feature.padding_idx is not None else 0
        )
        self.vocab_size = self.item_history_feature.vocab_size
        self.max_seq_len = max_seq_len

        if sequence_mode != "autoregressive":
            raise ValueError("[HSTU Error] HSTU currently only supports sequence_mode='autoregressive'.")

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
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        if tie_embeddings:
            self.output_proj.weight = self.item_embedding.weight

        self.register_regularization_weights(
            embedding_attr="item_embedding",
            include_modules=["layers", "output_proj"],
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        seq, padding_mask, valid_mask, attention_mask = self.prepare_sequence_batch(
            x=x,
            sequence_name=self.item_history_feature.name,
            max_seq_len=self.max_seq_len,
            padding_idx=self.padding_idx,
        )
        B, T = seq.shape
        device = seq.device
        token_emb = self.item_embedding(seq)  # [B, T, D]
        hidden_states = self.emb_dropout(token_emb * (self.hidden_dim**0.5))

        if attention_mask is not None and attention_mask.dtype == torch.bool:
            additive_mask = torch.zeros((T, T), device=device, dtype=hidden_states.dtype)
            additive_mask = additive_mask.masked_fill(attention_mask, float("-inf"))
            attention_mask = additive_mask

        timestamps = x.get("timestamps")

        for layer in self.layers:
            hidden_states = layer(
                x=hidden_states,
                attention_mask=attention_mask,
                key_padding_mask=padding_mask,
                timestamps=timestamps,
            )
        hidden_states = self.final_norm(hidden_states)  # [B, T, D]

        hidden_states = hidden_states * valid_mask
        logits = self.output_proj(hidden_states)  # [B, T, vocab_size]
        return logits
