"""
Date: create on 02/05/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Chen Z, Wang R, Jiang C, et al. HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.models.sequential.base import BaseSequentialModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import SequenceModeName, TaskTypeInput


class HLLMRelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_seq_len: int, num_buckets: int = 32):
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_seq_len = int(max_seq_len)
        self.num_buckets = int(num_buckets)
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        relative_position = torch.clamp(relative_position, min=0)
        max_exact = self.num_buckets // 2
        is_small = relative_position < max_exact
        large = (
            max_exact
            + (
                torch.log(relative_position.float() / max_exact + 1e-6)
                / torch.log(
                    torch.tensor(float(max(self.max_seq_len, max_exact + 1)), device=relative_position.device)
                    / max_exact
                )
                * (self.num_buckets - max_exact)
            ).long()
        )
        large = torch.clamp(large, max=self.num_buckets - 1)
        return torch.where(is_small, relative_position.long(), large)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        relative_position = positions.unsqueeze(1) - positions.unsqueeze(0)
        buckets = self._relative_position_bucket(relative_position)
        bias = self.relative_attention_bias(buckets)
        return bias.permute(2, 0, 1).unsqueeze(0)


class HLLMTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"[HLLMTransformerBlock Error] hidden_dim({hidden_dim}) must be divisible by num_heads({num_heads})."
            )
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        x_norm = self.attn_norm(x)

        q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(batch_size, 1, 1, seq_len), torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        x = residual + self.dropout(self.out_proj(attn_output))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HLLM(BaseSequentialModel):
    @property
    def model_name(self) -> str:
        return "HLLM"

    @property
    def default_task(self) -> str:
        return "generative"

    def __init__(
        self,
        sequence_features: list[SequenceFeature] | None = None,
        dense_features: Optional[list[DenseFeature]] = None,
        sparse_features: Optional[list[SparseFeature]] = None,
        item_history_name: str = "item_history",
        item_embeddings: torch.Tensor | str | Path | None = None,
        vocab_size: int | None = None,
        hidden_dim: Optional[int] = None,
        d_model: Optional[int] = None,
        num_heads: int | None = None,
        n_heads: int | None = None,
        num_layers: int | None = None,
        n_layers: int | None = None,
        max_seq_len: Optional[int] = None,
        dropout_rate: float | None = 0.1,
        dropout: float | None = None,
        use_rel_pos_bias: bool = True,
        num_position_buckets: int = 32,
        use_time_embedding: bool = True,
        num_time_buckets: int = 2048,
        time_bucket_fn: str = "sqrt",
        temperature: float = 1.0,
        trainable_item_embeddings: bool = False,
        sequence_mode: SequenceModeName = "autoregressive",
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        **kwargs,
    ):
        if dropout is not None:
            dropout_rate = dropout
        dropout_rate = float(dropout_rate if dropout_rate is not None else 0.1)
        hidden_dim = hidden_dim or d_model
        num_heads = num_heads if num_heads is not None else (n_heads or 8)
        num_layers = num_layers if num_layers is not None else (n_layers or 4)

        if isinstance(item_embeddings, torch.Tensor):
            loaded_item_embeddings = item_embeddings.detach().clone()
        elif item_embeddings is None:
            loaded_item_embeddings = None
        else:
            loaded_item_embeddings = torch.load(item_embeddings, map_location="cpu")
            if isinstance(loaded_item_embeddings, dict):
                for key in ("item_embeddings", "embeddings", "weight"):
                    if key in loaded_item_embeddings:
                        loaded_item_embeddings = loaded_item_embeddings[key]
                        break
            if not isinstance(loaded_item_embeddings, torch.Tensor):
                raise TypeError(f"[HLLM Error] Expected item_embeddings tensor, got {type(loaded_item_embeddings)}.")

        if loaded_item_embeddings is not None:
            vocab_size = int(loaded_item_embeddings.size(0))
            inferred_hidden_dim = int(loaded_item_embeddings.size(1))
        else:
            inferred_hidden_dim = int(hidden_dim or 64)

        if kwargs.get("multimodal_features") is not None:
            raise ValueError(
                "[HLLM Error] multimodal_features is no longer supported. "
                "Precompute multimodal content as item_embeddings/dense features, "
                "or convert semantic IDs to a SequenceFeature before constructing HLLM."
            )

        if sequence_features:
            self.item_history_feature = select_feature_objects(
                sequence_features,
                [item_history_name],
                "item_history_name",
            )[0]
            vocab_size = int(vocab_size or self.item_history_feature.vocab_size)
            max_seq_len = int(max_seq_len or self.item_history_feature.max_len)
        else:
            if vocab_size is None:
                raise ValueError("[HLLM Error] HLLM requires sequence_features, item_embeddings, or vocab_size.")
            max_seq_len = int(max_seq_len or 50)
            sequence_features = [
                SequenceFeature(
                    name=item_history_name,
                    vocab_size=int(vocab_size),
                    max_len=max_seq_len,
                    embedding_dim=inferred_hidden_dim,
                    padding_idx=0,
                )
            ]
            self.item_history_feature = sequence_features[0]

        self.vocab_size = int(vocab_size)
        self.padding_idx = (
            self.item_history_feature.padding_idx if self.item_history_feature.padding_idx is not None else 0
        )
        self.max_seq_len = int(max_seq_len)
        self.hidden_dim = int(hidden_dim or inferred_hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.use_time_embedding = bool(use_time_embedding)
        self.num_time_buckets = int(num_time_buckets)
        self.time_bucket_fn = time_bucket_fn
        self.temperature = float(temperature)
        self.trainable_item_embeddings = bool(trainable_item_embeddings)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"[HLLM Error] hidden_dim({self.hidden_dim}) must be divisible by num_heads({self.num_heads})."
            )

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

        if loaded_item_embeddings is not None:
            if loaded_item_embeddings.size(1) != self.hidden_dim:
                self.item_input_proj = nn.Linear(int(loaded_item_embeddings.size(1)), self.hidden_dim, bias=False)
            else:
                self.item_input_proj = nn.Identity()
            if self.trainable_item_embeddings:
                self.item_embedding = nn.Embedding.from_pretrained(
                    loaded_item_embeddings.float(), freeze=False, padding_idx=self.padding_idx
                )
            else:
                self.register_buffer("item_embedding_weight", loaded_item_embeddings.float())
                self.item_embedding = None
        else:
            self.item_embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.padding_idx)
            self.item_input_proj = nn.Identity()

        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.time_embedding = (
            nn.Embedding(self.num_time_buckets + 1, self.hidden_dim, padding_idx=0) if self.use_time_embedding else None
        )
        self.rel_pos_bias = (
            HLLMRelativePositionBias(
                num_heads=self.num_heads,
                max_seq_len=self.max_seq_len,
                num_buckets=num_position_buckets,
            )
            if use_rel_pos_bias
            else None
        )
        self.blocks = nn.ModuleList(
            [
                HLLMTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.register_regularization_weights(
            embedding_attr="item_embedding",
            include_modules=["item_input_proj", "position_embedding", "time_embedding", "rel_pos_bias", "blocks"],
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "item_embedding.weight" in name and self.trainable_item_embeddings:
                continue
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _time_diff_to_bucket(self, time_diffs: torch.Tensor) -> torch.Tensor:
        time_diffs = torch.clamp(time_diffs.float() / 60.0, min=1e-6)
        if self.time_bucket_fn == "sqrt":
            buckets = torch.sqrt(time_diffs).long()
        elif self.time_bucket_fn == "log":
            buckets = torch.log(time_diffs).long()
        else:
            raise ValueError(f"[HLLM Error] Unsupported time_bucket_fn={self.time_bucket_fn!r}.")
        return torch.clamp(buckets, min=0, max=self.num_time_buckets - 1)

    def encode(
        self,
        seq_tokens: torch.Tensor,
        time_diffs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = seq_tokens.long()[:, -self.max_seq_len :]
        padding_mask = seq.eq(self.padding_idx)
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        batch_size, seq_len = seq.shape
        device = seq.device

        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        item_emb = self.item_embedding(seq) if self.item_embedding is not None else self.item_embedding_weight.to(device)[seq]
        x = self.item_input_proj(item_emb) + self.position_embedding(positions)

        if self.time_embedding is not None:
            if time_diffs is None:
                time_diffs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            else:
                time_diffs = time_diffs.long()[:, -seq_len:].to(device)
            x = x + self.time_embedding(self._time_diff_to_bucket(time_diffs))

        x = self.dropout(x)
        attention_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        rel_pos_bias = self.rel_pos_bias(seq_len, device=device) if self.rel_pos_bias is not None else None
        for block in self.blocks:
            x = block(
                x,
                attention_mask=attention_mask,
                key_padding_mask=padding_mask,
                rel_pos_bias=rel_pos_bias,
            )
            x = x * valid_mask
        return self.final_norm(x) * valid_mask, padding_mask

    def score_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        item_weight = self.item_embedding.weight if self.item_embedding is not None else self.item_embedding_weight
        item_weight = self.item_input_proj(item_weight.to(hidden_states.device))
        return torch.matmul(hidden_states, item_weight.t()) / self.temperature

    def forward(
        self,
        x: dict[str, torch.Tensor] | torch.Tensor,
        time_diffs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        called_with_dict = isinstance(x, dict)
        if called_with_dict:
            seq_tokens = x[self.item_history_feature.name]
            time_diffs = x.get("time_diffs", x.get("timestamps", time_diffs))
        else:
            seq_tokens = x

        hidden_states, _ = self.encode(seq_tokens, time_diffs=time_diffs)
        logits = self.score_hidden_states(hidden_states)

        if called_with_dict:
            return logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.long().reshape(-1),
                ignore_index=self.padding_idx,
            )
        return logits, loss
