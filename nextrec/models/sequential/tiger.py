"""
Date: create on 02/05/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Rajput S, Mehta N, Singh A, et al. Recommender Systems with Generative Retrieval. NeurIPS 2023.

TIGER is an encoder-decoder generative retrieval model. It encodes the
user/history semantic-id sequence and decodes the next item's semantic-id codes
with teacher forcing during training and constrained beam search during inference.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SemanticIdFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import SemanticIdEmbedding
from nextrec.models.sequential.base import BaseSequentialModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import SequenceModeName, TaskTypeInput


class TrieNode(defaultdict):
    """Trie node used for constrained semantic-id decoding."""

    def __init__(self):
        super().__init__(TrieNode)
        self.is_end = False


class TigerOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None


class TigerGenerationOutput(NamedTuple):
    sem_ids: torch.Tensor
    log_probas: torch.Tensor


def build_trie(valid_semantic_ids: torch.Tensor) -> TrieNode:
    """Build a prefix tree over valid semantic-id code sequences."""
    root = TrieNode()
    if valid_semantic_ids.dim() == 1:
        valid_semantic_ids = valid_semantic_ids.unsqueeze(0)
    elif valid_semantic_ids.dim() > 2:
        valid_semantic_ids = valid_semantic_ids.reshape(-1, valid_semantic_ids.size(-1))

    for seq in valid_semantic_ids.long().cpu().tolist():
        node = root
        for token in seq:
            node = node[int(token)]
        node.is_end = True
    return root


class UserIdEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.emb = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids.long().remainder(self.num_embeddings))


class TigerTransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


class Tiger(BaseSequentialModel):
    @property
    def model_name(self) -> str:
        return "Tiger"

    @property
    def default_task(self) -> str:
        return "generative"

    def __init__(
        self,
        sequence_features: list[SequenceFeature] | None = None,
        dense_features: Optional[list[DenseFeature]] = None,
        sparse_features: Optional[list[SparseFeature]] = None,
        semantic_id_features: Optional[list[SemanticIdFeature]] = None,
        item_history_name: str = "item_history",
        user_id_name: str | None = "user_id",
        embedding_dim: Optional[int] = None,
        attn_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        n_layers: int | None = None,
        num_layers: int | None = None,
        num_blocks: int | None = None,
        dim_feedforward: int | None = None,
        ff_hidden_dim: Optional[int] = None,
        num_item_embeddings: int | None = None,
        num_user_embeddings: int | None = None,
        sem_id_dim: int | None = None,
        num_codebooks: int = 3,
        max_pos: int | None = None,
        d_kv: int = 0,
        share_position_bias: bool = False,
        scale_attn: bool = True,
        add_final_norm: bool = False,
        decoder_bidirectional: bool = True,
        max_seq_len: Optional[int] = None,
        dropout_rate: float | None = 0.1,
        dropout: float | None = None,
        tie_embeddings: bool = False,
        sequence_mode: SequenceModeName = "seq2seq",
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        **kwargs,
    ):
        sequence_features = sequence_features or []
        dense_features = dense_features or []
        semantic_id_features = semantic_id_features or []
        if "multimodal_features" in kwargs:
            raise ValueError(
                "[Tiger Error] multimodal_features is no longer supported. "
                "Convert multimodal content to dense features before model input, "
                "or pass semantic IDs through SemanticIdFeature."
            )
        if sequence_mode != "seq2seq":
            raise ValueError("[Tiger Error] GenRec-compatible Tiger requires sequence_mode='seq2seq'.")

        if sem_id_dim is not None:
            num_codebooks = sem_id_dim
        if num_codebooks < 1:
            raise ValueError(f"[Tiger Error] num_codebooks must be >= 1, got {num_codebooks}.")
        if dropout is not None:
            dropout_rate = dropout
        dropout_rate = float(dropout_rate if dropout_rate is not None else 0.1)

        item_history_candidates = list(semantic_id_features)
        if not item_history_candidates:
            raise ValueError(
                "[Tiger Error] Tiger requires a SemanticIdFeature. "
                "Semantic IDs should be prepared upstream with shape [batch, history_len, num_codebooks]."
            )
        self.item_history_feature = select_feature_objects(
            item_history_candidates,
            [item_history_name],
            "item_history_name",
        )[0]
        self.max_seq_len = int(max_seq_len or self.item_history_feature.max_len)
        self.num_codebooks = int(num_codebooks)
        if self.item_history_feature.num_codebooks != self.num_codebooks:
            raise ValueError(
                f"[Tiger Error] num_codebooks({self.num_codebooks}) must match SemanticIdFeature "
                f"num_codebooks({self.item_history_feature.num_codebooks})."
            )

        self.codebook_sizes = list(self.item_history_feature.codebook_sizes)
        if num_item_embeddings is not None:
            self.codebook_sizes = [int(num_item_embeddings)] * self.num_codebooks
        self.num_item_embeddings = max(self.codebook_sizes)
        self.vocab_size = sum(self.codebook_sizes) + 1
        self.padding_idx = self.vocab_size - 1
        self.max_pos = int(max_pos or self.max_seq_len)
        self.d_kv = int(d_kv)
        self.share_position_bias = bool(share_position_bias)
        self.scale_attn = bool(scale_attn)
        self.add_final_norm = bool(add_final_norm)
        self.decoder_bidirectional = bool(decoder_bidirectional)

        base_embedding_dim = int(embedding_dim or hidden_dim or self.item_history_feature.embedding_dim)
        self.hidden_dim = int(attn_dim or hidden_dim or base_embedding_dim)
        self.embedding_dim = base_embedding_dim
        self.use_proj = self.embedding_dim != self.hidden_dim
        self.ff_hidden_dim = int(dim_feedforward or ff_hidden_dim or (self.hidden_dim * 4))

        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"[Tiger Error] hidden_dim({self.hidden_dim}) must be divisible by num_heads({num_heads})."
            )

        layers = n_layers if n_layers is not None else num_layers
        if layers is None:
            layers = num_blocks if num_blocks is not None else 2
        num_encoder_layers = max(int(layers) // 2, 1)
        num_decoder_layers = max(int(layers) - num_encoder_layers, 1)

        self.user_id_name = user_id_name
        self.disable_default_head = True
        user_feature = None
        if user_id_name:
            matched = [feat for feat in (sparse_features or []) if feat.name == user_id_name]
            user_feature = matched[0] if matched else None
        self.num_user_embeddings = int(
            num_user_embeddings
            or (user_feature.vocab_size if user_feature is not None else max(self.num_item_embeddings, 1))
        )

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            semantic_id_features=semantic_id_features,
            target=target,
            task=task or self.default_task,
            sequence_mode=sequence_mode,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        self.bos_embedding = nn.Parameter(torch.randn(self.embedding_dim))
        self.sem_id_embedding = SemanticIdEmbedding(
            codebook_sizes=self.codebook_sizes,
            embedding_dim=self.embedding_dim,
            combiner=self.item_history_feature.combiner,
        )
        self.user_id_embedding = UserIdEmbedding(
            num_embeddings=self.num_user_embeddings,
            embedding_dim=self.embedding_dim,
        )
        self.encoder_norm = nn.LayerNorm(self.embedding_dim, eps=1e-8)
        self.decoder_norm = nn.LayerNorm(self.embedding_dim, eps=1e-8)
        self.drop = nn.Dropout(dropout_rate)
        self.in_proj_context = (
            nn.Linear(self.embedding_dim, self.hidden_dim, bias=False) if self.use_proj else nn.Identity()
        )
        self.in_proj = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False) if self.use_proj else nn.Identity()
        self.transformer = TigerTransformerEncoderDecoder(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=self.ff_hidden_dim,
            dropout=dropout_rate,
        )
        self.output_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        if tie_embeddings:
            raise ValueError("[Tiger Error] tie_embeddings=True is not supported for hierarchical semantic IDs.")

        self.register_regularization_weights(
            embedding_attr="sem_id_embedding",
            include_modules=["transformer", "output_head", "in_proj"],
        )

    def _prepare_target(
        self,
        x: dict[str, torch.Tensor],
        target_input_ids: Optional[torch.Tensor] = None,
        target_token_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if target_input_ids is None:
            target_input_ids = x.get("target_input_ids")
        if target_token_type_ids is None:
            target_token_type_ids = x.get("target_token_type_ids")
        if target_input_ids is None:
            return None, None

        # Targets are semantic-id code sequences: [B, num_codebooks].
        target_input_ids = target_input_ids.long().to(next(self.parameters()).device)
        target_input_ids = target_input_ids.reshape(target_input_ids.size(0), -1)
        if target_input_ids.size(1) != self.num_codebooks:
            raise ValueError(
                f"[Tiger Error] target semantic ids must have shape [batch, {self.num_codebooks}], "
                f"got {tuple(target_input_ids.shape)}."
            )
        if target_token_type_ids is None:
            target_token_type_ids = (
                torch.arange(target_input_ids.size(1), device=target_input_ids.device)
                .unsqueeze(0)
                .expand(target_input_ids.size(0), -1)
            )
        else:
            target_token_type_ids = target_token_type_ids.long().to(target_input_ids.device)
            target_token_type_ids = target_token_type_ids.reshape(target_input_ids.size(0), -1)
        return target_input_ids, target_token_type_ids

    def _encode_context(
        self,
        x: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode user id plus historical item semantic IDs.
        Returns Transformer memory and its key padding mask.
        """
        item_input_ids = x["item_input_ids"] if "item_input_ids" in x else x[self.item_history_feature.name]
        item_input_ids = item_input_ids.long()[:, -self.max_seq_len :, :]
        if item_input_ids.dim() != 3:
            raise ValueError(
                f"[Tiger Error] item semantic ids must have shape [batch, seq_len, {self.num_codebooks}], "
                f"got {tuple(item_input_ids.shape)}."
            )
        item_input_ids = item_input_ids.to(self.device)
        batch_size = item_input_ids.size(0)

        if "seq_mask" in x:
            seq_mask = x["seq_mask"].long()[:, -item_input_ids.size(1) :].to(self.device)
        else:
            padding_code_ids = self.sem_id_embedding.padding_code_ids.to(item_input_ids.device).view(1, 1, -1)
            invalid = (item_input_ids < 0) | (item_input_ids >= padding_code_ids)
            # A position is valid if at least one codebook id is in range.
            seq_mask = (~invalid.all(dim=-1)).long()

        if "user_input_ids" in x:
            user_input_ids = x["user_input_ids"].long().to(item_input_ids.device)
        elif self.user_id_name and self.user_id_name in x:
            user_input_ids = x[self.user_id_name].long().to(item_input_ids.device).reshape(batch_size, -1)[:, :1]
        else:
            user_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=item_input_ids.device)

        user_emb = self.user_id_embedding(user_input_ids)
        item_emb = self.sem_id_embedding(item_input_ids)
        encoder_input = torch.cat([user_emb, item_emb], dim=1)

        encoder_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=seq_mask.dtype, device=seq_mask.device),
                seq_mask,
            ],
            dim=1,
        )
        key_padding_mask = ~encoder_mask.bool()
        encoder_input = self.in_proj_context(self.drop(self.encoder_norm(encoder_input)))
        memory = self.transformer.encoder(encoder_input, src_key_padding_mask=key_padding_mask)
        return memory, key_padding_mask

    def _decode(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        target_input_ids: Optional[torch.Tensor],
        target_token_type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decode semantic-id codes from BOS plus optional teacher-forced target prefix."""
        batch_size = memory.size(0)
        bos = self.bos_embedding.view(1, 1, -1).expand(batch_size, 1, -1)
        if target_input_ids is None:
            decoder_input = bos
        else:
            target_emb = self.sem_id_embedding.embed_level_tokens(target_input_ids, target_token_type_ids)
            decoder_input = torch.cat([bos, target_emb], dim=1)

        decoder_input = self.in_proj(self.drop(self.decoder_norm(decoder_input)))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.size(1),
            device=decoder_input.device,
        )
        decoder_out = self.transformer.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_head(decoder_out)

    def forward(
        self,
        x: Optional[dict[str, torch.Tensor]] = None,
        *,
        user_input_ids: Optional[torch.Tensor] = None,
        item_input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        target_token_type_ids: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | TigerOutput:
        called_with_dict = isinstance(x, dict)
        if x is None:
            x = {}
        if user_input_ids is not None:
            x["user_input_ids"] = user_input_ids
        if item_input_ids is not None:
            x["item_input_ids"] = item_input_ids
        if token_type_ids is not None:
            x["token_type_ids"] = token_type_ids
        if target_input_ids is not None:
            x["target_input_ids"] = target_input_ids
        if target_token_type_ids is not None:
            x["target_token_type_ids"] = target_token_type_ids
        if seq_mask is not None:
            x["seq_mask"] = seq_mask

        target_input_ids, target_token_type_ids = self._prepare_target(
            x,
            target_input_ids=target_input_ids,
            target_token_type_ids=target_token_type_ids,
        )
        if called_with_dict:
            self._last_seq2seq_input = x
        memory, memory_key_padding_mask = self._encode_context(x)
        logits = self._decode(memory, memory_key_padding_mask, target_input_ids, target_token_type_ids)

        if called_with_dict:
            return logits[:, :-1, :] if target_input_ids is not None else logits

        loss = None
        if target_input_ids is not None:
            loss = self._compute_seq2seq_loss(logits=logits[:, :-1, :], target_input_ids=target_input_ids)
        return TigerOutput(logits=logits, loss=loss)

    def _compute_seq2seq_loss(self, logits: torch.Tensor, target_input_ids: torch.Tensor) -> torch.Tensor:

        target_vocab_ids = self.sem_id_embedding.target_to_flat_vocab(target_input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_vocab_ids.reshape(-1),
            ignore_index=self.padding_idx,
            reduction="mean",
        )

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor | None) -> torch.Tensor:
        if y_true is None:
            raise ValueError("[Tiger-compute_loss Error] Ground truth semantic IDs are required.")
        x = getattr(self, "_last_seq2seq_input", None)
        if x is None:
            if not isinstance(y_pred, torch.Tensor):
                raise ValueError("[Tiger-compute_loss Error] Missing cached seq2seq input.")
            return super().compute_loss(y_pred, y_true)

        target_input_ids, target_token_type_ids = self._prepare_target(
            x,
            target_input_ids=y_true,
            target_token_type_ids=None,
        )
        memory, memory_key_padding_mask = self._encode_context(x)
        logits = self._decode(memory, memory_key_padding_mask, target_input_ids, target_token_type_ids)
        return self._compute_seq2seq_loss(logits=logits[:, :-1, :], target_input_ids=target_input_ids)

    def _decode_step(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        generated_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return logits for the next semantic-id code level."""
        if generated_ids is None or generated_ids.numel() == 0:
            return self._decode(memory, memory_key_padding_mask, None, None)[:, -1, :]
        token_type_ids = torch.arange(generated_ids.size(1), device=generated_ids.device).unsqueeze(0)
        token_type_ids = token_type_ids.expand(generated_ids.size(0), -1)
        return self._decode(memory, memory_key_padding_mask, generated_ids, token_type_ids)[:, -1, :]

    def generate(
        self,
        x: Optional[dict[str, torch.Tensor]] = None,
        *,
        user_input_ids: Optional[torch.Tensor] = None,
        item_input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        n_top_k_candidates: int = 10,
        valid_item_ids: Optional[torch.Tensor] = None,
        valid_semantic_ids: Optional[torch.Tensor] = None,
        use_trie: bool = True,
    ) -> TigerGenerationOutput:
        if x is None:
            x = {}
        if user_input_ids is not None:
            x["user_input_ids"] = user_input_ids
        if item_input_ids is not None:
            x["item_input_ids"] = item_input_ids
        if token_type_ids is not None:
            x["token_type_ids"] = token_type_ids
        if seq_mask is not None:
            x["seq_mask"] = seq_mask

        if n_top_k_candidates < 1:
            raise ValueError("[Tiger-generate Error] n_top_k_candidates must be >= 1.")
        if temperature <= 0:
            raise ValueError("[Tiger-generate Error] temperature must be > 0.")
        valid_ids = valid_semantic_ids if valid_semantic_ids is not None else valid_item_ids
        if use_trie and valid_ids is None:
            raise ValueError("[Tiger-generate Error] valid_item_ids/valid_semantic_ids is required when use_trie=True.")
        if valid_ids is not None and valid_ids.size(-1) != self.num_codebooks:
            raise ValueError(
                f"[Tiger-generate Error] valid semantic id width must be {self.num_codebooks}, got {valid_ids.size(-1)}."
            )

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                memory, memory_key_padding_mask = self._encode_context(x)
                batch_size = memory.size(0)
                beam_size = int(n_top_k_candidates)
                device = memory.device

                memory = (
                    memory.unsqueeze(1)
                    .expand(-1, beam_size, -1, -1)
                    .reshape(
                        batch_size * beam_size,
                        memory.size(1),
                        memory.size(2),
                    )
                )
                memory_key_padding_mask = (
                    memory_key_padding_mask.unsqueeze(1)
                    .expand(-1, beam_size, -1)
                    .reshape(
                        batch_size * beam_size,
                        memory_key_padding_mask.size(1),
                    )
                )

                trie_root = build_trie(valid_ids) if use_trie and valid_ids is not None else None
                beam_scores = torch.zeros(batch_size, beam_size, device=device)
                if beam_size > 1:
                    beam_scores[:, 1:] = -torch.inf
                beam_tokens = torch.empty(batch_size, beam_size, 0, dtype=torch.long, device=device)
                beam_nodes = [trie_root for _ in range(batch_size * beam_size)] if trie_root is not None else None

                for step in range(self.num_codebooks):
                    # Decode one semantic-id codebook level at a time.
                    flat_generated = beam_tokens.reshape(batch_size * beam_size, -1)
                    logits = self._decode_step(
                        memory=memory,
                        memory_key_padding_mask=memory_key_padding_mask,
                        generated_ids=flat_generated,
                    )

                    if trie_root is not None and beam_nodes is not None:
                        legal_mask = torch.zeros_like(logits, dtype=torch.bool)
                        offset = int(self.sem_id_embedding.level_offsets[step].item())
                        for row, node in enumerate(beam_nodes):
                            if node:
                                # Constrain next code to valid trie children for the current prefix.
                                legal_mask[
                                    row, offset + torch.tensor(list(node.keys()), dtype=torch.long, device=device)
                                ] = True
                        logits = logits.masked_fill(~legal_mask, -torch.inf)
                    else:
                        legal_mask = torch.zeros_like(logits, dtype=torch.bool)
                        start = int(self.sem_id_embedding.level_offsets[step].item())
                        legal_mask[:, start : start + self.codebook_sizes[step]] = True
                        logits = logits.masked_fill(~legal_mask, -torch.inf)

                    log_probs = torch.log_softmax(logits / temperature, dim=-1)
                    candidate_log_probs, candidate_vocab_ids = torch.topk(log_probs, k=beam_size, dim=-1)
                    candidate_tokens = candidate_vocab_ids - int(self.sem_id_embedding.level_offsets[step].item())
                    candidate_log_probs = candidate_log_probs.view(batch_size, beam_size, beam_size)
                    candidate_tokens = candidate_tokens.view(batch_size, beam_size, beam_size)

                    total_scores = (beam_scores.unsqueeze(-1) + candidate_log_probs).reshape(batch_size, -1)
                    candidate_tokens = candidate_tokens.reshape(batch_size, -1)
                    parent_beams = (
                        torch.arange(beam_size, device=device)
                        .view(1, beam_size, 1)
                        .expand(batch_size, beam_size, beam_size)
                        .reshape(batch_size, -1)
                    )
                    next_scores, top_indices = torch.topk(total_scores, k=beam_size, dim=-1)
                    next_tokens = torch.gather(candidate_tokens, dim=1, index=top_indices)
                    next_parents = torch.gather(parent_beams, dim=1, index=top_indices)

                    next_beam_tokens = torch.empty(
                        batch_size,
                        beam_size,
                        step + 1,
                        dtype=torch.long,
                        device=device,
                    )
                    next_nodes = [] if trie_root is not None else None
                    for batch_idx in range(batch_size):
                        for beam_idx in range(beam_size):
                            parent_idx = int(next_parents[batch_idx, beam_idx].item())
                            token = int(next_tokens[batch_idx, beam_idx].item())
                            if step > 0:
                                next_beam_tokens[batch_idx, beam_idx, :step] = beam_tokens[batch_idx, parent_idx]
                            next_beam_tokens[batch_idx, beam_idx, step] = token
                            if next_nodes is not None and beam_nodes is not None:
                                next_nodes.append(beam_nodes[batch_idx * beam_size + parent_idx][token])

                    beam_scores = next_scores
                    beam_tokens = next_beam_tokens
                    beam_nodes = next_nodes
        finally:
            if was_training:
                self.train()

        return TigerGenerationOutput(sem_ids=beam_tokens, log_probas=beam_scores)
