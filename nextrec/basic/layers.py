"""
Layer implementations used across NextRec models.

Date: create on 27/10/2025
Checkpoint: edit on 20/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

from collections import OrderedDict
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.activation import activation_layer
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.utils.torch_utils import get_initializer


class PredictionLayer(nn.Module):
    def __init__(
        self,
        task_type: str | list[str] = "binary",
        task_dims: int | list[int] | None = None,
        use_bias: bool = True,
        return_logits: bool = False,
    ):
        """
        Prediction layer supporting binary and regression outputs.

        Args:
            task_type: A string or list of strings specifying the type of each task. supported types are "binary" and "regression".
            task_dims: An integer or list of integers specifying the output dimension for each task.
                If None, defaults to 1 for each task. If a single integer is provided, it is shared across all tasks.
            use_bias: Whether to include a bias term in the prediction layer.
            return_logits: If True, returns raw logits without applying activation functions.
        """
        super().__init__()
        self.task_types = [task_type] if isinstance(task_type, str) else list(task_type)
        if len(self.task_types) == 0:
            raise ValueError("At least one task_type must be specified.")

        if task_dims is None:
            dims = [1] * len(self.task_types)
        elif isinstance(task_dims, int):
            dims = [task_dims]
        else:
            dims = list(task_dims)
        if len(dims) not in (1, len(self.task_types)):
            raise ValueError(
                "[PredictionLayer Error]: task_dims must be None, a single int (shared), or a sequence of the same length as task_type."
            )
        if len(dims) == 1 and len(self.task_types) > 1:
            dims = dims * len(self.task_types)
        self.task_dims = dims
        self.total_dim = sum(self.task_dims)
        self.return_logits = return_logits

        # slice offsets per task
        start = 0
        self.task_slices: list[tuple[int, int]] = []
        for dim in self.task_dims:
            if dim < 1:
                raise ValueError("Each task dimension must be >= 1.")
            self.task_slices.append((start, start + dim))
            start += dim
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.total_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1 * total_dim)
        if x.shape[-1] != self.total_dim:
            raise ValueError(
                f"[PredictionLayer Error]: Input last dimension ({x.shape[-1]}) does not match expected total dimension ({self.total_dim})."
            )
        logits = x if self.bias is None else x + self.bias
        outputs = []
        for task_type, (start, end) in zip(self.task_types, self.task_slices):
            task_logits = logits[..., start:end]  # logits for the current task
            if self.return_logits:
                outputs.append(task_logits)
                continue
            task = task_type.lower()
            if task == "binary":
                outputs.append(torch.sigmoid(task_logits))
            elif task == "regression":
                outputs.append(task_logits)
            else:
                raise ValueError(
                    f"[PredictionLayer Error]: Unsupported task_type '{task_type}'."
                )
        result = torch.cat(
            outputs, dim=-1
        )  # single: (N,1), multi-task/multi-class: (N,total_dim)
        return result


class EmbeddingLayer(nn.Module):
    def __init__(self, features: list):
        super().__init__()
        self.features = list(features)
        self.embed_dict = nn.ModuleDict()
        self.dense_transforms = nn.ModuleDict()
        self.dense_input_dims: dict[str, int] = {}

        for feature in self.features:
            if isinstance(feature, (SparseFeature, SequenceFeature)):
                if feature.embedding_name in self.embed_dict:
                    continue
                if getattr(feature, "pretrained_weight", None) is not None:
                    weight = feature.pretrained_weight  # type: ignore[assignment]
                    if weight.shape != (feature.vocab_size, feature.embedding_dim):  # type: ignore[assignment]
                        raise ValueError(f"[EmbeddingLayer Error]: Pretrained weight for '{feature.embedding_name}' has shape {weight.shape}, expected ({feature.vocab_size}, {feature.embedding_dim}).")  # type: ignore[assignment]
                    embedding = nn.Embedding.from_pretrained(embeddings=weight, freeze=feature.freeze_pretrained, padding_idx=feature.padding_idx)  # type: ignore[assignment]
                    embedding.weight.requires_grad = feature.trainable and not feature.freeze_pretrained  # type: ignore[assignment]
                else:
                    embedding = nn.Embedding(
                        num_embeddings=feature.vocab_size,
                        embedding_dim=feature.embedding_dim,
                        padding_idx=feature.padding_idx,
                    )
                    embedding.weight.requires_grad = feature.trainable
                    initialization = get_initializer(
                        init_type=feature.init_type,
                        activation="linear",
                        param=feature.init_params,
                    )
                    initialization(embedding.weight)
                self.embed_dict[feature.embedding_name] = embedding
            elif isinstance(feature, DenseFeature):
                if not feature.use_embedding:
                    self.dense_input_dims[feature.name] = max(
                        int(getattr(feature, "input_dim", 1)), 1
                    )
                    continue
                if feature.name in self.dense_transforms:
                    continue
                in_dim = max(int(getattr(feature, "input_dim", 1)), 1)
                out_dim = max(int(getattr(feature, "embedding_dim", None) or in_dim), 1)
                dense_linear = nn.Linear(in_dim, out_dim, bias=True)
                nn.init.xavier_uniform_(dense_linear.weight)
                nn.init.zeros_(dense_linear.bias)
                self.dense_transforms[feature.name] = dense_linear
                self.dense_input_dims[feature.name] = in_dim
            else:
                raise TypeError(
                    f"[EmbeddingLayer Error]: Unsupported feature type: {type(feature)}"
                )
        self.output_dim = self.compute_output_dim()

    def forward(
        self,
        x: dict[str, torch.Tensor],
        features: list[object],
        squeeze_dim: bool = False,
    ) -> torch.Tensor:
        sparse_embeds: list[torch.Tensor] = []
        dense_embeds: list[torch.Tensor] = []

        for feature in features:
            if isinstance(feature, SparseFeature):
                embed = self.embed_dict[feature.embedding_name]
                sparse_embeds.append(embed(x[feature.name].long()).unsqueeze(1))

            elif isinstance(feature, SequenceFeature):
                seq_input = x[feature.name].long()
                if feature.max_len is not None and seq_input.size(1) > feature.max_len:
                    seq_input = seq_input[:, -feature.max_len :]

                embed = self.embed_dict[feature.embedding_name]
                seq_emb = embed(seq_input)  # [B, seq_len, emb_dim]

                if feature.combiner == "mean":
                    pooling_layer = AveragePooling()
                elif feature.combiner == "sum":
                    pooling_layer = SumPooling()
                elif feature.combiner == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError(
                        f"[EmbeddingLayer Error]: Unknown combiner for {feature.name}: {feature.combiner}"
                    )
                feature_mask = InputMask()(x, feature, seq_input)
                sparse_embeds.append(pooling_layer(seq_emb, feature_mask).unsqueeze(1))

            elif isinstance(feature, DenseFeature):
                dense_embeds.append(self.project_dense(feature, x))

        if squeeze_dim:
            flattened_sparse = [emb.flatten(start_dim=1) for emb in sparse_embeds]
            pieces = []
            if flattened_sparse:
                pieces.append(torch.cat(flattened_sparse, dim=1))
            if dense_embeds:
                pieces.append(torch.cat(dense_embeds, dim=1))
            if not pieces:
                raise ValueError(
                    "[EmbeddingLayer Error]: No input features found for EmbeddingLayer."
                )
            return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

        # squeeze_dim=False requires embeddings with identical last dimension
        output_embeddings = list(sparse_embeds)
        if dense_embeds:
            if output_embeddings:
                target_dim = output_embeddings[0].shape[-1]
                for emb in dense_embeds:
                    if emb.shape[-1] != target_dim:
                        raise ValueError(
                            f"[EmbeddingLayer Error]: squeeze_dim=False requires all dense feature dimensions to match the embedding dimension of sparse/sequence features ({target_dim}), but got {emb.shape[-1]}."
                        )
                output_embeddings.extend(emb.unsqueeze(1) for emb in dense_embeds)
            else:
                dims = {emb.shape[-1] for emb in dense_embeds}
                if len(dims) != 1:
                    raise ValueError(
                        f"[EmbeddingLayer Error]: squeeze_dim=False requires all dense features to have identical dimensions when no sparse/sequence features are present, but got dimensions {dims}."
                    )
                output_embeddings = [emb.unsqueeze(1) for emb in dense_embeds]
        if not output_embeddings:
            raise ValueError(
                "[EmbeddingLayer Error]: squeeze_dim=False requires at least one sparse/sequence feature or dense features with identical projected dimensions."
            )
        return torch.cat(output_embeddings, dim=1)

    def project_dense(
        self, feature: DenseFeature, x: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if feature.name not in x:
            raise KeyError(
                f"[EmbeddingLayer Error]:Dense feature '{feature.name}' is missing from input."
            )
        value = x[feature.name].float()
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        else:
            value = value.view(value.size(0), -1)
        expected_in_dim = self.dense_input_dims.get(
            feature.name, max(int(getattr(feature, "input_dim", 1)), 1)
        )
        if value.shape[1] != expected_in_dim:
            raise ValueError(
                f"[EmbeddingLayer Error]:Dense feature '{feature.name}' expects {expected_in_dim} inputs but got {value.shape[1]}."
            )
        if not feature.use_embedding:
            return value
        dense_layer = self.dense_transforms[feature.name]
        return dense_layer(value)

    def compute_output_dim(
        self,
        features: list[DenseFeature | SequenceFeature | SparseFeature] | None = None,
    ) -> int:
        candidates = list(features) if features is not None else self.features
        unique_feats = OrderedDict((feat.name, feat) for feat in candidates)  # type: ignore[assignment]
        dim = 0
        for feat in unique_feats.values():
            if isinstance(feat, DenseFeature):
                in_dim = max(int(getattr(feat, "input_dim", 1)), 1)
                if getattr(feat, "use_embedding", False):
                    emb_dim = getattr(feat, "embedding_dim", None)
                    out_dim = max(int(emb_dim), 1) if emb_dim else in_dim
                else:
                    out_dim = in_dim
                dim += out_dim
            elif isinstance(feat, SequenceFeature) and feat.combiner == "concat":
                dim += feat.embedding_dim * feat.max_len
            else:
                dim += feat.embedding_dim  # type: ignore[assignment]
        return dim

    def get_input_dim(self, features: list[object] | None = None) -> int:
        return self.compute_output_dim(features)  # type: ignore[assignment]

    @property
    def input_dim(self) -> int:
        return self.output_dim


class InputMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: dict[str, torch.Tensor],
        feature: SequenceFeature,
        seq_tensor: torch.Tensor | None = None,
    ):
        if seq_tensor is not None:
            values = seq_tensor
        else:
            values = x[feature.name]
        values = values.long()
        padding_idx = feature.padding_idx if feature.padding_idx is not None else 0
        mask = values != padding_idx

        if mask.dim() == 1:
            # [B] -> [B, 1, 1]
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 2:
            # [B, L] -> [B, 1, L]
            mask = mask.unsqueeze(1)
        elif mask.dim() == 3:
            # [B, 1, L]
            # [B, L, 1]  -> [B, L] -> [B, 1, L]
            if mask.size(1) != 1 and mask.size(2) == 1:
                mask = mask.squeeze(-1).unsqueeze(1)
        else:
            raise ValueError(
                f"InputMask only supports 1D/2D/3D tensors, got shape {values.shape}"
            )
        return mask.float()


class LR(nn.Module):
    def __init__(self, input_dim: int, sigmoid: bool = False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)


class ConcatPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return x.flatten(start_dim=1, end_dim=2)


class AveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            return torch.sum(x, dim=1)
        else:
            return torch.bmm(mask, x).squeeze(1)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_layer: bool = True,
        dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        use_norm: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        if dims is None:
            dims = []
        layers = []
        current_dim = input_dim
        for i_dim in dims:
            layers.append(nn.Linear(current_dim, i_dim))
            if use_norm:
                if norm_type == "batch_norm":
                    # **IMPORTANT** be careful when using BatchNorm1d in distributed training, nextrec does not support sync batch norm now
                    layers.append(nn.BatchNorm1d(i_dim))
                elif norm_type == "layer_norm":
                    layers.append(nn.LayerNorm(i_dim))
                else:
                    raise ValueError(f"Unsupported norm_type: {norm_type}")

            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            current_dim = i_dim
        # output layer
        if output_layer:
            layers.append(nn.Linear(current_dim, 1))
            self.output_dim = 1
        else:
            self.output_dim = current_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FM(nn.Module):
    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int):
        super(CrossLayer, self).__init__()
        self.w = torch.nn.Linear(input_dim, 1, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, x_0: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        x = self.w(x_i) * x_0 + self.b
        return x


class SENETLayer(nn.Module):
    def __init__(self, num_fields: int, reduction_ratio: int = 3):
        super(SENETLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.mean(x, dim=-1, out=None)
        a = self.mlp(z)
        v = x * a.unsqueeze(-1)
        return v


class BiLinearInteractionLayer(nn.Module):
    def __init__(
        self, input_dim: int, num_fields: int, bilinear_type: str = "field_interaction"
    ):
        super(BiLinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(input_dim, input_dim, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList(
                [nn.Linear(input_dim, input_dim, bias=False) for i in range(num_fields)]
            )
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList(
                [
                    nn.Linear(input_dim, input_dim, bias=False)
                    for i, j in combinations(range(num_fields), 2)
                ]
            )
        else:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_emb = torch.split(x, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [
                self.bilinear_layer(v_i) * v_j
                for v_i, v_j in combinations(feature_emb, 2)
            ]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb[i]) * feature_emb[j] for i, j in combinations(range(len(feature_emb)), 2)]  # type: ignore[assignment]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1] for i, v in enumerate(combinations(feature_emb, 2))]  # type: ignore[assignment]
        return torch.cat(bilinear_list, dim=1)


class HadamardInteractionLayer(nn.Module):
    """Hadamard interaction layer for Deep-FiBiNET (0 case in 01/11)."""

    def __init__(self, num_fields: int):
        super().__init__()
        self.num_fields = num_fields

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, D]
        feature_emb = torch.split(x, 1, dim=1)  # list of F tensors [B,1,D]

        hadamard_list = [v_i * v_j for (v_i, v_j) in combinations(feature_emb, 2)]
        return torch.cat(hadamard_list, dim=1)  # [B, num_pairs, D]


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer with Flash Attention support.
    Uses PyTorch 2.0+ scaled_dot_product_attention when available for better performance.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 2,
        dropout: float = 0.0,
        use_residual: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"[MultiHeadSelfAttention Error]: embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_residual = use_residual
        self.dropout_rate = dropout

        self.W_Q = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )  # Query projection
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)  # Key projection
        self.W_V = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )  # Value projection
        self.W_O = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )  # Output projection

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)
        else:
            self.layer_norm = None

        self.dropout = nn.Dropout(dropout)
        # Check if Flash Attention is available
        self.use_flash_attention = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [Batch, Length, Dim]
        B, L, D = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [Batch, Heads, Length, head_dim]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # [B,L], 1=valid, 0=pad
                key_padding_mask = ~attention_mask.bool()
                attn_mask = key_padding_mask[:, None, None, :]
                attn_mask = attn_mask.expand(B, 1, L, L)
            elif attention_mask.dim() == 3:  # [B,L,L], 1=allowed, 0=masked
                attn_mask = (~attention_mask.bool()).view(B, 1, L, L)
            else:
                raise ValueError("attention_mask must be [B,L] or [B,L,L]")
        else:
            attn_mask = None

        if self.use_flash_attention:
            attn = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )  # [B,H,L,dh]
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn = torch.matmul(attn_weights, V)  # [B,H,L,dh]

        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_O(attn)

        if self.use_residual:
            out = out + x
        if self.layer_norm is not None:
            out = self.layer_norm(out)

        if key_padding_mask is not None:
            out = out * (~key_padding_mask).unsqueeze(-1)

        return out


class AttentionPoolingLayer(nn.Module):
    """
    Attention pooling layer for DIN/DIEN
    Computes attention weights between query (candidate item) and keys (user behavior sequence)
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_units: list = [80, 40],
        activation: str = "sigmoid",
        use_softmax: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax
        # Build attention network
        # Input: [query, key, query-key, query*key] -> 4 * embedding_dim
        input_dim = 4 * embedding_dim
        layers = []
        for hidden_unit in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_unit))
            layers.append(activation_layer(activation, emb_size=hidden_unit))
            input_dim = hidden_unit
        layers.append(nn.Linear(input_dim, 1))
        self.attention_net = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        keys_length: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            query: [batch_size, embedding_dim] - candidate item embedding
            keys: [batch_size, seq_len, embedding_dim] - user behavior sequence
            keys_length: [batch_size] - actual length of each sequence (optional)
            mask: [batch_size, seq_len, 1] - mask for padding (optional)
        Returns:
            output: [batch_size, embedding_dim] - attention pooled representation
        """
        batch_size, sequence_length, embedding_dim = keys.shape
        assert query.shape == (
            batch_size,
            embedding_dim,
        ), f"query shape {query.shape} != ({batch_size}, {embedding_dim})"
        if mask is None and keys_length is not None:
            # keys_length: (batch_size,)
            device = keys.device
            seq_range = torch.arange(sequence_length, device=device).unsqueeze(
                0
            )  # (1, sequence_length)
            mask = (seq_range < keys_length.unsqueeze(1)).unsqueeze(-1).float()
        if mask is not None:
            if mask.dim() == 2:
                # (B, L)
                mask = mask.unsqueeze(-1)
            elif (
                mask.dim() == 3
                and mask.shape[1] == 1
                and mask.shape[2] == sequence_length
            ):
                # (B, 1, L) -> (B, L, 1)
                mask = mask.transpose(1, 2)
            elif (
                mask.dim() == 3
                and mask.shape[1] == sequence_length
                and mask.shape[2] == 1
            ):
                pass
            else:
                raise ValueError(
                    f"[AttentionPoolingLayer Error]: Unsupported mask shape: {mask.shape}"
                )
            mask = mask.to(keys.dtype)
        # Expand query to (B, L, D)
        query_expanded = query.unsqueeze(1).expand(-1, sequence_length, -1)
        # [query, key, query-key, query*key] -> (B, L, 4D)
        attention_input = torch.cat(
            [query_expanded, keys, query_expanded - keys, query_expanded * keys],
            dim=-1,
        )
        attention_scores = self.attention_net(attention_input)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # Get attention weights
        if self.use_softmax:
            # softmax over seq_len
            attention_weights = F.softmax(attention_scores, dim=1)  # (B, L, 1)
        else:
            attention_weights = torch.sigmoid(attention_scores)
            if mask is not None:
                attention_weights = attention_weights * mask
        # Weighted sum over keys: (B, L, 1) * (B, L, D) -> (B, D)
        output = torch.sum(attention_weights * keys, dim=1)
        return output


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt(mean(x^2) + eps)
        variance = torch.mean(x**2, dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normalized
