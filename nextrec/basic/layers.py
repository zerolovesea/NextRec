"""
Layer implementations used across NextRec models.

Date: create on 27/10/2025
Checkpoint: edit on 29/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""
from __future__ import annotations

from itertools import combinations
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.utils.initializer import get_initializer
from nextrec.basic.activation import activation_layer

__all__ = [
    "PredictionLayer",
    "EmbeddingLayer",
    "InputMask",
    "LR",
    "ConcatPooling",
    "AveragePooling",
    "SumPooling",
    "MLP",
    "FM",
    "CrossLayer",
    "SENETLayer",
    "BiLinearInteractionLayer",
    "MultiHeadSelfAttention",
    "AttentionPoolingLayer",
]

class PredictionLayer(nn.Module):
    def __init__(
        self,
        task_type: str | list[str] = "binary",
        task_dims: int | list[int] | None = None,
        use_bias: bool = True,
        return_logits: bool = False,
    ):
        super().__init__()
        if isinstance(task_type, str):
            self.task_types = [task_type]
        else:
            self.task_types = list(task_type)
        if len(self.task_types) == 0:
            raise ValueError("At least one task_type must be specified.")
        if task_dims is None:
            dims = [1] * len(self.task_types)
        elif isinstance(task_dims, int):
            dims = [task_dims]
        else:
            dims = list(task_dims)
        if len(dims) not in (1, len(self.task_types)):
            raise ValueError("[PredictionLayer Error]: task_dims must be None, a single int (shared), or a sequence of the same length as task_type.")
        if len(dims) == 1 and len(self.task_types) > 1:
            dims = dims * len(self.task_types)
        self.task_dims = dims
        self.total_dim = sum(self.task_dims)
        self.return_logits = return_logits

        # Keep slice offsets per task
        start = 0
        self._task_slices: list[tuple[int, int]] = []
        for dim in self.task_dims:
            if dim < 1:
                raise ValueError("Each task dimension must be >= 1.")
            self._task_slices.append((start, start + dim))
            start += dim
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.total_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1 * total_dim)
        if x.shape[-1] != self.total_dim:
            raise ValueError(f"[PredictionLayer Error]: Input last dimension ({x.shape[-1]}) does not match expected total dimension ({self.total_dim}).")
        logits = x if self.bias is None else x + self.bias
        outputs = []
        for task_type, (start, end) in zip(self.task_types, self._task_slices):
            task_logits = logits[..., start:end] # Extract logits for the current task
            if self.return_logits:
                outputs.append(task_logits)
                continue
            activation = self._get_activation(task_type)
            outputs.append(activation(task_logits))
        result = torch.cat(outputs, dim=-1)
        if result.shape[-1] == 1:
            result = result.squeeze(-1)
        return result

    def _get_activation(self, task_type: str):
        task = task_type.lower()
        if task == 'binary':
            return torch.sigmoid
        if task == 'regression':
            return lambda x: x
        if task == 'multiclass':
            return lambda x: torch.softmax(x, dim=-1)
        raise ValueError(f"[PredictionLayer Error]: Unsupported task_type '{task_type}'.")

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
                    weight = feature.pretrained_weight # type: ignore[assignment]
                    if weight.shape != (feature.vocab_size, feature.embedding_dim): # type: ignore[assignment]
                        raise ValueError(f"[EmbeddingLayer Error]: Pretrained weight for '{feature.embedding_name}' has shape {weight.shape}, expected ({feature.vocab_size}, {feature.embedding_dim}).") # type: ignore[assignment]
                    embedding = nn.Embedding.from_pretrained(embeddings=weight, freeze=feature.freeze_pretrained, padding_idx=feature.padding_idx) # type: ignore[assignment]
                    embedding.weight.requires_grad = feature.trainable and not feature.freeze_pretrained # type: ignore[assignment] 
                else:
                    embedding = nn.Embedding(num_embeddings=feature.vocab_size, embedding_dim=feature.embedding_dim, padding_idx=feature.padding_idx)
                    embedding.weight.requires_grad = feature.trainable
                    initialization = get_initializer(init_type=feature.init_type, activation="linear", param=feature.init_params)
                    initialization(embedding.weight)
                self.embed_dict[feature.embedding_name] = embedding
            elif isinstance(feature, DenseFeature):
                if not feature.use_embedding:
                    self.dense_input_dims[feature.name] = max(int(getattr(feature, "input_dim", 1)), 1)
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
                raise TypeError(f"[EmbeddingLayer Error]: Unsupported feature type: {type(feature)}")
        self.output_dim = self._compute_output_dim()

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
                    raise ValueError(f"[EmbeddingLayer Error]: Unknown combiner for {feature.name}: {feature.combiner}")
                feature_mask = InputMask()(x, feature, seq_input)
                sparse_embeds.append(pooling_layer(seq_emb, feature_mask).unsqueeze(1))

            elif isinstance(feature, DenseFeature):
                dense_embeds.append(self._project_dense(feature, x))

        if squeeze_dim:
            flattened_sparse = [emb.flatten(start_dim=1) for emb in sparse_embeds]
            pieces = []
            if flattened_sparse:
                pieces.append(torch.cat(flattened_sparse, dim=1))
            if dense_embeds:
                pieces.append(torch.cat(dense_embeds, dim=1))
            if not pieces:
                raise ValueError("[EmbeddingLayer Error]: No input features found for EmbeddingLayer.")
            return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        
        # squeeze_dim=False requires embeddings with identical last dimension
        output_embeddings = list(sparse_embeds)
        if dense_embeds:
            if output_embeddings:
                target_dim = output_embeddings[0].shape[-1]
                for emb in dense_embeds:
                    if emb.shape[-1] != target_dim:
                        raise ValueError(f"[EmbeddingLayer Error]: squeeze_dim=False requires all dense feature dimensions to match the embedding dimension of sparse/sequence features ({target_dim}), but got {emb.shape[-1]}.")
                output_embeddings.extend(emb.unsqueeze(1) for emb in dense_embeds)
            else:
                dims = {emb.shape[-1] for emb in dense_embeds}
                if len(dims) != 1:
                    raise ValueError(f"[EmbeddingLayer Error]: squeeze_dim=False requires all dense features to have identical dimensions when no sparse/sequence features are present, but got dimensions {dims}.")
                output_embeddings = [emb.unsqueeze(1) for emb in dense_embeds]
        if not output_embeddings:
            raise ValueError("[EmbeddingLayer Error]: squeeze_dim=False requires at least one sparse/sequence feature or dense features with identical projected dimensions.")
        return torch.cat(output_embeddings, dim=1)

    def _project_dense(self, feature: DenseFeature, x: dict[str, torch.Tensor]) -> torch.Tensor:
        if feature.name not in x:
            raise KeyError(f"[EmbeddingLayer Error]:Dense feature '{feature.name}' is missing from input.")
        value = x[feature.name].float()
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        else:
            value = value.view(value.size(0), -1)
        expected_in_dim = self.dense_input_dims.get(feature.name, max(int(getattr(feature, "input_dim", 1)), 1))
        if value.shape[1] != expected_in_dim:
            raise ValueError(f"[EmbeddingLayer Error]:Dense feature '{feature.name}' expects {expected_in_dim} inputs but got {value.shape[1]}.")
        if not feature.use_embedding:
            return value        
        dense_layer = self.dense_transforms[feature.name]
        return dense_layer(value)

    def _compute_output_dim(self, features: list[DenseFeature | SequenceFeature | SparseFeature] | None = None) -> int:
        """
        Compute flattened embedding dimension for provided features or all tracked features.
        Deduplicates by feature name to avoid double-counting shared embeddings.
        """
        candidates = list(features) if features is not None else self.features
        unique_feats = OrderedDict((feat.name, feat) for feat in candidates) # type: ignore[assignment]
        dim = 0
        for feat in unique_feats.values():
            if isinstance(feat, DenseFeature):
                in_dim = max(int(getattr(feat, "input_dim", 1)), 1)
                emb_dim = getattr(feat, "embedding_dim", None)
                out_dim = max(int(emb_dim), 1) if emb_dim else in_dim
                dim += out_dim
            elif isinstance(feat, SequenceFeature) and feat.combiner == "concat":
                dim += feat.embedding_dim * feat.max_len
            else:
                dim += feat.embedding_dim # type: ignore[assignment]
        return dim

    def get_input_dim(self, features: list[object] | None = None) -> int:
        return self._compute_output_dim(features) # type: ignore[assignment]

    @property
    def input_dim(self) -> int:
        return self.output_dim

class InputMask(nn.Module):
    """Utility module to build sequence masks for pooling layers."""
    def __init__(self):
        super().__init__()

    def forward(self, x: dict[str, torch.Tensor], feature: SequenceFeature, seq_tensor: torch.Tensor | None = None):
        values = seq_tensor if seq_tensor is not None else x[feature.name]
        if feature.padding_idx is not None:
            mask = (values.long() != feature.padding_idx)
        else:
            mask = (values.long() != 0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        return mask.unsqueeze(1).float()

class LR(nn.Module):
    """Wide component from Wide&Deep (Cheng et al., 2016)."""
    def __init__(
            self, 
            input_dim: int, 
            sigmoid: bool = False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)

class ConcatPooling(nn.Module):
    """Concatenates sequence embeddings along the temporal dimension."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x.flatten(start_dim=1, end_dim=2) 

class AveragePooling(nn.Module):
    """Mean pooling with optional padding mask."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)

class SumPooling(nn.Module):
    """Sum pooling with optional padding mask."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return torch.sum(x, dim=1)
        else:
            return torch.bmm(mask, x).squeeze(1)

class MLP(nn.Module):
    """Stacked fully connected layers used in the deep component."""
    def __init__(
            self, 
            input_dim: int, 
            output_layer: bool = True, 
            dims: list[int] | None = None, 
            dropout: float = 0.0, 
            activation: str = "relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class FM(nn.Module):
    """Factorization Machine (Rendle, 2010) second-order interaction term."""
    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_of_sum = torch.sum(x, dim=1)**2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class CrossLayer(nn.Module):
    """Single cross layer used in DCN (Wang et al., 2017)."""
    def __init__(self, input_dim: int):
        super(CrossLayer, self).__init__()
        self.w = torch.nn.Linear(input_dim, 1, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, x_0: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        x = self.w(x_i) * x_0 + self.b
        return x

class SENETLayer(nn.Module):
    """Squeeze-and-Excitation block adopted by FiBiNET (Huang et al., 2019)."""
    def __init__(
            self, 
            num_fields: int, 
            reduction_ratio: int = 3):
        super(SENETLayer, self).__init__()
        reduced_size = max(1, int(num_fields/ reduction_ratio))
        self.mlp = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(reduced_size, num_fields, bias=False),
                                 nn.ReLU())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.mean(x, dim=-1, out=None)
        a = self.mlp(z)
        v = x*a.unsqueeze(-1)
        return v

class BiLinearInteractionLayer(nn.Module):
    """Bilinear feature interaction from FiBiNET (Huang et al., 2019)."""
    def __init__(
            self, 
            input_dim: int, 
            num_fields: int, 
            bilinear_type: str = "field_interaction"):
        super(BiLinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(input_dim, input_dim, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for i,j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_emb = torch.split(x, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i)*v_j for v_i, v_j in combinations(feature_emb, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb[i])*feature_emb[j] for i,j in combinations(range(len(feature_emb)), 2)]  # type: ignore[assignment]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0])*v[1] for i,v in enumerate(combinations(feature_emb, 2))] # type: ignore[assignment]
        return torch.cat(bilinear_list, dim=1)

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer from AutoInt (Song et al., 2019)."""
    def __init__(
            self, 
            embedding_dim: int, 
            num_heads: int = 2, 
            dropout: float = 0.0, 
            use_residual: bool = True):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"[MultiHeadSelfAttention Error]: embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_residual = use_residual
        self.W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.use_residual:
            self.W_Res = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, num_fields, embedding_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_fields, embedding_dim)
        """
        batch_size, num_fields, _ = x.shape
        Q = self.W_Q(x)  # [batch_size, num_fields, embedding_dim]
        K = self.W_K(x)
        V = self.W_V(x)
        # Split into multiple heads: [batch_size, num_heads, num_fields, head_dim]
        Q = Q.view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, num_fields, head_dim]
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, num_fields, self.embedding_dim)
        # Residual connection
        if self.use_residual:
            output = attention_output + self.W_Res(x)
        else:
            output = attention_output
        output = F.relu(output)
        return output

class AttentionPoolingLayer(nn.Module):
    """
    Attention pooling layer for DIN/DIEN
    Computes attention weights between query (candidate item) and keys (user behavior sequence)
    """
    def __init__(
            self, 
            embedding_dim: int, 
            hidden_units: list = [80, 40], 
            activation: str ='sigmoid', 
            use_softmax: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax
        # Build attention network
        # Input: [query, key, query-key, query*key] -> 4 * embedding_dim
        input_dim = 4 * embedding_dim
        layers = []
        for hidden_unit in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_unit))
            layers.append(activation_layer(activation))
            input_dim = hidden_unit
        layers.append(nn.Linear(input_dim, 1))
        self.attention_net = nn.Sequential(*layers)
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, keys_length: torch.Tensor | None = None, mask: torch.Tensor | None = None):
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
        assert query.shape == (batch_size, embedding_dim), f"query shape {query.shape} != ({batch_size}, {embedding_dim})"
        if mask is None and keys_length is not None:
            # keys_length: (batch_size,)
            device = keys.device
            seq_range = torch.arange(sequence_length, device=device).unsqueeze(0)  # (1, sequence_length)
            mask = (seq_range < keys_length.unsqueeze(1)).unsqueeze(-1).float()
        if mask is not None:
            if mask.dim() == 2:
                # (B, L)
                mask = mask.unsqueeze(-1)
            elif mask.dim() == 3 and mask.shape[1] == 1 and mask.shape[2] == sequence_length:
                # (B, 1, L) -> (B, L, 1)
                mask = mask.transpose(1, 2)
            elif mask.dim() == 3 and mask.shape[1] == sequence_length and mask.shape[2] == 1:
                pass
            else:
                raise ValueError(f"[AttentionPoolingLayer Error]: Unsupported mask shape: {mask.shape}")
            mask = mask.to(keys.dtype)
        # Expand query to (B, L, D)
        query_expanded = query.unsqueeze(1).expand(-1, sequence_length, -1)
        # [query, key, query-key, query*key] -> (B, L, 4D)
        attention_input = torch.cat([query_expanded, keys, query_expanded - keys, query_expanded * keys], dim=-1,)
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
