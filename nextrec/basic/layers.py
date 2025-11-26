"""
Layer implementations used across NextRec models.

Date: create on 27/10/2025, update on 19/11/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.activation import activation_layer
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.utils.initializer import get_initializer

Feature = Union[DenseFeature, SparseFeature, SequenceFeature]

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
    "FFM",
    "CEN",
    "CIN",
    "CrossLayer",
    "CrossNetwork",
    "CrossNetV2",
    "CrossNetMix",
    "SENETLayer",
    "BiLinearInteractionLayer",
    "MultiInterestSA",
    "CapsuleNetwork",
    "MultiHeadSelfAttention",
    "AttentionPoolingLayer",
    "DynamicGRU",
    "AUGRU",
]


class PredictionLayer(nn.Module):
    def __init__(
        self,
        task_type: Union[str, Sequence[str]] = "binary",
        task_dims: Union[int, Sequence[int], None] = None,
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
            raise ValueError(
                "task_dims must be None, a single int (shared), or a sequence of the same length as task_type."
            )

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
            x = x.unsqueeze(-1)

        if x.shape[-1] != self.total_dim:
            raise ValueError(
                f"Input last dimension ({x.shape[-1]}) does not match expected total dimension ({self.total_dim})."
            )

        logits = x if self.bias is None else x + self.bias
        outputs: list[torch.Tensor] = []

        for task_type, (start, end) in zip(self.task_types, self._task_slices):
            task_logits = logits[..., start:end]
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
        if task in ['binary','multiclass']:
            return torch.sigmoid
        if task in ['regression']:
            return lambda x: x
        if task in ['multiclass']:
            return lambda x: torch.softmax(x, dim=-1)
        raise ValueError(f"Unsupported task_type '{task_type}'.")


class EmbeddingLayer(nn.Module):
    def __init__(self, features: Sequence[Feature]):
        super().__init__()
        self.features = list(features)
        self.embed_dict = nn.ModuleDict()
        self.dense_transforms = nn.ModuleDict()
        self.dense_input_dims: dict[str, int] = {}

        for feature in self.features:
            if isinstance(feature, (SparseFeature, SequenceFeature)):
                if feature.embedding_name in self.embed_dict:
                    continue

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
                raise TypeError(f"Unsupported feature type: {type(feature)}")
        self.output_dim = self._compute_output_dim()

    def forward(
        self,
        x: dict[str, torch.Tensor],
        features: Sequence[Feature],
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
                    raise ValueError(f"Unknown combiner for {feature.name}: {feature.combiner}")

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
                raise ValueError("No input features found for EmbeddingLayer.")

            return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

        # squeeze_dim=False requires embeddings with identical last dimension
        output_embeddings = list(sparse_embeds)
        if dense_embeds:
            target_dim = None
            if output_embeddings:
                target_dim = output_embeddings[0].shape[-1]
            elif len({emb.shape[-1] for emb in dense_embeds}) == 1:
                target_dim = dense_embeds[0].shape[-1]

            if target_dim is not None:
                aligned_dense = [
                    emb.unsqueeze(1) for emb in dense_embeds if emb.shape[-1] == target_dim
                ]
                output_embeddings.extend(aligned_dense)

        if not output_embeddings:
            raise ValueError(
                "squeeze_dim=False requires at least one sparse/sequence feature or "
                "dense features with identical projected dimensions."
            )

        return torch.cat(output_embeddings, dim=1)

    def _project_dense(self, feature: DenseFeature, x: dict[str, torch.Tensor]) -> torch.Tensor:
        if feature.name not in x:
            raise KeyError(f"Dense feature '{feature.name}' is missing from input.")

        value = x[feature.name].float()
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        else:
            value = value.view(value.size(0), -1)

        dense_layer = self.dense_transforms[feature.name]
        expected_in_dim = self.dense_input_dims[feature.name]
        if value.shape[1] != expected_in_dim:
            raise ValueError(
                f"Dense feature '{feature.name}' expects {expected_in_dim} inputs but "
                f"got {value.shape[1]}."
            )

        return dense_layer(value)

    def _compute_output_dim(self):
        return

class InputMask(nn.Module):
    """Utility module to build sequence masks for pooling layers."""

    def __init__(self):
        super().__init__()

    def forward(self, x, fea, seq_tensor=None):
        values = seq_tensor if seq_tensor is not None else x[fea.name]
        if fea.padding_idx is not None:
            mask = (values.long() != fea.padding_idx)
        else:
            mask = (values.long() != 0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        return mask.unsqueeze(1).float()


class LR(nn.Module):
    """Wide component from Wide&Deep (Cheng et al., 2016)."""

    def __init__(self, input_dim, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)


class ConcatPooling(nn.Module):
    """Concatenates sequence embeddings along the temporal dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x.flatten(start_dim=1, end_dim=2) 


class AveragePooling(nn.Module):
    """Mean pooling with optional padding mask."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
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

    def forward(self, x, mask=None):
        if mask is None:
            return torch.sum(x, dim=1)
        else:
            return torch.bmm(mask, x).squeeze(1)


class MLP(nn.Module):
    """Stacked fully connected layers used in the deep component."""

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
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

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1)**2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class CIN(nn.Module):
    """Compressed Interaction Network from xDeepFM (Lian et al., 2018)."""

    def __init__(self, input_dim, cin_size, split_half=True):
        super().__init__()
        self.num_layers = len(cin_size)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cin_size[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1, stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))

class CrossLayer(nn.Module):
    """Single cross layer used in DCN (Wang et al., 2017)."""

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.w = torch.nn.Linear(input_dim, 1, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, x_0, x_i):
        x = self.w(x_i) * x_0 + self.b
        return x


class CrossNetwork(nn.Module):
    """Stacked Cross Layers from DCN (Wang et al., 2017)."""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

class CrossNetV2(nn.Module):
    """Vector-wise cross network proposed in DCN V2 (Wang et al., 2021)."""
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])


    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            x =x0*self.w[i](x) + self.b[i] + x
        return x

class CrossNetMix(nn.Module):
    """Mixture of low-rank cross experts from DCN V2 (Wang et al., 2021)."""

    def __init__(self, input_dim, num_layers=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        # U: (input_dim, low_rank)
        self.u_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, input_dim, low_rank))) for i in range(self.num_layers)])
        # V: (input_dim, low_rank)
        self.v_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, input_dim, low_rank))) for i in range(self.num_layers)])
        # C: (low_rank, low_rank)
        self.c_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, low_rank, low_rank))) for i in range(self.num_layers)])
        self.gating = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for i in range(self.num_experts)])

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(
            torch.empty(input_dim, 1))) for i in range(self.num_layers)])

    def forward(self, x):
        x_0 = x.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.num_layers):
            output_of_experts = []
            gating_score_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.v_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.c_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.u_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_experts = torch.stack(gating_score_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l

class SENETLayer(nn.Module):
    """Squeeze-and-Excitation block adopted by FiBiNET (Huang et al., 2019)."""

    def __init__(self, num_fields, reduction_ratio=3):
        super(SENETLayer, self).__init__()
        reduced_size = max(1, int(num_fields/ reduction_ratio))
        self.mlp = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(reduced_size, num_fields, bias=False),
                                 nn.ReLU())
    def forward(self, x):
        z = torch.mean(x, dim=-1, out=None)
        a = self.mlp(z)
        v = x*a.unsqueeze(-1)
        return v

class BiLinearInteractionLayer(nn.Module):
    """Bilinear feature interaction from FiBiNET (Huang et al., 2019)."""

    def __init__(self, input_dim, num_fields, bilinear_type = "field_interaction"):
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

    def forward(self, x):
        feature_emb = torch.split(x, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i)*v_j for v_i, v_j in combinations(feature_emb, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb[i])*feature_emb[j] for i,j in combinations(range(len(feature_emb)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0])*v[1] for i,v in enumerate(combinations(feature_emb, 2))]
        return torch.cat(bilinear_list, dim=1)


class MultiInterestSA(nn.Module):
    """Multi-interest self-attention extractor from MIND (Li et al., 2019)."""

    def __init__(self, embedding_dim, interest_num, hidden_dim=None):
        super(MultiInterestSA, self).__init__()
        self.embedding_dim = embedding_dim
        self.interest_num = interest_num
        if hidden_dim == None:
            self.hidden_dim = self.embedding_dim * 4
        self.W1 = torch.nn.Parameter(torch.rand(self.embedding_dim, self.hidden_dim), requires_grad=True)
        self.W2 = torch.nn.Parameter(torch.rand(self.hidden_dim, self.interest_num), requires_grad=True)
        self.W3 = torch.nn.Parameter(torch.rand(self.embedding_dim, self.embedding_dim), requires_grad=True)

    def forward(self, seq_emb, mask=None):
        H = torch.einsum('bse, ed -> bsd', seq_emb, self.W1).tanh()
        if mask != None:
            A = torch.einsum('bsd, dk -> bsk', H, self.W2) + -1.e9 * (1 - mask.float())
            A = F.softmax(A, dim=1)
        else:
            A = F.softmax(torch.einsum('bsd, dk -> bsk', H, self.W2), dim=1)
        A = A.permute(0, 2, 1)
        multi_interest_emb = torch.matmul(A, seq_emb)
        return multi_interest_emb


class CapsuleNetwork(nn.Module):
    """Dynamic routing capsule network used in MIND (Li et al., 2019)."""

    def __init__(self, embedding_dim, seq_len, bilinear_type=2, interest_num=4, routing_times=3, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.embedding_dim = embedding_dim  # h
        self.seq_len = seq_len  # s
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times

        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), nn.ReLU())
        if self.bilinear_type == 0:  # MIND
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim * self.interest_num, bias=False)
        else:
            self.w = nn.Parameter(torch.Tensor(1, self.seq_len, self.interest_num * self.embedding_dim, self.embedding_dim))
            nn.init.xavier_uniform_(self.w)

    def forward(self, item_eb, mask):
        if self.bilinear_type == 0:
            item_eb_hat = self.linear(item_eb)
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:
            u = torch.unsqueeze(item_eb, dim=2)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u, dim=3)

        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.embedding_dim))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.embedding_dim))

        if self.stop_grad:
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        if self.bilinear_type > 0:
            capsule_weight = torch.zeros(item_eb_hat.shape[0],
                                         self.interest_num,
                                         self.seq_len,
                                         device=item_eb.device,
                                         requires_grad=False)
        else:
            capsule_weight = torch.randn(item_eb_hat.shape[0],
                                         self.interest_num,
                                         self.seq_len,
                                         device=item_eb.device,
                                         requires_grad=False)

        for i in range(self.routing_times):  # 动态路由传播3次
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_eb_hat_iter, torch.transpose(interest_capsule, 2, 3).contiguous())
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.embedding_dim))

        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)

        return interest_capsule


class FFM(nn.Module):
    """Field-aware Factorization Machine (Juan et al., 2016)."""

    def __init__(self, num_fields, reduce_sum=True):
        super().__init__()        
        self.num_fields = num_fields
        self.reduce_sum = reduce_sum

    def forward(self, x):
        # compute (non-redundant) second order field-aware feature crossings
        crossed_embeddings = []
        for i in range(self.num_fields-1):
            for j in range(i+1, self.num_fields):
                crossed_embeddings.append(x[:, i, j, :] *  x[:, j, i, :])        
        crossed_embeddings = torch.stack(crossed_embeddings, dim=1)
        
        # if reduce_sum is true, the crossing operation is effectively inner product, other wise Hadamard-product
        if self.reduce_sum:
            crossed_embeddings = torch.sum(crossed_embeddings, dim=-1, keepdim=True)
        return crossed_embeddings


class CEN(nn.Module):
    """Field-attentive interaction network from FAT-DeepFFM (Wang et al., 2020)."""

    def __init__(self, embed_dim, num_field_crosses, reduction_ratio):
        super().__init__()        
        
        # convolution weight (Eq.7 FAT-DeepFFM)
        self.u = torch.nn.Parameter(torch.rand(num_field_crosses, embed_dim), requires_grad=True)

        # two FC layers that computes the field attention
        self.mlp_att = MLP(num_field_crosses, dims=[num_field_crosses//reduction_ratio, num_field_crosses], output_layer=False, activation="relu")
        

    def forward(self, em):  
        # compute descriptor vector (Eq.7 FAT-DeepFFM), output shape [batch_size, num_field_crosses]
        d = F.relu((self.u.squeeze(0) * em).sum(-1))
        
        # compute field attention (Eq.9), output shape [batch_size, num_field_crosses]    
        s = self.mlp_att(d)                             

        # rescale original embedding with field attention (Eq.10), output shape [batch_size, num_field_crosses, embed_dim]
        aem = s.unsqueeze(-1) * em                 
        return aem.flatten(start_dim=1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer from AutoInt (Song et al., 2019)."""
    
    def __init__(self, embedding_dim, num_heads=2, dropout=0.0, use_residual=True):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        
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
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_fields, embedding_dim]
        Returns:
            output: [batch_size, num_fields, embedding_dim]
        """
        batch_size, num_fields, _ = x.shape
        
        # Linear projections
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
        
        # Apply attention to values
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
    
    def __init__(self, embedding_dim, hidden_units=[80, 40], activation='sigmoid', use_softmax=True):
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
    
    def forward(self, query, keys, keys_length=None, mask=None):
        """
        Args:
            query: [batch_size, embedding_dim] - candidate item embedding
            keys: [batch_size, seq_len, embedding_dim] - user behavior sequence
            keys_length: [batch_size] - actual length of each sequence (optional)
            mask: [batch_size, seq_len, 1] - mask for padding (optional)
        Returns:
            output: [batch_size, embedding_dim] - attention pooled representation
        """
        batch_size, seq_len, emb_dim = keys.shape
        
        # Expand query to match sequence length: [batch_size, seq_len, embedding_dim]
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute attention features: [query, key, query-key, query*key]
        attention_input = torch.cat([
            query_expanded,
            keys,
            query_expanded - keys,
            query_expanded * keys
        ], dim=-1)  # [batch_size, seq_len, 4*embedding_dim]
        
        # Compute attention scores
        attention_scores = self.attention_net(attention_input)  # [batch_size, seq_len, 1]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        if self.use_softmax:
            attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        else:
            attention_weights = attention_scores
        
        # Weighted sum of keys
        output = torch.sum(attention_weights * keys, dim=1)  # [batch_size, embedding_dim]
        
        return output


class DynamicGRU(nn.Module):
    """Dynamic GRU unit with auxiliary loss path from DIEN (Zhou et al., 2019)."""
    """
    GRU with dynamic routing for DIEN
    """
    
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # GRU parameters
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, att_scores=None):
        """
        Args:
            x: [batch_size, seq_len, input_size]
            att_scores: [batch_size, seq_len] - attention scores for auxiliary loss
        Returns:
            output: [batch_size, seq_len, hidden_size]
            hidden: [batch_size, hidden_size] - final hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]
            
            # GRU computation
            gi = F.linear(x_t, self.weight_ih, self.bias_ih)
            gh = F.linear(h, self.weight_hh, self.bias_hh)
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)
            
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h = newgate + inputgate * (h - newgate)
            
            outputs.append(h.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
        
        return output, h


class AUGRU(nn.Module):
    """Attention-aware GRU update gate used in DIEN (Zhou et al., 2019)."""
    """
    Attention-based GRU for DIEN
    Uses attention scores to weight the update of hidden states
    """
    
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, att_scores):
        """
        Args:
            x: [batch_size, seq_len, input_size]
            att_scores: [batch_size, seq_len, 1] - attention scores
        Returns:
            output: [batch_size, seq_len, hidden_size]
            hidden: [batch_size, hidden_size] - final hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]
            att_t = att_scores[:, t, :]  # [batch_size, 1]
            
            gi = F.linear(x_t, self.weight_ih, self.bias_ih)
            gh = F.linear(h, self.weight_hh, self.bias_hh)
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)
            
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            
            # Use attention score to control update
            h = (1 - att_t) * h + att_t * newgate
            
            outputs.append(h.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        
        return output, h        
