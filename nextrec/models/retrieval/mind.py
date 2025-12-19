"""
Date: create on 09/11/2025
Checkpoint: edit on 18/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Li C, Liu Z, Wu M, et al. Multi-interest network with dynamic routing for recommendation at Tmall[C]
//Proceedings of the 28th ACM international conference on information and knowledge management. 2019: 2615-2623.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.model import BaseMatchModel


class MultiInterestSA(nn.Module):
    """Multi-interest self-attention extractor from MIND (Li et al., 2019)."""

    def __init__(self, embedding_dim, interest_num, hidden_dim=None):
        super(MultiInterestSA, self).__init__()
        self.embedding_dim = embedding_dim
        self.interest_num = interest_num
        if hidden_dim is None:
            self.hidden_dim = self.embedding_dim * 4
        self.W1 = torch.nn.Parameter(
            torch.rand(self.embedding_dim, self.hidden_dim), requires_grad=True
        )
        self.W2 = torch.nn.Parameter(
            torch.rand(self.hidden_dim, self.interest_num), requires_grad=True
        )
        self.W3 = torch.nn.Parameter(
            torch.rand(self.embedding_dim, self.embedding_dim), requires_grad=True
        )

    def forward(self, seq_emb, mask=None):
        H = torch.einsum("bse, ed -> bsd", seq_emb, self.W1).tanh()
        if mask is not None:
            A = torch.einsum("bsd, dk -> bsk", H, self.W2) + -1.0e9 * (1 - mask.float())
            A = F.softmax(A, dim=1)
        else:
            A = F.softmax(torch.einsum("bsd, dk -> bsk", H, self.W2), dim=1)
        A = A.permute(0, 2, 1)
        multi_interest_emb = torch.matmul(A, seq_emb)
        return multi_interest_emb


class CapsuleNetwork(nn.Module):
    """Dynamic routing capsule network used in MIND (Li et al., 2019)."""

    def __init__(
        self,
        embedding_dim,
        seq_len,
        bilinear_type=2,
        interest_num=4,
        routing_times=3,
        relu_layer=False,
    ):
        super(CapsuleNetwork, self).__init__()
        self.embedding_dim = embedding_dim  # h
        self.seq_len = seq_len  # s
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times

        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), nn.ReLU()
        )
        if self.bilinear_type == 0:  # MIND
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(
                self.embedding_dim, self.embedding_dim * self.interest_num, bias=False
            )
        else:
            self.w = nn.Parameter(
                torch.Tensor(
                    1,
                    self.seq_len,
                    self.interest_num * self.embedding_dim,
                    self.embedding_dim,
                )
            )
            nn.init.xavier_uniform_(self.w)

    def forward(self, item_eb, mask):
        if self.bilinear_type == 0:
            item_eb_hat = self.linear(item_eb)
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:
            u = torch.unsqueeze(item_eb, dim=2)
            item_eb_hat = torch.sum(self.w[:, : self.seq_len, :, :] * u, dim=3)

        item_eb_hat = torch.reshape(
            item_eb_hat, (-1, self.seq_len, self.interest_num, self.embedding_dim)
        )
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(
            item_eb_hat, (-1, self.interest_num, self.seq_len, self.embedding_dim)
        )

        if self.stop_grad:
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        if self.bilinear_type > 0:
            capsule_weight = torch.zeros(
                item_eb_hat.shape[0],
                self.interest_num,
                self.seq_len,
                device=item_eb.device,
                requires_grad=False,
            )
        else:
            capsule_weight = torch.randn(
                item_eb_hat.shape[0],
                self.interest_num,
                self.seq_len,
                device=item_eb.device,
                requires_grad=False,
            )

        for i in range(self.routing_times):  # 动态路由传播3次
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(
                torch.eq(atten_mask, 0), paddings, capsule_softmax_weight
            )
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = torch.matmul(
                    capsule_softmax_weight, item_eb_hat_iter
                )
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(
                    item_eb_hat_iter,
                    torch.transpose(interest_capsule, 2, 3).contiguous(),
                )
                delta_weight = torch.reshape(
                    delta_weight, (-1, self.interest_num, self.seq_len)
                )
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(
            interest_capsule, (-1, self.interest_num, self.embedding_dim)
        )

        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)

        return interest_capsule


class MIND(BaseMatchModel):
    @property
    def model_name(self) -> str:
        return "MIND"

    @property
    def support_training_modes(self) -> list[str]:
        """MIND only supports pointwise training mode"""
        return ["pointwise"]

    def __init__(
        self,
        user_dense_features: list[DenseFeature] | None = None,
        user_sparse_features: list[SparseFeature] | None = None,
        user_sequence_features: list[SequenceFeature] | None = None,
        item_dense_features: list[DenseFeature] | None = None,
        item_sparse_features: list[SparseFeature] | None = None,
        item_sequence_features: list[SequenceFeature] | None = None,
        embedding_dim: int = 64,
        num_interests: int = 4,
        capsule_bilinear_type: int = 2,
        routing_times: int = 3,
        relu_layer: bool = False,
        item_dnn_hidden_units: list[int] = [256, 128],
        dnn_activation: str = "relu",
        dnn_dropout: float = 0.0,
        training_mode: Literal["pointwise", "pairwise", "listwise"] = "pointwise",
        num_negative_samples: int = 100,
        temperature: float = 1.0,
        similarity_metric: Literal["dot", "cosine", "euclidean"] = "dot",
        device: str = "cpu",
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        early_stop_patience: int = 20,
        optimizer: str | torch.optim.Optimizer = "adam",
        optimizer_params: dict | None = None,
        scheduler: (
            str
            | torch.optim.lr_scheduler._LRScheduler
            | type[torch.optim.lr_scheduler._LRScheduler]
            | None
        ) = None,
        scheduler_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        **kwargs,
    ):

        super(MIND, self).__init__(
            user_dense_features=user_dense_features,
            user_sparse_features=user_sparse_features,
            user_sequence_features=user_sequence_features,
            item_dense_features=item_dense_features,
            item_sparse_features=item_sparse_features,
            item_sequence_features=item_sequence_features,
            training_mode=training_mode,
            num_negative_samples=num_negative_samples,
            temperature=temperature,
            similarity_metric=similarity_metric,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            early_stop_patience=early_stop_patience,
            **kwargs,
        )

        self.embedding_dim = embedding_dim
        self.num_interests = num_interests
        self.item_dnn_hidden_units = item_dnn_hidden_units

        user_features = []
        if user_dense_features:
            user_features.extend(user_dense_features)
        if user_sparse_features:
            user_features.extend(user_sparse_features)
        if user_sequence_features:
            user_features.extend(user_sequence_features)

        if len(user_features) > 0:
            self.user_embedding = EmbeddingLayer(user_features)

            if not user_sequence_features or len(user_sequence_features) == 0:
                raise ValueError("MIND requires at least one user sequence feature")

            seq_max_len = (
                user_sequence_features[0].max_len
                if user_sequence_features[0].max_len
                else 50
            )
            seq_embedding_dim = user_sequence_features[0].embedding_dim

            # Capsule Network for multi-interest extraction
            self.capsule_network = CapsuleNetwork(
                embedding_dim=seq_embedding_dim,
                seq_len=seq_max_len,
                bilinear_type=capsule_bilinear_type,
                interest_num=num_interests,
                routing_times=routing_times,
                relu_layer=relu_layer,
            )

            if seq_embedding_dim != embedding_dim:
                self.interest_projection = nn.Linear(
                    seq_embedding_dim, embedding_dim, bias=False
                )
                nn.init.xavier_uniform_(self.interest_projection.weight)
            else:
                self.interest_projection = None

        # Item tower
        item_features = []
        if item_dense_features:
            item_features.extend(item_dense_features)
        if item_sparse_features:
            item_features.extend(item_sparse_features)
        if item_sequence_features:
            item_features.extend(item_sequence_features)

        if len(item_features) > 0:
            self.item_embedding = EmbeddingLayer(item_features)

            item_input_dim = 0
            for feat in item_dense_features or []:
                item_input_dim += 1
            for feat in item_sparse_features or []:
                item_input_dim += feat.embedding_dim
            for feat in item_sequence_features or []:
                item_input_dim += feat.embedding_dim

            # Item DNN
            if len(item_dnn_hidden_units) > 0:
                item_dnn_units = item_dnn_hidden_units + [embedding_dim]
                self.item_dnn = MLP(
                    input_dim=item_input_dim,
                    dims=item_dnn_units,
                    output_layer=False,
                    dropout=dnn_dropout,
                    activation=dnn_activation,
                )
            else:
                self.item_dnn = None

        self.register_regularization_weights(
            embedding_attr="user_embedding", include_modules=["capsule_network"]
        )
        self.register_regularization_weights(
            embedding_attr="item_embedding",
            include_modules=["item_dnn"] if self.item_dnn else [],
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            loss=loss,
            loss_params=loss_params,
        )

        self.to(device)

    def user_tower(self, user_input: dict) -> torch.Tensor:
        """
        User tower with multi-interest extraction

        Returns:
            user_interests: [batch_size, num_interests, embedding_dim]
        """
        seq_feature = self.user_sequence_features[0]
        seq_input = user_input[seq_feature.name]

        embed = self.user_embedding.embed_dict[seq_feature.embedding_name]
        seq_emb = embed(seq_input.long())  # [batch_size, seq_len, embedding_dim]

        mask = (seq_input != seq_feature.padding_idx).float()  # [batch_size, seq_len]

        multi_interests = self.capsule_network(
            seq_emb, mask
        )  # [batch_size, num_interests, seq_embedding_dim]

        if self.interest_projection is not None:
            multi_interests = self.interest_projection(
                multi_interests
            )  # [batch_size, num_interests, embedding_dim]

        # L2 normalization
        multi_interests = F.normalize(multi_interests, p=2, dim=-1)

        return multi_interests

    def item_tower(self, item_input: dict) -> torch.Tensor:
        """Item tower"""
        all_item_features = (
            self.item_dense_features
            + self.item_sparse_features
            + self.item_sequence_features
        )
        item_emb = self.item_embedding(item_input, all_item_features, squeeze_dim=True)

        if self.item_dnn is not None:
            item_emb = self.item_dnn(item_emb)

        # L2 normalization
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return item_emb

    def compute_similarity(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> torch.Tensor:
        item_emb_expanded = item_emb.unsqueeze(1)

        if self.similarity_metric == "dot":
            similarities = torch.sum(user_emb * item_emb_expanded, dim=-1)
        elif self.similarity_metric == "cosine":
            similarities = F.cosine_similarity(user_emb, item_emb_expanded, dim=-1)
        elif self.similarity_metric == "euclidean":
            similarities = -torch.sum((user_emb - item_emb_expanded) ** 2, dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        max_similarity, _ = torch.max(similarities, dim=1)  # [batch_size]
        max_similarity = max_similarity / self.temperature

        return max_similarity
