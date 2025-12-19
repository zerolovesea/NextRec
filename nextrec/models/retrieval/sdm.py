"""
Date: create on 09/11/2025
Checkpoint: edit on 18/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Ying H, Zhuang F, Zhang F, et al. Sequential recommender system based on hierarchical attention networks[C]
//IJCAI. 2018: 3926-3932.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.model import BaseMatchModel


class SDM(BaseMatchModel):
    @property
    def model_name(self) -> str:
        return "SDM"

    @property
    def support_training_modes(self) -> list[str]:
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
        rnn_type: Literal["GRU", "LSTM"] = "GRU",
        rnn_hidden_size: int = 64,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        use_short_term: bool = True,
        use_long_term: bool = True,
        item_dnn_hidden_units: list[int] = [256, 128],
        dnn_activation: str = "relu",
        dnn_dropout: float = 0.0,
        training_mode: Literal["pointwise", "pairwise", "listwise"] = "pointwise",
        num_negative_samples: int = 4,
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

        super(SDM, self).__init__(
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
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.use_short_term = use_short_term
        self.use_long_term = use_long_term
        self.item_dnn_hidden_units = item_dnn_hidden_units

        # User tower
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
                raise ValueError("SDM requires at least one user sequence feature")

            seq_emb_dim = user_sequence_features[0].embedding_dim

            if rnn_type == "GRU":
                self.rnn = nn.GRU(
                    input_size=seq_emb_dim,
                    hidden_size=rnn_hidden_size,
                    num_layers=rnn_num_layers,
                    batch_first=True,
                    dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
                )
            elif rnn_type == "LSTM":
                self.rnn = nn.LSTM(
                    input_size=seq_emb_dim,
                    hidden_size=rnn_hidden_size,
                    num_layers=rnn_num_layers,
                    batch_first=True,
                    dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
                )
            else:
                raise ValueError(f"Unknown RNN type: {rnn_type}")

            user_final_dim = 0
            if use_long_term:
                user_final_dim += rnn_hidden_size
            if use_short_term:
                user_final_dim += seq_emb_dim

            for feat in user_dense_features or []:
                user_final_dim += 1
            for feat in user_sparse_features or []:
                user_final_dim += feat.embedding_dim

            # User DNN to final embedding
            self.user_dnn = MLP(
                input_dim=user_final_dim,
                dims=[rnn_hidden_size * 2, embedding_dim],
                output_layer=False,
                dropout=dnn_dropout,
                activation=dnn_activation,
            )

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
            embedding_attr="user_embedding", include_modules=["rnn", "user_dnn"]
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
        seq_feature = self.user_sequence_features[0]
        seq_input = user_input[seq_feature.name]

        embed = self.user_embedding.embed_dict[seq_feature.embedding_name]
        seq_emb = embed(seq_input.long())  # [batch_size, seq_len, seq_emb_dim]

        if self.rnn_type == "GRU":
            rnn_output, hidden = self.rnn(
                seq_emb
            )  # hidden: [num_layers, batch, hidden_size]
        elif self.rnn_type == "LSTM":
            rnn_output, (hidden, cell) = self.rnn(seq_emb)

        features_list = []

        if self.use_long_term:
            if self.rnn.num_layers > 1:
                long_term = hidden[-1, :, :]  # [batch_size, hidden_size]
            else:
                long_term = hidden.squeeze(0)  # [batch_size, hidden_size]
            features_list.append(long_term)

        if self.use_short_term:
            mask = (
                seq_input != seq_feature.padding_idx
            ).float()  # [batch_size, seq_len]
            seq_lengths = mask.sum(dim=1).long() - 1  # [batch_size]
            seq_lengths = torch.clamp(seq_lengths, min=0)

            batch_size = seq_emb.size(0)
            batch_indices = torch.arange(batch_size, device=seq_emb.device)
            short_term = seq_emb[
                batch_indices, seq_lengths, :
            ]  # [batch_size, seq_emb_dim]
            features_list.append(short_term)

        if self.user_dense_features:
            dense_features = []
            for feat in self.user_dense_features:
                if feat.name in user_input:
                    val = user_input[feat.name].float()
                    if val.dim() == 1:
                        val = val.unsqueeze(1)
                    dense_features.append(val)
            if dense_features:
                features_list.append(torch.cat(dense_features, dim=1))

        if self.user_sparse_features:
            sparse_features = []
            for feat in self.user_sparse_features:
                if feat.name in user_input:
                    embed = self.user_embedding.embed_dict[feat.embedding_name]
                    sparse_emb = embed(user_input[feat.name].long())
                    sparse_features.append(sparse_emb)
            if sparse_features:
                features_list.append(torch.cat(sparse_features, dim=1))

        user_features = torch.cat(features_list, dim=1)
        user_emb = self.user_dnn(user_features)
        user_emb = F.normalize(user_emb, p=2, dim=1)

        return user_emb

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
