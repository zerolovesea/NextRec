"""
Feature definitions

Date: create on 27/10/2025
Checkpoint: edit on 20/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch

from nextrec.utils.embedding import get_auto_embedding_dim
from nextrec.utils.feature import normalize_to_list


class BaseFeature:
    def __repr__(self):
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class EmbeddingFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        vocab_size: int,
        embedding_name: str = "",
        embedding_dim: int | None = 4,
        padding_idx: int | None = None,
        init_type: str = "normal",
        init_params: dict | None = None,
        l1_reg: float = 0.0,
        l2_reg: float = 1e-5,
        trainable: bool = True,
        pretrained_weight: torch.Tensor | None = None,
        freeze_pretrained: bool = False,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_name = embedding_name or name
        self.embedding_dim = (
            get_auto_embedding_dim(vocab_size)
            if embedding_dim is None
            else embedding_dim
        )

        self.init_type = init_type
        self.init_params = init_params or {}
        self.padding_idx = padding_idx
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.trainable = trainable
        self.pretrained_weight = pretrained_weight
        self.freeze_pretrained = freeze_pretrained


class SequenceFeature(EmbeddingFeature):
    def __init__(
        self,
        name: str,
        vocab_size: int,
        max_len: int = 20,
        embedding_name: str = "",
        embedding_dim: int | None = 4,
        combiner: str = "mean",
        padding_idx: int | None = None,
        init_type: str = "normal",
        init_params: dict | None = None,
        l1_reg: float = 0.0,
        l2_reg: float = 1e-5,
        trainable: bool = True,
        pretrained_weight: torch.Tensor | None = None,
        freeze_pretrained: bool = False,
    ):
        super().__init__(
            name=name,
            vocab_size=vocab_size,
            embedding_name=embedding_name,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_type=init_type,
            init_params=init_params,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            trainable=trainable,
            pretrained_weight=pretrained_weight,
            freeze_pretrained=freeze_pretrained,
        )
        self.max_len = max_len
        self.combiner = combiner


class SparseFeature(EmbeddingFeature):
    pass


class DenseFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        embedding_dim: int | None = 1,
        input_dim: int = 1,
        use_embedding: bool = False,
    ):
        self.name = name
        self.input_dim = max(int(input_dim or 1), 1)
        self.embedding_dim = self.input_dim if embedding_dim is None else embedding_dim
        if use_embedding and self.embedding_dim == 0:
            raise ValueError(
                "[Features Error] DenseFeature: use_embedding=True is incompatible with embedding_dim=0"
            )
        if embedding_dim is not None and embedding_dim > 1:
            self.use_embedding = True
        else:
            self.use_embedding = use_embedding  # user decides for dim <= 1


class FeatureSet:
    def set_all_features(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: str | list[str] | None = None,
        id_columns: str | list[str] | None = None,
    ) -> None:
        self.dense_features = list(dense_features) if dense_features else []
        self.sparse_features = list(sparse_features) if sparse_features else []
        self.sequence_features = list(sequence_features) if sequence_features else []

        self.all_features = (
            self.dense_features + self.sparse_features + self.sequence_features
        )
        self.feature_names = [feat.name for feat in self.all_features]
        self.target_columns = normalize_to_list(target)
        self.id_columns = normalize_to_list(id_columns)

    def set_target_id(
        self,
        target: str | list[str] | None = None,
        id_columns: str | list[str] | None = None,
    ) -> None:
        self.target_columns = normalize_to_list(target)
        self.id_columns = normalize_to_list(id_columns)
