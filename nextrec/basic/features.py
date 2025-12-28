"""
Feature definitions for NextRec models.

Date: create on 27/10/2025
Checkpoint: edit on 27/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch

from typing import Literal

from nextrec.utils.embedding import get_auto_embedding_dim
from nextrec.utils.feature import to_list


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
        embedding_dim: int | None = None,
        padding_idx: int = 0,
        init_type: Literal[
            "normal",
            "uniform",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "orthogonal",
        ] = "normal",
        init_params: dict | None = None,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
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
        max_len: int = 50,
        embedding_name: str = "",
        embedding_dim: int | None = None,
        combiner: Literal[
            "mean",
            "sum",
            "concat",
            "dot_attention",
            "self_attention",
        ] = "mean",
        padding_idx: int = 0,
        init_type: Literal[
            "normal",
            "uniform",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "orthogonal",
        ] = "normal",
        init_params: dict | None = None,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        trainable: bool = True,
        pretrained_weight: torch.Tensor | None = None,
        freeze_pretrained: bool = False,
    ):
        """
        Sequence feature for variable-length categorical id sequences.

        Args:
            name: Feature name used as input key.
            vocab_size: Number of unique ids in the sequence vocabulary.
            max_len: Maximum sequence length for padding/truncation.
            embedding_name: Shared embedding table name. Defaults to ``name``.
            embedding_dim: Embedding dimension. Set to ``None`` for auto sizing.
            combiner: Pooling method for sequence embeddings, e.g. ``"mean"`` or ``"sum"``.
            padding_idx: Index used for padding tokens.
            init_type: Embedding initializer type.
            init_params: Initializer parameters.
            l1_reg: L1 regularization weight on embedding.
            l2_reg: L2 regularization weight on embedding.
            trainable: Whether the embedding is trainable. [TODO] This is for representation learning.
            pretrained_weight: Optional pretrained embedding weights. [TODO] This is for representation learning.
            freeze_pretrained: If True, keep pretrained weights frozen. [TODO] This is for representation learning.
        """
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

    def __init__(
        self,
        name: str,
        vocab_size: int,
        embedding_name: str = "",
        embedding_dim: int | None = None,
        padding_idx: int = 0,
        init_type: Literal[
            "normal",
            "uniform",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "orthogonal",
        ] = "normal",
        init_params: dict | None = None,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        trainable: bool = True,
        pretrained_weight: torch.Tensor | None = None,
        freeze_pretrained: bool = False,
    ):
        """
        Sparse feature for categorical ids.

        Args:
            name: Feature name used as input key.
            vocab_size: Number of unique categorical ids.
            embedding_name: Shared embedding table name. Defaults to ``name``.
            embedding_dim: Embedding dimension. Set to ``None`` for auto sizing.
            padding_idx: Index used for padding tokens.
            init_type: Embedding initializer type.
            init_params: Initializer parameters.
            l1_reg: L1 regularization weight on embedding.
            l2_reg: L2 regularization weight on embedding.
            trainable: Whether the embedding is trainable.
            pretrained_weight: Optional pretrained embedding weights.
            freeze_pretrained: If True, keep pretrained weights frozen.
        """
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


class DenseFeature(BaseFeature):

    def __init__(
        self,
        name: str,
        input_dim: int = 1,
        proj_dim: int | None = 0,
        use_projection: bool = False,
        trainable: bool = True,
        pretrained_weight: torch.Tensor | None = None,
        freeze_pretrained: bool = False,
    ):
        """
        Dense feature for continuous values.

        Args:
            name: Feature name used as input key.
            input_dim: Input dimension for continuous values.
            proj_dim: Projection dimension. If None or 0, no projection is applied.
            use_projection: Whether to project inputs to higher dimension.
            trainable: Whether the projection is trainable.
            pretrained_weight: Optional pretrained projection weights.
            freeze_pretrained: If True, keep pretrained weights frozen.
        """
        self.name = name
        self.input_dim = max(int(input_dim), 1)
        self.proj_dim = self.input_dim if proj_dim is None else proj_dim
        if use_projection and self.proj_dim == 0:
            raise ValueError(
                "[Features Error] DenseFeature: use_projection=True is incompatible with proj_dim=0"
            )
        if proj_dim is not None and proj_dim > 1:
            self.use_projection = True
        else:
            self.use_projection = use_projection
        self.embedding_dim = (
            self.input_dim if not self.use_projection else self.proj_dim
        )  # for compatibility

        self.trainable = trainable
        self.pretrained_weight = pretrained_weight
        self.freeze_pretrained = freeze_pretrained


class FeatureSet:
    def set_all_features(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: str | list[str] | None = None,
        id_columns: str | list[str] | None = None,
    ):
        self.dense_features = list(dense_features) if dense_features else []
        self.sparse_features = list(sparse_features) if sparse_features else []
        self.sequence_features = list(sequence_features) if sequence_features else []

        self.all_features = (
            self.dense_features + self.sparse_features + self.sequence_features
        )
        self.feature_names = [feat.name for feat in self.all_features]
        self.target_columns = to_list(target)
        self.id_columns = to_list(id_columns)

    def set_target_id(
        self,
        target: str | list[str] | None = None,
        id_columns: str | list[str] | None = None,
    ) -> None:
        self.target_columns = to_list(target)
        self.id_columns = to_list(id_columns)
