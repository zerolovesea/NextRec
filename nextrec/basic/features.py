"""
Feature definitions

Date: create on 27/10/2025
Author: Yang Zhou,zyaztec@gmail.com
"""
from __future__ import annotations
from typing import List, Sequence, Optional    
from nextrec.utils.embedding import get_auto_embedding_dim

class BaseFeature(object):
    def __repr__(self):
        params = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") 
        }
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"

class SequenceFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        vocab_size: int,
        max_len: int = 20,
        embedding_name: str = '',
        embedding_dim: int | None = 4,
        combiner: str = "mean",
        padding_idx: int | None = None,
        init_type: str='normal',
        init_params: dict|None = None,
        l1_reg: float = 0.0,
        l2_reg: float = 1e-5,
        trainable: bool = True,
    ):

        self.name = name
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_name = embedding_name or name
        self.embedding_dim = embedding_dim or get_auto_embedding_dim(vocab_size)

        self.init_type = init_type
        self.init_params = init_params or {}
        self.combiner = combiner
        self.padding_idx = padding_idx
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.trainable = trainable
    
class SparseFeature(BaseFeature):
    def __init__(self, 
                 name: str, 
                 vocab_size: int, 
                 embedding_name: str = '', 
                 embedding_dim: int | None  = 4, 
                 padding_idx: int | None = None,
                 init_type: str='normal',
                 init_params: dict|None = None,
                 l1_reg: float = 0.0,                 
                 l2_reg: float = 1e-5,
                 trainable: bool = True):
        
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_name = embedding_name or name
        self.embedding_dim = embedding_dim or get_auto_embedding_dim(vocab_size)

        self.init_type = init_type
        self.init_params = init_params or {}
        self.padding_idx = padding_idx
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.trainable = trainable

class DenseFeature(BaseFeature):
    def __init__(self, 
                 name: str, 
                 embedding_dim: int = 1):

        self.name = name
        self.embedding_dim = embedding_dim


class FeatureSpecMixin:
    """
    Mixin that normalizes dense/sparse/sequence feature lists and target/id columns.
    """

    def _set_feature_config(
        self,
        dense_features: Sequence[DenseFeature] | None = None,
        sparse_features: Sequence[SparseFeature] | None = None,
        sequence_features: Sequence[SequenceFeature] | None = None,
        target: str | Sequence[str] | None = None,
        id_columns: str | Sequence[str] | None = None,
    ) -> None:
        self.dense_features: List[DenseFeature] = list(dense_features) if dense_features else []
        self.sparse_features: List[SparseFeature] = list(sparse_features) if sparse_features else []
        self.sequence_features: List[SequenceFeature] = list(sequence_features) if sequence_features else []

        self.all_features = self.dense_features + self.sparse_features + self.sequence_features
        self.feature_names = [feat.name for feat in self.all_features]
        self.target_columns = self._normalize_to_list(target)
        self.id_columns = self._normalize_to_list(id_columns)

    def _set_target_id_config(
        self,
        target: str | Sequence[str] | None = None,
        id_columns: str | Sequence[str] | None = None,
    ) -> None:
        self.target_columns = self._normalize_to_list(target)
        self.id_columns = self._normalize_to_list(id_columns)

    @staticmethod
    def _normalize_to_list(value: str | Sequence[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

