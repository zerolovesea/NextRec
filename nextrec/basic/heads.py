"""
Task head implementations for NextRec models.

Date: create on 23/12/2025
Checkpoint: edit on 21/03/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.layers import PredictionLayer
from nextrec.utils.types import TaskTypeName


class TaskHead(nn.Module):
    """
    Task head for pointwise binary/regression and multi-task outputs.

    Args:
        task_type: The type of task(s) this head is responsible for.
            Supported task types are "binary" and "regression".
            Vocabulary-sized generative matching does not use TaskHead;
            use GenerativeMatchingHead instead.
        task_dims: The dimensionality of each task's output.
        use_bias: Whether to include a bias term in the prediction layer.
        return_logits: Whether to return raw logits or apply activation.
    """

    def __init__(
        self,
        task_type: TaskTypeName | list[TaskTypeName] = "binary",
        task_dims: int | list[int] | None = None,
        use_bias: bool = True,
        return_logits: bool = False,
    ) -> None:
        super().__init__()
        self.prediction = PredictionLayer(
            task_type=task_type,
            task_dims=task_dims,
            use_bias=use_bias,
            return_logits=return_logits,
        )
        # Expose commonly used attributes for compatibility with PredictionLayer.
        self.task_types = self.prediction.task_types
        self.task_dims = self.prediction.task_dims
        self.task_slices = self.prediction.task_slices
        self.total_dim = self.prediction.total_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction(x)


class GenerativeMatchingHead(nn.Module):
    """
    Head for generative matching over a full item vocabulary.

    Unlike TaskHead, this head treats the last dimension as a vocabulary/class
    axis rather than a per-task output slice, so it is suitable for
    generative matching trained with autoregressive classification loss such
    as CrossEntropyLoss over the item vocabulary.

    Args:
        vocab_size: Number of candidate items/classes.
        use_bias: Whether to include an additive bias per item.
        return_logits: Whether to return raw logits or normalized probabilities.
    """

    def __init__(
        self,
        vocab_size: int,
        use_bias: bool = True,
        return_logits: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError(f"[GenerativeMatchingHead Error] vocab_size must be >= 2, got {vocab_size}.")
        self.vocab_size = int(vocab_size)
        self.return_logits = return_logits
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if not torch.onnx.is_in_onnx_export() and x.shape[-1] != self.vocab_size:
            raise ValueError(
                f"[GenerativeMatchingHead Error] Input last dimension ({x.shape[-1]}) does not match vocab_size ({self.vocab_size})."
            )
        logits = x if self.bias is None else x + self.bias
        if self.return_logits:
            return logits
        return F.softmax(logits, dim=-1)


class MatchingHead(nn.Module):
    """
    Matching head for two-tower models.

    It computes similarity for pointwise training/inference and returns
    a structured payload containing embeddings plus scores.

    Args:
        similarity_metric: The metric used to compute similarity between embeddings.
        temperature: Scaling factor for similarity scores.
        training_mode: The training mode, which can be pointwise, pairwise, or listwise.
        sampling_mode: The sampling mode for pairwise/listwise matching, either explicit or inbatch.
        apply_sigmoid: Whether to apply sigmoid activation to the similarity scores in pointwise mode.
    """

    def __init__(
        self,
        similarity_metric: Literal["dot", "cosine", "euclidean"] = "dot",
        temperature: float = 1.0,
        training_mode: Literal["pointwise", "pairwise", "listwise"] = "pointwise",
        sampling_mode: Literal["explicit", "inbatch"] = "explicit",
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.training_mode = training_mode
        self.sampling_mode = sampling_mode
        self.apply_sigmoid = apply_sigmoid

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        similarity_fn=None,
    ) -> dict[str, torch.Tensor]:
        if similarity_fn is not None:
            similarity = similarity_fn(user_emb, item_emb)
        else:
            if user_emb.dim() == 2 and item_emb.dim() == 3:
                user_emb = user_emb.unsqueeze(1)

            if self.similarity_metric == "dot":
                similarity = torch.sum(user_emb * item_emb, dim=-1)
            elif self.similarity_metric == "cosine":
                similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
            elif self.similarity_metric == "euclidean":
                similarity = -torch.sum((user_emb - item_emb) ** 2, dim=-1)
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

            similarity = similarity / self.temperature
        raw_scores = similarity
        if self.training_mode == "pointwise" and self.apply_sigmoid:
            similarity = torch.sigmoid(similarity)
        return {
            "user_emb": user_emb,
            "item_emb": item_emb,
            "raw_scores": raw_scores,
            "logits": raw_scores,
            "scores": similarity,
        }
