"""
Task head implementations for NextRec models.

Date: create on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.layers import PredictionLayer


class TaskHead(nn.Module):
    """
    Unified task head for ranking/regression/multi-task outputs.

    This wraps PredictionLayer so models can depend on a "Head" abstraction
    without changing their existing forward signatures.
    """

    def __init__(
        self,
        task_type: str | list[str] = "binary",
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


class RetrievalHead(nn.Module):
    """
    Retrieval head for two-tower models.

    It computes similarity for pointwise training/inference, and returns
    raw embeddings for in-batch negative sampling in pairwise/listwise modes.
    """

    def __init__(
        self,
        similarity_metric: Literal["dot", "cosine", "euclidean"] = "dot",
        temperature: float = 1.0,
        training_mode: Literal["pointwise", "pairwise", "listwise"] = "pointwise",
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.training_mode = training_mode
        self.apply_sigmoid = apply_sigmoid

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        similarity_fn=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.training_mode in {"pairwise", "listwise"}:
            return user_emb, item_emb

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
        if self.training_mode == "pointwise" and self.apply_sigmoid:
            return torch.sigmoid(similarity)
        return similarity
