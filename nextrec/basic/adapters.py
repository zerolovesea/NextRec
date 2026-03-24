"""
Training adapters for NextRec models.

These adapters provide flexible interfaces between NextRec's training loop
and various model architectures and loss functions, enabling support for
pointwise, pairwise, listwise, representation, and two-tower models with in-batch negatives.

Date: create on 22/03/2026
Checkpoint: edit on 22/03/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.loss.listwise import InfoNCELoss, SampledSoftmaxLoss
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss

if TYPE_CHECKING:
    from nextrec.basic.model import BaseModel


class TrainingAdapter:
    # if the adapter requires labels for loss computation
    def needs_labels(self, model: "BaseModel") -> bool:
        return True

    def forward(self, model: "BaseModel", X_input: dict[str, torch.Tensor]):
        raw_output = model.call_model(X_input)
        return model.format_model_output(raw_output)

    def compute_loss(
        self,
        model: "BaseModel",
        y_pred: Any,
        y_true: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return None

    # whether the adapter's output format is compatible with generic metric computation
    # true for scorers with pointwise outputs,
    # false for models with structured outputs or custom losses
    # (e.g. two-tower with in-batch negatives or representation models)
    def supports_metrics(self, y_pred: Any, y_true: torch.Tensor | None) -> bool:
        return y_true is not None and isinstance(y_pred, torch.Tensor) and y_pred.dim() <= 2 and y_true.dim() <= 2


class RepresentationAdapter(TrainingAdapter):
    """
    Adapter for representation / generative models.

    These models usually expose structured raw outputs such as reconstructions,
    semantic IDs, quantization losses, or latent states, and typically manage
    their own training objectives instead of using BaseModel's pointwise/ranking
    loss dispatch. The adapter therefore keeps outputs untouched and disables
    generic metric collection by default.
    """

    def forward(self, model: "BaseModel", X_input: dict[str, torch.Tensor]):
        return model.call_model(X_input)

    def supports_metrics(self, y_pred: Any, y_true: torch.Tensor | None) -> bool:
        return False


class CandidateListAdapter(TrainingAdapter):
    """
    Adapter for pairwise and listwise ranking models with explicit candidate lists.

    These models expect input features to include a shared candidate-list axis,
    and the adapter handles reshaping between flat batch features and candidate-list inputs.
    It also supports in-batch negative sampling for two-tower models when sampling_mode='inbatch',
    and can compute appropriate losses based on the model's configured loss function and similarity metric.
    """

    def feature_has_candidate_axis(
        self,
        feature: DenseFeature | SparseFeature | SequenceFeature,
        tensor: torch.Tensor,
    ) -> bool:
        if isinstance(feature, SparseFeature):
            return tensor.dim() >= 2
        if isinstance(feature, SequenceFeature):
            return tensor.dim() >= 3
        if isinstance(feature, DenseFeature):
            return tensor.dim() >= 3
        return False

    def list_size(
        self,
        model: "BaseModel",
        X_input: dict[str, torch.Tensor],
    ) -> int:
        list_sizes = set()
        for feature in model.all_features:
            tensor = X_input.get(feature.name)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if self.feature_has_candidate_axis(feature, tensor):
                list_sizes.add(int(tensor.shape[1]))
        if not list_sizes:
            raise ValueError(
                "[CandidateListAdapter-input Error] sampling_mode='explicit' requires candidate-list features with a shared list axis."
            )
        if len(list_sizes) != 1:
            raise ValueError(
                f"[CandidateListAdapter-input Error] Candidate list sizes must be consistent across features, got {sorted(list_sizes)}."
            )
        list_size = next(iter(list_sizes))
        if list_size <= 1:
            raise ValueError(
                f"[CandidateListAdapter-input Error] sampling_mode='explicit' requires list_size >= 2, got {list_size}."
            )
        return list_size

    def prepare_list_input(
        self,
        model: "BaseModel",
        X_input: dict[str, torch.Tensor],
    ) -> tuple[int, int, dict[str, torch.Tensor]]:
        list_size = self.list_size(model, X_input)
        batch_size = next(iter(X_input.values())).shape[0]

        flat_input = {}
        for feature in model.all_features:
            tensor = X_input[feature.name]
            has_candidate_axis = self.feature_has_candidate_axis(feature, tensor)

            if has_candidate_axis:
                # Explicit candidate lists arrive as [B, L, ...]. Flatten them to
                # [B*L, ...] so downstream models can keep their usual single-pair
                # forward logic without knowing about the list axis.
                flat_input[feature.name] = tensor.reshape(batch_size * list_size, *tensor.shape[2:])
            else:
                # Features without a list axis (for example user features shaped
                # [B, ...]) are shared by all candidates in the row, so expand them
                # to [B, L, ...] first and then flatten to [B*L, ...].
                expanded = tensor.unsqueeze(1).expand(batch_size, list_size, *tensor.shape[1:])
                flat_input[feature.name] = expanded.reshape(batch_size * list_size, *tensor.shape[1:])
        return list_size, batch_size, flat_input

    def reshape_list_output(self, output: Any, batch_size: int, list_size: int) -> Any:
        if isinstance(output, torch.Tensor):
            if output.dim() == 1:
                return output.reshape(batch_size, list_size)
            return output.reshape(batch_size, list_size, *output.shape[1:])
        if isinstance(output, tuple):
            return tuple(self.reshape_list_output(item, batch_size, list_size) for item in output)
        if isinstance(output, list):
            return [self.reshape_list_output(item, batch_size, list_size) for item in output]
        return output

    def forward(self, model: "BaseModel", X_input: dict[str, torch.Tensor]):
        list_size, batch_size, flat_input = self.prepare_list_input(model, X_input)
        raw_output = model.call_model(flat_input)
        # Restore the candidate-list axis after the model scores each flattened
        # user-candidate pair independently, e.g. [B*L] -> [B, L].
        raw_output = self.reshape_list_output(raw_output, batch_size=batch_size, list_size=list_size)
        return model.format_model_output(raw_output)


class TwoTowerAdapter(CandidateListAdapter):
    """
    Adapter for two-tower models with in-batch negative sampling.

    This adapter extends CandidateListAdapter to support in-batch negatives for pairwise and listwise objectives.
    When sampling_mode='inbatch', it treats the batch as implicitly containing one positive and multiple
    negatives for each instance, and computes losses accordingly
    based on the model's configured loss function and similarity metric.
    """

    def needs_labels(self, model: "BaseModel") -> bool:
        return False

    def sample_inbatch_negatives(
        self,
        negatives: torch.Tensor,
        max_negatives: int | None,
    ) -> torch.Tensor:
        if max_negatives is None or max_negatives <= 0:
            return negatives
        if negatives.dim() < 2:
            return negatives

        available_negatives = negatives.size(1)
        if max_negatives >= available_negatives:
            return negatives

        random_scores = torch.rand(
            negatives.size(0),
            available_negatives,
            device=negatives.device,
        )
        sampled_indices = torch.topk(random_scores, k=max_negatives, dim=1).indices
        if negatives.dim() == 2:
            return negatives.gather(1, sampled_indices)

        expanded_indices = sampled_indices.unsqueeze(-1).expand(-1, -1, negatives.size(-1))
        return negatives.gather(1, expanded_indices)

    def forward(self, model: "BaseModel", X_input: dict[str, torch.Tensor]):
        for feature in model.all_features:
            tensor = X_input.get(feature.name)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if self.feature_has_candidate_axis(feature, tensor):
                raise ValueError(
                    "[TwoTowerAdapter-input Error] sampling_mode='inbatch' expects flat batch features without an explicit candidate-list axis."
                )
        raw_output = model.call_model(X_input)
        return model.format_model_output(raw_output)

    def compute_loss(
        self,
        model: "BaseModel",
        y_pred: Any,
        y_true: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not isinstance(y_pred, (tuple, list)) or len(y_pred) != 2:
            return None

        user_emb, item_emb = y_pred
        batch_size = user_emb.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=user_emb.device)

        logits = model.inbatch_logits(user_emb, item_emb)  # type: ignore[attr-defined]
        eye = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        pos_logits = logits.diag()
        neg_logits = logits.masked_select(~eye).view(batch_size, batch_size - 1)
        neg_logits = self.sample_inbatch_negatives(
            neg_logits,
            getattr(model, "num_negative_samples", None),
        )

        loss_fn = model.loss_fn[0] if hasattr(model, "loss_fn") and model.loss_fn else None
        if isinstance(loss_fn, SampledSoftmaxLoss):
            loss = loss_fn(pos_logits, neg_logits)
        elif isinstance(loss_fn, (BPRLoss, HingeLoss)):
            loss = loss_fn(pos_logits, neg_logits)
        elif isinstance(loss_fn, TripletLoss):
            neg_emb = item_emb.masked_select(~eye.unsqueeze(-1)).view(batch_size, batch_size - 1, item_emb.size(-1))
            neg_emb = self.sample_inbatch_negatives(
                neg_emb,
                getattr(model, "num_negative_samples", None),
            )
            loss = loss_fn(user_emb, item_emb, neg_emb)
        elif isinstance(loss_fn, InfoNCELoss) and getattr(model, "similarity_metric", None) == "dot":
            neg_emb = item_emb.masked_select(~eye.unsqueeze(-1)).view(batch_size, batch_size - 1, item_emb.size(-1))
            neg_emb = self.sample_inbatch_negatives(
                neg_emb,
                getattr(model, "num_negative_samples", None),
            )
            loss = loss_fn(user_emb, item_emb, neg_emb)
        elif isinstance(loss_fn, nn.CrossEntropyLoss):
            targets = torch.arange(batch_size, device=logits.device)
            loss = loss_fn(logits, targets)
        else:
            raise ValueError(
                "[TwoTowerAdapter-loss Error] sampling_mode='inbatch' does not support the configured loss. "
                "Use one of: CrossEntropyLoss/ce, SampledSoftmaxLoss, BPRLoss, HingeLoss, TripletLoss, "
                "or InfoNCELoss with similarity_metric='dot'."
            )

        if model.loss_weights is not None:
            loss *= float(model.loss_weights[0])
        return loss
