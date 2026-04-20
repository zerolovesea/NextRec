"""
Training adapters for NextRec models.

These adapters provide flexible interfaces between NextRec's training loop
and various model architectures and loss functions, enabling support for
pointwise, pairwise, listwise, pretrain, and two-tower models with in-batch negatives.

Date: create on 22/03/2026
Checkpoint: edit on 19/04/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn

from nextrec.data.data_processing import get_column_data
from nextrec.loss.listwise import InfoNCELoss, SampledSoftmaxLoss
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss
from nextrec.utils.torch_utils import to_tensor


class TrainingAdapter:
    # if the adapter requires labels for loss computation
    def needs_labels(self, model) -> bool:
        return True

    def forward(self, model, X_input: dict[str, torch.Tensor]):
        model_output = model.call_model(X_input)
        return self.adapt_output(model, model_output)

    def adapt_output(self, model, model_output: Any):
        return model_output

    def compute_loss(
        self,
        model,
        y_pred: Any,
        y_true: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return None

    def build_target_tensor(
        self,
        model,
        input_data: dict[str, Any],
        target_name: str,
        require_labels: bool,
    ) -> torch.Tensor | None:
        label_source = input_data.get("labels")
        if label_source is None or target_name not in label_source:
            if require_labels:
                raise KeyError(f"[BaseModel-input Error] Target column '{target_name}' not found in input data.")
            return None

        target_data = get_column_data(label_source, target_name)
        if target_data is None:
            if require_labels:
                raise ValueError(f"[BaseModel-input Error] Target column '{target_name}' contains no data.")
            return None

        problem_type = self.resolve_problem_type(model, target_name)
        target_dtype = torch.long if problem_type == "generative" else torch.float32
        target_tensor = to_tensor(target_data, dtype=target_dtype, device=model.device)
        return target_tensor.reshape(target_tensor.size(0), -1)

    def resolve_problem_type(self, model, target_name: str) -> str:
        if isinstance(model.task, list):
            if target_name in getattr(model, "target_columns", []):
                target_index = model.target_columns.index(target_name)
                if target_index < len(model.task):
                    return model.task[target_index]
            return model.task[0]
        return model.task


class PretrainAdapter(TrainingAdapter):
    """
    Adapter for pretrain / generative models.

    These models usually expose structured raw outputs such as reconstructions,
    semantic IDs, quantization losses, or latent states, and typically manage
    their own training objectives instead of using BaseModel's pointwise/ranking
    loss dispatch. The adapter therefore keeps outputs untouched and disables
    generic metric collection by default.
    """

    def forward(self, model, X_input: dict[str, torch.Tensor]):
        return model.call_model(X_input)


class MatchingAdapter(TrainingAdapter):
    """
    Adapter for matching models that emit user/item embeddings and use a
    MatchingHead to convert them into final scores.
    """

    def adapt_output(self, model, model_output: Any):
        if isinstance(model_output, dict) and "scores" in model_output:
            return model_output["scores"]
        return model_output


class SequentialAdapter(TrainingAdapter):
    """
    Adapter for sequential recommendation models.

    Sequential models use the base adapter for output formatting and loss / metric
    routing, while sequence-specific batch utilities live in BaseSequentialModel.
    """

    def build_target_tensor(
        self,
        model,
        input_data: dict[str, Any],
        target_name: str,
        require_labels: bool,
    ) -> torch.Tensor | None:
        target_tensor = super().build_target_tensor(
            model=model,
            input_data=input_data,
            target_name=target_name,
            require_labels=require_labels,
        )
        return target_tensor.reshape(target_tensor.size(0), -1) if target_tensor is not None else None


class MaskedSequentialAdapter(SequentialAdapter):
    """
    Adapter for masked sequential pretraining models.

    These models synthesize labels from the input sequence during forward(), so
    the generic input pipeline should not require explicit labels.
    """

    def needs_labels(self, model) -> bool:
        return False


class CandidateListAdapter(TrainingAdapter):
    """
    Adapter for pairwise and listwise ranking models with explicit candidate lists.

    These models expect input features to include a shared candidate-list axis,
    and the adapter handles reshaping between flat batch features and candidate-list inputs.
    
    It also supports in-batch negative sampling for two-tower models when sampling_mode='inbatch',
    and can compute appropriate losses based on the model's configured loss function and similarity metric.
    """

    def get_batch_schema(self, model) -> dict[str, Any]:
        schema = getattr(model, "current_batch_schema", None)
        if not isinstance(schema, dict):
            raise ValueError(
                "[CandidateListAdapter-input Error] Explicit candidate-list training requires batch schema."
            )
        return schema

    def prepare_list_input(
        self,
        model,
        X_input: dict[str, torch.Tensor],
    ) -> tuple[int, int, dict[str, torch.Tensor]]:
        schema = self.get_batch_schema(model)
        list_size = int(schema.get("list_size") or 0)
        if list_size < 2:
            raise ValueError(
                f"[CandidateListAdapter-input Error] sampling_mode='explicit' requires schema.list_size >= 2, got {list_size}."
            )
        batch_size = next(iter(X_input.values())).shape[0]
        feature_scopes = schema.get("feature_scopes", {})

        flat_input = {}
        for feature in model.all_features:
            tensor = X_input[feature.name]
            feature_scope = feature_scopes.get(feature.name, "shared")
            has_candidate_axis = feature_scope == "candidate"

            if has_candidate_axis:
                if tensor.shape[1] != list_size:
                    raise ValueError(
                        f"[CandidateListAdapter-input Error] Feature '{feature.name}' expected candidate axis size {list_size}, got shape {tuple(tensor.shape)}."
                    )
                flat_input[feature.name] = tensor.reshape(batch_size * list_size, *tensor.shape[2:])
            else:
                expanded = tensor.unsqueeze(1).expand(batch_size, list_size, *tensor.shape[1:])
                flat_input[feature.name] = expanded.reshape(batch_size * list_size, *tensor.shape[1:])
        return list_size, batch_size, flat_input

    def reshape_list_output(self, output: Any, batch_size: int, list_size: int) -> Any:
        if isinstance(output, torch.Tensor):
            if output.dim() == 1:
                return output.reshape(batch_size, list_size)
            return output.reshape(batch_size, list_size, *output.shape[1:])
        if isinstance(output, dict):
            return {key: self.reshape_list_output(value, batch_size, list_size) for key, value in output.items()}
        if isinstance(output, tuple):
            return tuple(self.reshape_list_output(item, batch_size, list_size) for item in output)
        if isinstance(output, list):
            return [self.reshape_list_output(item, batch_size, list_size) for item in output]
        return output

    def adapt_output(self, model, model_output: Any):
        if isinstance(model_output, dict) and "scores" in model_output:
            return model_output["scores"]
        return model_output

    def forward(self, model, X_input: dict[str, torch.Tensor]):
        list_size, batch_size, flat_input = self.prepare_list_input(model, X_input)
        model_output = model.call_model(flat_input)
        # Restore the candidate-list axis after the model scores each flattened
        # user-candidate pair independently, e.g. [B*L] -> [B, L].
        model_output = self.reshape_list_output(model_output, batch_size=batch_size, list_size=list_size)
        return self.adapt_output(model, model_output)


class TwoTowerAdapter(CandidateListAdapter):
    """
    Adapter for two-tower models with in-batch negative sampling.

    This adapter extends CandidateListAdapter to support in-batch negatives for pairwise and listwise objectives.
    When sampling_mode='inbatch', it treats the batch as implicitly containing one positive and multiple
    negatives for each instance, and computes losses accordingly
    based on the model's configured loss function and similarity metric.
    """

    def needs_labels(self, model) -> bool:
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

    def forward(self, model, X_input: dict[str, torch.Tensor]):
        schema = getattr(model, "current_batch_schema", None) or {}
        if schema.get("label_format") != "implicit_inbatch":
            raise ValueError(
                "[TwoTowerAdapter-input Error] sampling_mode='inbatch' requires batch schema label_format='implicit_inbatch'."
            )
        raw_output = model.call_model(X_input)
        return self.adapt_output(model, raw_output)

    def adapt_output(self, model, model_output: Any):
        return model_output

    def compute_loss(
        self,
        model,
        y_pred: Any,
        y_true: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not isinstance(y_pred, dict):
            return None
        user_emb = y_pred.get("user_emb")
        item_emb = y_pred.get("item_emb")
        if not isinstance(user_emb, torch.Tensor) or not isinstance(item_emb, torch.Tensor):
            return None
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
