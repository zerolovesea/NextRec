"""
Training adapters for NextRec models.

These adapters provide flexible interfaces between NextRec's training loop
and various model architectures and loss functions, enabling support for
pointwise, pairwise, listwise, pretrain, and two-tower models with in-batch negatives.

Date: create on 22/03/2026
Checkpoint: edit on 22/03/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.heads import GenerativeRetrievalHead, TaskHead
from nextrec.data.data_processing import get_column_data
from nextrec.data.dataloader import build_shifted_sequence_column
from nextrec.loss.listwise import InfoNCELoss, SampledSoftmaxLoss
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss
from nextrec.utils.torch_utils import to_tensor


class TrainingAdapter:
    # if the adapter requires labels for loss computation
    def needs_labels(self, model) -> bool:
        return True

    def get_target_shift_config(self, model) -> tuple[str | None, int]:
        """
        Return the optional target source and shift used when labels are not
        provided explicitly in the batch.

        Non-sequential adapters default to no shift-based label synthesis.
        """
        return None, 1

    def forward(self, model, X_input: dict[str, torch.Tensor]):
        raw_output = model.call_model(X_input)
        return self.format_model_output(model, raw_output)

    def build_prediction_layer(self, model) -> nn.Module | None:
        if model.training_modes[0] != "pointwise":
            return None
        task_type = model.task[0] if isinstance(model.task, list) else model.task
        if task_type == "generative":
            return GenerativeRetrievalHead(vocab_size=int(model.vocab_size), return_logits=True)
        return TaskHead(task_type=model.task)

    def format_model_output(self, model, raw_output: Any):
        if model.training_modes[0] != "pointwise":
            return raw_output
        if isinstance(raw_output, torch.Tensor) and model.prediction_layer is not None:
            return model.prediction_layer(raw_output)
        return raw_output

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

        target_tensor = to_tensor(target_data, dtype=torch.float32, device=model.device)
        return target_tensor.reshape(target_tensor.size(0), -1)


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

    def build_prediction_layer(self, model) -> nn.Module | None:
        return None

    def format_model_output(self, model, raw_output: Any):
        return raw_output
class RetrievalAdapter(TrainingAdapter):
    """
    Adapter for retrieval models that emit user/item embeddings and use a
    RetrievalHead to convert them into final scores.
    """

    def build_prediction_layer(self, model) -> nn.Module | None:
        return None

    def format_model_output(self, model, raw_output: Any):
        if not isinstance(raw_output, (tuple, list)) or len(raw_output) != 2:
            return super().format_model_output(model, raw_output)
        user_emb, item_emb = raw_output
        return model.head(user_emb, item_emb, similarity_fn=model.compute_similarity)


class SequentialAdapter(TrainingAdapter):
    """
    Adapter for sequential recommendation models.

    Sequential models use the base adapter for output formatting and loss / metric
    routing, while sequence-specific batch utilities live in BaseSequentialModel.
    """

    def get_target_shift_config(self, model) -> tuple[str | None, int]:
        target_source = getattr(model, "target_source", None)
        target_shift_steps = int(getattr(model, "target_shift_steps", 1))
        return target_source, target_shift_steps

    def build_target_tensor(
        self,
        model,
        input_data: dict[str, Any],
        target_name: str,
        require_labels: bool,
    ) -> torch.Tensor | None:
        # SequentialAdapter support given explicit labels
        # or create labels by shifting a source sequence feature.
        # if explicit labels are provided, use them directly.
        # Otherwise, fall back to shift-based label with target_source and target_shift_steps.
        target_tensor = super().build_target_tensor(
            model=model,
            input_data=input_data,
            target_name=target_name,
            require_labels=False,
        )
        if target_tensor is not None:
            return target_tensor

        target_source, target_shift_steps = self.get_target_shift_config(model)
        if target_source is None:
            if require_labels:
                raise KeyError(f"[BaseModel-input Error] Target column '{target_name}' not found in input data.")
            return None

        feature_source = input_data.get("features", {})
        feature_by_name = {feature.name: feature for feature in model.all_features}
        source_feature = feature_by_name.get(target_source)
        if source_feature is None or not isinstance(source_feature, SequenceFeature):
            raise KeyError(
                f"[BaseModel-input Error] target_source='{target_source}' requires a matching SequenceFeature."
            )

        source_data = get_column_data(feature_source, target_source)
        if source_data is None:
            raise KeyError(f"[BaseModel-input Error] target_source column '{target_source}' not found in input data.")

        target_data = build_shifted_sequence_column(
            source_data,
            source_feature,
            shift=target_shift_steps,
        )
        target_tensor = to_tensor(target_data, dtype=torch.long, device=model.device)
        return target_tensor.reshape(target_tensor.size(0), -1)

    def build_prediction_layer(self, model) -> nn.Module | None:
        return None

    def format_model_output(self, model, raw_output: Any):
        return raw_output


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
        model,
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
        model,
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

    def format_model_output(self, model, raw_output: Any):
        if not isinstance(raw_output, (tuple, list)) or len(raw_output) != 2:
            return super().format_model_output(model, raw_output)
        user_emb, item_emb = raw_output
        return model.head(user_emb, item_emb, similarity_fn=model.compute_similarity)

    def forward(self, model, X_input: dict[str, torch.Tensor]):
        list_size, batch_size, flat_input = self.prepare_list_input(model, X_input)
        raw_output = model.call_model(flat_input)
        # Restore the candidate-list axis after the model scores each flattened
        # user-candidate pair independently, e.g. [B*L] -> [B, L].
        raw_output = self.reshape_list_output(raw_output, batch_size=batch_size, list_size=list_size)
        return self.format_model_output(model, raw_output)


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
        for feature in model.all_features:
            tensor = X_input.get(feature.name)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if self.feature_has_candidate_axis(feature, tensor):
                raise ValueError(
                    "[TwoTowerAdapter-input Error] sampling_mode='inbatch' expects flat batch features without an explicit candidate-list axis."
                )
        raw_output = model.call_model(X_input)
        return self.format_model_output(model, raw_output)

    def compute_loss(
        self,
        model,
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

    def build_prediction_layer(self, model) -> nn.Module | None:
        return None
