"""
Base model class for NextRec, providing common functionalities
for training, validation, prediction, and exporting.

Date: create on 15/04/2025
Checkpoint: edit on 15/04/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from nextrec import __version__
from nextrec.basic.heads import TaskHead
from nextrec.basic.adapters import CandidateListAdapter, TrainingAdapter
from nextrec.basic.asserts import (
    assert_model_family,
    assert_sampling_mode,
    assert_task,
    assert_task_family_compatibility,
    assert_training_mode,
)
from nextrec.basic.features import DenseFeature, FeatureSet, SequenceFeature, SparseFeature
from nextrec.basic.session import create_session, get_save_path
from nextrec.basic.summary import SummarySet
from nextrec.data.data_processing import get_column_data
from nextrec.engine.backends import InferenceBackend, OnnxInferenceBackend, TorchInferenceBackend
from nextrec.engine.exporter import Exporter
from nextrec.engine.predictor import BasePredictor
from nextrec.engine.trainer import BaseTrainer
from nextrec.engine.validator import BaseValidator
from nextrec.utils.torch_utils import get_device, to_tensor
from nextrec.utils.types import (
    ModelFamilyName,
    SamplingModeName,
    TaskTypeInput,
    TaskTypeName,
    TrainingModeName,
)
from nextrec.basic.callback import CallbackList


class Model(BaseTrainer, BaseValidator, BasePredictor, Exporter, SummarySet, FeatureSet, nn.Module):
    supported_sampling_modes = ["explicit"]

    @property
    def model_name(self) -> str:  # type: ignore[override]
        raise NotImplementedError

    @property
    def default_task(self) -> TaskTypeName | list[TaskTypeName]:
        raise NotImplementedError

    @property
    def model_family(self) -> ModelFamilyName:
        raise NotImplementedError

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: list[str] | str | None = None,
        key_columns: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        training_mode: TrainingModeName | None = None,
        sampling_mode: SamplingModeName = "explicit",
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        device: str = "cpu",
        session_id: str | None = None,
    ):
        """
        Initialize a base model.

        Args:
            dense_features: DenseFeature definitions.
            sparse_features: SparseFeature definitions.
            sequence_features: SequenceFeature definitions.
            target: Target column name. e.g., 'label_ctr' or ['label_ctr', 'label_cvr'].
            key_columns: Key column name(s) carried through dataloaders for grouping, reporting, or output. e.g., 'user_id'.
            task: Task types, e.g., 'binary', 'regression', or ['binary', 'regression']. If None, falls back to self.default_task.
            training_mode: Training mode shared by all tasks. e.g., 'pointwise'.
            sampling_mode: Candidate organization mode used by pairwise/listwise training.
                BaseModel supports 'explicit' only, where candidates/negatives are provided explicitly.

            embedding_l1_reg: L1 regularization strength for embedding params. e.g., 1e-6.
            dense_l1_reg: L1 regularization strength for dense params. e.g., 1e-5.
            embedding_l2_reg: L2 regularization strength for embedding params. e.g., 1e-5.
            dense_l2_reg: L2 regularization strength for dense params. e.g., 1e-4.

            device: Torch device string or torch.device. e.g., 'cpu', 'cuda:0'.
            session_id: Session id for logging. If None, a default id with timestamps will be created. e.g., 'session_tutorial'.
        Note:
            Optimizer, scheduler, and loss are configured via compile().
        """
        super().__init__()

        self.device = get_device(device)

        self.session_id = session_id
        self.session = create_session(session_id)
        self.session_path = self.session.root
        self.checkpoint_path = os.path.join(self.session_path, self.model_name.upper() + "_checkpoint.pt")
        self.best_path = os.path.join(self.session_path, self.model_name.upper() + "_best.pt")
        self.features_config_path = os.path.join(self.session_path, "features_config.pkl")
        self.set_all_features(dense_features, sparse_features, sequence_features, target, key_columns)

        self.task = self.default_task if task is None else task
        self.nums_task = len(self.task) if isinstance(self.task, list) else 1
        self.training_mode = training_mode or "pointwise"

        # assertions
        assert_task(self.task, self.nums_task, model_name=self.model_name)
        assert_model_family(self.model_family, model_name=self.model_name)
        assert_training_mode(self.training_mode, model_name=self.model_name)
        assert_sampling_mode(sampling_mode, model_name=self.model_name)
        assert_task_family_compatibility(
            self.task,
            self.model_family,
            self.training_mode,
            sampling_mode,
            model_name=self.model_name,
        )
        if sampling_mode not in self.supported_sampling_modes:
            raise ValueError(
                f"[BaseModel-init Error] Unsupported sampling_mode='{sampling_mode}' for model_family='{self.model_family}'. "
                f"Supported: {self.supported_sampling_modes}"
            )
        self.sampling_mode = sampling_mode
        self.current_batch_schema = None

        self.embedding_l1_reg = embedding_l1_reg
        self.dense_l1_reg = dense_l1_reg
        self.embedding_l2_reg = embedding_l2_reg
        self.dense_l2_reg = dense_l2_reg
        self.regularization_weights = []
        self.embedding_params = []

        self.ignore_label = None
        self.compiled = False
        self.grad_norm_config = None
        self.grad_norm_shared_modules = []

        self.max_gradient_norm = 1.0
        self.training_logger = None
        self.steps_per_epoch = None
        self.metrics = None
        self.task_specific_metrics = None
        self.best_metrics_mode = None

        self.callbacks = CallbackList()

        self.set_adapter()
        self.set_head()

        # summary for training/validation data distribution
        self.train_data_summary = None
        self.valid_data_summary = None
        self.note = None
        self.logger_initialized = False

        self.inference_artifact_path = None
        self.inference_backend: InferenceBackend = TorchInferenceBackend(self)

    def set_head(self):
        """
        Set the default task head for base models.

        Model families with specialized heads override this method in their own
        base classes.
        """
        if self.training_mode != "pointwise":
            self.head = None
            return
        self.head = TaskHead(task_type=self.task)

    def set_adapter(self):
        """
        Set the default training adapter for base models, used to process output from taskheads.

        BaseModel uses CandidateListAdapter only for pairwise/listwise training
        with explicit candidate lists. All other cases fall back to the generic
        TrainingAdapter.

        Matching or sequential models will override this method in their own base classes.
        """
        is_candidate_list_mode = self.training_mode in {"pairwise", "listwise"} and self.sampling_mode == "explicit"
        self.training_adapter = CandidateListAdapter() if is_candidate_list_mode else TrainingAdapter()

    def call_model(self, X_input: dict[str, torch.Tensor]):
        """
        Call the model to get raw output before head processing.
        """
        raw_output = self(X_input)
        if self.head is None:
            return raw_output
        return self.head(raw_output)

    def register_regularization_weights(
        self,
        embedding_attr: str = "embedding",
        exclude_modules: list[str] | None = None,
        include_modules: list[str] | None = None,
    ):
        """
        Register parameters for regularization.
        By default, all nn.Linear weights (excluding those in BatchNorm/Dropout layers) and
        embedding weights under `embedding_attr` are registered.

        Args:
            embedding_attr: Attribute name of the embedding layer/module.
            exclude_modules: List of module name substrings to exclude from regularization.
            include_modules: List of module name substrings to include for regularization.
                If provided, only modules containing these substrings are included.
        """
        exclude_modules = exclude_modules or []
        include_modules = include_modules or []

        embedding_layer = getattr(self, embedding_attr, None)
        embed_dict = getattr(embedding_layer, "embed_dict", None)
        # get embedding parameters from embed_dict if exists, or get weight from embedding_layer directly
        if embed_dict is not None:
            embedding_params = [embed.weight for embed in embed_dict.values() if hasattr(embed, "weight")]
        # from nn.Embedding or nn.EmbeddingBag layer
        else:
            weight = getattr(embedding_layer, "weight", None)
            embedding_params = [weight] if isinstance(weight, torch.Tensor) else []

        existing_embedding_ids = {id(param) for param in self.embedding_params}
        self.embedding_params.extend(param for param in embedding_params if id(param) not in existing_embedding_ids)

        # skip bn and dropout layers and linear layers in embedding layer
        skip_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.Dropout,
            nn.Dropout2d,
            nn.Dropout3d,
        )

        existing_reg_ids = {id(param) for param in self.regularization_weights}
        for name, module in self.named_modules():
            in_embedding_subtree = name == embedding_attr or name.startswith(f"{embedding_attr}.")
            is_dense_projection = in_embedding_subtree and ".dense_transforms." in name
            if (
                module is self
                or (in_embedding_subtree and not is_dense_projection)
                or isinstance(module, skip_types)
                or not isinstance(module, nn.Linear)
                or (include_modules and not any(inc in name for inc in include_modules))
                or (exclude_modules and any(exc in name for exc in exclude_modules))
                or id(module.weight) in existing_reg_ids
            ):
                continue
            self.regularization_weights.append(module.weight)
            existing_reg_ids.add(id(module.weight))

    def add_reg_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)

        if self.embedding_l1_reg > 0:
            reg_loss += self.embedding_l1_reg * sum(param.abs().sum() for param in self.embedding_params)
        if self.embedding_l2_reg > 0:
            reg_loss += self.embedding_l2_reg * sum((param**2).sum() for param in self.embedding_params)

        if self.dense_l1_reg > 0:
            reg_loss += self.dense_l1_reg * sum(param.abs().sum() for param in self.regularization_weights)
        if self.dense_l2_reg > 0:
            reg_loss += self.dense_l2_reg * sum((param**2).sum() for param in self.regularization_weights)
        return reg_loss

    def get_input(self, input_data: dict, require_labels: bool = True):
        """
        Prepare unified input features and labels from the given input data.

        Args:
            input_data: Input data dict containing 'features', 'labels', and optionally 'schema'.
                        e.g., {'features': {'feat1': [...], 'feat2': [...]}, 'labels': {'label': [...]}, 'schema': {...}}.
            require_labels: Whether labels are required in the input data.
                        Default is True: for training and evaluation with labels.
                        Set to False for prediction/auto-regressive generation tasks.
        """
        feature_source = input_data.get("features", {})
        label_source = input_data.get("labels")
        batch_schema = input_data.get("schema")
        self.current_batch_schema = batch_schema

        X_input = {}
        for feature in self.all_features:
            if feature.name not in feature_source:
                raise KeyError(f"[BaseModel-input Error] Feature '{feature.name}' not found in input data.")
            feature_data = get_column_data(feature_source, feature.name)
            X_input[feature.name] = to_tensor(
                feature_data,
                dtype=(torch.float32 if isinstance(feature, DenseFeature) else torch.long),
                device=self.device,
            )

        # if labels are not required or not found in the input, return None for y to indicate no labels
        needs_labels = bool(self.target_columns) and (
            require_labels or (label_source and any(name in label_source for name in self.target_columns))
        )
        if not needs_labels:
            return X_input, None

        target_tensors = []
        for target_name in self.target_columns:
            target_tensor = self.training_adapter.build_target_tensor(
                model=self,
                input_data=input_data,
                target_name=target_name,
                require_labels=require_labels,
            )
            if target_tensor is not None:
                target_tensors.append(target_tensor)

        if not target_tensors:
            if require_labels:
                raise ValueError("[BaseModel-input Error] Labels are required but none were found in the input batch.")
            return X_input, None

        label_format = batch_schema.get("label_format") if isinstance(batch_schema, dict) else None

        # candidate-list need to assign k candidates for each sample,
        # so the target tensors should be stacked to shape [batch_size, k, ...]
        if label_format == "candidate_list" and len({tensor.shape[1] for tensor in target_tensors}) > 1:
            raise ValueError("[BaseModel-input Error] candidate-list labels must share the same list width.")

        # [batch size, seq_len] for sequence labels, [batch size, k] for candidate list labels
        # samples should be stacked
        if label_format in {"candidate_list", "sequence"}:
            y = target_tensors[0] if len(target_tensors) == 1 else torch.stack(target_tensors, dim=-1)
        else:
            y = torch.cat(target_tensors, dim=1)
            if y.shape[1] == 1:
                y = y.reshape(-1)
        return X_input, y

    # todo: support resume from ckpt, and save optimizer/scheduler state in ckpt model files
    def save_model(
        self,
        save_path: str | Path | None = None,
        add_timestamp: bool | None = None,
        verbose: bool = True,
    ) -> Path:
        add_timestamp = False if add_timestamp is None else add_timestamp
        target_path = get_save_path(
            path=save_path,
            default_dir=self.session_path,
            default_name=self.model_name.upper(),
            suffix=".pt",
            add_timestamp=add_timestamp,
        )
        model_path = Path(target_path)

        torch.save(self.state_dict(), model_path)

        config_path = self.features_config_path
        features_config = {
            "all_features": self.all_features,
            "target": self.target_columns,
            "key_columns": self.key_columns,
            "version": __version__,
        }
        with open(config_path, "wb") as file_obj:
            pickle.dump(features_config, file_obj)
        self.features_config_path = str(config_path)
        if verbose:
            logging.info(
                f"Model saved to: {model_path}, features config saved to: {config_path}, NextRec version: {__version__}"
            )
        return model_path

    def load_model(
        self,
        save_path: str | Path,
        map_location: str | torch.device | None = "cpu",
    ):
        self.to(self.device)
        base_path = Path(save_path)
        if base_path.is_dir():
            model_files = sorted(base_path.glob("*.pt"))
            if not model_files:
                raise FileNotFoundError(f"[BaseModel-load-model Error] No *.pt file found in directory: {base_path}")
            model_path = model_files[-1]
            config_dir = base_path
        else:
            model_path = base_path.with_suffix(".pt") if base_path.suffix == "" else base_path
            config_dir = model_path.parent
        if not model_path.exists():
            raise FileNotFoundError(f"[BaseModel-load-model Error] Model file does not exist: {model_path}")

        state_dict = torch.load(model_path, map_location=map_location)
        self.load_state_dict(state_dict)

        features_config_path = config_dir / "features_config.pkl"
        if not features_config_path.exists():
            raise FileNotFoundError(f"[BaseModel-load-model Error] features_config.pkl not found in: {config_dir}")
        with open(features_config_path, "rb") as file_obj:
            features_config = pickle.load(file_obj)

        all_features = features_config.get("all_features", [])
        target = features_config.get("target", [])
        key_columns = features_config.get("key_columns", [])
        dense_features = [feature for feature in all_features if isinstance(feature, DenseFeature)]
        sparse_features = [feature for feature in all_features if isinstance(feature, SparseFeature)]
        sequence_features = [feature for feature in all_features if isinstance(feature, SequenceFeature)]
        self.set_all_features(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            key_columns=key_columns,
        )

        cfg_version = features_config.get("version")
        logging.info(
            f"Model weights loaded from: {model_path}, features config loaded from: {features_config_path}, NextRec version: {cfg_version}"
        )
        self.set_inference_backend()

    def set_inference_backend(
        self,
        artifact_path: str | Path | None = None,
    ) -> InferenceBackend:
        """
        Configure the inference backend for this loaded model.

        Args:
            artifact_path: None for the current PyTorch model, or an ONNX artifact
                path for ONNX Runtime inference.
        """
        if artifact_path is None:
            self.inference_artifact_path = None
            self.inference_backend = TorchInferenceBackend(self)
            return self.inference_backend

        model_path = Path(artifact_path)
        if model_path.suffix.lower() != ".onnx":
            raise ValueError("[BaseModel-backend Error] artifact_path must be None or an ONNX model path.")
        if not model_path.exists():
            raise FileNotFoundError(f"[BaseModel-backend Error] ONNX model file does not exist: {model_path}")
        self.inference_artifact_path = str(model_path)
        self.inference_backend = OnnxInferenceBackend(self, onnx_path=model_path)
        return self.inference_backend

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | torch.device | None = "cpu",
        device: str | torch.device = "cpu",
        session_id: str | None = None,
        **kwargs: Any,
    ):
        """
        Load a model from a checkpoint directory or file.

        Args:
            checkpoint_path: Path to the checkpoint directory or file.
                        If the path is a directory, the latest checkpoint file (*.pt or *.onnx) will be used.
                        If the path is a file(*.pt or *.onnx), will be used directly.
            map_location: Device mapping for loading the PyTorch model.
            device: Device to place the model on.
            session_id: Optional session ID for the model.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the model loaded from the checkpoint.
        """
        base_path = Path(checkpoint_path)
        if base_path.is_dir():
            model_candidates = sorted(base_path.glob("*.pt"))
            onnx_candidates = sorted(base_path.glob("*.onnx"))
            if model_candidates:
                model_file = model_candidates[-1]
            elif onnx_candidates:
                model_file = onnx_candidates[-1]
            else:
                raise FileNotFoundError(
                    f"[BaseModel-from-checkpoint Error] No *.pt or *.onnx file found under: {base_path}"
                )
            config_dir = base_path
        else:
            model_file = base_path.with_suffix(".pt") if base_path.suffix == "" else base_path
            config_dir = model_file.parent
        features_config_path = config_dir / "features_config.pkl"
        if not features_config_path.exists():
            raise FileNotFoundError(
                f"[BaseModel-from-checkpoint Error] features_config.pkl not found next to checkpoint: {features_config_path}"
            )
        with open(features_config_path, "rb") as file_obj:
            features_config = pickle.load(file_obj)
        all_features = features_config.get("all_features", [])
        target = features_config.get("target", [])
        key_columns = features_config.get("key_columns", [])

        dense_features = [feature for feature in all_features if isinstance(feature, DenseFeature)]
        sparse_features = [feature for feature in all_features if isinstance(feature, SparseFeature)]
        sequence_features = [feature for feature in all_features if isinstance(feature, SequenceFeature)]

        model = cls(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            key_columns=key_columns,
            device=str(device),
            session_id=session_id,
            **kwargs,
        )
        if model_file.suffix.lower() == ".onnx":
            model.set_inference_backend(model_file)
            logging.info(
                f"ONNX inference model loaded from: {model_file}, features config loaded from: {features_config_path}, NextRec version: {features_config.get('version')}"
            )
        else:
            model.load_model(model_file, map_location=map_location)
        return model
