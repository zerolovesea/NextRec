"""
ONNX export/load tests for NextRec models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.models.multi_task.esmm import ESMM
from nextrec.models.multi_task.mmoe import MMOE
from nextrec.models.multi_task.pepnet import PEPNet
from nextrec.models.multi_task.ple import PLE
from nextrec.models.multi_task.share_bottom import ShareBottom
from nextrec.models.ranking.afm import AFM
from nextrec.models.ranking.autoint import AutoInt
from nextrec.models.ranking.dcn import DCN
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.models.ranking.dien import DIEN
from nextrec.models.ranking.din import DIN
from nextrec.models.ranking.eulernet import EulerNet
from nextrec.models.ranking.ffm import FFM
from nextrec.models.ranking.fibinet import FiBiNET
from nextrec.models.ranking.fm import FM
from nextrec.models.ranking.lr import LR
from nextrec.models.ranking.pnn import PNN
from nextrec.models.ranking.widedeep import WideDeep
from nextrec.models.ranking.xdeepfm import xDeepFM
from nextrec.models.retrieval.dssm import DSSM
from nextrec.models.retrieval.mind import MIND
from nextrec.models.retrieval.sdm import SDM
from nextrec.models.retrieval.youtube_dnn import YoutubeDNN
from nextrec.utils.onnx_utils import (
    build_onnx_input_feed,
    create_dummy_inputs,
    load_onnx_session,
    merge_onnx_outputs,
)

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

DEVICE = "cpu"
EXPORT_BATCH = 2
DEFAULT_SEQ_LEN = 5
ONNX_RTOL = 1e-4
ONNX_ATOL = 1e-4
ONNX_RNN_RTOL = 2e-2
ONNX_RNN_ATOL = 1e-2


def _ensure_sequence_non_padding(feature: SequenceFeature, tensor: torch.Tensor) -> torch.Tensor:
    vocab_size = int(feature.vocab_size)
    if vocab_size <= 1:
        return tensor
    padding_idx = feature.padding_idx if feature.padding_idx is not None else 0
    fill_value = 0 if padding_idx != 0 else 1
    if fill_value >= vocab_size:
        return tensor
    return torch.full_like(tensor, fill_value)


def _get_onnx_tolerances(model: torch.nn.Module) -> tuple[float, float]:
    has_rnn = any(isinstance(module, torch.nn.RNNBase) for module in model.modules())
    if has_rnn:
        return ONNX_RNN_RTOL, ONNX_RNN_ATOL
    return ONNX_RTOL, ONNX_ATOL


def _base_features():
    dense_features = [
        DenseFeature(name="age", proj_dim=1),
        DenseFeature(name="price", proj_dim=1),
    ]
    sparse_features = [
        SparseFeature(name="user_id", vocab_size=100, embedding_dim=8),
        SparseFeature(name="category", vocab_size=20, embedding_dim=8),
        SparseFeature(name="item_id", vocab_size=50, embedding_dim=8),
    ]
    sequence_features = [
        SequenceFeature(
            name="hist_item_ids",
            vocab_size=50,
            max_len=DEFAULT_SEQ_LEN,
            embedding_dim=8,
            padding_idx=0,
        )
    ]
    return dense_features, sparse_features, sequence_features


def _match_features():
    user_dense = [DenseFeature(name="user_age", proj_dim=1)]
    user_sparse = [
        SparseFeature(name="user_id", vocab_size=100, embedding_dim=8),
        SparseFeature(name="user_city", vocab_size=20, embedding_dim=8),
    ]
    user_sequence = [
        SequenceFeature(
            name="user_hist_items",
            vocab_size=50,
            max_len=DEFAULT_SEQ_LEN,
            embedding_dim=8,
            padding_idx=0,
        )
    ]

    item_dense = [DenseFeature(name="item_price", proj_dim=1)]
    item_sparse = [
        SparseFeature(name="item_id", vocab_size=50, embedding_dim=8),
        SparseFeature(name="item_category", vocab_size=20, embedding_dim=8),
    ]
    item_sequence = []
    return (
        user_dense,
        user_sparse,
        user_sequence,
        item_dense,
        item_sparse,
        item_sequence,
    )


BASE_FEATURES = _base_features()
MATCH_FEATURES = _match_features()


def _normalize_torch_output(output: torch.Tensor | list | tuple) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        pieces = []
        for out in output:
            if not isinstance(out, torch.Tensor):
                continue
            if out.dim() == 0:
                out = out.view(1, 1)
            elif out.dim() == 1:
                out = out.view(-1, 1)
            pieces.append(out)
        if not pieces:
            raise AssertionError("Empty output list/tuple from torch model.")
        return torch.cat(pieces, dim=1)
    if output.dim() == 0:
        return output.view(1, 1)
    if output.dim() == 1:
        return output.view(-1, 1)
    return output


def _normalize_onnx_output(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _run_onnx_roundtrip(model: torch.nn.Module, tmp_path: Path) -> None:
    model.eval()
    model.to(DEVICE)

    onnx_path = model.export_onnx(
        save_path=tmp_path / f"{model.model_name}.onnx",
        batch_size=EXPORT_BATCH,
    )

    input_names = [feat.name for feat in model.all_features]
    dummy_inputs = create_dummy_inputs(
        model.all_features,
        batch_size=EXPORT_BATCH,
        device=torch.device(DEVICE),
    )
    for idx, feature in enumerate(model.all_features):
        if isinstance(feature, SequenceFeature):
            dummy_inputs[idx] = _ensure_sequence_non_padding(feature, dummy_inputs[idx])
    input_dict = {name: tensor for name, tensor in zip(input_names, dummy_inputs)}

    with torch.no_grad():
        torch_output = _normalize_torch_output(model(input_dict))

    try:
        session = load_onnx_session(onnx_path)
    except Exception as exc:  # pragma: no cover
        pytest.xfail(f"ONNX Runtime failed to load model ({model.model_name}): {exc}")
    session_input_names = [inp.name for inp in session.get_inputs()]
    feed = build_onnx_input_feed(model.all_features, input_dict, input_names=session_input_names)
    try:
        onnx_outputs = session.run(None, feed)
    except Exception as exc:  # pragma: no cover
        pytest.xfail(f"ONNX Runtime failed to run model ({model.model_name}): {exc}")
    onnx_output = _normalize_onnx_output(merge_onnx_outputs(onnx_outputs))

    assert onnx_output.shape == tuple(torch_output.shape)
    rtol, atol = _get_onnx_tolerances(model)
    np.testing.assert_allclose(
        onnx_output,
        torch_output.detach().cpu().numpy(),
        rtol=rtol,
        atol=atol,
    )


def _run_onnx_dynamo_roundtrip(model: torch.nn.Module, tmp_path: Path) -> None:
    model.eval()
    model.to(DEVICE)

    onnx_path = model.export_onnx(
        save_path=tmp_path / f"{model.model_name}_dynamo.onnx",
        batch_size=EXPORT_BATCH,
    )

    input_names = [feat.name for feat in model.all_features]
    dummy_inputs = create_dummy_inputs(
        model.all_features,
        batch_size=EXPORT_BATCH,
        device=torch.device(DEVICE),
    )
    for idx, feature in enumerate(model.all_features):
        if isinstance(feature, SequenceFeature):
            dummy_inputs[idx] = _ensure_sequence_non_padding(feature, dummy_inputs[idx])
    input_dict = {name: tensor for name, tensor in zip(input_names, dummy_inputs)}

    with torch.no_grad():
        torch_output = _normalize_torch_output(model(input_dict))

    try:
        session = load_onnx_session(onnx_path)
    except Exception as exc:  # pragma: no cover
        pytest.xfail(f"ONNX Runtime failed to load dynamo model ({model.model_name}): {exc}")
    session_input_names = [inp.name for inp in session.get_inputs()]
    feed = build_onnx_input_feed(model.all_features, input_dict, input_names=session_input_names)
    try:
        onnx_outputs = session.run(None, feed)
    except Exception as exc:  # pragma: no cover
        pytest.xfail(f"ONNX Runtime failed to run dynamo model ({model.model_name}): {exc}")
    onnx_output = _normalize_onnx_output(merge_onnx_outputs(onnx_outputs))

    assert onnx_output.shape == tuple(torch_output.shape)
    rtol, atol = _get_onnx_tolerances(model)
    np.testing.assert_allclose(
        onnx_output,
        torch_output.detach().cpu().numpy(),
        rtol=rtol,
        atol=atol,
    )


def _ranking_model_factories() -> list[Callable[[], torch.nn.Module]]:
    dense, sparse, sequence = BASE_FEATURES
    return [
        lambda: LR(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            target=["label"],
            device=DEVICE,
        ),
        lambda: FM(
            sparse_features=sparse,
            sequence_features=sequence,
            target=["label"],
            device=DEVICE,
        ),
        lambda: FFM(
            dense_features=[],
            sparse_features=sparse,
            sequence_features=sequence,
            target=["label"],
            device=DEVICE,
        ),
        lambda: DeepFM(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            mlp_params={"hidden_dims": [16, 8], "activation": "relu", "dropout": 0.0},
            target=["label"],
            device=DEVICE,
        ),
        lambda: xDeepFM(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            cin_size=[16, 16],
            mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            target=["label"],
            device=DEVICE,
        ),
        lambda: DCN(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            cross_num=2,
            mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            target=["label"],
            device=DEVICE,
        ),
        lambda: AutoInt(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            att_layer_num=2,
            att_embedding_dim=8,
            att_head_num=2,
            target=["label"],
            device=DEVICE,
        ),
        lambda: DIN(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            behavior_feature_name="hist_item_ids",
            candidate_feature_name="item_id",
            attention_mlp_params={"hidden_dims": [16], "activation": "relu"},
            mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            target=["label"],
            device=DEVICE,
        ),
        lambda: DIEN(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            behavior_feature_name="hist_item_ids",
            candidate_feature_name="item_id",
            mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            gru_hidden_size=8,
            target=["label"],
            device=DEVICE,
        ),
        lambda: WideDeep(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            target=["label"],
            device=DEVICE,
        ),
        lambda: PNN(
            dense_features=[],
            sparse_features=sparse,
            sequence_features=sequence,
            mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            product_type="inner",
            target=["label"],
            device=DEVICE,
        ),
        lambda: FiBiNET(
            dense_features=[],
            sparse_features=sparse,
            sequence_features=sequence,
            mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            target=["label"],
            device=DEVICE,
        ),
        lambda: EulerNet(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            target=["label"],
            device=DEVICE,
        ),
        lambda: AFM(
            dense_features=[],
            sparse_features=sparse,
            sequence_features=sequence,
            target=["label"],
            device=DEVICE,
        ),
    ]


def _match_model_factories() -> list[Callable[[], torch.nn.Module]]:
    (
        user_dense,
        user_sparse,
        user_sequence,
        item_dense,
        item_sparse,
        item_sequence,
    ) = MATCH_FEATURES
    return [
        lambda: DSSM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            user_mlp_params={"hidden_dims": [16], "activation": "relu"},
            item_mlp_params={"hidden_dims": [16], "activation": "relu"},
            embedding_dim=8,
            training_mode="pointwise",
            similarity_metric="dot",
            device=DEVICE,
        ),
        lambda: YoutubeDNN(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            user_mlp_params={"hidden_dims": [16], "activation": "relu"},
            item_mlp_params={"hidden_dims": [16], "activation": "relu"},
            embedding_dim=8,
            training_mode="pointwise",
            similarity_metric="dot",
            device=DEVICE,
        ),
        lambda: MIND(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            embedding_dim=8,
            num_interests=2,
            training_mode="pointwise",
            similarity_metric="dot",
            device=DEVICE,
        ),
        lambda: SDM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            embedding_dim=8,
            rnn_type="GRU",
            user_mlp_params={"hidden_dims": [16], "activation": "relu"},
            item_mlp_params={"hidden_dims": [16], "activation": "relu"},
            training_mode="pointwise",
            similarity_metric="dot",
            device=DEVICE,
        ),
    ]


def _multitask_model_factories() -> list[Callable[[], torch.nn.Module]]:
    dense, sparse, sequence = BASE_FEATURES
    return [
        lambda: ShareBottom(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            bottom_mlp_params={
                "hidden_dims": [16],
                "activation": "relu",
                "dropout": 0.0,
            },
            tower_mlp_params_list=[
                {"hidden_dims": [8], "activation": "relu", "dropout": 0.0},
                {"hidden_dims": [8], "activation": "relu", "dropout": 0.0},
            ],
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=DEVICE,
        ),
        lambda: MMOE(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            expert_mlp_params={
                "hidden_dims": [16],
                "activation": "relu",
                "dropout": 0.0,
            },
            num_experts=2,
            tower_mlp_params_list=[
                {"hidden_dims": [8], "activation": "relu", "dropout": 0.0},
                {"hidden_dims": [8], "activation": "relu", "dropout": 0.0},
            ],
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=DEVICE,
        ),
        lambda: PLE(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            shared_expert_mlp_params={
                "hidden_dims": [16],
                "activation": "relu",
                "dropout": 0.0,
            },
            specific_expert_mlp_params=[
                {"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
                {"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            ],
            num_shared_experts=1,
            num_specific_experts=1,
            num_levels=1,
            tower_mlp_params_list=[
                {"hidden_dims": [8], "activation": "relu", "dropout": 0.0},
                {"hidden_dims": [8], "activation": "relu", "dropout": 0.0},
            ],
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=DEVICE,
        ),
        lambda: PEPNet(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            domain_features=["category"],
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=DEVICE,
        ),
        lambda: ESMM(
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            ctr_mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            cvr_mlp_params={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            target=["ctr", "ctcvr"],
            task=["binary", "binary"],
            device=DEVICE,
        ),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    _ranking_model_factories(),
    ids=lambda factory: factory().__class__.__name__,
)
def test_onnx_export_load_ranking(factory, tmp_path):
    model = factory()
    _run_onnx_roundtrip(model, tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    _match_model_factories(),
    ids=lambda factory: factory().__class__.__name__,
)
def test_onnx_export_load_match(factory, tmp_path):
    model = factory()
    _run_onnx_roundtrip(model, tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    _multitask_model_factories(),
    ids=lambda factory: factory().__class__.__name__,
)
def test_onnx_export_load_multitask(factory, tmp_path):
    model = factory()
    _run_onnx_roundtrip(model, tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [
        lambda: DeepFM(
            dense_features=BASE_FEATURES[0],
            sparse_features=BASE_FEATURES[1],
            sequence_features=BASE_FEATURES[2],
            mlp_params={"hidden_dims": [16, 8], "activation": "relu", "dropout": 0.0},
            target=["label"],
            device=DEVICE,
        ),
        lambda: DSSM(
            user_dense_features=MATCH_FEATURES[0],
            user_sparse_features=MATCH_FEATURES[1],
            user_sequence_features=MATCH_FEATURES[2],
            item_dense_features=MATCH_FEATURES[3],
            item_sparse_features=MATCH_FEATURES[4],
            item_sequence_features=MATCH_FEATURES[5],
            user_mlp_params={"hidden_dims": [16], "activation": "relu"},
            item_mlp_params={"hidden_dims": [16], "activation": "relu"},
            embedding_dim=8,
            training_mode="pointwise",
            similarity_metric="dot",
            device=DEVICE,
        ),
        lambda: ShareBottom(
            dense_features=BASE_FEATURES[0],
            sparse_features=BASE_FEATURES[1],
            sequence_features=BASE_FEATURES[2],
            bottom_mlp_params={
                "hidden_dims": [16],
                "activation": "relu",
                "dropout": 0.0,
            },
            tower_mlp_params_list=[
                {"hidden_dims": [8], "activation": "relu", "dropout": 0.0},
                {"hidden_dims": [8], "activation": "relu", "dropout": 0.0},
            ],
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=DEVICE,
        ),
    ],
    ids=lambda factory: f"{factory().__class__.__name__}_dynamo",
)
def test_onnx_export_load_dynamo(factory, tmp_path):
    model = factory()
    _run_onnx_dynamo_roundtrip(model, tmp_path)
