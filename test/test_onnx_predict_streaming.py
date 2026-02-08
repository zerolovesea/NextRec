"""
Tests for ONNX streaming prediction paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nextrec.basic.model as model_module
from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.lr import LR

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def test_predict_onnx_streaming_multiprocess_csv(tmp_path):
    dense_features = [
        DenseFeature(name="age", proj_dim=1),
        DenseFeature(name="price", proj_dim=1),
    ]
    sparse_features = [
        SparseFeature(name="user_id", vocab_size=200, embedding_dim=8),
        SparseFeature(name="category", vocab_size=50, embedding_dim=8),
    ]
    model = LR(
        dense_features=dense_features,
        sparse_features=sparse_features,
        target=["label"],
        id_columns=["user_id"],
        device="cpu",
    )

    rows = 121
    df = pd.DataFrame(
        {
            "age": np.random.rand(rows).astype(np.float32),
            "price": np.random.rand(rows).astype(np.float32),
            "user_id": np.random.randint(1, 200, size=rows),
            "category": np.random.randint(1, 50, size=rows),
            "label": np.random.randint(0, 2, size=rows).astype(np.float32),
        }
    )
    input_path = tmp_path / "predict_input.csv"
    output_path = tmp_path / "predict_output.csv"
    onnx_path = tmp_path / "lr.onnx"
    df.to_csv(input_path, index=False)

    model.export_onnx(save_path=onnx_path, batch_size=8)
    result = model.predict_onnx(
        onnx_path=onnx_path,
        data=str(input_path),
        batch_size=20,
        save_path=str(output_path),
        save_format="csv",
        include_ids=True,
        return_dataframe=False,
        num_workers=0,
        num_processes=2,
    )

    assert str(result) == str(output_path)
    assert output_path.exists()
    pred_df = pd.read_csv(output_path)
    assert len(pred_df) == rows
    assert "user_id" in pred_df.columns
    assert "label" in pred_df.columns


def test_predict_onnx_streaming_disables_worker_progress(tmp_path, monkeypatch):
    dense_features = [
        DenseFeature(name="age", proj_dim=1),
        DenseFeature(name="price", proj_dim=1),
    ]
    sparse_features = [
        SparseFeature(name="user_id", vocab_size=200, embedding_dim=8),
        SparseFeature(name="category", vocab_size=50, embedding_dim=8),
    ]
    model = LR(
        dense_features=dense_features,
        sparse_features=sparse_features,
        target=["label"],
        id_columns=["user_id"],
        device="cpu",
    )

    rows = 16
    df = pd.DataFrame(
        {
            "age": np.random.rand(rows).astype(np.float32),
            "price": np.random.rand(rows).astype(np.float32),
            "user_id": np.random.randint(1, 200, size=rows),
            "category": np.random.randint(1, 50, size=rows),
            "label": np.random.randint(0, 2, size=rows).astype(np.float32),
        }
    )
    onnx_path = tmp_path / "lr_worker_progress.onnx"
    output_path = tmp_path / "predict_output_worker_progress.csv"
    model.export_onnx(save_path=onnx_path, batch_size=8)

    progress_disable_flags: list[bool] = []

    def fake_progress(iterable, **kwargs):
        progress_disable_flags.append(bool(kwargs.get("disable", False)))
        for item in iterable:
            yield item

    monkeypatch.setattr(model_module, "progress", fake_progress)

    result = model.predict_onnx_streaming(
        onnx_path=onnx_path,
        data=df,
        batch_size=8,
        save_path=output_path,
        save_format="csv",
        include_ids=False,
        return_dataframe=False,
        id_columns=None,
        num_workers=0,
        num_processes=1,
        shard_rank=1,
        shard_count=2,
    )

    assert str(result) == str(output_path)
    assert progress_disable_flags and progress_disable_flags[-1] is True
