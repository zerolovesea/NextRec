import pandas as pd
import torch
import torch.nn as nn
import math
import pytest

from nextrec.basic.features import DenseFeature
from nextrec.basic.model import BaseModel
from nextrec.utils.data import expand_tabular_rows, get_expand_columns


class _DummyPredictExpandModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummyPredictExpand"

    @property
    def default_task(self) -> str:
        return "binary"

    def forward(self, X_input):
        return X_input["product"].reshape(-1, 1).float()

    def __init__(self):
        super().__init__(
            dense_features=[DenseFeature(name="product", proj_dim=1)],
            sparse_features=[],
            sequence_features=[],
            target=[],
            task="binary",
            id_columns=["uid"],
        )
        self.bias = nn.Parameter(torch.zeros(1))


def test_get_expand_columns_rejects_invalid_values():
    try:
        get_expand_columns({"product": "p1"})
    except TypeError:
        pass
    else:
        raise AssertionError("get_expand_columns should reject scalar candidate values")


def test_expand_tabular_rows_pandas_cartesian_product():
    df = pd.DataFrame({"uid": ["u1", "u2"], "base": [10, 20]})
    expanded = expand_tabular_rows(df, {"product": [1, 2], "channel": ["app", "h5"]})

    assert len(expanded) == 8
    assert expanded["product"].tolist()[:4] == [1, 1, 2, 2]
    assert expanded["channel"].tolist()[:4] == ["app", "h5", "app", "h5"]


def test_predict_expands_rows_and_includes_expand_column():
    model = _DummyPredictExpandModel()
    data = pd.DataFrame({"uid": ["u1", "u2"]})

    result = model.predict(
        data=data,
        batch_size=2,
        return_dataframe=True,
        expand={"product": [1, 2, 3]},
    )

    assert list(result.columns) == ["uid", "product", "pred_0"]
    assert result["uid"].tolist() == ["u1", "u1", "u1", "u2", "u2", "u2"]
    assert result["product"].tolist() == ["1", "2", "3", "1", "2", "3"]
    expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    assert result["pred_0"].tolist() == pytest.approx([1 / (1 + math.exp(-x)) for x in expected])


def test_predict_streaming_expands_rows_per_chunk(tmp_path):
    model = _DummyPredictExpandModel()
    input_path = tmp_path / "pred.csv"
    output_path = tmp_path / "pred_out.csv"
    pd.DataFrame({"uid": ["u1", "u2"]}).to_csv(input_path, index=False)

    saved = model.predict(
        data=str(input_path),
        batch_size=2,
        save_path=str(output_path),
        save_format="csv",
        return_dataframe=False,
        stream_chunk_size=1,
        expand={"product": [4, 5]},
    )

    result = pd.read_csv(saved)
    assert result["uid"].tolist() == ["u1", "u1", "u2", "u2"]
    assert result["product"].astype(str).tolist() == ["4", "5", "4", "5"]
    expected = [4.0, 5.0, 4.0, 5.0]
    assert result["pred_0"].tolist() == pytest.approx([1 / (1 + math.exp(-x)) for x in expected])
