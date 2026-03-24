import numpy as np
import pandas as pd
import polars as pl
import torch

from nextrec.basic.model import BaseModel


class _DummySplitModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummySplit"

    @property
    def default_task(self):  # type: ignore[override]
        return "binary"

    def forward(self, X_input):  # pragma: no cover
        return torch.zeros(1, 1)

    def __init__(self):
        super().__init__(
            dense_features=[],
            sparse_features=[],
            sequence_features=[],
            target=None,
            task="binary",
        )


def _assert_group_disjoint(train_groups, valid_groups):
    train_set = set(np.asarray(train_groups).tolist())
    valid_set = set(np.asarray(valid_groups).tolist())
    assert train_set.isdisjoint(valid_set)


def test_handle_valid_split_dict_group_by_no_leakage():
    model = _DummySplitModel()
    data = {
        "x": [1, 2, 3, 4],
        "label": [0, 1, 0, 1],
        "group": ["u1", "u1", "u2", "u2"],
    }

    train_split, valid_split = model.handle_valid_split(
        data,
        valid_split=0.5,
        split_group_by="group",
    )

    assert len(train_split["x"]) + len(valid_split["x"]) == len(data["x"])
    _assert_group_disjoint(train_split["group"], valid_split["group"])


def test_handle_valid_split_pandas_group_by_no_leakage():
    model = _DummySplitModel()
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
            "group": ["u1", "u1", "u2", "u2"],
        }
    )

    train_split, valid_split = model.handle_valid_split(
        df,
        valid_split=0.5,
        split_group_by="group",
    )

    assert len(train_split) + len(valid_split) == len(df)
    _assert_group_disjoint(train_split["group"], valid_split["group"])


def test_handle_valid_split_polars_group_by_no_leakage():
    model = _DummySplitModel()
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
            "group": ["u1", "u1", "u2", "u2"],
        }
    )

    train_split, valid_split = model.handle_valid_split(
        df,
        valid_split=0.5,
        split_group_by="group",
    )

    assert train_split.height + valid_split.height == df.height
    _assert_group_disjoint(train_split.get_column("group"), valid_split.get_column("group"))
