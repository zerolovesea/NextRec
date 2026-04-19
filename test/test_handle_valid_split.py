import numpy as np
import pandas as pd
import polars as pl
import pytest

from nextrec.data.data_processing import split_train_valid
from nextrec.utils.data import split_path_files


def _assert_group_disjoint(train_groups, valid_groups):
    train_set = set(np.asarray(train_groups).tolist())
    valid_set = set(np.asarray(valid_groups).tolist())
    assert train_set.isdisjoint(valid_set)


def test_handle_valid_split_dict_group_by_no_leakage():
    data = {
        "x": [1, 2, 3, 4],
        "label": [0, 1, 0, 1],
        "group": ["u1", "u1", "u2", "u2"],
    }

    train_split, valid_split = split_train_valid(
        data,
        valid_split=0.5,
        split_group_by="group",
    )

    assert len(train_split["x"]) + len(valid_split["x"]) == len(data["x"])
    _assert_group_disjoint(train_split["group"], valid_split["group"])


def test_handle_valid_split_pandas_group_by_no_leakage():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
            "group": ["u1", "u1", "u2", "u2"],
        }
    )

    train_split, valid_split = split_train_valid(
        df,
        valid_split=0.5,
        split_group_by="group",
    )

    assert len(train_split) + len(valid_split) == len(df)
    _assert_group_disjoint(train_split["group"], valid_split["group"])


def test_handle_valid_split_polars_group_by_no_leakage():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
            "group": ["u1", "u1", "u2", "u2"],
        }
    )

    train_split, valid_split = split_train_valid(
        df,
        valid_split=0.5,
        split_group_by="group",
    )

    assert train_split.height + valid_split.height == df.height
    _assert_group_disjoint(train_split.get_column("group"), valid_split.get_column("group"))


def test_handle_valid_split_path_files(tmp_path):
    file_paths = []
    for idx in range(5):
        path = tmp_path / f"part_{idx}.csv"
        path.write_text("x,label\n1,0\n", encoding="utf-8")
        file_paths.append(str(path))

    train_files, valid_files = split_path_files(file_paths, valid_split=0.4)

    assert train_files == file_paths[:3]
    assert valid_files == file_paths[3:]


def test_handle_valid_split_path_files_requires_multiple_files(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("x,label\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires at least 2 files"):
        split_path_files(str(path), valid_split=0.2)
