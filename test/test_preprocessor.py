import ast
from pathlib import Path

import numpy as np
import pandas as pd

from nextrec.data.preprocessor import DataProcessor


def _build_processor():
    processor = DataProcessor()
    processor.add_numeric_feature("age", scaler="minmax")
    processor.add_sparse_feature("user_id", encode_method="label")
    processor.add_sequence_feature("hist", encode_method="label", max_len=3, pad_value=0, separator=",")
    processor.add_target("label", target_type="binary")
    return processor


def _sample_dataframe():
    return pd.DataFrame(
        {
            "age": [10, 20, 30],
            "user_id": ["u1", "u2", "u1"],
            "hist": ["a,b", "c", ""],
            "label": [1, 0, 1],
        }
    )


def test_fit_transform_in_memory_sets_encoders_and_shapes():
    df = _sample_dataframe()
    processor = _build_processor()

    output = processor.fit_transform(df, return_dict=True)

    assert processor.is_fitted
    assert set(output.keys()) == {
        "age",
        "user_id",
        "hist",
        "label",
        "age_minmax",
        "user_id_label",
        "hist_label",
    }

    # numeric minmax scaled between 0 and 1
    assert np.isclose(output["age_minmax"].min(), 0.0)
    assert np.isclose(output["age_minmax"].max(), 1.0)

    # sparse and sequence encodings should be integer arrays
    assert np.issubdtype(output["user_id_label"].dtype, np.integer)
    assert output["hist_label"].shape == (len(df), 3)
    assert np.issubdtype(output["hist_label"].dtype, np.integer)

    # target binary floats
    label_values = pd.to_numeric(output["label"], errors="coerce")
    assert not np.isnan(label_values).any()
    assert set(np.unique(label_values).tolist()) <= {0.0, 1.0}


def test_transform_path_writes_files(tmp_path: Path):
    df = _sample_dataframe()
    processor = _build_processor().fit(df)

    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, index=False)

    saved_paths = processor.transform(
        str(input_path),
        return_dict=False,
        output_path=tmp_path,
        save_format="csv",
    )

    assert len(saved_paths) == 1
    saved_file = Path(saved_paths[0])
    assert saved_file.exists()

    loaded = pd.read_csv(saved_file)
    assert set(loaded.columns) == {
        "age",
        "user_id",
        "hist",
        "label",
        "age_minmax",
        "user_id_label",
        "hist_label",
    }
    assert len(loaded) == len(df)

    # encoded sequence column should remain length-3 lists when round-tripped through CSV
    def _parse(seq_str):
        if pd.isna(seq_str):
            return []
        text = str(seq_str)
        try:
            parsed = ast.literal_eval(text)
            return list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else []
        except Exception:
            tokens = text.strip(" []").replace(",", " ").split()
            return [int(t) for t in tokens] if tokens else []

    parsed_hist = loaded["hist_label"].apply(_parse)
    assert all(len(seq) == 3 for seq in parsed_hist)


def test_fit_from_path_streams_and_transforms(tmp_path: Path):
    df = _sample_dataframe()
    input_path = tmp_path / "stream.csv"
    df.to_csv(input_path, index=False)

    processor = _build_processor()
    processor.fit(str(input_path))

    assert processor.is_fitted
    assert "age_minmax" in processor.scalers
    assert processor.sparse_features["user_id_label"].get("vocab_size", 0) > 0

    transformed = processor.transform(df, return_dict=True)
    assert transformed["age_minmax"].shape == (len(df),)
    assert transformed["hist_label"].shape == (len(df), 3)


def test_sparse_and_sequence_row_filters_are_applied_in_fit_and_transform():
    df = pd.DataFrame(
        {
            "age": [10, 20, 30, 40],
            "user_id": ["u1", "u2", "u3", "u2"],
            "hist": ["a,b", "x,y", "b,c", "a"],
            "label": [1, 0, 1, 0],
        }
    )
    processor = DataProcessor()
    processor.add_numeric_feature("age", scaler="minmax")
    processor.add_sparse_feature("user_id", encode_method="label", keep_value=["u1", "u2"])
    processor.add_sequence_feature(
        "hist",
        encode_method="label",
        max_len=3,
        pad_value=0,
        separator=",",
        filter_value=["x"],
    )
    processor.add_target("label", target_type="binary")

    output = processor.fit_transform(df, return_dict=True)

    # Remaining rows should be:
    # - user_id in {u1, u2}
    # - hist does not contain token x
    assert output["age_minmax"].shape == (2,)
    assert output["hist_label"].shape == (2, 3)
    assert set(output["user_id"].tolist()) <= {"u1", "u2"}
    assert all("x" not in str(v) for v in output["hist"].tolist())
    assert processor.sparse_features["user_id_label"]["vocab_size"] == 3  # u1, u2, <UNK>


def test_sparse_row_filters_support_contains_match_mode():
    df = pd.DataFrame(
        {
            "user_id": ["usr_100", "guest_001", "usr_200"],
            "label": [1, 0, 1],
        }
    )
    processor = DataProcessor()
    processor.add_sparse_feature(
        "user_id",
        encode_method="label",
        keep_value=["usr_"],
        match_mode="contains",
    )
    processor.add_target("label", target_type="binary")

    output = processor.fit_transform(df, return_dict=True)

    assert output["user_id"].shape == (2,)
    assert set(output["user_id"].tolist()) == {"usr_100", "usr_200"}


def test_sequence_row_filters_support_regex_match_mode():
    df = pd.DataFrame(
        {
            "hist": ["a,b", "x1,y", "x9,z", "b,c"],
            "label": [1, 0, 1, 0],
        }
    )
    processor = DataProcessor()
    processor.add_sequence_feature(
        "hist",
        encode_method="label",
        max_len=3,
        pad_value=0,
        separator=",",
        filter_value=[r"^x\d$"],
        match_mode="regex",
    )
    processor.add_target("label", target_type="binary")

    output = processor.fit_transform(df, return_dict=True)

    assert output["hist"].shape == (2,)
    assert set(output["hist"].tolist()) == {"a,b", "b,c"}


def test_keep_row_filters_across_features_use_or_semantics():
    df = pd.DataFrame(
        {
            "seq_a": ["ZQ-001,x", "x,y", "none"],
            "seq_b": ["none", "GP-777,z", "none"],
            "label": [1, 0, 1],
        }
    )
    processor = DataProcessor()
    processor.add_sequence_feature(
        "seq_a",
        encode_method="label",
        max_len=4,
        separator=",",
        keep_value=["ZQ-"],
        match_mode="contains",
    )
    processor.add_sequence_feature(
        "seq_b",
        encode_method="label",
        max_len=4,
        separator=",",
        keep_value=["GP-"],
        match_mode="contains",
    )
    processor.add_target("label", target_type="binary")

    output = processor.fit_transform(df, return_dict=True)

    assert output["seq_a"].shape == (2,)
    assert output["seq_b"].shape == (2,)
    assert output["label"].shape == (2,)


def test_numeric_scaler_pipeline_supports_ordered_transform_list():
    df = pd.DataFrame({"age": [0.0, 1.0, 9.0, 99.0], "label": [0, 1, 0, 1]})
    processor = DataProcessor()
    processor.add_numeric_feature("age", scaler=["log", "minmax"])
    processor.add_target("label", target_type="binary")

    output = processor.fit_transform(df, return_dict=True)

    assert "age_log_minmax" in output
    transformed = output["age_log_minmax"]
    assert np.isclose(transformed.min(), 0.0)
    assert np.isclose(transformed.max(), 1.0)
