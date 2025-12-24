import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nextrec.data.preprocessor import DataProcessor


def _build_processor():
    processor = DataProcessor()
    processor.add_numeric_feature("age", scaler="minmax")
    processor.add_sparse_feature("user_id", encode_method="label")
    processor.add_sequence_feature(
        "hist", encode_method="label", max_len=3, pad_value=0, separator=","
    )
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
    assert set(output.keys()) == {"age", "user_id", "hist", "label"}

    # numeric minmax scaled between 0 and 1
    assert np.isclose(output["age"].min(), 0.0)
    assert np.isclose(output["age"].max(), 1.0)

    # sparse and sequence encodings should be integer arrays
    assert output["user_id"].dtype == np.int64
    assert output["hist"].shape == (len(df), 3)
    assert output["hist"].dtype == np.int64

    # target binary floats
    assert set(np.unique(output["label"]).tolist()) <= {0.0, 1.0}


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
    assert list(loaded.columns) == ["age", "user_id", "hist", "label"]
    assert len(loaded) == len(df)

    # sequence column should remain length-3 lists when round-tripped through CSV
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

    parsed_hist = loaded["hist"].apply(_parse)
    assert all(len(seq) == 3 for seq in parsed_hist)


def test_fit_from_path_streams_and_transforms(tmp_path: Path):
    df = _sample_dataframe()
    input_path = tmp_path / "stream.csv"
    df.to_csv(input_path, index=False)

    processor = _build_processor()
    processor.fit(str(input_path), chunk_size=2)

    assert processor.is_fitted
    assert "age" in processor.scalers
    assert processor.sparse_features["user_id"].get("vocab_size", 0) > 0

    transformed = processor.transform(df, return_dict=True)
    assert transformed["age"].shape == (len(df),)
    assert transformed["hist"].shape == (len(df), 3)


def _write_non_streaming(df: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "feather":
        pytest.importorskip("pyarrow")
        df.to_feather(path)
        return
    if fmt == "excel":
        pytest.importorskip("openpyxl", reason="openpyxl required for Excel I/O")
        df.to_excel(path, index=False)
        return
    if fmt == "hdf5":
        pytest.importorskip("tables", reason="tables required for HDF5 I/O")
        df.to_hdf(path, key="data", mode="w")
        return
    raise ValueError(f"Unsupported format in test: {fmt}")


@pytest.mark.parametrize(
    "fmt,suffix",
    [("feather", ".feather"), ("excel", ".xlsx"), ("hdf5", ".h5")],
)
def test_fit_from_path_non_streaming_raises(tmp_path: Path, fmt: str, suffix: str):
    df = _sample_dataframe()
    processor = _build_processor()
    input_path = tmp_path / f"non_stream{suffix}"
    _write_non_streaming(df, input_path, fmt)

    with pytest.raises(ValueError, match="does not support streaming"):
        processor.fit(str(input_path), chunk_size=2)
