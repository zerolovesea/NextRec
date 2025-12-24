import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from nextrec.utils import data as data_utils


def _make_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_resolve_file_paths_file_and_dir(tmp_path):
    csv_path = tmp_path / "data.csv"
    _make_df().to_csv(csv_path, index=False)

    file_paths, file_type = data_utils.resolve_file_paths(str(csv_path))
    assert file_paths == [str(csv_path)]
    assert file_type == "csv"

    (tmp_path / "a.csv").write_text("a\n1\n")
    (tmp_path / "b.csv").write_text("a\n2\n")
    file_paths, file_type = data_utils.resolve_file_paths(str(tmp_path))
    assert file_type == "csv"
    assert file_paths == sorted(file_paths)


def test_resolve_file_paths_parquet_and_errors(tmp_path):
    parquet_path = tmp_path / "data.parquet"
    _make_df().to_parquet(parquet_path)
    file_paths, file_type = data_utils.resolve_file_paths(str(parquet_path))
    assert file_paths == [str(parquet_path)]
    assert file_type == "parquet"

    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    (mixed_dir / "a.csv").write_text("a\n1\n")
    _make_df().to_parquet(mixed_dir / "b.parquet")
    with pytest.raises(ValueError):
        data_utils.resolve_file_paths(str(mixed_dir))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        data_utils.resolve_file_paths(str(empty_dir))

    with pytest.raises(ValueError):
        data_utils.resolve_file_paths(str(tmp_path / "missing"))

    bad_ext = tmp_path / "data.json"
    bad_ext.write_text('{"a": 1}\n')
    with pytest.raises(ValueError):
        data_utils.resolve_file_paths(str(bad_ext))


def test_read_table_csv_and_parquet(tmp_path):
    df = _make_df()
    csv_path = tmp_path / "data.csv"
    parquet_path = tmp_path / "data.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path)

    assert_frame_equal(data_utils.read_table(csv_path), df)
    assert_frame_equal(data_utils.read_table(parquet_path), df)
    assert_frame_equal(data_utils.read_table(csv_path, data_format="csv"), df)

    parquet_dir = tmp_path / "parquet_dir"
    parquet_dir.mkdir()
    df.to_parquet(parquet_dir / "part.parquet")
    assert_frame_equal(data_utils.read_table(parquet_dir), df)

    with pytest.raises(ValueError):
        data_utils.read_table(csv_path, data_format="json")


def test_load_dataframes(tmp_path):
    df = _make_df()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    dataframes = data_utils.load_dataframes([str(csv_path)], "csv")
    assert len(dataframes) == 1
    assert_frame_equal(dataframes[0], df)


def test_iter_file_chunks_csv_and_parquet(tmp_path):
    df = pd.DataFrame({"a": list(range(5))})
    csv_path = tmp_path / "data.csv"
    parquet_path = tmp_path / "data.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path)

    csv_chunks = list(data_utils.iter_file_chunks(str(csv_path), "csv", chunk_size=2))
    assert sum(len(chunk) for chunk in csv_chunks) == len(df)

    parquet_chunks = list(
        data_utils.iter_file_chunks(str(parquet_path), "parquet", chunk_size=2)
    )
    assert sum(len(chunk) for chunk in parquet_chunks) == len(df)


def test_iter_file_chunks_orc(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.orc as orc

    df = pd.DataFrame({"a": list(range(5))})
    orc_path = tmp_path / "data.orc"
    table = pa.Table.from_pandas(df)
    orc.write_table(table, str(orc_path))

    with pytest.raises(ValueError):
        list(data_utils.iter_file_chunks(str(orc_path), "orc", chunk_size=2))


def test_iter_file_chunks_non_streaming_raises():
    for fmt in ["feather", "excel", "hdf5"]:
        with pytest.raises(ValueError):
            list(data_utils.iter_file_chunks("unused", fmt, chunk_size=2))


def test_default_output_dir(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n")
    assert (
        data_utils.default_output_dir(str(csv_path)) == tmp_path / "data_preprocessed"
    )

    dir_path = tmp_path / "input"
    dir_path.mkdir()
    assert (
        data_utils.default_output_dir(str(dir_path)) == tmp_path / "input_preprocessed"
    )


def test_read_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("a: 1\nb: test\n", encoding="utf-8")
    assert data_utils.read_yaml(yaml_path) == {"a": 1, "b": "test"}

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    assert data_utils.read_yaml(empty_yaml) == {}


def test_generate_ranking_data_basic():
    df, dense_features, sparse_features, sequence_features = (
        data_utils.generate_ranking_data(
            n_samples=8,
            n_dense=2,
            n_sparse=4,
            n_sequences=2,
            sequence_max_len=6,
            seed=1,
        )
    )

    assert df.shape[0] == 8
    assert len(dense_features) == 2
    assert len(sparse_features) == 4
    assert len(sequence_features) == 2
    assert "label" in df.columns
    assert "sequence_0" in df.columns
    assert all(len(seq) == 6 for seq in df["sequence_0"])
    assert sequence_features[0].embedding_name == "item_emb"
    assert sequence_features[1].embedding_name == "sparse_0_emb"


def test_generate_ranking_data_custom_sparse_features():
    df, _, sparse_features, sequence_features = data_utils.generate_ranking_data(
        n_samples=6,
        n_dense=1,
        n_sparse=4,
        n_sequences=2,
        sequence_max_len=6,
        seed=2,
        custom_sparse_features={"gender": 2, "category": 5},
    )

    assert "hist_items" in df.columns
    assert "hist_categories" in df.columns
    sparse_names = {feat.name for feat in sparse_features}
    assert "category" in sparse_names
    assert sequence_features[0].embedding_name == "item_emb"
    assert sequence_features[1].embedding_name == "category_emb"


def test_generate_match_data_shapes():
    (
        df,
        user_dense,
        user_sparse,
        user_seq,
        item_dense,
        item_sparse,
        item_seq,
    ) = data_utils.generate_match_data(n_samples=5, sequence_max_len=12, seed=3)

    assert df.shape[0] == 5
    assert len(user_dense) == 1
    assert len(user_sparse) == 6
    assert len(user_seq) == 2
    assert len(item_dense) == 1
    assert len(item_sparse) == 6
    assert len(item_seq) == 0
    assert all(len(seq) == 12 for seq in df["user_hist_items"])


def test_generate_multitask_data_shapes():
    df, dense, sparse, seq = data_utils.generate_multitask_data(
        n_samples=6, n_dense=2, n_sparse=4, n_sequences=2, sequence_max_len=6, seed=4
    )
    assert df.shape[0] == 6
    assert len(dense) == 2
    assert len(sparse) == 4
    assert len(seq) == 2
    assert {"click", "conversion", "ctcvr"}.issubset(df.columns)


def test_generate_distributed_ranking_data():
    df, dense, sparse, seq = data_utils.generate_distributed_ranking_data(
        num_samples=5, num_users=10, num_items=11, seed=5
    )
    assert df.shape[0] == 5
    assert len(dense) == 5
    assert len(sparse) == 6
    assert len(seq) == 2
    sparse_names = {feat.name for feat in sparse}
    assert {"gender", "age_group", "category", "city"}.issubset(sparse_names)


def test_generate_synthetic_embeddings():
    item_ids, embeddings = data_utils.generate_synthetic_embeddings(
        num_samples=4, embedding_dim=3
    )
    assert item_ids.tolist() == [0, 1, 2, 3]
    assert embeddings.shape == (4, 3)
    assert embeddings.dtype == torch.float32


def test_read_table_feather_and_orc(tmp_path):
    df = _make_df()
    pytest.importorskip("pyarrow")

    feather_path = tmp_path / "data.feather"
    df.to_feather(feather_path)
    assert_frame_equal(data_utils.read_table(feather_path), df, check_dtype=False)

    import pyarrow as pa
    import pyarrow.orc as orc

    orc_path = tmp_path / "data.orc"
    orc.write_table(pa.Table.from_pandas(df), str(orc_path))
    with pytest.raises(ValueError):
        data_utils.read_table(orc_path)


def test_read_table_excel(tmp_path):
    df = _make_df()
    pytest.importorskip("openpyxl", reason="openpyxl required for Excel I/O")

    excel_path = tmp_path / "data.xlsx"
    df.to_excel(excel_path, index=False)
    assert_frame_equal(data_utils.read_table(excel_path), df, check_dtype=False)


def test_read_table_hdf5(tmp_path):
    df = _make_df()
    pytest.importorskip("tables", reason="tables required for HDF5 I/O")

    hdf_path = tmp_path / "data.h5"
    df.to_hdf(hdf_path, key="data", mode="w")
    assert_frame_equal(data_utils.read_table(hdf_path), df, check_dtype=False)
