import pandas as pd
import polars as pl
import pytest
import torch
from pandas.testing import assert_frame_equal

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.data.dataloader import RecDataLoader, prepare_sequence_column
from nextrec.data.data_processing import to_column_names
from nextrec.utils import data as data_utils


def _make_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_prepare_sequence_column_parses_list_strings():
    feature = SequenceFeature(name="sequence_0", vocab_size=100, embedding_dim=8, max_len=5, padding_idx=0)
    column = pd.Series(["[1, 2, 3]", "[4]", "[]"])

    actual = prepare_sequence_column(column, feature)

    assert actual.tolist() == [
        [1, 2, 3, 0, 0],
        [4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_get_file_paths_file_and_dir(tmp_path):
    csv_path = tmp_path / "data.csv"
    _make_df().to_csv(csv_path, index=False)

    file_paths, file_type = data_utils.get_file_paths(str(csv_path))
    assert file_paths == [str(csv_path)]
    assert file_type == "csv"

    (tmp_path / "a.csv").write_text("a\n1\n")
    (tmp_path / "b.csv").write_text("a\n2\n")
    file_paths, file_type = data_utils.get_file_paths(str(tmp_path))
    assert file_type == "csv"
    assert file_paths == sorted(file_paths)


def test_get_file_paths_parquet_and_errors(tmp_path):
    parquet_path = tmp_path / "data.parquet"
    _make_df().to_parquet(parquet_path)
    file_paths, file_type = data_utils.get_file_paths(str(parquet_path))
    assert file_paths == [str(parquet_path)]
    assert file_type == "parquet"

    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    (mixed_dir / "a.csv").write_text("a\n1\n")
    _make_df().to_parquet(mixed_dir / "b.parquet")
    with pytest.raises(ValueError):
        data_utils.get_file_paths(str(mixed_dir))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        data_utils.get_file_paths(str(empty_dir))

    with pytest.raises(ValueError):
        data_utils.get_file_paths(str(tmp_path / "missing"))

    bad_ext = tmp_path / "data.json"
    bad_ext.write_text('{"a": 1}\n')
    with pytest.raises(ValueError):
        data_utils.get_file_paths(str(bad_ext))


def test_read_table_csv_and_parquet(tmp_path):
    df = _make_df()
    csv_path = tmp_path / "data.csv"
    parquet_path = tmp_path / "data.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path)

    assert_frame_equal(data_utils.read_table(csv_path, data_format="csv", engine="pandas"), df)
    assert_frame_equal(data_utils.read_table(parquet_path, data_format="parquet", engine="pandas"), df)
    assert_frame_equal(data_utils.read_table(csv_path, data_format="csv", engine="pandas"), df)

    parquet_dir = tmp_path / "parquet_dir"
    parquet_dir.mkdir()
    df.to_parquet(parquet_dir / "part.parquet")
    assert_frame_equal(data_utils.read_table(parquet_dir, engine="pandas"), df)

    with pytest.raises(ValueError):
        data_utils.read_table(csv_path, data_format="json", engine="pandas")


def test_read_table_polars_engine(tmp_path):
    """Test read_table with polars engine."""
    df = _make_df()
    csv_path = tmp_path / "data.csv"
    parquet_path = tmp_path / "data.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path)

    # Test CSV reading with polars engine (default)
    pl_csv = data_utils.read_table(csv_path, data_format="csv", engine="polars")
    assert isinstance(pl_csv, pl.DataFrame)
    assert pl_csv.shape == (3, 2)
    assert pl_csv.columns == ["a", "b"]

    # Test parquet reading with polars engine
    pl_parquet = data_utils.read_table(parquet_path, data_format="parquet", engine="polars")
    assert isinstance(pl_parquet, pl.DataFrame)
    assert pl_parquet.shape == (3, 2)

    # Test directory reading with polars engine
    parquet_dir = tmp_path / "parquet_dir"
    parquet_dir.mkdir()
    df.to_parquet(parquet_dir / "part.parquet")
    pl_dir = data_utils.read_table(parquet_dir, engine="polars")
    assert isinstance(pl_dir, pl.DataFrame)
    assert pl_dir.shape == (3, 2)

    # Test default engine is polars
    pl_default = data_utils.read_table(csv_path, data_format="csv")
    assert isinstance(pl_default, pl.DataFrame)


def test_to_column_names_handles_scalar_sequence_and_none():
    assert to_column_names(None) == []
    assert to_column_names("user_id") == ["user_id"]
    assert to_column_names(("user_id", 2)) == ["user_id", "2"]


def test_iter_file_chunks_csv_and_parquet(tmp_path):
    df = pd.DataFrame({"a": list(range(5))})
    csv_path = tmp_path / "data.csv"
    parquet_path = tmp_path / "data.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path)

    csv_chunks = list(data_utils.iter_file_chunks(str(csv_path), "csv", chunk_size=2))
    assert sum(len(chunk) for chunk in csv_chunks) == len(df)

    parquet_chunks = list(data_utils.iter_file_chunks(str(parquet_path), "parquet", chunk_size=2))
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
    with pytest.raises(ValueError):
        list(data_utils.iter_file_chunks("unused", "excel", chunk_size=2))


def test_read_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("a: 1\nb: test\n", encoding="utf-8")
    assert data_utils.read_yaml(yaml_path) == {"a": 1, "b": "test"}

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    assert data_utils.read_yaml(empty_yaml) == {}


def test_generate_ranking_data_basic():
    df, dense_features, sparse_features, sequence_features = data_utils.generate_ranking_data(
        n_samples=8,
        n_dense=2,
        n_sparse=4,
        n_sequences=2,
        sequence_max_len=6,
        seed=1,
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


def test_generate_synthetic_embeddings():
    item_ids, embeddings = data_utils.generate_synthetic_embeddings(num_samples=4, embedding_dim=3)
    assert item_ids.tolist() == [0, 1, 2, 3]
    assert embeddings.shape == (4, 3)
    assert embeddings.dtype == torch.float32


def test_read_table_orc(tmp_path):
    df = _make_df()
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.orc as orc

    orc_path = tmp_path / "data.orc"
    orc.write_table(pa.Table.from_pandas(df), str(orc_path))
    with pytest.raises(ValueError):
        data_utils.read_table(orc_path)


def test_streaming_single_csv_sharded_keeps_all_rows(tmp_path):
    rows = 121
    df = pd.DataFrame(
        {
            "user_id": list(range(rows)),
            "age": [float(i % 10) for i in range(rows)],
            "label": [float(i % 2) for i in range(rows)],
        }
    )
    csv_path = tmp_path / "single.csv"
    df.to_csv(csv_path, index=False)

    rec_loader = RecDataLoader(
        dense_features=[DenseFeature(name="age", proj_dim=1)],
        sparse_features=[SparseFeature(name="user_id", vocab_size=500, embedding_dim=8)],
        target=["label"],
        key_columns=["user_id"],
    )

    shard_ids = []
    for shard_rank in (0, 1):
        loader = rec_loader.create_dataloader(
            data=str(csv_path),
            batch_size=1,
            shuffle=False,
            streaming=True,
            chunk_size=20,
            num_workers=0,
            shard_rank=shard_rank,
            shard_count=2,
        )
        ids = []
        for batch in loader:
            id_batch = batch["keys"]["user_id"]
            ids.extend([int(x) for x in id_batch.tolist()])
        shard_ids.append(set(ids))

    assert len(shard_ids[0]) > 0
    assert len(shard_ids[1]) > 0
    assert shard_ids[0].isdisjoint(shard_ids[1])
    assert len(shard_ids[0] | shard_ids[1]) == rows
