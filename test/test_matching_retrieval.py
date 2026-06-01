import numpy as np
import pandas as pd
import pytest

from nextrec.basic.features import SparseFeature
from nextrec.models.matching.dssm import DSSM
from nextrec.utils.matching import EmbeddingTable, FaissVectorIndex


def build_simple_dssm():
    return DSSM(
        user_sparse_features=[SparseFeature(name="user_id", vocab_size=10, embedding_dim=4)],
        item_sparse_features=[SparseFeature(name="item_id", vocab_size=10, embedding_dim=4)],
        user_mlp_params={"hidden_dims": [8], "output_dim": 4},
        item_mlp_params={"hidden_dims": [8], "output_dim": 4},
        embedding_dim=4,
        similarity_metric="dot",
        device="cpu",
    )


def test_embedding_table_roundtrip(tmp_path):
    table = EmbeddingTable(
        ids=np.asarray(["a", "b", "c"]),
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        id_column="item_id",
        tower="item",
        metric="dot",
    )

    artifact_path = table.save(tmp_path, save_format="csv")
    loaded = EmbeddingTable.load(artifact_path)

    assert loaded.ids.tolist() == ["a", "b", "c"]
    assert np.allclose(loaded.embeddings, table.embeddings)
    assert loaded.id_column == "item_id"


def test_matching_model_exports_embedding_table_and_searches(tmp_path):
    pytest.importorskip("faiss")
    model = build_simple_dssm()
    item_df = pd.DataFrame({"item_id": [1, 2, 3]})
    user_df = pd.DataFrame({"user_id": [1, 2]})

    artifact_path = model.export_embedding_table(
        tower="item",
        data=item_df,
        id_column="item_id",
        save_dir=tmp_path,
        save_format="csv",
        batch_size=2,
    )
    index = model.build_item_index(embedding_table=artifact_path, id_column="item_id")
    result = model.search(index=index, user_data=user_df, user_id_column="user_id", top_k=2)

    assert set(result.columns) == {"query_index", "item_id", "score", "rank", "user_id"}
    assert len(result) == 4


def test_faiss_search_clamps_top_k_to_index_size():
    class FakeFaissIndex:
        ntotal = 2

        def search(self, queries, k):
            assert k == self.ntotal
            return (
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                np.asarray([[0, 1]], dtype=np.int64),
            )

    index = object.__new__(FaissVectorIndex)
    index.metric = "dot"
    index.ids = np.asarray(["a", "b"])
    index.index = FakeFaissIndex()

    retrieved_ids, scores = index.search(
        query_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        top_k=5,
    )

    assert retrieved_ids.shape == (1, 2)
    assert scores.shape == (1, 2)
    assert retrieved_ids[0].tolist() == ["a", "b"]


def test_matching_model_streams_embedding_table_export(tmp_path):
    model = build_simple_dssm()
    item_df = pd.DataFrame({"item_id": [1, 2, 3, 4]})

    artifact_path = model.export_embedding_table(
        tower="item",
        data=item_df,
        id_column="item_id",
        save_dir=tmp_path,
        save_format="csv",
        batch_size=2,
        streaming=True,
    )
    loaded = EmbeddingTable.load(artifact_path)

    assert loaded.ids.tolist() == ["1", "2", "3", "4"]
    assert loaded.embeddings.shape == (4, 4)
    assert loaded.id_column == "item_id"
    assert loaded.tower == "item"


def test_encode_user_with_ids_returns_single_vector_matrix():
    model = build_simple_dssm()

    user_embeddings = model.encode_user(pd.DataFrame({"user_id": [1, 2]}), id_column="user_id")

    assert user_embeddings.shape == (2, 4)


def test_encode_tower_requires_id_column():
    model = build_simple_dssm()

    with pytest.raises(ValueError, match="id_column is required"):
        model.encode_tower("user", pd.DataFrame({"user_id": [1, 2]}), id_column=None)


def test_full_corpus_evaluate_returns_topk_metrics():
    pytest.importorskip("faiss")
    model = build_simple_dssm()
    item_df = pd.DataFrame({"item_id": [1, 2, 3, 4]})
    eval_df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [1, 2, 3]})

    metrics = model.full_corpus_evaluate(
        eval_data=eval_df,
        item_data=item_df,
        user_col="user_id",
        item_col="item_id",
        top_k=(1, 2),
        batch_size=2,
    )

    assert set(metrics) == {
        "recall@1",
        "hitrate@1",
        "ndcg@1",
        "mrr@1",
        "recall@2",
        "hitrate@2",
        "ndcg@2",
        "mrr@2",
    }
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_search_streams_user_batches_to_file(tmp_path):
    pytest.importorskip("faiss")
    model = build_simple_dssm()
    item_df = pd.DataFrame({"item_id": [1, 2, 3]})
    user_df = pd.DataFrame({"user_id": [1, 2, 3, 4]})
    index = model.build_item_index(item_data=item_df, id_column="item_id", batch_size=2)

    output_path = model.search(
        index=index,
        user_data=user_df,
        user_id_column="user_id",
        save_path=tmp_path / "retrieval.csv",
        top_k=2,
        batch_size=2,
        return_dataframe=False,
    )

    result = pd.read_csv(output_path)
    assert len(result) == 8
    assert set(result.columns) == {"user_id", "item_id", "score", "rank"}


def test_faiss_streaming_index_save_and_load(tmp_path):
    pytest.importorskip("faiss")
    model = build_simple_dssm()
    item_df = pd.DataFrame({"item_id": [1, 2, 3, 4]})
    user_embeddings = model.encode_user(pd.DataFrame({"user_id": [1]}), id_column="user_id")

    index = model.build_item_index(
        item_data=item_df,
        id_column="item_id",
        batch_size=2,
        save_dir=tmp_path / "faiss_index",
    )
    ids_before, scores_before = index.search(user_embeddings, top_k=2)
    loaded = FaissVectorIndex.load(tmp_path / "faiss_index")
    ids_after, scores_after = loaded.search(user_embeddings, top_k=2)

    assert ids_before.tolist() == ids_after.tolist()
    assert np.allclose(scores_before, scores_after)


def test_tower_onnx_export_requires_onnx_dependency(tmp_path):
    model = build_simple_dssm()
    pytest.importorskip("onnx")

    user_path = model.export_tower_onnx("user", save_path=tmp_path / "user_tower.onnx")
    item_path = model.export_tower_onnx("item", save_path=tmp_path / "item_tower.onnx")

    assert user_path.exists()
    assert item_path.exists()
