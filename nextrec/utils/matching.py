"""
Retrieval utilities for matching models.

Date: create on 09/05/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

RetrievalMetric = Literal["dot", "cosine", "euclidean"]


@dataclass
class EmbeddingTable:
    """In-memory table that binds entity IDs to tower embeddings."""

    ids: np.ndarray
    embeddings: np.ndarray
    id_column: str
    tower: Literal["user", "item"]
    metric: RetrievalMetric = "dot"

    def __post_init__(self) -> None:
        """Normalize array dtypes and validate row alignment."""
        self.ids = np.asarray(self.ids)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError("[EmbeddingTable Error] embeddings must be a 2D array.")
        if self.ids.shape[0] != self.embeddings.shape[0]:
            raise ValueError(
                "[EmbeddingTable Error] ids and embeddings must have the same row count. "
                f"Got {self.ids.shape[0]} ids and {self.embeddings.shape[0]} embeddings."
            )

    @property
    def dim(self) -> int:
        """Return the embedding vector dimension."""
        return int(self.embeddings.shape[1])

    def save(
        self,
        save_dir: str | Path,
        save_format: Literal["parquet", "csv"] = "parquet",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save embeddings and metadata under a target directory.

        The data file is named by tower, for example item_embeddings.parquet,
        and metadata.json stores tower, ID column, metric, row count, and dim.
        """
        target_dir = Path(save_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "tower": self.tower,
            "id_column": self.id_column,
            "metric": self.metric,
            "row_count": int(self.embeddings.shape[0]),
            "dim": self.dim,
            "format": save_format,
            **dict(metadata or {}),
        }
        (target_dir / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        df = self.to_dataframe()
        if save_format == "parquet":
            path = target_dir / f"{self.tower}_embeddings.parquet"
            df.to_parquet(path, index=False)
            return path
        if save_format == "csv":
            path = target_dir / f"{self.tower}_embeddings.csv"
            df.to_csv(path, index=False)
            return path
        raise ValueError("[EmbeddingTable Error] save_format must be one of: parquet, csv.")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the table to columns [id_column, emb_0, emb_1, ...]."""
        columns = [f"emb_{idx}" for idx in range(self.dim)]
        df = pd.DataFrame(self.embeddings, columns=columns)
        df.insert(0, self.id_column, self.ids.astype(str))
        return df

    @classmethod
    def load(
        cls,
        path: str | Path,
        tower: Literal["user", "item"] | None = None,
        id_column: str | None = None,
        metric: RetrievalMetric | None = None,
    ) -> "EmbeddingTable":
        """
        Load an embedding table from a parquet/csv file or artifact directory.

        Explicit arguments override values read from metadata.json.
        """
        path_obj = Path(path)
        metadata_path = path_obj / "metadata.json" if path_obj.is_dir() else path_obj.parent / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if path_obj.is_dir():
            candidates = sorted(path_obj.glob("*_embeddings.parquet")) or sorted(path_obj.glob("*_embeddings.csv"))
            if not candidates:
                raise FileNotFoundError(f"[EmbeddingTable Error] No embedding artifact found under: {path_obj}")
            path_obj = candidates[0]

        resolved_tower = tower or metadata.get("tower") or "item"
        resolved_id_column = id_column or metadata.get("id_column") or "id"
        resolved_metric = metric or metadata.get("metric") or "dot"

        if path_obj.suffix == ".parquet":
            df = pd.read_parquet(path_obj)
            ids, embeddings = dataframe_to_embedding_arrays(df, resolved_id_column)
        elif path_obj.suffix == ".csv":
            df = pd.read_csv(path_obj)
            ids, embeddings = dataframe_to_embedding_arrays(df, resolved_id_column)
        else:
            raise ValueError("[EmbeddingTable Error] Supported artifact formats: parquet, csv.")

        return cls(
            ids=ids,
            embeddings=embeddings,
            id_column=resolved_id_column,
            tower=resolved_tower,
            metric=resolved_metric,
        )

    @classmethod
    def iter_batches(
        cls,
        path: str | Path,
        id_column: str | None = None,
        tower: Literal["user", "item"] | None = None,
        metric: RetrievalMetric | None = None,
        batch_size: int = 10000,
    ):
        """
        Yield embedding-table batches from a parquet/csv artifact.

        Use this for large item tables when building a vector index without
        loading the full artifact into memory.
        """
        path_obj = Path(path)
        metadata_path = path_obj / "metadata.json" if path_obj.is_dir() else path_obj.parent / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if path_obj.is_dir():
            candidates = sorted(path_obj.glob("*_embeddings.parquet")) or sorted(path_obj.glob("*_embeddings.csv"))
            if not candidates:
                raise FileNotFoundError(f"[EmbeddingTable Error] No embedding artifact found under: {path_obj}")
            path_obj = candidates[0]

        resolved_tower = tower or metadata.get("tower") or "item"
        resolved_id_column = id_column or metadata.get("id_column") or "id"
        resolved_metric = metric or metadata.get("metric") or "dot"
        batch_size = max(int(batch_size), 1)

        if path_obj.suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path_obj)
            for record_batch in parquet_file.iter_batches(batch_size=batch_size):
                df = record_batch.to_pandas()
                ids, embeddings = dataframe_to_embedding_arrays(df, resolved_id_column)
                yield cls(
                    ids=ids,
                    embeddings=embeddings,
                    id_column=resolved_id_column,
                    tower=resolved_tower,
                    metric=resolved_metric,
                )
            return

        if path_obj.suffix == ".csv":
            for df in pd.read_csv(path_obj, chunksize=batch_size):
                ids, embeddings = dataframe_to_embedding_arrays(df, resolved_id_column)
                yield cls(
                    ids=ids,
                    embeddings=embeddings,
                    id_column=resolved_id_column,
                    tower=resolved_tower,
                    metric=resolved_metric,
                )
            return

        raise ValueError("[EmbeddingTable Error] Supported artifact formats: parquet, csv.")


def dataframe_to_embedding_arrays(df: pd.DataFrame, id_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract ID and emb_* columns from a dataframe artifact."""
    if id_column not in df.columns:
        raise KeyError(f"[EmbeddingTable Error] id_column '{id_column}' not found in embedding table.")
    emb_cols = [col for col in df.columns if str(col).startswith("emb_")]
    if not emb_cols:
        raise ValueError("[EmbeddingTable Error] embedding table must contain emb_0, emb_1, ... columns.")
    return df[id_column].astype(str).to_numpy(), df[emb_cols].to_numpy(dtype=np.float32)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows for cosine similarity search."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


class FaissVectorIndex:
    """FAISS flat vector index with ID mapping managed in NumPy."""

    def __init__(self, metric: RetrievalMetric = "dot", index_type: str = "flat"):
        """Initialize an empty FAISS index wrapper."""
        try:
            import faiss  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "[FaissVectorIndex Error] faiss is required for backend='faiss'. " "Install faiss-cpu or faiss-gpu."
            ) from exc
        self.faiss = faiss
        self.metric = metric
        self.index_type = index_type
        self.ids: np.ndarray | None = None
        self.index = None
        self.id_column = "item_id"
        self.id_to_positions: dict[str, list[int]] | None = None

    def _create_index(self, dim: int):
        """Create the FAISS index object for the configured metric."""
        if self.index_type != "flat":
            raise NotImplementedError("[FaissVectorIndex Error] Only index_type='flat' is currently supported.")
        if self.metric in {"dot", "cosine"}:
            return self.faiss.IndexFlatIP(int(dim))
        if self.metric == "euclidean":
            return self.faiss.IndexFlatL2(int(dim))
        raise ValueError(f"[FaissVectorIndex Error] Unknown metric: {self.metric}")

    def _prepare_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        """Validate vectors and normalize them when cosine search is used."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("[FaissVectorIndex Error] embeddings must be a 2D array.")
        if self.metric == "cosine":
            vectors = normalize_embeddings(vectors).astype(np.float32)
        return vectors

    def add(self, ids: np.ndarray, embeddings: np.ndarray) -> "FaissVectorIndex":
        """Append vectors and IDs to the FAISS index."""
        vectors = self._prepare_vectors(embeddings)
        new_ids = np.asarray(ids).astype(str)
        if new_ids.shape[0] != vectors.shape[0]:
            raise ValueError("[FaissVectorIndex Error] ids and embeddings must have the same row count.")
        if self.index is None:
            self.index = self._create_index(vectors.shape[1])
            self.ids = new_ids
        else:
            if vectors.shape[1] != self.index.d:
                raise ValueError(
                    f"[FaissVectorIndex Error] embedding dim mismatch: index dim={self.index.d}, got {vectors.shape[1]}."
                )
            self.ids = np.concatenate([self.ids, new_ids]) if self.ids is not None else new_ids
        self.index.add(vectors)
        self.id_to_positions = None
        return self

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        exclude_ids: list[set[str]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search top-k IDs and scores for each query vector.

        Per-query exclusions use an exact matrix score over reconstructed FAISS vectors
        because flat FAISS does not support arbitrary masked search directly.
        """
        if self.index is None or self.ids is None:
            raise ValueError("[FaissVectorIndex Error] Index is empty. Call add() before search().")
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError("[FaissVectorIndex Error] top_k must be a positive integer.")
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2:
            raise ValueError("[FaissVectorIndex Error] query_embeddings must be a 2D array.")
        if self.metric == "cosine":
            queries = normalize_embeddings(queries).astype(np.float32)

        blocked_positions_by_row = None
        if exclude_ids is not None:
            if len(exclude_ids) != queries.shape[0]:
                raise ValueError(
                    "[FaissVectorIndex Error] exclude_ids length must match the number of query embeddings."
                )
            if self.id_to_positions is None:
                self.id_to_positions = {}
                for position, item_id in enumerate(self.ids.astype(str).tolist()):
                    self.id_to_positions.setdefault(item_id, []).append(position)
            blocked_positions_by_row = []
            max_blocked_positions = 0
            for blocked_ids in exclude_ids:
                blocked_positions = set()
                for item_id in blocked_ids:
                    blocked_positions.update(self.id_to_positions.get(str(item_id), []))
                blocked_positions_by_row.append(blocked_positions)
                max_blocked_positions = max(max_blocked_positions, len(blocked_positions))
            k = min(top_k + max_blocked_positions, int(self.index.ntotal))
        else:
            k = min(top_k, int(self.index.ntotal))

        scores, positions = self.index.search(queries, k)
        if self.metric == "euclidean":
            scores = -scores

        if blocked_positions_by_row is not None:
            result_ids = np.full((queries.shape[0], min(top_k, int(self.index.ntotal))), "", dtype=object)
            result_scores = np.full(result_ids.shape, -np.inf, dtype=np.float32)
            for row_idx, blocked_positions in enumerate(blocked_positions_by_row):
                write_idx = 0
                for position, score in zip(positions[row_idx], scores[row_idx]):
                    if position < 0 or int(position) in blocked_positions:
                        continue
                    result_ids[row_idx, write_idx] = self.ids[position]
                    result_scores[row_idx, write_idx] = score
                    write_idx += 1
                    if write_idx == result_ids.shape[1]:
                        break
            return result_ids, result_scores
        return self.ids[positions], scores

    def save(self, save_dir: str | Path) -> Path:
        """Persist the FAISS index, ID mapping, and metadata to a directory."""
        if self.index is None or self.ids is None:
            raise ValueError("[FaissVectorIndex Error] Cannot save an empty index.")
        target_dir = Path(save_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        index_path = target_dir / "index.faiss"
        self.faiss.write_index(self.index, str(index_path))
        np.save(target_dir / "ids.npy", self.ids.astype(str))
        metadata = {
            "backend": "faiss",
            "metric": self.metric,
            "index_type": self.index_type,
            "id_column": getattr(self, "id_column", "item_id"),
            "row_count": int(self.ids.shape[0]),
            "dim": int(self.index.d),
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return index_path

    @classmethod
    def load(cls, index_dir: str | Path) -> "FaissVectorIndex":
        """Load a FAISS index directory created by save()."""
        target_dir = Path(index_dir)
        metadata = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(metric=metadata.get("metric", "dot"), index_type=metadata.get("index_type", "flat"))
        obj.index = obj.faiss.read_index(str(target_dir / "index.faiss"))
        obj.ids = np.load(target_dir / "ids.npy", allow_pickle=False).astype(str)
        obj.id_column = metadata.get("id_column", "item_id")
        obj.id_to_positions = None
        return obj


def exact_vector_search(
    ids: np.ndarray,
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    metric: RetrievalMetric,
    top_k: int = 10,
    exclude_ids: list[set[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run exact top-k scoring over an in-memory vector matrix."""
    item_ids = np.asarray(ids).astype(str)
    item_vectors = np.asarray(embeddings, dtype=np.float32)
    queries = np.asarray(query_embeddings, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if metric == "cosine":
        queries = normalize_embeddings(queries)
        item_vectors = normalize_embeddings(item_vectors)
        scores = queries @ item_vectors.T
    elif metric == "dot":
        scores = queries @ item_vectors.T
    elif metric == "euclidean":
        q_sq = np.sum(queries**2, axis=1, keepdims=True)
        i_sq = np.sum(item_vectors**2, axis=1, keepdims=True).T
        scores = -(q_sq + i_sq - 2.0 * queries @ item_vectors.T)
    else:
        raise ValueError(f"[Retrieval Error] Unknown metric: {metric}")

    if exclude_ids is not None:
        if len(exclude_ids) != queries.shape[0]:
            raise ValueError("[Retrieval Error] exclude_ids length must match the number of query embeddings.")
        id_to_pos: dict[str, list[int]] = {}
        for idx, item_id in enumerate(item_ids.tolist()):
            id_to_pos.setdefault(item_id, []).append(idx)
        for row_idx, blocked_ids in enumerate(exclude_ids):
            blocked_positions = [position for item_id in blocked_ids for position in id_to_pos.get(str(item_id), [])]
            if blocked_positions:
                scores[row_idx, blocked_positions] = -np.inf

    k = min(int(top_k), scores.shape[1])
    order = np.argpartition(-scores, kth=np.arange(k), axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    sorted_idx = np.argsort(-scores[row, order], axis=1)
    top_pos = order[row, sorted_idx]
    return item_ids[top_pos], scores[row, top_pos]


def compute_full_corpus_recall_metrics(
    query_ids: np.ndarray,
    positive_ids: list[set[str]],
    retrieved_ids: np.ndarray,
    top_k: list[int] | tuple[int, ...],
) -> dict[str, float]:
    """Compute recall, hitrate, NDCG, and MRR for full-corpus retrieval."""
    metrics: dict[str, float] = {}
    if len(query_ids) != len(positive_ids):
        raise ValueError("[RetrievalMetric Error] query_ids and positive_ids must have the same length.")

    valid_rows = [idx for idx, ids in enumerate(positive_ids) if ids]
    if not valid_rows:
        return {f"{name}@{k}": 0.0 for k in top_k for name in ("recall", "hitrate", "ndcg", "mrr")}

    max_width = retrieved_ids.shape[1] if retrieved_ids.ndim == 2 else 0
    for k in top_k:
        k_eff = min(int(k), max_width)
        recall_values = []
        hit_values = []
        ndcg_values = []
        mrr_values = []
        for row_idx in valid_rows:
            positives = positive_ids[row_idx]
            ranked = [str(item_id) for item_id in retrieved_ids[row_idx, :k_eff]]
            hits = [1 if item_id in positives else 0 for item_id in ranked]
            hit_count = sum(hits)
            recall_values.append(hit_count / float(len(positives)))
            hit_values.append(1.0 if hit_count > 0 else 0.0)
            if hit_count:
                discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
                dcg = float(np.sum(np.asarray(hits, dtype=np.float32) * discounts))
                ideal_hits = min(len(positives), k_eff)
                ideal_dcg = float(np.sum(discounts[:ideal_hits]))
                ndcg_values.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
                first_hit = hits.index(1) + 1
                mrr_values.append(1.0 / first_hit)
            else:
                ndcg_values.append(0.0)
                mrr_values.append(0.0)
        metrics[f"recall@{k}"] = float(np.mean(recall_values))
        metrics[f"hitrate@{k}"] = float(np.mean(hit_values))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_values))
        metrics[f"mrr@{k}"] = float(np.mean(mrr_values))
    return metrics
