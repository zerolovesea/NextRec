"""
Data utilities for NextRec.

This module provides file I/O helpers and synthetic data generation.

Date: create on 19/12/2025
Checkpoint: edit on 24/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

FILE_FORMAT_CONFIG = {
    "csv": {
        "extension": [".csv", ".txt"],
        "streaming": True,
    },
    "parquet": {
        "extension": [".parquet"],
        "streaming": True,
    },
    "feather": {
        "extension": [".feather", ".ftr"],
        "streaming": False,
    },
    "excel": {
        "extension": [".xlsx", ".xls"],
        "streaming": False,
    },
    "hdf5": {
        "extension": [".h5", ".hdf5"],
        "streaming": False,
    },
}


def get_file_format_from_extension(ext: str) -> str | None:
    """Get file format from extension."""
    return {
        ext.lstrip("."): fmt
        for fmt, config in FILE_FORMAT_CONFIG.items()
        for ext in config["extension"]
    }.get(ext.lower().lstrip("."))


def check_streaming_support(file_format: str) -> bool:
    """Check if a format supports streaming."""
    file_format = file_format.lower()
    if file_format not in FILE_FORMAT_CONFIG:
        return False
    return FILE_FORMAT_CONFIG[file_format].get("streaming", False)


def resolve_file_paths(path: str) -> tuple[list[str], str]:
    """
    Resolve file or directory path into a sorted list of files and file type.

    Args: path: Path to a file or directory
    Returns: tuple: (list of file paths, file type)
    """
    path_obj = Path(path)

    if path_obj.is_file():
        file_format = get_file_format_from_extension(path_obj.suffix)
        if file_format is None:
            raise ValueError(
                f"Unsupported file extension: {path_obj.suffix}. "
                f"Supported formats: {', '.join(FILE_FORMAT_CONFIG.keys())}"
            )
        return [str(path_obj)], file_format

    if path_obj.is_dir():
        collected_files = [p for p in path_obj.iterdir() if p.is_file()]
        # Group files by format
        format_groups: Dict[str, List[str]] = {}
        for file in collected_files:
            file_format = get_file_format_from_extension(file.suffix)
            if file_format:
                format_groups.setdefault(file_format, []).append(str(file))

        if len(format_groups) > 1:
            formats = ", ".join(format_groups.keys())
            raise ValueError(
                f"Directory contains mixed file formats: {formats}. "
                "Please keep a single format per directory."
            )

        if not format_groups:
            raise ValueError(
                f"No supported data files found in directory: {path}. "
                f"Supported formats: {', '.join(FILE_FORMAT_CONFIG.keys())}"
            )

        file_type = list(format_groups.keys())[0]
        file_paths = format_groups[file_type]
        file_paths.sort()
        return file_paths, file_type

    raise ValueError(f"Invalid path: {path}")


def read_table(path: str | Path, data_format: str | None = None) -> pd.DataFrame:
    data_path = Path(path)

    # Determine format
    if data_format:
        fmt = data_format.lower()
    elif data_path.is_dir():
        _, fmt = resolve_file_paths(str(data_path))
    else:
        fmt = get_file_format_from_extension(data_path.suffix)
        if fmt is None:
            raise ValueError(
                f"Cannot determine format for {data_path}. "
                f"Please specify data_format parameter."
            )

    if data_path.is_dir():
        file_paths, _ = resolve_file_paths(str(data_path))
        dataframes = [read_table(fp, fmt) for fp in file_paths]
        if not dataframes:
            raise ValueError(f"No supported data files found in directory: {data_path}")
        if len(dataframes) == 1:
            return dataframes[0]
        return pd.concat(dataframes, ignore_index=True)

    # Read based on format
    try:
        if fmt == "hdf5":
            # HDF5 requires a key; use the first available key
            with pd.HDFStore(data_path, mode="r") as store:
                if len(store.keys()) == 0:
                    raise ValueError(f"HDF5 file {data_path} contains no datasets")
                return pd.read_hdf(data_path, key=store.keys()[0])
        reader = {
            "parquet": pd.read_parquet,
            "csv": lambda p: pd.read_csv(p, low_memory=False),
            "feather": pd.read_feather,
            "excel": pd.read_excel,
        }.get(fmt)
        if reader:
            return reader(data_path)
        raise ValueError(
            f"Unsupported format: {fmt}. "
            f"Supported: {', '.join(FILE_FORMAT_CONFIG.keys())}"
        )
    except ImportError as e:
        raise ImportError(
            f"Format '{fmt}' requires additional dependencies. "
            f"Install with: pip install pandas[{fmt}] or check documentation. "
            f"Original error: {e}"
        ) from e


def load_dataframes(file_paths: list[str], file_type: str) -> list[pd.DataFrame]:
    return [read_table(fp, file_type) for fp in file_paths]


def iter_file_chunks(
    file_path: str, file_type: str, chunk_size: int
) -> Generator[pd.DataFrame, None, None]:
    """Iterate over file in chunks for streaming reading.

    Args:
        file_path: Path to the file
        file_type: Format type (csv, parquet)
        chunk_size: Number of rows per chunk

    Yields:
        DataFrame chunks

    Raises:
        ValueError: If format doesn't support streaming
    """
    file_type = file_type.lower()
    if not check_streaming_support(file_type):
        raise ValueError(
            f"Format '{file_type}' does not support streaming reads. "
            "Formats with streaming support: csv, parquet"
        )

    try:
        if file_type == "csv":
            yield from pd.read_csv(file_path, chunksize=chunk_size)
        elif file_type == "parquet":
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()
        else:
            raise ValueError(
                f"Format '{file_type}' does not support streaming. "
                f"Use read_table() to load the entire file into memory."
            )
    except ImportError as e:
        raise ImportError(
            f"Streaming format '{file_type}' requires additional dependencies. "
            f"Install with: pip install pandas[{file_type}] pyarrow. "
            f"Original error: {e}"
        ) from e


def default_output_dir(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_file():
        return path_obj.parent / f"{path_obj.stem}_preprocessed"
    return path_obj.with_name(f"{path_obj.name}_preprocessed")


def read_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def generate_ranking_data(
    n_samples: int = 10000,
    n_dense: int = 5,
    n_sparse: int = 8,
    n_sequences: int = 2,
    user_vocab_size: int = 1000,
    item_vocab_size: int = 500,
    sparse_vocab_size: int = 50,
    sequence_max_len: int = 20,
    embedding_dim: int = 16,
    seed: int = 42,
    custom_sparse_features: Optional[Dict[str, int]] = None,
    use_simple_names: bool = True,
) -> Tuple[pd.DataFrame, List, List, List]:
    """
    Generate synthetic data for ranking tasks (CTR prediction)

    Returns:
        tuple: (dataframe, dense_features, sparse_features, sequence_features)
    """
    print(f"Generating {n_samples} synthetic ranking samples...")

    np.random.seed(seed)
    data = {}

    for i in range(n_dense):
        data[f"dense_{i}"] = np.random.randn(n_samples).astype(np.float32)

    # Generate basic sparse features (always include user_id and item_id)
    data["user_id"] = np.random.randint(1, user_vocab_size, n_samples)
    data["item_id"] = np.random.randint(1, item_vocab_size, n_samples)

    # Generate additional sparse features
    if custom_sparse_features:
        for feat_name, vocab_size in custom_sparse_features.items():
            data[feat_name] = np.random.randint(0, vocab_size, n_samples)
    else:
        for i in range(n_sparse - 2):
            data[f"sparse_{i}"] = np.random.randint(1, sparse_vocab_size, n_samples)

    # Generate sequence features (list of IDs)
    sequence_names = []
    sequence_vocabs = []

    for i in range(n_sequences):
        sequences = []
        for _ in range(n_samples):
            seq_len = np.random.randint(5, sequence_max_len + 1)
            if i == 0:
                # First sequence uses item vocabulary
                seq = np.random.randint(0, item_vocab_size, seq_len).tolist()
                seq_vocab = item_vocab_size
                if custom_sparse_features:
                    seq_name = "hist_items"
                else:
                    seq_name = "sequence_0"
            else:
                # Other sequences use category vocabulary
                if custom_sparse_features and "category" in custom_sparse_features:
                    seq_vocab = custom_sparse_features["category"]
                    seq = np.random.randint(0, seq_vocab, seq_len).tolist()
                    seq_name = "hist_categories" if i == 1 else f"sequence_{i}"
                else:
                    seq_vocab = sparse_vocab_size
                    seq = np.random.randint(0, seq_vocab, seq_len).tolist()
                    seq_name = f"sequence_{i}"

            # Padding
            seq = seq + [0] * (sequence_max_len - len(seq))
            sequences.append(seq)

        data[seq_name] = sequences
        sequence_names.append(seq_name)
        sequence_vocabs.append(seq_vocab)

    if "gender" in data and "dense_0" in data:
        dense_1 = data.get("dense_1", 0)
        # Complex label generation with feature correlation
        label_probs = 1 / (
            1
            + np.exp(
                -(
                    data["dense_0"] * 0.3
                    + dense_1 * 0.2
                    + (data["gender"] - 0.5) * 0.5
                    + np.random.randn(n_samples) * 0.1
                )
            )
        )
        data["label"] = (label_probs > 0.5).astype(np.float32)
    else:
        data["label"] = np.random.randint(0, 2, n_samples).astype(np.float32)

    df = pd.DataFrame(data)
    print(f"Generated data shape: {df.shape}")
    if "gender" in data:
        print(f"Positive rate: {data['label'].mean():.4f}")

    # Import here to avoid circular import
    from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature

    # Create feature definitions
    # Use input_dim for dense features to be compatible with both simple and complex scenarios
    dense_features = [
        DenseFeature(name=f"dense_{i}", input_dim=1) for i in range(n_dense)
    ]

    # Create sparse features
    sparse_features = [
        SparseFeature(
            name="user_id",
            embedding_name="user_emb",
            vocab_size=user_vocab_size,
            embedding_dim=embedding_dim,
        ),
        SparseFeature(
            name="item_id",
            embedding_name="item_emb",
            vocab_size=item_vocab_size,
            embedding_dim=embedding_dim,
        ),
    ]

    if custom_sparse_features:
        # Add custom sparse features with proper vocab sizes
        for feat_name, vocab_size in custom_sparse_features.items():
            sparse_features.append(
                SparseFeature(
                    name=feat_name,
                    embedding_name=f"{feat_name}_emb",
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                )
            )
    else:
        # Add generic sparse features
        sparse_features.extend(
            [
                SparseFeature(
                    name=f"sparse_{i}",
                    embedding_name=f"sparse_{i}_emb",
                    vocab_size=sparse_vocab_size,
                    embedding_dim=embedding_dim,
                )
                for i in range(n_sparse - 2)
            ]
        )

    # Create sequence features
    sequence_features = []
    for i, (seq_name, seq_vocab) in enumerate(zip(sequence_names, sequence_vocabs)):
        if i == 0:
            # First sequence shares embedding with item_id
            embedding_name = "item_emb"
        elif (
            custom_sparse_features
            and "category" in custom_sparse_features
            and seq_name == "hist_categories"
        ):
            # hist_categories shares embedding with category
            embedding_name = "category_emb"
        else:
            # Other sequences share with sparse_0
            embedding_name = "sparse_0_emb"
        sequence_features.append(
            SequenceFeature(
                name=seq_name,
                vocab_size=seq_vocab,
                max_len=sequence_max_len,
                embedding_dim=embedding_dim,
                padding_idx=0,
                embedding_name=embedding_name,
            )
        )
    return df, dense_features, sparse_features, sequence_features


def generate_match_data(
    n_samples: int = 10000,
    user_vocab_size: int = 1000,
    item_vocab_size: int = 5000,
    category_vocab_size: int = 100,
    brand_vocab_size: int = 200,
    city_vocab_size: int = 100,
    user_feature_vocab_size: int = 50,
    item_feature_vocab_size: int = 50,
    sequence_max_len: int = 50,
    user_embedding_dim: int = 32,
    item_embedding_dim: int = 32,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List, List, List, List, List, List]:
    """
    Generate synthetic data for match/retrieval tasks

    Returns:
        tuple: (dataframe, user_dense_features, user_sparse_features, user_sequence_features,
                item_dense_features, item_sparse_features, item_sequence_features)
    """
    print(f"Generating {n_samples} synthetic match samples...")

    np.random.seed(seed)
    data = {}

    # User features
    data["user_id"] = np.random.randint(1, user_vocab_size, n_samples)
    data["user_age"] = np.random.randn(n_samples).astype(np.float32)
    data["user_gender"] = np.random.randint(0, 2, n_samples)
    data["user_city"] = np.random.randint(0, city_vocab_size, n_samples)

    for i in range(3):
        data[f"user_feature_{i}"] = np.random.randint(
            1, user_feature_vocab_size, n_samples
        )

    # User behavior sequences
    user_hist_items = []
    user_hist_categories = []
    for _ in range(n_samples):
        seq_len = np.random.randint(10, sequence_max_len + 1)
        hist_items = np.random.randint(1, item_vocab_size, seq_len).tolist()
        hist_items = hist_items + [0] * (sequence_max_len - len(hist_items))
        user_hist_items.append(hist_items)

        hist_cats = np.random.randint(1, category_vocab_size, seq_len).tolist()
        hist_cats = hist_cats + [0] * (sequence_max_len - len(hist_cats))
        user_hist_categories.append(hist_cats)

    data["user_hist_items"] = user_hist_items
    data["user_hist_categories"] = user_hist_categories

    # Item features
    data["item_id"] = np.random.randint(1, item_vocab_size, n_samples)
    data["item_price"] = np.random.randn(n_samples).astype(np.float32)
    data["item_category"] = np.random.randint(1, category_vocab_size, n_samples)
    data["item_brand"] = np.random.randint(1, brand_vocab_size, n_samples)

    for i in range(3):
        data[f"item_feature_{i}"] = np.random.randint(
            1, item_feature_vocab_size, n_samples
        )

    # Generate labels with some correlation to features
    label_probs = 1 / (
        1
        + np.exp(
            -(
                data["user_age"] * 0.2
                + (data["user_gender"] - 0.5) * 0.3
                + data["item_price"] * 0.15
                + np.random.randn(n_samples) * 0.5
            )
        )
    )
    data["label"] = (label_probs > 0.5).astype(np.float32)

    df = pd.DataFrame(data)
    print(f"Generated data shape: {df.shape}")
    print(f"Positive rate: {data['label'].mean():.4f}")

    # Import here to avoid circular import
    from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature

    # User dense features
    user_dense_features = [DenseFeature(name="user_age", input_dim=1)]

    # User sparse features
    user_sparse_features = [
        SparseFeature(
            name="user_id", vocab_size=user_vocab_size, embedding_dim=user_embedding_dim
        ),
        SparseFeature(name="user_gender", vocab_size=2, embedding_dim=8),
        SparseFeature(name="user_city", vocab_size=city_vocab_size, embedding_dim=16),
    ]
    user_sparse_features.extend(
        [
            SparseFeature(
                name=f"user_feature_{i}",
                vocab_size=user_feature_vocab_size,
                embedding_dim=8,
            )
            for i in range(3)
        ]
    )

    # User sequence features
    user_sequence_features = [
        SequenceFeature(
            name="user_hist_items",
            vocab_size=item_vocab_size,
            max_len=sequence_max_len,
            embedding_dim=user_embedding_dim,
            padding_idx=0,
        ),
        SequenceFeature(
            name="user_hist_categories",
            vocab_size=category_vocab_size,
            max_len=sequence_max_len,
            embedding_dim=16,
            padding_idx=0,
        ),
    ]

    # Item dense features
    item_dense_features = [DenseFeature(name="item_price", input_dim=1)]

    # Item sparse features
    item_sparse_features = [
        SparseFeature(
            name="item_id", vocab_size=item_vocab_size, embedding_dim=item_embedding_dim
        ),
        SparseFeature(
            name="item_category", vocab_size=category_vocab_size, embedding_dim=16
        ),
        SparseFeature(name="item_brand", vocab_size=brand_vocab_size, embedding_dim=16),
    ]
    item_sparse_features.extend(
        [
            SparseFeature(
                name=f"item_feature_{i}",
                vocab_size=item_feature_vocab_size,
                embedding_dim=8,
            )
            for i in range(3)
        ]
    )

    # Item sequence features (empty for most match models)
    item_sequence_features = []

    return (
        df,
        user_dense_features,
        user_sparse_features,
        user_sequence_features,
        item_dense_features,
        item_sparse_features,
        item_sequence_features,
    )


def generate_multitask_data(
    n_samples: int = 10000,
    n_dense: int = 5,
    n_sparse: int = 8,
    n_sequences: int = 2,
    user_vocab_size: int = 1000,
    item_vocab_size: int = 500,
    sparse_vocab_size: int = 50,
    sequence_max_len: int = 20,
    embedding_dim: int = 16,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List, List, List]:
    """
    Generate synthetic data for multi-task learning

    Returns:
        tuple: (dataframe, dense_features, sparse_features, sequence_features)
    """
    print(f"Generating {n_samples} synthetic multi-task samples...")

    np.random.seed(seed)
    data = {}

    # Generate dense features
    for i in range(n_dense):
        data[f"dense_{i}"] = np.random.randn(n_samples).astype(np.float32)

    # Generate sparse features
    data["user_id"] = np.random.randint(1, user_vocab_size, n_samples)
    data["item_id"] = np.random.randint(1, item_vocab_size, n_samples)

    for i in range(n_sparse - 2):
        data[f"sparse_{i}"] = np.random.randint(1, sparse_vocab_size, n_samples)

    # Generate sequence features
    sequence_names = []
    sequence_vocabs = []

    for i in range(n_sequences):
        sequences = []
        for _ in range(n_samples):
            seq_len = np.random.randint(5, sequence_max_len + 1)
            if i == 0:
                seq = np.random.randint(0, item_vocab_size, seq_len).tolist()
                seq_vocab = item_vocab_size
                seq_name = "sequence_0"
            else:
                seq = np.random.randint(0, sparse_vocab_size, seq_len).tolist()
                seq_vocab = sparse_vocab_size
                seq_name = f"sequence_{i}"

            seq = seq + [0] * (sequence_max_len - len(seq))
            sequences.append(seq)

        data[seq_name] = sequences
        sequence_names.append(seq_name)
        sequence_vocabs.append(seq_vocab)

    # Generate multi-task labels with correlation
    # CTR (click) is relatively easier to predict
    dense_0 = data.get("dense_0", 0)
    dense_1 = data.get("dense_1", 0)
    dense_2 = data.get("dense_2", 0)
    dense_3 = data.get("dense_3", 0)
    ctr_logits = dense_0 * 0.3 + dense_1 * 0.2 + np.random.randn(n_samples) * 0.5
    data["click"] = (1 / (1 + np.exp(-ctr_logits)) > 0.5).astype(np.float32)

    # CVR (conversion) depends on click and is harder
    cvr_logits = (
        dense_2 * 0.2
        + dense_3 * 0.15
        + data["click"] * 1.5  # Strong dependency on click
        + np.random.randn(n_samples) * 0.8
    )
    data["conversion"] = (1 / (1 + np.exp(-cvr_logits)) > 0.3).astype(np.float32)

    # CTCVR = click AND conversion
    data["ctcvr"] = (data["click"] * data["conversion"]).astype(np.float32)

    df = pd.DataFrame(data)
    print(f"Generated data shape: {df.shape}")
    print(f"Click rate: {data['click'].mean():.4f}")
    print(f"Conversion rate (overall): {data['conversion'].mean():.4f}")
    if data["click"].sum() > 0:
        print(
            f"Conversion rate (given click): {data['conversion'][data['click'] == 1].mean():.4f}"
        )
    print(f"CTCVR rate: {data['ctcvr'].mean():.4f}")

    # Import here to avoid circular import
    from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature

    # Create feature definitions
    dense_features = [
        DenseFeature(name=f"dense_{i}", input_dim=1) for i in range(n_dense)
    ]

    # Create sparse features
    sparse_features = [
        SparseFeature(
            name="user_id",
            embedding_name="user_emb",
            vocab_size=user_vocab_size,
            embedding_dim=embedding_dim,
        ),
        SparseFeature(
            name="item_id",
            embedding_name="item_emb",
            vocab_size=item_vocab_size,
            embedding_dim=embedding_dim,
        ),
    ]
    sparse_features.extend(
        [
            SparseFeature(
                name=f"sparse_{i}",
                embedding_name=f"sparse_{i}_emb",
                vocab_size=sparse_vocab_size,
                embedding_dim=embedding_dim,
            )
            for i in range(n_sparse - 2)
        ]
    )

    # Create sequence features
    sequence_features = []
    for i, (seq_name, seq_vocab) in enumerate(zip(sequence_names, sequence_vocabs)):
        if i == 0:
            embedding_name = "item_emb"
        else:
            embedding_name = "sparse_0_emb"
        sequence_features.append(
            SequenceFeature(
                name=seq_name,
                vocab_size=seq_vocab,
                max_len=sequence_max_len,
                embedding_dim=embedding_dim,
                padding_idx=0,
                embedding_name=embedding_name,
            )
        )

    return df, dense_features, sparse_features, sequence_features


def generate_distributed_ranking_data(
    num_samples: int = 100000,
    num_users: int = 10000,
    num_items: int = 5000,
    num_categories: int = 20,
    num_cities: int = 100,
    max_seq_len: int = 50,
    embedding_dim: int = 32,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List, List, List]:
    """
    Generate synthetic data for distributed training scenarios

    Returns:
        tuple: (dataframe, dense_features, sparse_features, sequence_features)
    """
    return generate_ranking_data(
        n_samples=num_samples,
        n_dense=5,
        n_sparse=6,  # user_id, item_id + 4 custom features
        n_sequences=2,
        user_vocab_size=num_users + 1,
        item_vocab_size=num_items + 1,
        sequence_max_len=max_seq_len,
        embedding_dim=embedding_dim,
        seed=seed,
        custom_sparse_features={
            "gender": 2,
            "age_group": 7,
            "category": num_categories,
            "city": num_cities,
        },
        use_simple_names=False,
    )


def generate_synthetic_embeddings(num_samples=1000, embedding_dim=768):
    """
    Generate synthetic multimodal embeddings for demonstration.

    In practice, you would load real embeddings from:
    - Image embeddings (e.g., from ResNet, ViT)
    - Text embeddings (e.g., from BERT, GPT)
    - Audio embeddings (e.g., from wav2vec)
    """
    # Generate random embeddings
    embeddings = torch.randn(num_samples, embedding_dim)

    # Create item IDs
    item_ids = torch.arange(num_samples)

    return item_ids, embeddings
