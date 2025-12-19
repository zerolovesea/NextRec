"""
Pytest Configuration and Shared Fixtures

This module provides common test fixtures and configurations for all test modules.
"""

import logging

import numpy as np
import pytest
import torch

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def device():
    """
    Fixture: Determine the device (CPU/CUDA) for testing

    Returns:
        str: 'cuda' if available, else 'cpu'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    return device


@pytest.fixture(scope="session")
def batch_size():
    """
    Fixture: Standard batch size for tests

    Returns:
        int: Batch size
    """
    return 32


@pytest.fixture(scope="function")
def sample_dense_features():
    """
    Fixture: Create sample dense features for testing

    Returns:
        List[DenseFeature]: List of dense features
    """
    logger.info("Creating sample dense features")
    return [
        DenseFeature(name="age", embedding_dim=1),
        DenseFeature(name="price", embedding_dim=1),
        DenseFeature(name="score", embedding_dim=1),
    ]


@pytest.fixture(scope="function")
def sample_sparse_features():
    """
    Fixture: Create sample sparse features for testing

    Returns:
        List[SparseFeature]: List of sparse features
    """
    logger.info("Creating sample sparse features")
    return [
        SparseFeature(name="user_id", vocab_size=1000, embedding_dim=16),
        SparseFeature(name="item_id", vocab_size=500, embedding_dim=16),
        SparseFeature(
            name="category", vocab_size=50, embedding_dim=16
        ),  # Changed to 16 for FM compatibility
        SparseFeature(
            name="city", vocab_size=100, embedding_dim=16
        ),  # Changed to 16 for FM compatibility
    ]


@pytest.fixture(scope="function")
def sample_sequence_features():
    """
    Fixture: Create sample sequence features for testing

    Returns:
        List[SequenceFeature]: List of sequence features
    """
    logger.info("Creating sample sequence features")
    return [
        SequenceFeature(
            name="hist_item_ids",
            vocab_size=500,
            max_len=20,
            embedding_dim=16,
            padding_idx=0,
        ),
        SequenceFeature(
            name="hist_categories",
            vocab_size=50,
            max_len=20,
            embedding_dim=16,  # Changed to 16 to match other embeddings for FM
            padding_idx=0,
        ),
    ]


@pytest.fixture(scope="function")
def sample_batch_data(
    batch_size, sample_dense_features, sample_sparse_features, sample_sequence_features
):
    """
    Fixture: Generate sample batch data for testing

    Args:
        batch_size: Batch size
        sample_dense_features: Dense features
        sample_sparse_features: Sparse features
        sample_sequence_features: Sequence features

    Returns:
        Dict: Sample batch data
    """
    logger.info(f"Generating sample batch data with batch_size={batch_size}")

    data = {}

    # Generate dense feature data
    for feat in sample_dense_features:
        data[feat.name] = torch.randn(batch_size, 1)

    # Generate sparse feature data
    for feat in sample_sparse_features:
        data[feat.name] = torch.randint(1, feat.vocab_size, (batch_size,))

    # Generate sequence feature data
    for feat in sample_sequence_features:
        data[feat.name] = torch.randint(0, feat.vocab_size, (batch_size, feat.max_len))

    # Generate labels
    data["label"] = torch.randint(0, 2, (batch_size,)).float()

    return data


@pytest.fixture(scope="function")
def sample_match_batch_data(batch_size):
    """
    Fixture: Generate sample batch data for match models

    Args:
        batch_size: Batch size

    Returns:
        Dict: Sample match batch data with user and item features
    """
    logger.info(f"Generating sample match batch data with batch_size={batch_size}")

    data = {
        # User features
        "user_age": torch.randn(batch_size, 1),
        "user_id": torch.randint(1, 1000, (batch_size,)),
        "user_city": torch.randint(1, 100, (batch_size,)),
        "user_hist_items": torch.randint(0, 500, (batch_size, 20)),
        # Item features
        "item_id": torch.randint(1, 500, (batch_size,)),
        "item_category": torch.randint(1, 50, (batch_size,)),
        "item_price": torch.randn(batch_size, 1),
        # Labels
        "label": torch.randint(0, 2, (batch_size,)).float(),
    }

    return data


@pytest.fixture(scope="function")
def sample_multitask_batch_data(
    batch_size, sample_dense_features, sample_sparse_features, sample_sequence_features
):
    """
    Fixture: Generate sample batch data for multi-task models

    Args:
        batch_size: Batch size
        sample_dense_features: Dense features
        sample_sparse_features: Sparse features
        sample_sequence_features: Sequence features

    Returns:
        Dict: Sample batch data with multiple task labels
    """
    logger.info(f"Generating sample multi-task batch data with batch_size={batch_size}")

    data = {}

    # Generate feature data
    for feat in sample_dense_features:
        data[feat.name] = torch.randn(batch_size, 1)

    for feat in sample_sparse_features:
        data[feat.name] = torch.randint(1, feat.vocab_size, (batch_size,))

    for feat in sample_sequence_features:
        data[feat.name] = torch.randint(0, feat.vocab_size, (batch_size, feat.max_len))

    # Generate multiple task labels
    data["label_ctr"] = torch.randint(0, 2, (batch_size,)).float()
    data["label_cvr"] = torch.randint(0, 2, (batch_size,)).float()
    data["label_ctcvr"] = data["label_ctr"] * data["label_cvr"]

    return data


@pytest.fixture(scope="function")
def set_random_seed():
    """
    Fixture: Set random seed for reproducibility
    """
    seed = 42
    logger.info(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pytest_configure(config):
    """
    Pytest configuration hook
    """
    logger.info("=" * 80)
    logger.info("Starting NextRec Unit Tests")
    logger.info("=" * 80)


def pytest_unconfigure(config):
    """
    Pytest teardown hook
    """
    logger.info("=" * 80)
    logger.info("NextRec Unit Tests Completed")
    logger.info("=" * 80)
