"""
Unit Tests for Match Models

This module contains unit tests for all matching models including:
- DSSM (Deep Structured Semantic Model)
- YoutubeDNN
- MIND
- SDM (Sequential Deep Matching)

Tests cover model initialization, forward pass, training, and inference.
"""

import logging
from test.helpers import (
    assert_model_output_range,
    assert_model_output_shape,
    assert_no_nan_or_inf,
    count_parameters,
    run_model_inference,
)

import pytest
import torch

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.models.retrieval.dssm import DSSM
from nextrec.models.retrieval.mind import MIND
from nextrec.models.retrieval.sdm import SDM
from nextrec.models.retrieval.youtube_dnn import YoutubeDNN

logger = logging.getLogger(__name__)


class TestDSSM:
    """Test suite for DSSM (Deep Structured Semantic Model)"""

    @pytest.fixture
    def user_features(self):
        """Create user features for DSSM"""
        logger.info("Creating user features for DSSM")
        user_dense = [DenseFeature(name="user_age", embedding_dim=1)]
        user_sparse = [
            SparseFeature(name="user_id", vocab_size=1000, embedding_dim=16),
            SparseFeature(name="user_city", vocab_size=100, embedding_dim=8),
        ]
        user_sequence = [
            SequenceFeature(
                name="user_hist_items",
                vocab_size=500,
                max_len=20,
                embedding_dim=16,
                padding_idx=0,
            )
        ]
        return user_dense, user_sparse, user_sequence

    @pytest.fixture
    def item_features(self):
        """Create item features for DSSM"""
        logger.info("Creating item features for DSSM")
        item_dense = [DenseFeature(name="item_price", embedding_dim=1)]
        item_sparse = [
            SparseFeature(name="item_id", vocab_size=500, embedding_dim=16),
            SparseFeature(name="item_category", vocab_size=50, embedding_dim=8),
        ]
        return item_dense, item_sparse, []

    def test_dssm_initialization(self, user_features, item_features, device):
        """Test DSSM model initialization"""
        logger.info("=" * 80)
        logger.info("Testing DSSM initialization")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = user_features
        item_dense, item_sparse, item_sequence = item_features

        model = DSSM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            user_dnn_hidden_units=[128, 64],
            item_dnn_hidden_units=[128, 64],
            embedding_dim=32,
            training_mode="pointwise",
            similarity_metric="cosine",
            device=device,
        )

        assert model is not None
        assert model.model_name == "DSSM"
        assert model.embedding_dim == 32
        logger.info("DSSM initialization successful")

        # Count parameters
        num_params = count_parameters(model)
        assert num_params > 0

    def test_dssm_forward_pass(
        self, user_features, item_features, device, batch_size, set_random_seed
    ):
        """Test DSSM forward pass"""
        logger.info("=" * 80)
        logger.info("Testing DSSM forward pass")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = user_features
        item_dense, item_sparse, item_sequence = item_features

        model = DSSM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            user_dnn_hidden_units=[128, 64],
            item_dnn_hidden_units=[128, 64],
            embedding_dim=32,
            training_mode="pointwise",
            similarity_metric="cosine",
            device=device,
        )

        # Create sample data
        data = {
            "user_age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "user_city": torch.randint(1, 100, (batch_size,)).to(device),
            "user_hist_items": torch.randint(0, 500, (batch_size, 20)).to(device),
            "item_price": torch.randn(batch_size, 1).to(device),
            "item_id": torch.randint(1, 500, (batch_size,)).to(device),
            "item_category": torch.randint(1, 50, (batch_size,)).to(device),
        }

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(data)

        # Assertions
        assert_model_output_shape(output, (batch_size,), "DSSM output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "DSSM output")

        logger.info("DSSM forward pass successful")

    def test_dssm_towers(self, user_features, item_features, device, batch_size):
        """Test DSSM user and item towers separately"""
        logger.info("=" * 80)
        logger.info("Testing DSSM user and item towers")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = user_features
        item_dense, item_sparse, item_sequence = item_features

        model = DSSM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            embedding_dim=32,
            device=device,
        )

        # User data
        user_data = {
            "user_age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "user_city": torch.randint(1, 100, (batch_size,)).to(device),
            "user_hist_items": torch.randint(0, 500, (batch_size, 20)).to(device),
        }

        # Item data
        item_data = {
            "item_price": torch.randn(batch_size, 1).to(device),
            "item_id": torch.randint(1, 500, (batch_size,)).to(device),
            "item_category": torch.randint(1, 50, (batch_size,)).to(device),
        }

        # Test user tower
        model.eval()
        with torch.no_grad():
            user_emb = model.user_tower(user_data)

        assert_model_output_shape(user_emb, (batch_size, 32), "User embedding shape")
        assert_no_nan_or_inf(user_emb, "User embedding")

        # Test item tower
        with torch.no_grad():
            item_emb = model.item_tower(item_data)

        assert_model_output_shape(item_emb, (batch_size, 32), "Item embedding shape")
        assert_no_nan_or_inf(item_emb, "Item embedding")

        logger.info("DSSM towers test successful")

    @pytest.mark.parametrize("similarity_metric", ["cosine", "dot"])
    def test_dssm_similarity_metrics(
        self, user_features, item_features, device, batch_size, similarity_metric
    ):
        """Test DSSM with different similarity metrics"""
        logger.info("=" * 80)
        logger.info(f"Testing DSSM with {similarity_metric} similarity")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = user_features
        item_dense, item_sparse, item_sequence = item_features

        model = DSSM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            similarity_metric=similarity_metric,
            device=device,
        )

        data = {
            "user_age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "user_city": torch.randint(1, 100, (batch_size,)).to(device),
            "user_hist_items": torch.randint(0, 500, (batch_size, 20)).to(device),
            "item_price": torch.randn(batch_size, 1).to(device),
            "item_id": torch.randint(1, 500, (batch_size,)).to(device),
            "item_category": torch.randint(1, 50, (batch_size,)).to(device),
        }

        model.eval()
        with torch.no_grad():
            output = model(data)

        assert_model_output_shape(output, (batch_size,))
        assert_no_nan_or_inf(output, f"DSSM output ({similarity_metric})")

        logger.info(f"DSSM {similarity_metric} similarity test successful")


class TestYoutubeDNN:
    """Test suite for YoutubeDNN"""

    @pytest.fixture
    def user_features(self):
        """Create user features for YoutubeDNN"""
        logger.info("Creating user features for YoutubeDNN")
        user_dense = [DenseFeature(name="user_age", embedding_dim=1)]
        user_sparse = [SparseFeature(name="user_id", vocab_size=1000, embedding_dim=16)]
        user_sequence = [
            SequenceFeature(
                name="user_watch_history",
                vocab_size=10000,
                max_len=50,
                embedding_dim=32,
                padding_idx=0,
            )
        ]
        return user_dense, user_sparse, user_sequence

    @pytest.fixture
    def item_features(self):
        """Create item features for YoutubeDNN"""
        logger.info("Creating item features for YoutubeDNN")
        item_sparse = [
            SparseFeature(name="video_id", vocab_size=10000, embedding_dim=32),
            SparseFeature(name="video_category", vocab_size=100, embedding_dim=16),
        ]
        return [], item_sparse, []

    def test_youtube_dnn_initialization(self, user_features, item_features, device):
        """Test YoutubeDNN initialization"""
        logger.info("=" * 80)
        logger.info("Testing YoutubeDNN initialization")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = user_features
        item_dense, item_sparse, item_sequence = item_features

        model = YoutubeDNN(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            user_dnn_hidden_units=[256, 128],
            item_dnn_hidden_units=[256, 128],
            embedding_dim=64,
            training_mode="listwise",
            num_negative_samples=100,
            device=device,
        )

        assert model is not None
        assert model.model_name == "YouTubeDNN"
        assert model.embedding_dim == 64
        logger.info("YoutubeDNN initialization successful")

        count_parameters(model)

    def test_youtube_dnn_forward_pass(
        self, user_features, item_features, device, batch_size, set_random_seed
    ):
        """Test YoutubeDNN forward pass"""
        logger.info("=" * 80)
        logger.info("Testing YoutubeDNN forward pass")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = user_features
        item_dense, item_sparse, item_sequence = item_features

        model = YoutubeDNN(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            embedding_dim=64,
            training_mode="pointwise",
            device=device,
        )

        # Create sample data
        data = {
            "user_age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "user_watch_history": torch.randint(0, 10000, (batch_size, 50)).to(device),
            "video_id": torch.randint(1, 10000, (batch_size,)).to(device),
            "video_category": torch.randint(1, 100, (batch_size,)).to(device),
        }

        # Forward pass
        output = run_model_inference(model, data)

        # Assertions
        assert_model_output_shape(output, (batch_size,), "YoutubeDNN output shape")
        assert_model_output_range(output, 0.0, 1.0)

        logger.info("YoutubeDNN forward pass successful")

    def test_youtube_dnn_embeddings(
        self, user_features, item_features, device, batch_size
    ):
        """Test YoutubeDNN user and item embeddings"""
        logger.info("=" * 80)
        logger.info("Testing YoutubeDNN embeddings")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = user_features
        item_dense, item_sparse, item_sequence = item_features

        embedding_dim = 64
        model = YoutubeDNN(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            embedding_dim=embedding_dim,
            device=device,
        )

        # User data
        user_data = {
            "user_age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "user_watch_history": torch.randint(0, 10000, (batch_size, 50)).to(device),
        }

        # Item data
        item_data = {
            "video_id": torch.randint(1, 10000, (batch_size,)).to(device),
            "video_category": torch.randint(1, 100, (batch_size,)).to(device),
        }

        model.eval()
        with torch.no_grad():
            user_emb = model.user_tower(user_data)
            item_emb = model.item_tower(item_data)

        assert_model_output_shape(
            user_emb, (batch_size, embedding_dim), "User embedding"
        )
        assert_model_output_shape(
            item_emb, (batch_size, embedding_dim), "Item embedding"
        )

        # Check L2 normalization
        user_norms = torch.norm(user_emb, p=2, dim=1)
        item_norms = torch.norm(item_emb, p=2, dim=1)

        assert torch.allclose(
            user_norms, torch.ones_like(user_norms), atol=1e-5
        ), "User embeddings should be L2 normalized"
        assert torch.allclose(
            item_norms, torch.ones_like(item_norms), atol=1e-5
        ), "Item embeddings should be L2 normalized"

        logger.info("YoutubeDNN embeddings test successful")


class TestMatchModelsComparison:
    """Comparison tests for match models"""

    def test_models_deterministic(self, device, batch_size):
        """Test that models produce deterministic outputs with same random seed"""
        logger.info("=" * 80)
        logger.info("Testing match models determinism")
        logger.info("=" * 80)

        # Setup features
        user_sparse = [SparseFeature(name="user_id", vocab_size=100, embedding_dim=8)]
        item_sparse = [SparseFeature(name="item_id", vocab_size=100, embedding_dim=8)]

        # Create data
        data = {
            "user_id": torch.randint(1, 100, (batch_size,)).to(device),
            "item_id": torch.randint(1, 100, (batch_size,)).to(device),
        }

        # Test DSSM
        torch.manual_seed(42)
        model1 = DSSM(
            user_sparse_features=user_sparse,
            item_sparse_features=item_sparse,
            embedding_dim=16,
            device=device,
        )

        torch.manual_seed(42)
        model2 = DSSM(
            user_sparse_features=user_sparse,
            item_sparse_features=item_sparse,
            embedding_dim=16,
            device=device,
        )

        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(data)
            output2 = model2(data)

        assert torch.allclose(
            output1, output2, atol=1e-6
        ), "Models with same seed should produce identical outputs"

        logger.info("Match models determinism test successful")


class TestMIND:
    """Test suite for MIND (Multi-Interest Network with Dynamic Routing)"""

    @pytest.fixture
    def mind_user_features(self):
        """Create user features for MIND with sequence"""
        user_sparse = [SparseFeature(name="user_id", vocab_size=1000, embedding_dim=16)]
        user_sequence = [
            SequenceFeature(
                name="user_behavior",
                vocab_size=10000,
                max_len=50,
                embedding_dim=32,
                padding_idx=0,
            )
        ]
        return [], user_sparse, user_sequence

    @pytest.fixture
    def mind_item_features(self):
        """Create item features for MIND"""
        item_sparse = [
            SparseFeature(name="item_id", vocab_size=10000, embedding_dim=32),
        ]
        return [], item_sparse, []

    def test_mind_initialization(self, mind_user_features, mind_item_features, device):
        """Test MIND model initialization"""
        logger.info("=" * 80)
        logger.info("Testing MIND initialization")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = mind_user_features
        item_dense, item_sparse, item_sequence = mind_item_features

        model = MIND(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            embedding_dim=64,
            num_interests=4,
            training_mode="pointwise",
            device=device,
        )

        assert model is not None
        assert model.model_name == "MIND"
        assert model.num_interests == 4
        logger.info("MIND initialization successful")

        count_parameters(model)

    def test_mind_forward_pass(
        self,
        mind_user_features,
        mind_item_features,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test MIND forward pass with multi-interest extraction"""
        logger.info("=" * 80)
        logger.info("Testing MIND forward pass")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = mind_user_features
        item_dense, item_sparse, item_sequence = mind_item_features

        model = MIND(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            embedding_dim=64,
            num_interests=3,
            training_mode="pointwise",
            device=device,
        )

        data = {
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "user_behavior": torch.randint(0, 10000, (batch_size, 50)).to(device),
            "item_id": torch.randint(1, 10000, (batch_size,)).to(device),
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,), "MIND output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "MIND output")

        logger.info("MIND forward pass successful")

    @pytest.mark.parametrize("num_interests", [2, 4, 8])
    def test_mind_different_interests(
        self, mind_user_features, mind_item_features, device, batch_size, num_interests
    ):
        """Test MIND with different numbers of interests"""
        logger.info("=" * 80)
        logger.info(f"Testing MIND with {num_interests} interests")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = mind_user_features
        item_dense, item_sparse, item_sequence = mind_item_features

        model = MIND(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            embedding_dim=64,
            num_interests=num_interests,
            device=device,
        )

        data = {
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "user_behavior": torch.randint(0, 10000, (batch_size, 50)).to(device),
            "item_id": torch.randint(1, 10000, (batch_size,)).to(device),
        }

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info(f"MIND with {num_interests} interests test successful")


class TestSDM:
    """Test suite for SDM (Sequential Deep Matching)"""

    @pytest.fixture
    def sdm_user_features(self):
        """Create user features for SDM with sequence"""
        user_sparse = [SparseFeature(name="user_id", vocab_size=1000, embedding_dim=16)]
        user_sequence = [
            SequenceFeature(
                name="item_sequence",
                vocab_size=10000,
                max_len=50,
                embedding_dim=32,
                padding_idx=0,
            )
        ]
        return [], user_sparse, user_sequence

    @pytest.fixture
    def sdm_item_features(self):
        """Create item features for SDM"""
        item_sparse = [
            SparseFeature(name="item_id", vocab_size=10000, embedding_dim=32),
        ]
        return [], item_sparse, []

    def test_sdm_initialization(self, sdm_user_features, sdm_item_features, device):
        """Test SDM model initialization"""
        logger.info("=" * 80)
        logger.info("Testing SDM initialization")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = sdm_user_features
        item_dense, item_sparse, item_sequence = sdm_item_features

        model = SDM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            rnn_type="GRU",
            rnn_hidden_size=64,
            training_mode="pointwise",
            device=device,
        )

        assert model is not None
        assert model.model_name == "SDM"
        assert model.rnn_type == "GRU"
        logger.info("SDM initialization successful")

        count_parameters(model)

    def test_sdm_forward_pass(
        self, sdm_user_features, sdm_item_features, device, batch_size, set_random_seed
    ):
        """Test SDM forward pass with RNN"""
        logger.info("=" * 80)
        logger.info("Testing SDM forward pass")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = sdm_user_features
        item_dense, item_sparse, item_sequence = sdm_item_features

        model = SDM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            rnn_type="GRU",
            rnn_hidden_size=64,
            training_mode="pointwise",
            device=device,
        )

        data = {
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "item_sequence": torch.randint(0, 10000, (batch_size, 50)).to(device),
            "item_id": torch.randint(1, 10000, (batch_size,)).to(device),
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,), "SDM output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "SDM output")

        logger.info("SDM forward pass successful")

    @pytest.mark.parametrize("rnn_type", ["GRU", "LSTM"])
    def test_sdm_different_rnn_types(
        self, sdm_user_features, sdm_item_features, device, batch_size, rnn_type
    ):
        """Test SDM with different RNN types"""
        logger.info("=" * 80)
        logger.info(f"Testing SDM with {rnn_type}")
        logger.info("=" * 80)

        user_dense, user_sparse, user_sequence = sdm_user_features
        item_dense, item_sparse, item_sequence = sdm_item_features

        model = SDM(
            user_dense_features=user_dense,
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_dense_features=item_dense,
            item_sparse_features=item_sparse,
            item_sequence_features=item_sequence,
            rnn_type=rnn_type,
            rnn_hidden_size=64,
            device=device,
        )

        data = {
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "item_sequence": torch.randint(0, 10000, (batch_size, 50)).to(device),
            "item_id": torch.randint(1, 10000, (batch_size,)).to(device),
        }

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info(f"SDM with {rnn_type} test successful")


class TestTrainingModes:
    """Test suite for different training modes and loss functions"""

    @pytest.fixture
    def simple_features(self):
        """Create simple features for testing"""
        user_sparse = [SparseFeature(name="user_id", vocab_size=100, embedding_dim=8)]
        item_sparse = [SparseFeature(name="item_id", vocab_size=100, embedding_dim=8)]
        return user_sparse, item_sparse

    @pytest.mark.parametrize("training_mode", ["pointwise", "pairwise", "listwise"])
    def test_dssm_training_modes(
        self, simple_features, device, batch_size, training_mode
    ):
        """Test DSSM with different training modes"""
        logger.info("=" * 80)
        logger.info(f"Testing DSSM with {training_mode} training mode")
        logger.info("=" * 80)

        user_sparse, item_sparse = simple_features

        model = DSSM(
            user_sparse_features=user_sparse,
            item_sparse_features=item_sparse,
            embedding_dim=16,
            training_mode=training_mode,
            device=device,
        )

        # Compile with appropriate loss
        if training_mode == "pointwise":
            model.compile(loss="bce")
        elif training_mode == "pairwise":
            model.compile(loss="bpr")
        elif training_mode == "listwise":
            model.compile(loss="sampled_softmax")

        assert model.training_mode == training_mode
        assert training_mode in model.support_training_modes

        logger.info(f"DSSM {training_mode} training mode test successful")

    def test_mind_only_supports_pointwise(self, device):
        """Test that MIND only supports pointwise training mode"""
        logger.info("=" * 80)
        logger.info("Testing MIND training mode restrictions")
        logger.info("=" * 80)

        user_sparse = [SparseFeature(name="user_id", vocab_size=100, embedding_dim=8)]
        user_sequence = [
            SequenceFeature(
                name="user_hist",
                vocab_size=100,
                max_len=20,
                embedding_dim=8,
                padding_idx=0,
            )
        ]
        item_sparse = [SparseFeature(name="item_id", vocab_size=100, embedding_dim=8)]

        # Should work with pointwise
        model = MIND(
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_sparse_features=item_sparse,
            embedding_dim=16,
            training_mode="pointwise",
            device=device,
        )

        assert model.support_training_modes == ["pointwise"]

        # Should fail with pairwise
        with pytest.raises(ValueError, match="does not support training_mode"):
            model_pairwise = MIND(
                user_sparse_features=user_sparse,
                user_sequence_features=user_sequence,
                item_sparse_features=item_sparse,
                embedding_dim=16,
                training_mode="pairwise",
                device=device,
            )
            model_pairwise.compile(loss="bpr")

        logger.info("MIND training mode restriction test successful")

    def test_sdm_only_supports_pointwise(self, device):
        """Test that SDM only supports pointwise training mode"""
        logger.info("=" * 80)
        logger.info("Testing SDM training mode restrictions")
        logger.info("=" * 80)

        user_sparse = [SparseFeature(name="user_id", vocab_size=100, embedding_dim=8)]
        user_sequence = [
            SequenceFeature(
                name="item_seq",
                vocab_size=100,
                max_len=20,
                embedding_dim=8,
                padding_idx=0,
            )
        ]
        item_sparse = [SparseFeature(name="item_id", vocab_size=100, embedding_dim=8)]

        # Should work with pointwise
        model = SDM(
            user_sparse_features=user_sparse,
            user_sequence_features=user_sequence,
            item_sparse_features=item_sparse,
            embedding_dim=16,
            training_mode="pointwise",
            device=device,
        )

        assert model.support_training_modes == ["pointwise"]

        # Should fail with listwise
        with pytest.raises(ValueError, match="does not support training_mode"):
            model_listwise = SDM(
                user_sparse_features=user_sparse,
                user_sequence_features=user_sequence,
                item_sparse_features=item_sparse,
                embedding_dim=16,
                training_mode="listwise",
                device=device,
            )
            model_listwise.compile(loss="sampled_softmax")

        logger.info("SDM training mode restriction test successful")

    @pytest.mark.parametrize("loss_name", ["bpr", "hinge", "triplet"])
    def test_pairwise_losses(self, simple_features, device, batch_size, loss_name):
        """Test different pairwise loss functions"""
        logger.info("=" * 80)
        logger.info(f"Testing {loss_name} loss")
        logger.info("=" * 80)

        user_sparse, item_sparse = simple_features

        model = DSSM(
            user_sparse_features=user_sparse,
            item_sparse_features=item_sparse,
            embedding_dim=16,
            training_mode="pairwise",
            device=device,
        )

        model.compile(loss=loss_name)

        assert model.training_mode == "pairwise"

        logger.info(f"{loss_name} loss test successful")

    @pytest.mark.parametrize("loss_name", ["sampled_softmax", "infonce"])
    def test_listwise_losses(self, simple_features, device, batch_size, loss_name):
        """Test different listwise loss functions"""
        logger.info("=" * 80)
        logger.info(f"Testing {loss_name} loss")
        logger.info("=" * 80)

        user_sparse, item_sparse = simple_features

        model = DSSM(
            user_sparse_features=user_sparse,
            item_sparse_features=item_sparse,
            embedding_dim=16,
            training_mode="listwise",
            device=device,
        )

        model.compile(loss=loss_name)

        assert model.training_mode == "listwise"

        logger.info(f"{loss_name} loss test successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
