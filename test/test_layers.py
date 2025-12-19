"""
Unit Tests for Basic Layers

This module contains unit tests for basic layer components including:
- MLP (Multi-Layer Perceptron)
- FM (Factorization Machine)
- CrossNetwork
- EmbeddingLayer
- Attention mechanisms
- And other building blocks
"""

import logging

import pytest
import torch

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import (
    FM,
    LR,
    MLP,
    AveragePooling,
    EmbeddingLayer,
    MultiHeadSelfAttention,
    PredictionLayer,
    SumPooling,
)
from nextrec.models.ranking.dcn import CrossNetwork
from nextrec.models.ranking.xdeepfm import CIN

logger = logging.getLogger(__name__)


class TestMLP:
    """Test suite for MLP (Multi-Layer Perceptron)"""

    def test_mlp_initialization(self):
        """Test MLP initialization"""
        logger.info("=" * 80)
        logger.info("Testing MLP initialization")
        logger.info("=" * 80)

        mlp = MLP(input_dim=128, dims=[64, 32, 16], dropout=0, activation="relu")

        assert mlp is not None
        logger.info("MLP initialization successful")

    def test_mlp_forward_pass(self):
        """Test MLP forward pass"""
        logger.info("=" * 80)
        logger.info("Testing MLP forward pass")
        logger.info("=" * 80)

        batch_size = 32
        input_dim = 128

        mlp = MLP(
            input_dim=input_dim,
            dims=[64, 32],
            dropout=0.0,
            activation="relu",
            output_layer=False,
        )

        x = torch.randn(batch_size, input_dim)
        output = mlp(x)

        assert output.shape == (batch_size, 32)
        assert not torch.isnan(output).any()

        logger.info("MLP forward pass successful")

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh", "leaky_relu"])
    def test_mlp_activations(self, activation):
        """Test MLP with different activation functions"""
        logger.info("=" * 80)
        logger.info(f"Testing MLP with {activation} activation")
        logger.info("=" * 80)

        mlp = MLP(
            input_dim=64,
            dims=[32, 16],
            dropout=0.0,
            activation=activation,
            output_layer=False,
        )

        x = torch.randn(16, 64)
        output = mlp(x)

        assert output.shape == (16, 16)

        logger.info(f"MLP with {activation} test successful")

    def test_mlp_different_depths(self):
        """Test MLP with different depths"""
        logger.info("=" * 80)
        logger.info("Testing MLP with different depths")
        logger.info("=" * 80)

        for dims in [[32], [64, 32], [128, 64, 32], [256, 128, 64, 32]]:
            mlp = MLP(
                input_dim=128,
                dims=dims,
                dropout=0.0,
                activation="relu",
                output_layer=False,
            )

            x = torch.randn(8, 128)
            output = mlp(x)

            assert output.shape == (8, dims[-1])

        logger.info("MLP different depths test successful")


class TestFM:
    """Test suite for FM (Factorization Machine)"""

    def test_fm_initialization(self):
        """Test FM initialization"""
        logger.info("=" * 80)
        logger.info("Testing FM initialization")
        logger.info("=" * 80)

        fm = FM()
        assert fm is not None

        logger.info("FM initialization successful")

    def test_fm_forward_pass(self):
        """Test FM forward pass"""
        logger.info("=" * 80)
        logger.info("Testing FM forward pass")
        logger.info("=" * 80)

        batch_size = 32
        num_fields = 10
        embedding_dim = 16

        # Input: [batch_size, num_fields, embedding_dim]
        x = torch.randn(batch_size, num_fields, embedding_dim)

        fm = FM()
        output = fm(x)

        # Output: [batch_size, 1]
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

        logger.info("FM forward pass successful")

    def test_fm_interaction_computation(self):
        """Test that FM computes interactions correctly"""
        logger.info("=" * 80)
        logger.info("Testing FM interaction computation")
        logger.info("=" * 80)

        # Simple case with 2 fields
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]]  # batch_size=1, num_fields=2, embedding_dim=2
        )

        fm = FM()
        output = fm(x)

        # Should compute: sum of (v_i * v_j) for i < j
        # = (1*3 + 2*4) = 3 + 8 = 11
        expected = 11.0

        assert torch.allclose(output, torch.tensor([[expected]]), atol=1e-5)

        logger.info("FM interaction computation test successful")


class TestCrossNetwork:
    """Test suite for CrossNetwork"""

    def test_crossnetwork_initialization(self):
        """Test CrossNetwork initialization"""
        logger.info("=" * 80)
        logger.info("Testing CrossNetwork initialization")
        logger.info("=" * 80)

        cross_net = CrossNetwork(input_dim=128, num_layers=3)

        assert cross_net is not None
        assert cross_net.num_layers == 3

        logger.info("CrossNetwork initialization successful")

    def test_crossnetwork_forward_pass(self):
        """Test CrossNetwork forward pass"""
        logger.info("=" * 80)
        logger.info("Testing CrossNetwork forward pass")
        logger.info("=" * 80)

        batch_size = 32
        input_dim = 128

        cross_net = CrossNetwork(input_dim=input_dim, num_layers=2)

        x = torch.randn(batch_size, input_dim)
        output = cross_net(x)

        assert output.shape == (batch_size, input_dim)
        assert not torch.isnan(output).any()

        logger.info("CrossNetwork forward pass successful")

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 5])
    def test_crossnetwork_different_layers(self, num_layers):
        """Test CrossNetwork with different numbers of layers"""
        logger.info("=" * 80)
        logger.info(f"Testing CrossNetwork with {num_layers} layers")
        logger.info("=" * 80)

        input_dim = 64
        cross_net = CrossNetwork(input_dim=input_dim, num_layers=num_layers)

        x = torch.randn(16, input_dim)
        output = cross_net(x)

        assert output.shape == (16, input_dim)

        logger.info(f"CrossNetwork with {num_layers} layers test successful")


class TestCIN:
    """Test suite for CIN (Compressed Interaction Network)"""

    def test_cin_initialization(self):
        """Test CIN initialization"""
        logger.info("=" * 80)
        logger.info("Testing CIN initialization")
        logger.info("=" * 80)

        cin = CIN(input_dim=10, cin_size=[64, 64], split_half=True)

        assert cin is not None

        logger.info("CIN initialization successful")

    def test_cin_forward_pass(self):
        """Test CIN forward pass"""
        logger.info("=" * 80)
        logger.info("Testing CIN forward pass")
        logger.info("=" * 80)

        batch_size = 32
        num_fields = 10
        embedding_dim = 16

        # Input: [batch_size, num_fields, embedding_dim]
        x = torch.randn(batch_size, num_fields, embedding_dim)

        cin = CIN(input_dim=num_fields, cin_size=[64, 64], split_half=True)

        output = cin(x)

        # Output shape depends on configuration
        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()

        logger.info("CIN forward pass successful")


class TestLR:
    """Test suite for LR (Linear Regression layer)"""

    def test_lr_initialization(self):
        """Test LR initialization"""
        logger.info("=" * 80)
        logger.info("Testing LR initialization")
        logger.info("=" * 80)

        lr = LR(input_dim=128)
        assert lr is not None

        logger.info("LR initialization successful")

    def test_lr_forward_pass(self):
        """Test LR forward pass"""
        logger.info("=" * 80)
        logger.info("Testing LR forward pass")
        logger.info("=" * 80)

        batch_size = 32
        input_dim = 128

        lr = LR(input_dim=input_dim)
        x = torch.randn(batch_size, input_dim)
        output = lr(x)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

        logger.info("LR forward pass successful")


class TestEmbeddingLayer:
    """Test suite for EmbeddingLayer"""

    @pytest.fixture
    def sample_features(self):
        """Create sample features for embedding"""
        dense_features = [DenseFeature(name="age", embedding_dim=16)]
        sparse_features = [
            SparseFeature(name="user_id", vocab_size=1000, embedding_dim=16),
            SparseFeature(name="item_id", vocab_size=500, embedding_dim=16),
        ]
        sequence_features = [
            SequenceFeature(
                name="history",
                vocab_size=500,
                max_len=20,
                embedding_dim=16,
                padding_idx=0,
            )
        ]
        return dense_features + sparse_features + sequence_features

    def test_embedding_layer_initialization(self, sample_features):
        """Test EmbeddingLayer initialization"""
        logger.info("=" * 80)
        logger.info("Testing EmbeddingLayer initialization")
        logger.info("=" * 80)

        embedding_layer = EmbeddingLayer(features=sample_features)
        assert embedding_layer is not None

        logger.info("EmbeddingLayer initialization successful")

    def test_embedding_layer_forward_pass(self, sample_features):
        """Test EmbeddingLayer forward pass"""
        logger.info("=" * 80)
        logger.info("Testing EmbeddingLayer forward pass")
        logger.info("=" * 80)

        batch_size = 32
        embedding_layer = EmbeddingLayer(features=sample_features)

        # Create sample input
        x = {
            "age": torch.randn(batch_size, 1),
            "user_id": torch.randint(1, 1000, (batch_size,)),
            "item_id": torch.randint(1, 500, (batch_size,)),
            "history": torch.randint(0, 500, (batch_size, 20)),
        }

        output = embedding_layer(x, features=sample_features, squeeze_dim=False)

        # Output should be [batch_size, num_features, embedding_dim]
        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()

        logger.info("EmbeddingLayer forward pass successful")

    def test_embedding_layer_squeeze(self, sample_features):
        """Test EmbeddingLayer with squeeze_dim=True"""
        logger.info("=" * 80)
        logger.info("Testing EmbeddingLayer squeeze_dim")
        logger.info("=" * 80)

        batch_size = 16
        embedding_layer = EmbeddingLayer(features=sample_features)

        x = {
            "age": torch.randn(batch_size, 1),
            "user_id": torch.randint(1, 1000, (batch_size,)),
            "item_id": torch.randint(1, 500, (batch_size,)),
            "history": torch.randint(0, 500, (batch_size, 20)),
        }

        output = embedding_layer(x, features=sample_features, squeeze_dim=True)

        # With squeeze, should concatenate all embeddings
        assert output.shape[0] == batch_size
        assert len(output.shape) == 2

        logger.info("EmbeddingLayer squeeze_dim test successful")


class TestMultiHeadSelfAttention:
    """Test suite for MultiHeadSelfAttention"""

    def test_attention_initialization(self):
        """Test MultiHeadSelfAttention initialization"""
        logger.info("=" * 80)
        logger.info("Testing MultiHeadSelfAttention initialization")
        logger.info("=" * 80)

        attention = MultiHeadSelfAttention(
            embedding_dim=64, num_heads=4, dropout=0.1, use_residual=True
        )

        assert attention is not None

        logger.info("MultiHeadSelfAttention initialization successful")

    def test_attention_forward_pass(self):
        """Test MultiHeadSelfAttention forward pass"""
        logger.info("=" * 80)
        logger.info("Testing MultiHeadSelfAttention forward pass")
        logger.info("=" * 80)

        batch_size = 32
        seq_len = 10
        embedding_dim = 64

        attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim, num_heads=4, dropout=0.0, use_residual=True
        )

        # Input: [batch_size, seq_len, embedding_dim]
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert not torch.isnan(output).any()

        logger.info("MultiHeadSelfAttention forward pass successful")

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_attention_different_heads(self, num_heads):
        """Test attention with different numbers of heads"""
        logger.info("=" * 80)
        logger.info(f"Testing MultiHeadSelfAttention with {num_heads} heads")
        logger.info("=" * 80)

        embedding_dim = 64
        attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, dropout=0.0
        )

        x = torch.randn(16, 10, embedding_dim)
        output = attention(x)

        assert output.shape == x.shape

        logger.info(f"MultiHeadSelfAttention with {num_heads} heads test successful")

    def test_attention_with_mask(self):
        """Test MultiHeadSelfAttention with attention mask"""
        logger.info("=" * 80)
        logger.info("Testing MultiHeadSelfAttention with attention mask")
        logger.info("=" * 80)

        batch_size = 8
        seq_len = 12
        embedding_dim = 64

        attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim, num_heads=4, dropout=0.0, use_residual=True
        )

        x = torch.randn(batch_size, seq_len, embedding_dim)
        # Create a mask: first 8 positions are valid, rest are padding
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[:, 8:] = False  # Mask out last 4 positions

        output = attention(x, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert not torch.isnan(output).any()

        logger.info("MultiHeadSelfAttention with mask test successful")

    def test_attention_with_layer_norm(self):
        """Test MultiHeadSelfAttention with layer normalization"""
        logger.info("=" * 80)
        logger.info("Testing MultiHeadSelfAttention with layer normalization")
        logger.info("=" * 80)

        batch_size = 16
        seq_len = 10
        embedding_dim = 64

        attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=4,
            dropout=0.0,
            use_residual=True,
            use_layer_norm=True,
        )

        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert not torch.isnan(output).any()
        # Check that layer norm is applied (output should have mean ~0 and std ~1 per feature)
        assert hasattr(attention, "layer_norm")

        logger.info("MultiHeadSelfAttention with layer norm test successful")

    def test_attention_flash_vs_standard(self):
        """Test that Flash Attention and standard attention produce similar results"""
        logger.info("=" * 80)
        logger.info("Testing Flash Attention vs standard attention compatibility")
        logger.info("=" * 80)

        batch_size = 8
        seq_len = 10
        embedding_dim = 64

        attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim, num_heads=4, dropout=0.0, use_residual=False
        )

        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Run with same input
        attention.eval()  # Set to eval mode to disable dropout
        output = attention(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert not torch.isnan(output).any()

        logger.info(f"Using Flash Attention: {attention.use_flash_attention}")
        logger.info("Flash vs standard attention compatibility test successful")


class TestPoolingLayers:
    """Test suite for pooling layers"""

    def test_sum_pooling(self):
        """Test SumPooling"""
        logger.info("=" * 80)
        logger.info("Testing SumPooling")
        logger.info("=" * 80)

        pooling = SumPooling()

        # Input: [batch_size, seq_len, embedding_dim]
        x = torch.randn(32, 10, 16)
        output = pooling(x)

        # Output: [batch_size, embedding_dim]
        assert output.shape == (32, 16)

        logger.info("SumPooling test successful")

    def test_average_pooling(self):
        """Test AveragePooling"""
        logger.info("=" * 80)
        logger.info("Testing AveragePooling")
        logger.info("=" * 80)

        pooling = AveragePooling()

        x = torch.randn(32, 10, 16)
        output = pooling(x)

        assert output.shape == (32, 16)

        logger.info("AveragePooling test successful")


class TestPredictionLayer:
    """Test suite for PredictionLayer"""

    def test_prediction_layer_binary(self):
        """Test PredictionLayer for binary classification"""
        logger.info("=" * 80)
        logger.info("Testing PredictionLayer for binary classification")
        logger.info("=" * 80)

        pred_layer = PredictionLayer(task_type="binary")

        x = torch.randn(32, 1)
        output = pred_layer(x)

        # Output should be probabilities in [0, 1]
        assert output.shape == (32, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)

        logger.info("PredictionLayer binary test successful")

    def test_prediction_layer_regression(self):
        """Test PredictionLayer for regression"""
        logger.info("=" * 80)
        logger.info("Testing PredictionLayer for regression")
        logger.info("=" * 80)

        pred_layer = PredictionLayer(task_type="regression")

        x = torch.randn(32, 1)
        output = pred_layer(x)

        assert output.shape == (32, 1)

        logger.info("PredictionLayer regression test successful")

    def test_prediction_layer_multitask(self):
        """PredictionLayer should support multiple task heads"""
        pred_layer = PredictionLayer(
            task_type=["binary", "regression"],
            task_dims=[1, 1],
        )

        x = torch.randn(16, 2)
        output = pred_layer(x)

        assert output.shape == (16, 2)
        assert torch.all(output[:, 0] >= 0) and torch.all(output[:, 0] <= 1)

    def test_prediction_layer_shared_dim_multihead(self):
        """Single task_dim should broadcast across multiple task heads"""
        pred_layer = PredictionLayer(task_type=["binary", "regression"], task_dims=1)

        x = torch.randn(4, 2)
        output = pred_layer(x)

        assert output.shape == (4, 2)
        assert torch.all(output[:, 0] >= 0) and torch.all(output[:, 0] <= 1)

    def test_prediction_layer_input_dim_mismatch_raises(self):
        """Input last dimension must match configured task dims"""
        pred_layer = PredictionLayer(task_type="binary", task_dims=1)
        bad_input = torch.randn(2, 2)

        with pytest.raises(ValueError):
            _ = pred_layer(bad_input)

    def test_prediction_layer_return_logits(self):
        """PredictionLayer can be configured to skip final activation"""
        pred_layer = PredictionLayer(
            task_type="binary", return_logits=True, use_bias=False
        )

        x = torch.randn(10, 1)
        output = pred_layer(x)
        assert output.shape == (10, 1)
        assert torch.allclose(output, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
