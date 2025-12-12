"""
Unit Tests for Generative Models

This module contains unit tests for generative recommendation models including:
- HSTU (Hierarchical Sequential Transduction Unit)
- TIGER (Tokenized Item Generation for Recommendation)

Tests cover model initialization, forward pass, training, and inference.

Date: create on 11/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import pytest
import torch
import logging

from nextrec.basic.features import SequenceFeature
from nextrec.models.generative.hstu import (
    HSTU,
    HSTULayer,
    HSTUPointwiseAttention,
    RelativePositionBias,
    relative_position_bucket,
)

from test.test_utils import (
    assert_model_output_shape,
    assert_no_nan_or_inf,
    run_model_inference,
    count_parameters,
)

logger = logging.getLogger(__name__)


class TestRelativePositionBucket:
    """Test suite for relative position bucket function"""

    def test_relative_position_bucket_small_distances(self):
        """Test that small distances are mapped exactly"""
        logger.info("=" * 80)
        logger.info("Testing relative position bucket for small distances")
        logger.info("=" * 80)

        num_buckets = 32
        max_exact = num_buckets // 2

        # Test small negative distances (causal attention)
        rel_pos = torch.tensor([[-1, -2, -3, -5, -10]])
        buckets = relative_position_bucket(rel_pos, num_buckets=num_buckets)

        # For small distances (< max_exact=16), should map directly
        assert torch.all(buckets[0, :3] == torch.tensor([1, 2, 3]))
        logger.info(f"Small distance buckets: {buckets}")

    def test_relative_position_bucket_large_distances(self):
        """Test that large distances use log scaling"""
        logger.info("=" * 80)
        logger.info("Testing relative position bucket for large distances")
        logger.info("=" * 80)

        num_buckets = 32
        rel_pos = torch.tensor([[-50, -100, -200]])
        buckets = relative_position_bucket(
            rel_pos, num_buckets=num_buckets, max_distance=128
        )

        # Large distances should be in [16, 31] range
        assert torch.all(buckets >= 16)
        assert torch.all(buckets < num_buckets)
        logger.info(f"Large distance buckets: {buckets}")

    def test_relative_position_bucket_positive_clamping(self):
        """Test that positive positions (future) are clamped to 0"""
        logger.info("=" * 80)
        logger.info("Testing relative position bucket positive clamping")
        logger.info("=" * 80)

        rel_pos = torch.tensor([[0, 1, 5, 10]])
        buckets = relative_position_bucket(rel_pos)

        # Positive positions should all map to bucket 0
        assert torch.all(buckets == 0)
        logger.info("Positive positions correctly clamped to 0")


class TestRelativePositionBias:
    """Test suite for RelativePositionBias module"""

    def test_rab_initialization(self):
        """Test RelativePositionBias initialization"""
        logger.info("=" * 80)
        logger.info("Testing RelativePositionBias initialization")
        logger.info("=" * 80)

        num_heads = 8
        rab = RelativePositionBias(num_heads=num_heads)

        assert rab.num_heads == num_heads
        assert rab.num_buckets == 32  # default
        assert rab.embedding.num_embeddings == 32
        assert rab.embedding.embedding_dim == num_heads
        logger.info("RelativePositionBias initialization successful")

    def test_rab_forward(self, device):
        """Test RelativePositionBias forward pass"""
        logger.info("=" * 80)
        logger.info("Testing RelativePositionBias forward pass")
        logger.info("=" * 80)

        num_heads = 8
        seq_len = 20
        rab = RelativePositionBias(num_heads=num_heads)

        bias = rab(seq_len=seq_len, device=device)

        # Output shape: [1, num_heads, seq_len, seq_len]
        assert bias.shape == (1, num_heads, seq_len, seq_len)
        assert_no_nan_or_inf(bias, "RAB output")
        logger.info(f"RAB output shape: {bias.shape}")

    def test_rab_different_seq_lengths(self, device):
        """Test RAB with different sequence lengths"""
        logger.info("=" * 80)
        logger.info("Testing RAB with different sequence lengths")
        logger.info("=" * 80)

        rab = RelativePositionBias(num_heads=4)

        for seq_len in [10, 50, 100]:
            bias = rab(seq_len=seq_len, device=device)
            assert bias.shape == (1, 4, seq_len, seq_len)
            logger.info(f"RAB works for seq_len={seq_len}")


class TestHSTUPointwiseAttention:
    """Test suite for HSTUPointwiseAttention module"""

    def test_pointwise_attention_initialization(self):
        """Test HSTUPointwiseAttention initialization"""
        logger.info("=" * 80)
        logger.info("Testing HSTUPointwiseAttention initialization")
        logger.info("=" * 80)

        hidden_dim = 64
        num_heads = 8
        attn = HSTUPointwiseAttention(hidden_dim=hidden_dim, num_heads=num_heads)

        assert attn.hidden_dim == hidden_dim
        assert attn.num_heads == num_heads
        assert attn.head_dim == hidden_dim // num_heads
        logger.info("HSTUPointwiseAttention initialization successful")

    def test_pointwise_attention_invalid_config(self):
        """Test that invalid configurations raise errors"""
        logger.info("=" * 80)
        logger.info("Testing HSTUPointwiseAttention invalid configuration")
        logger.info("=" * 80)

        # hidden_dim not divisible by num_heads
        with pytest.raises(ValueError, match="hidden_dim.*num_heads"):
            HSTUPointwiseAttention(hidden_dim=63, num_heads=8)
        logger.info("Invalid configuration correctly rejected")

    def test_pointwise_attention_forward(self, device, batch_size):
        """Test HSTUPointwiseAttention forward pass"""
        logger.info("=" * 80)
        logger.info("Testing HSTUPointwiseAttention forward pass")
        logger.info("=" * 80)

        hidden_dim = 64
        num_heads = 8
        seq_len = 20

        attn = HSTUPointwiseAttention(hidden_dim=hidden_dim, num_heads=num_heads).to(
            device
        )
        x = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert_no_nan_or_inf(output, "Pointwise attention output")
        logger.info(f"Pointwise attention output shape: {output.shape}")

    def test_pointwise_attention_with_masks(self, device, batch_size):
        """Test HSTUPointwiseAttention with attention and padding masks"""
        logger.info("=" * 80)
        logger.info("Testing HSTUPointwiseAttention with masks")
        logger.info("=" * 80)

        hidden_dim = 64
        num_heads = 8
        seq_len = 20

        attn = HSTUPointwiseAttention(hidden_dim=hidden_dim, num_heads=num_heads).to(
            device
        )
        x = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Create causal mask
        attention_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

        # Create padding mask (last 5 tokens are padding)
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        key_padding_mask[:, -5:] = True

        output = attn(
            x, attention_mask=attention_mask, key_padding_mask=key_padding_mask
        )

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert_no_nan_or_inf(output, "Masked attention output")
        logger.info("Masked attention successful")

    def test_pointwise_attention_with_rab(self, device, batch_size):
        """Test HSTUPointwiseAttention with relative position bias"""
        logger.info("=" * 80)
        logger.info("Testing HSTUPointwiseAttention with RAB")
        logger.info("=" * 80)

        hidden_dim = 64
        num_heads = 8
        seq_len = 20

        attn = HSTUPointwiseAttention(hidden_dim=hidden_dim, num_heads=num_heads).to(
            device
        )
        x = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Create RAB
        rab = torch.randn(1, num_heads, seq_len, seq_len).to(device)

        output = attn(x, rab=rab)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert_no_nan_or_inf(output, "Attention with RAB output")
        logger.info("Attention with RAB successful")


class TestHSTULayer:
    """Test suite for HSTULayer module"""

    def test_hstu_layer_initialization(self):
        """Test HSTULayer initialization"""
        logger.info("=" * 80)
        logger.info("Testing HSTULayer initialization")
        logger.info("=" * 80)

        hidden_dim = 64
        num_heads = 8
        layer = HSTULayer(hidden_dim=hidden_dim, num_heads=num_heads)

        assert layer.attention is not None
        assert layer.rel_pos_bias is not None  # use_rab_pos=True by default
        logger.info("HSTULayer initialization successful")

    def test_hstu_layer_without_rab(self):
        """Test HSTULayer without relative position bias"""
        logger.info("=" * 80)
        logger.info("Testing HSTULayer without RAB")
        logger.info("=" * 80)

        layer = HSTULayer(hidden_dim=64, num_heads=8, use_rab_pos=False)

        assert layer.rel_pos_bias is None
        logger.info("HSTULayer without RAB initialization successful")

    def test_hstu_layer_forward(self, device, batch_size):
        """Test HSTULayer forward pass"""
        logger.info("=" * 80)
        logger.info("Testing HSTULayer forward pass")
        logger.info("=" * 80)

        hidden_dim = 64
        num_heads = 8
        seq_len = 20

        layer = HSTULayer(hidden_dim=hidden_dim, num_heads=num_heads).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        output = layer(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert_no_nan_or_inf(output, "HSTULayer output")
        logger.info(f"HSTULayer output shape: {output.shape}")

    def test_hstu_layer_residual_connection(self, device, batch_size):
        """Test that HSTULayer implements residual connection"""
        logger.info("=" * 80)
        logger.info("Testing HSTULayer residual connection")
        logger.info("=" * 80)

        hidden_dim = 64
        num_heads = 8
        seq_len = 20

        layer = HSTULayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.0).to(
            device
        )
        x = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        output = layer(x)

        # Output should maintain input shape due to residual connection
        assert output.shape == x.shape
        logger.info("Residual connection test completed")


class TestHSTU:
    """Test suite for HSTU model"""

    @pytest.fixture
    def hstu_sequence_feature(self):
        """Create sequence feature for HSTU"""
        return [
            SequenceFeature(
                name="item_history",
                vocab_size=10000,
                max_len=100,
                embedding_dim=64,
                padding_idx=0,
            )
        ]

    def test_hstu_initialization(self, hstu_sequence_feature, device):
        """Test HSTU model initialization"""
        logger.info("=" * 80)
        logger.info("Testing HSTU initialization")
        logger.info("=" * 80)

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=4,
            max_seq_len=100,
            target=["next_item"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "HSTU"
        assert model.default_task == "multiclass"
        assert model.hidden_dim == 64
        assert model.vocab_size == 10000
        assert len(model.layers) == 4
        logger.info("HSTU initialization successful")

        # Count parameters
        num_params = count_parameters(model)
        logger.info(f"HSTU total parameters: {num_params:,}")

    def test_hstu_requires_sequence_feature(self):
        """Test that HSTU requires at least one sequence feature"""
        logger.info("=" * 80)
        logger.info("Testing HSTU requires sequence feature")
        logger.info("=" * 80)

        with pytest.raises(ValueError, match="at least one SequenceFeature"):
            HSTU(sequence_features=[], device="cpu")
        logger.info("Sequence feature requirement correctly enforced")

    def test_hstu_forward_pass(self, hstu_sequence_feature, device, batch_size):
        """Test HSTU forward pass"""
        logger.info("=" * 80)
        logger.info("Testing HSTU forward pass")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size
        seq_len = 50

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            max_seq_len=100,
            target=["next_item"],
            device=device,
        )

        # Create input data
        data = {
            "item_history": torch.randint(0, vocab_size, (batch_size, seq_len)).to(
                device
            )
        }

        output = run_model_inference(model, data)

        # Output should be [B, vocab_size] logits for next item prediction
        assert_model_output_shape(output, (batch_size, vocab_size), "HSTU output shape")
        assert_no_nan_or_inf(output, "HSTU output")
        logger.info(f"HSTU forward pass successful, output shape: {output.shape}")

    def test_hstu_with_padding(self, hstu_sequence_feature, device, batch_size):
        """Test HSTU with padded sequences"""
        logger.info("=" * 80)
        logger.info("Testing HSTU with padding")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size
        seq_len = 50
        padding_idx = hstu_sequence_feature[0].padding_idx

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            max_seq_len=100,
            target=["next_item"],
            device=device,
        )

        # Create sequences with padding (last 20 tokens are padding)
        data = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
        data[:, -20:] = padding_idx

        input_data = {"item_history": data}
        output = run_model_inference(model, input_data)

        assert_model_output_shape(output, (batch_size, vocab_size))
        assert_no_nan_or_inf(output, "HSTU padded output")
        logger.info("HSTU with padding successful")

    def test_hstu_sequence_trimming(self, hstu_sequence_feature, device, batch_size):
        """Test that HSTU correctly trims long sequences"""
        logger.info("=" * 80)
        logger.info("Testing HSTU sequence trimming")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size
        max_seq_len = 50
        long_seq_len = 100  # Longer than max_seq_len

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            max_seq_len=max_seq_len,
            target=["next_item"],
            device=device,
        )

        # Create long sequence
        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, long_seq_len)).to(
                device
            )
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size, vocab_size))
        assert_no_nan_or_inf(output, "HSTU trimmed sequence output")
        logger.info("Sequence trimming successful")

    def test_hstu_training_step(self, hstu_sequence_feature, device, batch_size):
        """Test HSTU training with loss computation"""
        logger.info("=" * 80)
        logger.info("Testing HSTU training step")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size
        seq_len = 50

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            max_seq_len=100,
            target=["next_item"],
            optimizer="adam",
            device=device,
        )

        # Create training data
        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, seq_len)).to(
                device
            )
        }
        labels = torch.randint(1, vocab_size, (batch_size,)).to(device)

        # Forward pass
        model.train()
        logits = model(data)

        # Compute loss
        loss = model.compute_loss(logits, labels)

        assert loss is not None
        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Backward pass
        loss.backward()

        logger.info(f"Training step successful, loss: {loss.item():.4f}")

    def test_hstu_ignore_padding_in_loss(
        self, hstu_sequence_feature, device, batch_size
    ):
        """Test that HSTU ignores padding tokens in loss computation"""
        logger.info("=" * 80)
        logger.info("Testing HSTU ignores padding in loss")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size
        padding_idx = hstu_sequence_feature[0].padding_idx

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            target=["next_item"],
            device=device,
        )

        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, 50)).to(device)
        }

        # Create labels with some padding
        labels = torch.randint(1, vocab_size, (batch_size,)).to(device)
        labels[0] = padding_idx  # First sample has padding as label

        model.train()
        logits = model(data)
        loss = model.compute_loss(logits, labels)

        # Loss should still be computed (ignoring padded samples)
        assert not torch.isnan(loss)
        logger.info("Padding in loss correctly handled")

    @pytest.mark.parametrize("num_layers", [1, 2, 4, 8])
    def test_hstu_different_depths(
        self, hstu_sequence_feature, device, batch_size, num_layers
    ):
        """Test HSTU with different numbers of layers"""
        logger.info("=" * 80)
        logger.info(f"Testing HSTU with {num_layers} layers")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=num_layers,
            target=["next_item"],
            device=device,
        )

        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, 50)).to(device)
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size, vocab_size))
        assert_no_nan_or_inf(output, f"HSTU {num_layers}-layer output")
        logger.info(f"HSTU with {num_layers} layers successful")

    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    def test_hstu_different_heads(
        self, hstu_sequence_feature, device, batch_size, num_heads
    ):
        """Test HSTU with different numbers of attention heads"""
        logger.info("=" * 80)
        logger.info(f"Testing HSTU with {num_heads} heads")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size
        # Make hidden_dim divisible by num_heads
        hidden_dim = num_heads * 8

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            target=["next_item"],
            device=device,
        )

        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, 50)).to(device)
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size, vocab_size))
        assert_no_nan_or_inf(output, f"HSTU {num_heads}-head output")
        logger.info(f"HSTU with {num_heads} heads successful")

    def test_hstu_with_rms_norm(self, hstu_sequence_feature, device, batch_size):
        """Test HSTU with RMSNorm instead of LayerNorm"""
        logger.info("=" * 80)
        logger.info("Testing HSTU with RMSNorm")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            use_rms_norm=True,
            target=["next_item"],
            device=device,
        )

        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, 50)).to(device)
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size, vocab_size))
        assert_no_nan_or_inf(output, "HSTU with RMSNorm output")
        logger.info("HSTU with RMSNorm successful")

    def test_hstu_without_rab(self, hstu_sequence_feature, device, batch_size):
        """Test HSTU without relative position bias"""
        logger.info("=" * 80)
        logger.info("Testing HSTU without RAB")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            use_rab_pos=False,
            target=["next_item"],
            device=device,
        )

        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, 50)).to(device)
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size, vocab_size))
        assert_no_nan_or_inf(output, "HSTU without RAB output")
        logger.info("HSTU without RAB successful")

    def test_hstu_tied_embeddings(self, hstu_sequence_feature, device):
        """Test that HSTU ties embeddings between input and output when specified"""
        logger.info("=" * 80)
        logger.info("Testing HSTU tied embeddings")
        logger.info("=" * 80)

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            tie_embeddings=True,
            target=["next_item"],
            device=device,
        )

        # Check that weights are tied
        assert (
            model.lm_head.weight is model.token_embedding.weight
        ), "Embeddings should be tied"
        logger.info("Tied embeddings verified")

    def test_hstu_untied_embeddings(self, hstu_sequence_feature, device):
        """Test HSTU with separate input and output embeddings"""
        logger.info("=" * 80)
        logger.info("Testing HSTU untied embeddings")
        logger.info("=" * 80)

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            tie_embeddings=False,
            target=["next_item"],
            device=device,
        )

        # Check that weights are NOT tied
        assert (
            model.lm_head.weight is not model.token_embedding.weight
        ), "Embeddings should not be tied"
        logger.info("Untied embeddings verified")

    def test_hstu_deterministic(self, hstu_sequence_feature, device, batch_size):
        """Test that HSTU produces deterministic outputs with same random seed"""
        logger.info("=" * 80)
        logger.info("Testing HSTU determinism")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size

        # Create two models with same seed
        torch.manual_seed(42)
        model1 = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            target=["next_item"],
            device=device,
        )

        torch.manual_seed(42)
        model2 = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            target=["next_item"],
            device=device,
        )

        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, 50)).to(device)
        }

        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(data)
            output2 = model2(data)

        assert torch.allclose(
            output1, output2, atol=1e-6
        ), "Models with same seed should produce identical outputs"
        logger.info("HSTU determinism test successful")

    def test_hstu_gradient_flow(self, hstu_sequence_feature, device, batch_size):
        """Test that gradients flow through HSTU"""
        logger.info("=" * 80)
        logger.info("Testing HSTU gradient flow")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            target=["next_item"],
            device=device,
        )

        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, 50)).to(device)
        }
        labels = torch.randint(1, vocab_size, (batch_size,)).to(device)

        model.train()
        logits = model(data)
        loss = model.compute_loss(logits, labels)
        loss.backward()

        # Check that gradients exist and are non-zero for embedding layer
        assert model.token_embedding.weight.grad is not None
        grad_sum = model.token_embedding.weight.grad.abs().sum().item()
        assert grad_sum > 0, "Gradients should be non-zero"

        logger.info("Gradient flow verified")


class TestHSTUEdgeCases:
    """Test edge cases and error handling for HSTU"""

    @pytest.fixture
    def hstu_sequence_feature(self):
        """Create sequence feature for HSTU"""
        return [
            SequenceFeature(
                name="item_history",
                vocab_size=1000,
                max_len=100,
                embedding_dim=64,
                padding_idx=0,
            )
        ]

    def test_hstu_single_item_sequence(self, hstu_sequence_feature, device, batch_size):
        """Test HSTU with sequence length of 1"""
        logger.info("=" * 80)
        logger.info("Testing HSTU with single item sequence")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            target=["next_item"],
            device=device,
        )

        data = {
            "item_history": torch.randint(1, vocab_size, (batch_size, 1)).to(device)
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size, vocab_size))
        assert_no_nan_or_inf(output, "Single item sequence output")
        logger.info("Single item sequence handled correctly")

    def test_hstu_all_padding_sequence(self, hstu_sequence_feature, device):
        """Test HSTU with sequences that are all padding"""
        logger.info("=" * 80)
        logger.info("Testing HSTU with all padding sequence")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size
        padding_idx = hstu_sequence_feature[0].padding_idx

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            target=["next_item"],
            device=device,
        )

        # Create all-padding sequence
        data = {
            "item_history": torch.full((2, 50), padding_idx, dtype=torch.long).to(
                device
            )
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (2, vocab_size))
        # Output may have unusual values but should not be NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        logger.info("All-padding sequence handled without crashing")

    def test_hstu_batch_size_one(self, hstu_sequence_feature, device):
        """Test HSTU with batch size of 1"""
        logger.info("=" * 80)
        logger.info("Testing HSTU with batch size 1")
        logger.info("=" * 80)

        vocab_size = hstu_sequence_feature[0].vocab_size

        model = HSTU(
            sequence_features=hstu_sequence_feature,
            hidden_dim=64,
            num_heads=8,
            num_layers=2,
            target=["next_item"],
            device=device,
        )

        data = {"item_history": torch.randint(1, vocab_size, (1, 50)).to(device)}

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (1, vocab_size))
        assert_no_nan_or_inf(output, "Batch size 1 output")
        logger.info("Batch size 1 handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
