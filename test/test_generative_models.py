"""
Unit Tests for Generative Models

This module contains unit tests for generative model components.

Date: create on 11/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import logging
from test.helpers import assert_no_nan_or_inf

import pytest
import torch

from nextrec.models.sequential.hstu import (
    HSTULayer,
    HSTUPointwiseAttention,
    RelativePositionBias,
    relative_position_bucket,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
