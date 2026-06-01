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

from nextrec.basic.features import SequenceFeature
from nextrec.engine.predictor import BasePredictor
from nextrec.models.sequential.gru4rec import GRU4Rec
from nextrec.models.sequential.hllm import HLLM
from nextrec.models.sequential.hstu import (
    HSTU,
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
        buckets = relative_position_bucket(rel_pos, num_buckets=num_buckets, max_distance=128)

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

        assert rab.num_buckets == 32  # default
        assert rab.max_distance == 128  # default
        assert rab.relative_attention_bias is rab.embedding
        assert rab.embedding.num_embeddings == rab.num_buckets
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

        # Output shape: [num_heads, seq_len, seq_len]
        assert bias.shape == (num_heads, seq_len, seq_len)
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
            assert bias.shape == (4, seq_len, seq_len)
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

        attn = HSTUPointwiseAttention(hidden_dim=hidden_dim, num_heads=num_heads).to(device)
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

        attn = HSTUPointwiseAttention(hidden_dim=hidden_dim, num_heads=num_heads).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        # Create causal mask
        attention_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

        # Create padding mask (last 5 tokens are padding)
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        key_padding_mask[:, -5:] = True

        output = attn(x, attention_mask=attention_mask, key_padding_mask=key_padding_mask)

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

        attn = HSTUPointwiseAttention(hidden_dim=hidden_dim, num_heads=num_heads).to(device)
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

        layer = HSTULayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.0).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim).to(device)

        output = layer(x)

        # Output should maintain input shape due to residual connection
        assert output.shape == x.shape
        logger.info("Residual connection test completed")


class TestHSTUModel:
    def test_hstu_forward_and_loss(self):
        sequence_features = [
            SequenceFeature(
                name="item_history",
                vocab_size=16,
                max_len=5,
                embedding_dim=8,
                padding_idx=0,
            )
        ]
        model = HSTU(
            sequence_features=sequence_features,
            item_history_name="item_history",
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            ff_hidden_dim=16,
            max_seq_len=5,
            dropout_rate=0.0,
            use_rab_pos=True,
            use_temporal_bias=False,
            tie_embeddings=True,
            target=["next_item"],
            task="generative",
            device="cpu",
            session_id="hstu_test",
        )
        model.compile(loss="ce")

        x = {
            "item_history": torch.tensor(
                [
                    [1, 2, 3, 4, 0],
                    [2, 3, 4, 5, 0],
                ],
                dtype=torch.long,
            )
        }
        y_true = torch.tensor(
            [
                [2, 3, 4, 5, 0],
                [3, 4, 5, 6, 0],
            ],
            dtype=torch.long,
        )

        y_pred = model.forward(x)
        assert y_pred.shape == (2, 5, 16)
        assert_no_nan_or_inf(y_pred, "hstu_logits")

        loss = model.compute_loss(y_pred, y_true)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestSequentialPredictionHelpers:
    def test_hllm_hstu_keep_dataset_predict_api(self):
        assert HLLM.predict is BasePredictor.predict
        assert HSTU.predict is BasePredictor.predict

    def test_hllm_hstu_sequence_prediction_helpers(self):
        sequence_features = [
            SequenceFeature(
                name="item_history",
                vocab_size=20,
                max_len=5,
                embedding_dim=8,
                padding_idx=0,
            )
        ]
        x = {
            "item_history": torch.tensor(
                [
                    [1, 2, 3, 0, 0],
                    [4, 5, 0, 0, 0],
                ],
                dtype=torch.long,
            )
        }

        models = [
            HLLM(
                sequence_features=sequence_features,
                hidden_dim=8,
                num_heads=2,
                num_layers=1,
                use_time_embedding=False,
                device="cpu",
                session_id="hllm_predict_helpers_test",
            ),
            HSTU(
                sequence_features=sequence_features,
                hidden_dim=8,
                num_heads=2,
                num_layers=1,
                ff_hidden_dim=16,
                max_seq_len=5,
                dropout_rate=0.0,
                use_temporal_bias=False,
                device="cpu",
                session_id="hstu_predict_helpers_test",
            ),
        ]

        for model in models:
            logits = model.predict_next_logits(x)
            raw_topk = model.recommend_topk(x["item_history"], top_k=3)
            assert logits.shape == (2, 20)
            assert raw_topk.shape == (2, 3)
            assert torch.all(raw_topk.ne(model.padding_idx))


class TestGRU4RecModel:
    def test_gru4rec_forward_and_loss(self):
        sequence_features = [
            SequenceFeature(
                name="item_history",
                vocab_size=20,
                max_len=6,
                embedding_dim=8,
                padding_idx=0,
            )
        ]
        model = GRU4Rec(
            sequence_features=sequence_features,
            item_history_name="item_history",
            hidden_dim=8,
            num_layers=1,
            max_seq_len=6,
            dropout_rate=0.0,
            target=["next_item"],
            task="generative",
            device="cpu",
            session_id="gru4rec_test",
        )
        model.compile(loss="ce")

        x = {
            "item_history": torch.tensor(
                [
                    [1, 2, 3, 4, 0, 0],
                    [2, 3, 4, 5, 6, 0],
                ],
                dtype=torch.long,
            )
        }
        y_true = torch.tensor(
            [
                [2, 3, 4, 5, 0, 0],
                [3, 4, 5, 6, 7, 0],
            ],
            dtype=torch.long,
        )

        y_pred = model.forward(x)
        assert y_pred.shape == (2, 6, 20)
        assert_no_nan_or_inf(y_pred, "gru4rec_logits")

        loss = model.compute_loss(y_pred, y_true)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

        next_item_logits = model.predict_last(x)
        assert next_item_logits.shape == (2, 20)
        assert_no_nan_or_inf(next_item_logits, "gru4rec_last_logits")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
