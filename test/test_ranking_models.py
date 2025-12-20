"""
Unit Tests for Ranking Models

This module contains unit tests for all ranking/CTR prediction models including:
- LR (Logistic Regression)
- DeepFM (Deep Factorization Machine)
- FFM (Field-aware Factorization Machines)
- DIN (Deep Interest Network)
- DIEN (Deep Interest Evolution Network)
- DCN (Deep & Cross Network)
- AutoInt
- WideDeep
- xDeepFM (Extreme Deep Factorization Machine)
- EulerNet

Tests cover model initialization, forward pass, training, and inference.
"""

import logging
from test.helpers import (
    assert_model_output_range,
    assert_model_output_shape,
    assert_no_nan_or_inf,
    count_parameters,
    run_model_forward_backward,
    run_model_inference,
)

import pytest
import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.models.ranking.afm import AFM
from nextrec.models.ranking.autoint import AutoInt
from nextrec.models.ranking.dcn import DCN
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.models.ranking.dien import DIEN
from nextrec.models.ranking.din import DIN
from nextrec.models.ranking.eulernet import EulerNet
from nextrec.models.ranking.ffm import FFM
from nextrec.models.ranking.fibinet import FiBiNET
from nextrec.models.ranking.fm import FM
from nextrec.models.ranking.lr import LR
from nextrec.models.ranking.pnn import PNN
from nextrec.models.ranking.widedeep import WideDeep
from nextrec.models.ranking.xdeepfm import xDeepFM

logger = logging.getLogger(__name__)


class TestFMModel:
    """Test suite for FM ranking model"""

    def test_fm_initialization(
        self, sample_sparse_features, sample_sequence_features, device
    ):
        model = FM(
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            target=["label"],
            device=device,
        )
        assert model.model_name == "FM"
        assert model.task == "binary"

    def test_fm_forward_pass(
        self,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
    ):
        model = FM(
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            target=["label"],
            device=device,
        )
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,))
        assert_model_output_range(output, 0.0, 1.0)


class TestLRModel:
    """Test suite for LR ranking model"""

    def test_lr_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        model = LR(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            target=["label"],
            device=device,
        )
        assert model.model_name == "LR"
        assert model.task == "binary"

    def test_lr_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
    ):
        model = LR(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            target=["label"],
            device=device,
        )
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,))
        assert_model_output_range(output, 0.0, 1.0)


class TestEulerNet:
    """Test suite for EulerNet"""

    def test_eulernet_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        model = EulerNet(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            target=["label"],
            device=device,
        )
        assert model.model_name == "EulerNet"
        assert model.task == "binary"

    def test_eulernet_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
    ):
        model = EulerNet(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            target=["label"],
            device=device,
        )
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,))
        assert_model_output_range(output, 0.0, 1.0)


class TestDeepFM:
    """Test suite for DeepFM (Deep Factorization Machine)"""

    def test_deepfm_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test DeepFM model initialization"""
        logger.info("=" * 80)
        logger.info("Testing DeepFM initialization")
        logger.info("=" * 80)

        mlp_params = {
            "dims": [256, 128, 64],
            "dropout": 0.2,
            "activation": "relu",
        }

        model = DeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "DeepFM"
        assert model.task == "binary"
        logger.info("DeepFM initialization successful")

        # Count parameters
        num_params = count_parameters(model)
        assert num_params > 0

    def test_deepfm_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test DeepFM forward pass"""
        logger.info("=" * 80)
        logger.info("Testing DeepFM forward pass")
        logger.info("=" * 80)

        mlp_params = {
            "dims": [128, 64],
            "dropout": 0.0,
            "activation": "relu",
        }

        model = DeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )

        # Move data to device
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}

        # Forward pass
        output = run_model_inference(model, data)

        # Assertions
        assert_model_output_shape(output, (batch_size,), "DeepFM output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "DeepFM output")

        logger.info("DeepFM forward pass successful")

    def test_deepfm_training_step(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test DeepFM training step (forward + backward)"""
        logger.info("=" * 80)
        logger.info("Testing DeepFM training step")
        logger.info("=" * 80)

        mlp_params = {
            "dims": [64, 32],
            "dropout": 0.1,
            "activation": "relu",
        }

        model = DeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            optimizer="adam",
            optimizer_params={"lr": 0.001},
            loss="bce",
            device=device,
        )

        # Move data to device
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        target = sample_batch_data["label"].to(device)

        # Training step
        loss_fn = nn.BCELoss()
        result = run_model_forward_backward(model, data, target, loss_fn)

        assert result["loss"] > 0
        logger.info(f"Training loss: {result['loss']:.4f}")
        logger.info("DeepFM training step successful")

    @pytest.mark.parametrize("mlp_depth", [1, 2, 3])
    def test_deepfm_with_different_depths(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
        mlp_depth,
    ):
        """Test DeepFM with different MLP depths"""
        logger.info("=" * 80)
        logger.info(f"Testing DeepFM with MLP depth={mlp_depth}")
        logger.info("=" * 80)

        dims = [128] * mlp_depth
        mlp_params = {
            "dims": dims,
            "dropout": 0.0,
            "activation": "relu",
        }

        model = DeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )

        # Create sample data
        data = {}
        for feat in sample_dense_features:
            data[feat.name] = torch.randn(batch_size, 1).to(device)
        for feat in sample_sparse_features:
            data[feat.name] = torch.randint(1, feat.vocab_size, (batch_size,)).to(
                device
            )
        for feat in sample_sequence_features:
            data[feat.name] = torch.randint(
                0, feat.vocab_size, (batch_size, feat.max_len)
            ).to(device)

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info(f"DeepFM with depth={mlp_depth} test successful")


class TestDIN:
    """Test suite for DIN (Deep Interest Network)"""

    @pytest.fixture
    def din_features(self):
        """Create features for DIN model"""
        logger.info("Creating features for DIN")

        dense_features = [
            DenseFeature(name="price", embedding_dim=1),
            DenseFeature(name="age", embedding_dim=1),
        ]

        sparse_features = [
            SparseFeature(name="user_id", vocab_size=1000, embedding_dim=16),
            SparseFeature(name="gender", vocab_size=3, embedding_dim=4),
            SparseFeature(
                name="candidate_item", vocab_size=5000, embedding_dim=32
            ),  # Last one is candidate
        ]

        sequence_features = [
            SequenceFeature(
                name="behavior_sequence",
                vocab_size=5000,
                max_len=30,
                embedding_dim=32,
                padding_idx=0,
            )
        ]

        return dense_features, sparse_features, sequence_features

    def test_din_initialization(self, din_features, device):
        """Test DIN model initialization"""
        logger.info("=" * 80)
        logger.info("Testing DIN initialization")
        logger.info("=" * 80)

        dense_features, sparse_features, sequence_features = din_features

        mlp_params = {
            "dims": [256, 128, 64],
            "dropout": 0.2,
            "activation": "relu",
        }

        model = DIN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            behavior_feature_name="behavior_sequence",
            candidate_feature_name="candidate_item",
            mlp_params=mlp_params,
            attention_hidden_units=[80, 40],
            attention_activation="sigmoid",
            target=["label"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "DIN"
        assert model.task == "binary"
        logger.info("DIN initialization successful")

        # Count parameters
        num_params = count_parameters(model)
        assert num_params > 0

    def test_din_forward_pass(self, din_features, device, batch_size, set_random_seed):
        """Test DIN forward pass with attention mechanism"""
        logger.info("=" * 80)
        logger.info("Testing DIN forward pass")
        logger.info("=" * 80)

        dense_features, sparse_features, sequence_features = din_features

        mlp_params = {
            "dims": [128, 64],
            "dropout": 0.0,
            "activation": "relu",
        }

        model = DIN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            behavior_feature_name="behavior_sequence",
            candidate_feature_name="candidate_item",
            mlp_params=mlp_params,
            attention_hidden_units=[64, 32],
            attention_activation="sigmoid",
            target=["label"],
            device=device,
        )

        # Create sample data
        data = {
            "price": torch.randn(batch_size, 1).to(device),
            "age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "gender": torch.randint(1, 3, (batch_size,)).to(device),
            "candidate_item": torch.randint(1, 5000, (batch_size,)).to(device),
            "behavior_sequence": torch.randint(0, 5000, (batch_size, 30)).to(device),
        }

        # Forward pass
        output = run_model_inference(model, data)

        # Assertions
        assert_model_output_shape(output, (batch_size,), "DIN output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "DIN output")

        logger.info("DIN forward pass successful")

    def test_din_attention_weights(self, din_features, device, batch_size):
        """Test that DIN attention mechanism produces valid weights"""
        logger.info("=" * 80)
        logger.info("Testing DIN attention mechanism")
        logger.info("=" * 80)

        dense_features, sparse_features, sequence_features = din_features

        mlp_params = {
            "dims": [64],
            "dropout": 0.0,
            "activation": "relu",
        }

        model = DIN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            behavior_feature_name="behavior_sequence",
            candidate_feature_name="candidate_item",
            mlp_params=mlp_params,
            attention_hidden_units=[32],
            attention_activation="sigmoid",
            attention_use_softmax=True,
            target=["label"],
            device=device,
        )

        # Create sample data with some padding
        data = {
            "price": torch.randn(batch_size, 1).to(device),
            "age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "gender": torch.randint(1, 3, (batch_size,)).to(device),
            "candidate_item": torch.randint(1, 5000, (batch_size,)).to(device),
            "behavior_sequence": torch.randint(0, 5000, (batch_size, 30)).to(device),
        }

        # Set some positions to padding
        data["behavior_sequence"][:, 20:] = 0

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,))
        assert_no_nan_or_inf(output, "DIN output with padding")

        logger.info("DIN attention mechanism test successful")

    @pytest.mark.parametrize("attention_activation", ["sigmoid", "relu", "tanh"])
    def test_din_attention_activations(
        self, din_features, device, batch_size, attention_activation
    ):
        """Test DIN with different attention activation functions"""
        logger.info("=" * 80)
        logger.info(f"Testing DIN with {attention_activation} attention activation")
        logger.info("=" * 80)

        dense_features, sparse_features, sequence_features = din_features

        mlp_params = {"dims": [64], "dropout": 0.0, "activation": "relu"}

        model = DIN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            behavior_feature_name="behavior_sequence",
            candidate_feature_name="candidate_item",
            mlp_params=mlp_params,
            attention_hidden_units=[32],
            attention_activation=attention_activation,
            target=["label"],
            device=device,
        )

        data = {
            "price": torch.randn(batch_size, 1).to(device),
            "age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "gender": torch.randint(1, 3, (batch_size,)).to(device),
            "candidate_item": torch.randint(1, 5000, (batch_size,)).to(device),
            "behavior_sequence": torch.randint(0, 5000, (batch_size, 30)).to(device),
        }

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info(f"DIN with {attention_activation} activation test successful")


class TestRankingModelsComparison:
    """Comparison and integration tests for ranking models"""

    def test_deepfm_vs_din_output_consistency(self, device, batch_size):
        """Test that different models produce consistent output formats"""
        logger.info("=" * 80)
        logger.info("Testing ranking models output consistency")
        logger.info("=" * 80)

        # Simple features for both models
        dense_features = [DenseFeature(name="feat1", embedding_dim=1)]
        sparse_features = [
            SparseFeature(name="sparse1", vocab_size=100, embedding_dim=8),
            SparseFeature(name="sparse2", vocab_size=100, embedding_dim=8),
        ]
        sequence_features = [
            SequenceFeature(
                name="seq1", vocab_size=100, max_len=10, embedding_dim=8, padding_idx=0
            )
        ]

        # Data
        data = {
            "feat1": torch.randn(batch_size, 1).to(device),
            "sparse1": torch.randint(1, 100, (batch_size,)).to(device),
            "sparse2": torch.randint(1, 100, (batch_size,)).to(device),
            "seq1": torch.randint(0, 100, (batch_size, 10)).to(device),
        }

        # Test DeepFM
        deepfm = DeepFM(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            mlp_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            target=["label"],
            device=device,
        )

        deepfm_output = run_model_inference(deepfm, data)
        assert_model_output_shape(deepfm_output, (batch_size,))
        assert_model_output_range(deepfm_output, 0.0, 1.0)

        # Test DIN
        din = DIN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            behavior_feature_name="seq1",
            candidate_feature_name="sparse1",
            mlp_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            attention_hidden_units=[16],
            target=["label"],
            device=device,
        )

        din_output = run_model_inference(din, data)
        assert_model_output_shape(din_output, (batch_size,))
        assert_model_output_range(din_output, 0.0, 1.0)

        logger.info("Ranking models output consistency test successful")

    def test_model_with_empty_features(self, device, batch_size):
        """Test models with minimal feature configurations"""
        logger.info("=" * 80)
        logger.info("Testing models with minimal features")
        logger.info("=" * 80)

        # Only sparse features
        sparse_features = [
            SparseFeature(name="feat1", vocab_size=100, embedding_dim=8),
            SparseFeature(name="feat2", vocab_size=50, embedding_dim=8),
        ]

        model = DeepFM(
            dense_features=[],
            sparse_features=sparse_features,
            sequence_features=[],
            mlp_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            target=["label"],
            device=device,
        )

        data = {
            "feat1": torch.randint(1, 100, (batch_size,)).to(device),
            "feat2": torch.randint(1, 50, (batch_size,)).to(device),
        }

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info("Minimal features test successful")

    def test_models_save_and_load(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
        tmp_path,
    ):
        """Test saving and loading model state"""
        logger.info("=" * 80)
        logger.info("Testing model save and load")
        logger.info("=" * 80)

        mlp_params = {"dims": [32], "dropout": 0.0, "activation": "relu"}

        # Create and train model
        model1 = DeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )

        # Create test data
        data = {}
        for feat in sample_dense_features:
            data[feat.name] = torch.randn(batch_size, 1).to(device)
        for feat in sample_sparse_features:
            data[feat.name] = torch.randint(1, feat.vocab_size, (batch_size,)).to(
                device
            )
        for feat in sample_sequence_features:
            data[feat.name] = torch.randint(
                0, feat.vocab_size, (batch_size, feat.max_len)
            ).to(device)

        # Get output from original model
        output1 = run_model_inference(model1, data)

        # Save model
        save_path = tmp_path / "model.pth"
        torch.save(model1.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

        # Create new model and load weights
        model2 = DeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )

        model2.load_state_dict(torch.load(save_path, weights_only=True))
        logger.info("Model loaded from checkpoint")

        # Get output from loaded model
        output2 = run_model_inference(model2, data)

        # Outputs should be identical
        assert torch.allclose(
            output1, output2, atol=1e-6
        ), "Loaded model should produce identical outputs"

        logger.info("Model save and load test successful")


class TestAutoInt:
    """Test suite for AutoInt (Automatic Feature Interaction)"""

    def test_autoint_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test AutoInt model initialization"""
        logger.info("=" * 80)
        logger.info("Testing AutoInt initialization")
        logger.info("=" * 80)

        model = AutoInt(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            att_layer_num=3,
            att_embedding_dim=8,
            att_head_num=2,
            att_dropout=0.1,
            target=["label"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "AutoInt"
        assert model.att_layer_num == 3
        logger.info("AutoInt initialization successful")

        count_parameters(model)

    def test_autoint_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test AutoInt forward pass with self-attention"""
        logger.info("=" * 80)
        logger.info("Testing AutoInt forward pass")
        logger.info("=" * 80)

        model = AutoInt(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            att_layer_num=2,
            att_embedding_dim=16,
            att_head_num=2,
            att_dropout=0.0,
            target=["label"],
            device=device,
        )

        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,), "AutoInt output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "AutoInt output")

        logger.info("AutoInt forward pass successful")

    @pytest.mark.parametrize("att_head_num", [1, 2, 4])
    def test_autoint_different_heads(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
        att_head_num,
    ):
        """Test AutoInt with different numbers of attention heads"""
        logger.info("=" * 80)
        logger.info(f"Testing AutoInt with {att_head_num} attention heads")
        logger.info("=" * 80)

        model = AutoInt(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            att_layer_num=2,
            att_embedding_dim=16,
            att_head_num=att_head_num,
            target=["label"],
            device=device,
        )

        data = {}
        for feat in sample_dense_features:
            data[feat.name] = torch.randn(batch_size, 1).to(device)
        for feat in sample_sparse_features:
            data[feat.name] = torch.randint(1, feat.vocab_size, (batch_size,)).to(
                device
            )
        for feat in sample_sequence_features:
            data[feat.name] = torch.randint(
                0, feat.vocab_size, (batch_size, feat.max_len)
            ).to(device)

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info(f"AutoInt with {att_head_num} heads test successful")


class TestWideDeep:
    """Test suite for Wide&Deep"""

    def test_widedeep_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test Wide&Deep model initialization"""
        logger.info("=" * 80)
        logger.info("Testing Wide&Deep initialization")
        logger.info("=" * 80)

        mlp_params = {
            "dims": [256, 128, 64],
            "dropout": 0.2,
            "activation": "relu",
        }

        model = WideDeep(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "WideDeep"
        logger.info("Wide&Deep initialization successful")

        count_parameters(model)

    def test_widedeep_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test Wide&Deep forward pass"""
        logger.info("=" * 80)
        logger.info("Testing Wide&Deep forward pass")
        logger.info("=" * 80)

        mlp_params = {
            "dims": [128, 64],
            "dropout": 0.0,
            "activation": "relu",
        }

        model = WideDeep(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )

        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,), "Wide&Deep output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "Wide&Deep output")

        logger.info("Wide&Deep forward pass successful")


class TestDCN:
    """Test suite for DCN (Deep & Cross Network)"""

    def test_dcn_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test DCN model initialization"""
        logger.info("=" * 80)
        logger.info("Testing DCN initialization")
        logger.info("=" * 80)

        model = DCN(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            cross_num=3,
            mlp_params={"dims": [128, 64], "dropout": 0.2, "activation": "relu"},
            target=["label"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "DCN"
        logger.info("DCN initialization successful")

        count_parameters(model)

    def test_dcn_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test DCN forward pass with cross network"""
        logger.info("=" * 80)
        logger.info("Testing DCN forward pass")
        logger.info("=" * 80)

        model = DCN(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            cross_num=2,
            mlp_params={"dims": [64, 32], "dropout": 0.0, "activation": "relu"},
            target=["label"],
            device=device,
        )

        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,), "DCN output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "DCN output")

        logger.info("DCN forward pass successful")

    def test_dcn_without_dnn(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
    ):
        """Test DCN without DNN part (cross network only)"""
        logger.info("=" * 80)
        logger.info("Testing DCN without DNN")
        logger.info("=" * 80)

        model = DCN(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            cross_num=3,
            mlp_params=None,  # No DNN
            target=["label"],
            device=device,
        )

        data = {}
        for feat in sample_dense_features:
            data[feat.name] = torch.randn(batch_size, 1).to(device)
        for feat in sample_sparse_features:
            data[feat.name] = torch.randint(1, feat.vocab_size, (batch_size,)).to(
                device
            )
        for feat in sample_sequence_features:
            data[feat.name] = torch.randint(
                0, feat.vocab_size, (batch_size, feat.max_len)
            ).to(device)

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info("DCN without DNN test successful")

    @pytest.mark.parametrize("cross_num", [1, 2, 3, 5])
    def test_dcn_different_cross_layers(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
        cross_num,
    ):
        """Test DCN with different numbers of cross layers"""
        logger.info("=" * 80)
        logger.info(f"Testing DCN with {cross_num} cross layers")
        logger.info("=" * 80)

        model = DCN(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            cross_num=cross_num,
            target=["label"],
            device=device,
        )

        data = {}
        for feat in sample_dense_features:
            data[feat.name] = torch.randn(batch_size, 1).to(device)
        for feat in sample_sparse_features:
            data[feat.name] = torch.randint(1, feat.vocab_size, (batch_size,)).to(
                device
            )
        for feat in sample_sequence_features:
            data[feat.name] = torch.randint(
                0, feat.vocab_size, (batch_size, feat.max_len)
            ).to(device)

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info(f"DCN with {cross_num} cross layers test successful")


class TestxDeepFM:
    """Test suite for xDeepFM (Extreme Deep Factorization Machine)"""

    def test_xdeepfm_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test xDeepFM model initialization"""
        logger.info("=" * 80)
        logger.info("Testing xDeepFM initialization")
        logger.info("=" * 80)

        model = xDeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            cin_size=[128, 128],
            mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
            target=["label"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "xDeepFM"
        logger.info("xDeepFM initialization successful")

        count_parameters(model)

    def test_xdeepfm_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test xDeepFM forward pass with CIN"""
        logger.info("=" * 80)
        logger.info("Testing xDeepFM forward pass")
        logger.info("=" * 80)

        model = xDeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            cin_size=[64, 64],
            mlp_params={"dims": [128, 64], "dropout": 0.0, "activation": "relu"},
            target=["label"],
            device=device,
        )

        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,), "xDeepFM output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "xDeepFM output")

        logger.info("xDeepFM forward pass successful")

    @pytest.mark.parametrize("cin_layers", [[64], [64, 64], [128, 64, 32]])
    def test_xdeepfm_different_cin_configs(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
        cin_layers,
    ):
        """Test xDeepFM with different CIN configurations"""
        logger.info("=" * 80)
        logger.info(f"Testing xDeepFM with CIN layers: {cin_layers}")
        logger.info("=" * 80)

        model = xDeepFM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            cin_size=cin_layers,
            mlp_params={"dims": [64], "dropout": 0.0, "activation": "relu"},
            target=["label"],
            device=device,
        )

        data = {}
        for feat in sample_dense_features:
            data[feat.name] = torch.randn(batch_size, 1).to(device)
        for feat in sample_sparse_features:
            data[feat.name] = torch.randint(1, feat.vocab_size, (batch_size,)).to(
                device
            )
        for feat in sample_sequence_features:
            data[feat.name] = torch.randint(
                0, feat.vocab_size, (batch_size, feat.max_len)
            ).to(device)

        output = run_model_inference(model, data)
        assert_model_output_shape(output, (batch_size,))

        logger.info(f"xDeepFM with CIN {cin_layers} test successful")


class TestDIEN:
    """Test suite for DIEN (Deep Interest Evolution Network)"""

    @pytest.fixture
    def dien_features(self):
        """Create features for DIEN model"""
        logger.info("Creating features for DIEN")

        dense_features = [
            DenseFeature(name="price", embedding_dim=1),
            DenseFeature(name="age", embedding_dim=1),
        ]

        sparse_features = [
            SparseFeature(name="user_id", vocab_size=1000, embedding_dim=16),
            SparseFeature(name="gender", vocab_size=3, embedding_dim=4),
            SparseFeature(name="candidate_item", vocab_size=5000, embedding_dim=32),
        ]

        sequence_features = [
            SequenceFeature(
                name="behavior_sequence",
                vocab_size=5000,
                max_len=30,
                embedding_dim=32,
                padding_idx=0,
            )
        ]

        return dense_features, sparse_features, sequence_features

    def test_dien_initialization(self, dien_features, device):
        """Test DIEN model initialization"""
        logger.info("=" * 80)
        logger.info("Testing DIEN initialization")
        logger.info("=" * 80)

        dense_features, sparse_features, sequence_features = dien_features

        model = DIEN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            behavior_feature_name="behavior_sequence",
            candidate_feature_name="candidate_item",
            gru_hidden_size=32,
            attention_hidden_units=[80, 40],
            mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
            target=["label"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "DIEN"
        logger.info("DIEN initialization successful")

        count_parameters(model)

    def test_dien_forward_pass(
        self, dien_features, device, batch_size, set_random_seed
    ):
        """Test DIEN forward pass with GRU and attention"""
        logger.info("=" * 80)
        logger.info("Testing DIEN forward pass")
        logger.info("=" * 80)

        dense_features, sparse_features, sequence_features = dien_features

        model = DIEN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            behavior_feature_name="behavior_sequence",
            candidate_feature_name="candidate_item",
            gru_hidden_size=32,
            attention_hidden_units=[64, 32],
            mlp_params={"dims": [128, 64], "dropout": 0.0, "activation": "relu"},
            target=["label"],
            device=device,
        )

        data = {
            "price": torch.randn(batch_size, 1).to(device),
            "age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "gender": torch.randint(1, 3, (batch_size,)).to(device),
            "candidate_item": torch.randint(1, 5000, (batch_size,)).to(device),
            "behavior_sequence": torch.randint(0, 5000, (batch_size, 30)).to(device),
        }

        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,), "DIEN output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "DIEN output")

        logger.info("DIEN forward pass successful")

    def test_dien_neg_sampling_aux_loss(
        self, dien_features, device, batch_size, set_random_seed
    ):
        """Test DIEN negative sampling path with auxiliary loss"""
        dense_features, sparse_features, base_sequence_features = dien_features

        # Add negative behavior sequence for auxiliary loss
        neg_behavior_feature = SequenceFeature(
            name="neg_behavior_sequence",
            vocab_size=5000,
            max_len=30,
            embedding_dim=32,
            padding_idx=0,
        )
        sequence_features = base_sequence_features + [neg_behavior_feature]

        model = DIEN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            behavior_feature_name="behavior_sequence",
            candidate_feature_name="candidate_item",
            neg_behavior_feature_name="neg_behavior_sequence",
            use_negsampling=True,
            gru_hidden_size=32,
            attention_hidden_units=[64, 32],
            mlp_params={"dims": [64], "dropout": 0.0, "activation": "relu"},
            target=["label"],
            device=device,
        )

        data = {
            "price": torch.randn(batch_size, 1).to(device),
            "age": torch.randn(batch_size, 1).to(device),
            "user_id": torch.randint(1, 1000, (batch_size,)).to(device),
            "gender": torch.randint(1, 3, (batch_size,)).to(device),
            "candidate_item": torch.randint(1, 5000, (batch_size,)).to(device),
            "behavior_sequence": torch.randint(1, 5000, (batch_size, 30)).to(
                device
            ),  # avoid all padding
            "neg_behavior_sequence": torch.randint(1, 5000, (batch_size, 30)).to(
                device
            ),
        }

        model.train()
        output = model(data)
        assert_model_output_shape(output, (batch_size,), "DIEN neg-sampling output")

        aux_loss = model.compute_auxiliary_loss()
        assert torch.isfinite(aux_loss).all(), "Auxiliary loss contains NaN/Inf"
        assert aux_loss.item() >= 0.0
        logger.info("DIEN negative sampling auxiliary loss path successful")


class TestFiBiNET:
    """Test suite for FiBiNET"""

    def test_fibinet_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        mlp_params = {"dims": [128, 64], "dropout": 0.0, "activation": "relu"}
        model = FiBiNET(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )
        assert model.model_name == "FiBiNET"

    def test_fibinet_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
    ):
        mlp_params = {"dims": [64, 32], "dropout": 0.0, "activation": "relu"}
        model = FiBiNET(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            target=["label"],
            device=device,
        )
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,))
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "FiBiNET output")


class TestAFM:
    """Test suite for AFM"""

    def test_afm_initialization(
        self, sample_sparse_features, sample_sequence_features, device
    ):
        model = AFM(
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            attention_dim=16,
            target=["label"],
            device=device,
        )
        assert model.model_name == "AFM"

    def test_afm_forward_pass(
        self,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
    ):
        model = AFM(
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            attention_dim=8,
            target=["label"],
            device=device,
        )
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,))
        assert_model_output_range(output, 0.0, 1.0)


class TestFFM:
    """Test suite for FFM"""

    def test_ffm_initialization(
        self, sample_sparse_features, sample_sequence_features, device
    ):
        model = FFM(
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            target=["label"],
            device=device,
        )
        assert model.model_name == "FFM"

    def test_ffm_forward_pass(
        self,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
    ):
        model = FFM(
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            target=["label"],
            device=device,
        )
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,))
        assert_model_output_range(output, 0.0, 1.0)


class TestPNN:
    """Test suite for PNN"""

    @pytest.mark.parametrize("product_type,outer_dim", [("inner", None), ("outer", 8)])
    def test_pnn_forward_pass(
        self,
        product_type,
        outer_dim,
        sample_sparse_features,
        sample_sequence_features,
        sample_batch_data,
        device,
        batch_size,
    ):
        mlp_params = {"dims": [64], "dropout": 0.0, "activation": "relu"}
        model = PNN(
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            mlp_params=mlp_params,
            product_type=product_type,
            outer_product_dim=outer_dim,
            target=["label"],
            device=device,
        )
        data = {k: v.to(device) for k, v in sample_batch_data.items() if k != "label"}
        output = run_model_inference(model, data)

        assert_model_output_shape(output, (batch_size,))
        assert_model_output_range(output, 0.0, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
