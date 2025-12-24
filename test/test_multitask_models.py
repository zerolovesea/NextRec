"""
Unit Tests for Multi-Task Models

This module contains unit tests for all multi-task learning models including:
- ShareBottom (Shared-Bottom Multi-Task Learning)
- MMOE (Multi-gate Mixture-of-Experts)
- PLE (Progressive Layered Extraction)
- ESMM (Entire Space Multi-Task Model)

Tests cover model initialization, forward pass, multi-task learning, and task-specific predictions.
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

from nextrec.models.multi_task.esmm import ESMM
from nextrec.models.multi_task.mmoe import MMOE
from nextrec.models.multi_task.ple import PLE
from nextrec.models.multi_task.share_bottom import ShareBottom

logger = logging.getLogger(__name__)


class TestShareBottom:
    """Test suite for ShareBottom (Shared-Bottom Multi-Task Learning)"""

    def test_share_bottom_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test ShareBottom model initialization"""
        logger.info("=" * 80)
        logger.info("Testing ShareBottom initialization")
        logger.info("=" * 80)

        bottom_params = {
            "dims": [256, 128],
            "dropout": 0.2,
            "activation": "relu",
        }

        tower_params_list = [
            {"dims": [64, 32], "dropout": 0.1, "activation": "relu"},
            {"dims": [64, 32], "dropout": 0.1, "activation": "relu"},
        ]

        model = ShareBottom(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            bottom_params=bottom_params,
            tower_params_list=tower_params_list,
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "ShareBottom"
        assert model.nums_task == 2
        logger.info("ShareBottom initialization successful")

        # Count parameters
        num_params = count_parameters(model)
        assert num_params > 0

    def test_share_bottom_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_multitask_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test ShareBottom forward pass"""
        logger.info("=" * 80)
        logger.info("Testing ShareBottom forward pass")
        logger.info("=" * 80)

        bottom_params = {"dims": [128, 64], "dropout": 0.0, "activation": "relu"}
        tower_params_list = [
            {"dims": [32], "dropout": 0.0, "activation": "relu"},
            {"dims": [32], "dropout": 0.0, "activation": "relu"},
        ]

        model = ShareBottom(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            bottom_params=bottom_params,
            tower_params_list=tower_params_list,
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=device,
        )

        # Move data to device (exclude labels)
        data = {
            k: v.to(device)
            for k, v in sample_multitask_batch_data.items()
            if not k.startswith("label")
        }

        # Forward pass
        output = run_model_inference(model, data)

        # Assertions
        assert_model_output_shape(output, (batch_size, 2), "ShareBottom output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "ShareBottom output")

        logger.info("ShareBottom forward pass successful")

    def test_share_bottom_task_outputs(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
    ):
        """Test that ShareBottom produces separate outputs for each task"""
        logger.info("=" * 80)
        logger.info("Testing ShareBottom task outputs")
        logger.info("=" * 80)

        nums_task = 3
        bottom_params = {"dims": [64], "dropout": 0.0, "activation": "relu"}
        tower_params_list = [
            {"dims": [32], "dropout": 0.0, "activation": "relu"}
            for _ in range(nums_task)
        ]

        model = ShareBottom(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            bottom_params=bottom_params,
            tower_params_list=tower_params_list,
            target=["task1", "task2", "task3"],
            task=["binary"] * nums_task,
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

        assert_model_output_shape(output, (batch_size, nums_task))

        # Each task output should be independent
        for i in range(nums_task):
            task_output = output[:, i]
            assert_model_output_range(task_output, 0.0, 1.0)

        logger.info("ShareBottom task outputs test successful")


class TestMMOE:
    """Test suite for MMOE (Multi-gate Mixture-of-Experts)"""

    def test_mmoe_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test MMOE model initialization"""
        logger.info("=" * 80)
        logger.info("Testing MMOE initialization")
        logger.info("=" * 80)

        expert_params = {
            "dims": [256, 128],
            "dropout": 0.2,
            "activation": "relu",
        }

        tower_params_list = [
            {"dims": [64, 32], "dropout": 0.1, "activation": "relu"},
            {"dims": [64, 32], "dropout": 0.1, "activation": "relu"},
        ]

        model = MMOE(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            expert_params=expert_params,
            num_experts=4,
            tower_params_list=tower_params_list,
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "MMOE"
        assert model.nums_task == 2
        assert model.num_experts == 4
        logger.info("MMOE initialization successful")

        # Count parameters
        num_params = count_parameters(model)
        assert num_params > 0

    def test_mmoe_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_multitask_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test MMOE forward pass with expert gating"""
        logger.info("=" * 80)
        logger.info("Testing MMOE forward pass")
        logger.info("=" * 80)

        expert_params = {"dims": [128, 64], "dropout": 0.0, "activation": "relu"}
        tower_params_list = [
            {"dims": [32], "dropout": 0.0, "activation": "relu"},
            {"dims": [32], "dropout": 0.0, "activation": "relu"},
        ]

        model = MMOE(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            expert_params=expert_params,
            num_experts=3,
            tower_params_list=tower_params_list,
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=device,
        )

        # Move data to device
        data = {
            k: v.to(device)
            for k, v in sample_multitask_batch_data.items()
            if not k.startswith("label")
        }

        # Forward pass
        output = run_model_inference(model, data)

        # Assertions
        assert_model_output_shape(output, (batch_size, 2), "MMOE output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "MMOE output")

        logger.info("MMOE forward pass successful")

    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    def test_mmoe_with_different_expert_numbers(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
        num_experts,
    ):
        """Test MMOE with different numbers of experts"""
        logger.info("=" * 80)
        logger.info(f"Testing MMOE with {num_experts} experts")
        logger.info("=" * 80)

        expert_params = {"dims": [64], "dropout": 0.0, "activation": "relu"}
        tower_params_list = [
            {"dims": [32], "dropout": 0.0, "activation": "relu"},
            {"dims": [32], "dropout": 0.0, "activation": "relu"},
        ]

        model = MMOE(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            expert_params=expert_params,
            num_experts=num_experts,
            tower_params_list=tower_params_list,
            target=["task1", "task2"],
            task=["binary", "binary"],
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
        assert_model_output_shape(output, (batch_size, 2))

        logger.info(f"MMOE with {num_experts} experts test successful")


class TestPLE:
    """Test suite for PLE (Progressive Layered Extraction)"""

    def test_ple_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test PLE model initialization"""
        logger.info("=" * 80)
        logger.info("Testing PLE initialization")
        logger.info("=" * 80)

        shared_expert_params = {
            "dims": [128, 64],
            "dropout": 0.2,
            "activation": "relu",
        }

        specific_expert_params = {
            "dims": [128, 64],
            "dropout": 0.2,
            "activation": "relu",
        }

        tower_params_list = [
            {"dims": [32], "dropout": 0.1, "activation": "relu"},
            {"dims": [32], "dropout": 0.1, "activation": "relu"},
        ]

        model = PLE(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            shared_expert_params=shared_expert_params,
            specific_expert_params=specific_expert_params,
            num_shared_experts=2,
            num_specific_experts=2,
            num_levels=2,
            tower_params_list=tower_params_list,
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=device,
        )

        assert model is not None
        assert model.model_name == "PLE"
        assert model.nums_task == 2
        assert model.num_levels == 2
        logger.info("PLE initialization successful")

        # Count parameters
        num_params = count_parameters(model)
        assert num_params > 0

    def test_ple_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_multitask_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test PLE forward pass with progressive extraction"""
        logger.info("=" * 80)
        logger.info("Testing PLE forward pass")
        logger.info("=" * 80)

        shared_expert_params = {"dims": [64], "dropout": 0.0, "activation": "relu"}
        specific_expert_params = {"dims": [64], "dropout": 0.0, "activation": "relu"}
        tower_params_list = [
            {"dims": [32], "dropout": 0.0, "activation": "relu"},
            {"dims": [32], "dropout": 0.0, "activation": "relu"},
        ]

        model = PLE(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            shared_expert_params=shared_expert_params,
            specific_expert_params=specific_expert_params,
            num_shared_experts=2,
            num_specific_experts=1,
            num_levels=2,
            tower_params_list=tower_params_list,
            target=["label_ctr", "label_cvr"],
            task=["binary", "binary"],
            device=device,
        )

        # Move data to device
        data = {
            k: v.to(device)
            for k, v in sample_multitask_batch_data.items()
            if not k.startswith("label")
        }

        # Forward pass
        output = run_model_inference(model, data)

        # Assertions
        assert_model_output_shape(output, (batch_size, 2), "PLE output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "PLE output")

        logger.info("PLE forward pass successful")

    @pytest.mark.parametrize("num_levels", [1, 2, 3])
    def test_ple_with_different_levels(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
        num_levels,
    ):
        """Test PLE with different numbers of extraction levels"""
        logger.info("=" * 80)
        logger.info(f"Testing PLE with {num_levels} levels")
        logger.info("=" * 80)

        shared_expert_params = {"dims": [32], "dropout": 0.0, "activation": "relu"}
        specific_expert_params = {"dims": [32], "dropout": 0.0, "activation": "relu"}
        tower_params_list = [
            {"dims": [16], "dropout": 0.0, "activation": "relu"},
            {"dims": [16], "dropout": 0.0, "activation": "relu"},
        ]

        model = PLE(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            shared_expert_params=shared_expert_params,
            specific_expert_params=specific_expert_params,
            num_shared_experts=2,
            num_specific_experts=1,
            num_levels=num_levels,
            tower_params_list=tower_params_list,
            target=["task1", "task2"],
            task=["binary", "binary"],
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
        assert_model_output_shape(output, (batch_size, 2))

        logger.info(f"PLE with {num_levels} levels test successful")


class TestESMM:
    """Test suite for ESMM (Entire Space Multi-Task Model)"""

    def test_esmm_initialization(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
    ):
        """Test ESMM model initialization"""
        logger.info("=" * 80)
        logger.info("Testing ESMM initialization")
        logger.info("=" * 80)

        ctr_params = {
            "dims": [128, 64],
            "dropout": 0.2,
            "activation": "relu",
        }

        cvr_params = {
            "dims": [128, 64],
            "dropout": 0.2,
            "activation": "relu",
        }

        model = ESMM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            ctr_params=ctr_params,
            cvr_params=cvr_params,
            target=["label_ctr", "label_ctcvr"],  # CTR and CTCVR
            device=device,
        )

        assert model is not None
        assert model.model_name == "ESMM"
        assert len(model.target_columns) == 2
        logger.info("ESMM initialization successful")

        # Count parameters
        num_params = count_parameters(model)
        assert num_params > 0

    def test_esmm_forward_pass(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        sample_multitask_batch_data,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test ESMM forward pass with CTCVR calculation"""
        logger.info("=" * 80)
        logger.info("Testing ESMM forward pass")
        logger.info("=" * 80)

        ctr_params = {"dims": [64, 32], "dropout": 0.0, "activation": "relu"}
        cvr_params = {"dims": [64, 32], "dropout": 0.0, "activation": "relu"}

        model = ESMM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            ctr_params=ctr_params,
            cvr_params=cvr_params,
            target=["label_ctr", "label_ctcvr"],
            device=device,
        )

        # Move data to device
        data = {
            k: v.to(device)
            for k, v in sample_multitask_batch_data.items()
            if not k.startswith("label")
        }

        # Forward pass
        output = run_model_inference(model, data)

        # Assertions
        assert_model_output_shape(output, (batch_size, 2), "ESMM output shape")
        assert_model_output_range(output, 0.0, 1.0)
        assert_no_nan_or_inf(output, "ESMM output")

        # ESMM property: CTCVR = CTR * CVR, so CTCVR <= CTR
        ctr_output = output[:, 0]
        ctcvr_output = output[:, 1]

        # Allow small numerical errors
        assert torch.all(
            ctcvr_output <= ctr_output + 1e-5
        ), "CTCVR should be less than or equal to CTR"

        logger.info("ESMM forward pass successful")

    def test_esmm_ctcvr_constraint(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
        set_random_seed,
    ):
        """Test that ESMM maintains CTCVR = CTR * CVR constraint"""
        logger.info("=" * 80)
        logger.info("Testing ESMM CTCVR constraint")
        logger.info("=" * 80)

        ctr_params = {"dims": [32], "dropout": 0.0, "activation": "relu"}
        cvr_params = {"dims": [32], "dropout": 0.0, "activation": "relu"}

        model = ESMM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            ctr_params=ctr_params,
            cvr_params=cvr_params,
            target=["label_ctr", "label_ctcvr"],
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

        # Get intermediate representations by accessing towers directly
        model.eval()
        with torch.no_grad():
            input_flat = model.embedding(
                x=data, features=model.all_features, squeeze_dim=True
            )
            ctr_logit = model.ctr_tower(input_flat)
            cvr_logit = model.cvr_tower(input_flat)
            ctr = torch.sigmoid(ctr_logit)
            cvr = torch.sigmoid(cvr_logit)
            expected_ctcvr = ctr * cvr

            # Get actual output
            output = model(data)
            actual_ctcvr = output[:, 1:2]

        # Check constraint
        assert torch.allclose(
            actual_ctcvr, expected_ctcvr, atol=1e-5
        ), "CTCVR should equal CTR * CVR"

        logger.info("ESMM CTCVR constraint test successful")


class TestMultiTaskModelsComparison:
    """Comparison tests for multi-task models"""

    def test_models_output_consistency(
        self,
        sample_dense_features,
        sample_sparse_features,
        sample_sequence_features,
        device,
        batch_size,
    ):
        """Test that all multi-task models produce consistent output formats"""
        logger.info("=" * 80)
        logger.info("Testing multi-task models output consistency")
        logger.info("=" * 80)

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

        nums_task = 2

        # Test ShareBottom
        share_bottom = ShareBottom(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            bottom_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            tower_params_list=[
                {"dims": [16], "dropout": 0.0, "activation": "relu"},
                {"dims": [16], "dropout": 0.0, "activation": "relu"},
            ],
            target=["task1", "task2"],
            task=["binary", "binary"],
            device=device,
        )

        sb_output = run_model_inference(share_bottom, data)
        assert_model_output_shape(sb_output, (batch_size, nums_task))

        # Test MMOE
        mmoe = MMOE(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            expert_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            num_experts=2,
            tower_params_list=[
                {"dims": [16], "dropout": 0.0, "activation": "relu"},
                {"dims": [16], "dropout": 0.0, "activation": "relu"},
            ],
            target=["task1", "task2"],
            task=["binary", "binary"],
            device=device,
        )

        mmoe_output = run_model_inference(mmoe, data)
        assert_model_output_shape(mmoe_output, (batch_size, nums_task))

        # Test PLE
        ple = PLE(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            shared_expert_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            specific_expert_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            num_shared_experts=1,
            num_specific_experts=1,
            num_levels=1,
            tower_params_list=[
                {"dims": [16], "dropout": 0.0, "activation": "relu"},
                {"dims": [16], "dropout": 0.0, "activation": "relu"},
            ],
            target=["task1", "task2"],
            task=["binary", "binary"],
            device=device,
        )

        ple_output = run_model_inference(ple, data)
        assert_model_output_shape(ple_output, (batch_size, nums_task))

        # Test ESMM
        esmm = ESMM(
            dense_features=sample_dense_features,
            sparse_features=sample_sparse_features,
            sequence_features=sample_sequence_features,
            ctr_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            cvr_params={"dims": [32], "dropout": 0.0, "activation": "relu"},
            target=["label_ctr", "label_ctcvr"],
            device=device,
        )

        esmm_output = run_model_inference(esmm, data)
        assert_model_output_shape(esmm_output, (batch_size, nums_task))

        logger.info("Multi-task models output consistency test successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
