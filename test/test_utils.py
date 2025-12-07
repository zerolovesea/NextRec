"""
Test Utility Functions

This module provides utility functions for testing models.
"""

import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def assert_model_output_shape(
    output: torch.Tensor, expected_shape: tuple, message: str = ""
):
    """
    Assert that model output has the expected shape

    Args:
        output: Model output tensor
        expected_shape: Expected shape tuple
        message: Optional custom message
    """
    actual_shape = tuple(output.shape)
    normalized_expected = expected_shape
    if (
        len(expected_shape) == 1
        and len(actual_shape) == 2
        and actual_shape[-1] == 1
        and actual_shape[0] == expected_shape[0]
    ):
        normalized_expected = (
            actual_shape  # accept trailing singleton dim for single-output heads
        )
    assert (
        actual_shape == normalized_expected
    ), f"{message}\nExpected shape: {normalized_expected}, but got: {actual_shape}"
    logger.info(f"Output shape assertion passed: {actual_shape}")


def assert_model_output_range(
    output: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0
):
    """
    Assert that model output values are within expected range

    Args:
        output: Model output tensor
        min_val: Minimum expected value
        max_val: Maximum expected value
    """
    assert torch.all(output >= min_val) and torch.all(output <= max_val), (
        f"Output values should be in range [{min_val}, {max_val}], "
        f"but got min={output.min().item():.4f}, max={output.max().item():.4f}"
    )
    logger.info(
        f"Output range assertion passed: [{output.min().item():.4f}, {output.max().item():.4f}]"
    )


def assert_no_nan_or_inf(tensor: torch.Tensor, name: str = "tensor"):
    """
    Assert that tensor contains no NaN or Inf values

    Args:
        tensor: Input tensor
        name: Name of the tensor for error message
    """
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf values"
    logger.info(f"No NaN/Inf assertion passed for {name}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")
    return num_params


def run_model_forward_backward(
    model: torch.nn.Module,
    data: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Test forward and backward pass of a model

    Args:
        model: PyTorch model
        data: Input data dictionary
        targets: Target labels
        loss_fn: Loss function

    Returns:
        Dict: Dictionary containing loss and output
    """
    logger.info("Testing forward pass...")
    model.train()

    # Forward pass
    output = model(data)
    assert_no_nan_or_inf(output, "model_output")
    if isinstance(targets, torch.Tensor):
        if output.dim() == 2 and output.shape[-1] == 1 and targets.dim() == 1:
            targets = targets.view(-1, 1)
        elif output.shape != targets.shape and targets.numel() == output.numel():
            targets = targets.view_as(output)

    # Calculate loss
    logger.info("Testing backward pass...")
    loss = loss_fn(output, targets)
    assert_no_nan_or_inf(loss, "loss")

    # Backward pass
    loss.backward()

    # Check gradients
    logger.info("Checking gradients...")
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for parameter: {name}"
            assert_no_nan_or_inf(param.grad, f"gradient of {name}")
            has_grad = True

    assert has_grad, "No gradients computed"
    logger.info("Forward and backward pass test passed")

    return {"loss": loss.item(), "output": output.detach()}


def run_model_inference(
    model: torch.nn.Module, data: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Test model inference (eval mode)

    Args:
        model: PyTorch model
        data: Input data dictionary

    Returns:
        torch.Tensor: Model output
    """
    logger.info("Testing inference mode...")
    model.eval()

    with torch.no_grad():
        output = model(data)
        assert_no_nan_or_inf(output, "inference_output")

    logger.info("Inference test passed")
    return output


def compare_outputs(
    output1: torch.Tensor, output2: torch.Tensor, tolerance: float = 1e-5
):
    """
    Compare two model outputs

    Args:
        output1: First output tensor
        output2: Second output tensor
        tolerance: Tolerance for comparison
    """
    assert (
        output1.shape == output2.shape
    ), f"Output shapes don't match: {output1.shape} vs {output2.shape}"

    max_diff = torch.max(torch.abs(output1 - output2)).item()
    assert (
        max_diff < tolerance
    ), f"Outputs differ by {max_diff}, tolerance is {tolerance}"

    logger.info(f"Outputs match within tolerance (max_diff={max_diff:.2e})")
