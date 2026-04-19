import torch
import pytest

from nextrec.utils.torch_utils import (
    get_device,
    get_gain,
    get_nonlinearity,
    smart_inference_mode,
    to_float,
    to_list,
    to_numpy,
    to_tensor,
)


def test_smart_inference_mode_enables_inference_context():
    @smart_inference_mode()
    def probe():
        return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

    inference_enabled, grad_enabled = probe()

    assert inference_enabled is True
    assert grad_enabled is False


def test_smart_inference_mode_is_passthrough_when_already_enabled():
    @smart_inference_mode()
    def probe():
        return torch.is_inference_mode_enabled()

    with torch.inference_mode():
        assert probe() is True


def test_smart_inference_mode_preserves_function_metadata():
    @smart_inference_mode()
    def named_probe():
        return None

    assert named_probe.__name__ == "named_probe"


def test_get_device_rejects_multi_gpu_spec():
    with pytest.raises(ValueError, match="Multi-GPU training is no longer supported"):
        get_device("0,1")


def test_to_list_normalizes_scalar_and_none():
    assert to_list(None) == []
    assert to_list("label") == ["label"]
    assert to_list(("a", "b")) == ["a", "b"]


def test_to_float_handles_scalars_and_tensors():
    assert to_float(1) == 1.0
    assert to_float(torch.tensor(2.5)) == 2.5
    assert to_float(object()) is None


def test_to_numpy_and_to_tensor_round_trip():
    tensor = to_tensor([1, 2, 3], dtype=torch.float32)

    assert tensor.dtype == torch.float32
    assert to_numpy(tensor).tolist() == [1.0, 2.0, 3.0]


def test_initializer_helpers_use_consistent_names():
    assert get_nonlinearity("unsupported") == "linear"
    assert get_gain("relu", {}) == pytest.approx(torch.nn.init.calculate_gain("relu"))
    assert get_gain("relu", {"gain": 3.0}) == 3.0
