"""
Initialization utilities for NextRec

Date: create on 13/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from typing import Any, Dict, Set, cast

import torch.nn as nn
from torch.nn.init import _NonlinearityType

KNOWN_NONLINEARITIES: Set[str] = {
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
    "gelu",
}


def resolve_nonlinearity(activation: str | _NonlinearityType) -> _NonlinearityType:
    if isinstance(activation, str):
        if activation in KNOWN_NONLINEARITIES:
            return cast(_NonlinearityType, activation)
        # Fall back to linear for custom activations (gain handled separately).
        return "linear"
    return activation


def resolve_gain(activation: str | _NonlinearityType, param: Dict[str, Any]) -> float:
    if "gain" in param:
        return param["gain"]
    nonlinearity = resolve_nonlinearity(activation)
    try:
        return nn.init.calculate_gain(nonlinearity, param.get("param"))
    except ValueError:
        return 1.0  # custom activation with no gain estimate available


def get_initializer(
    init_type: str = "normal",
    activation: str | _NonlinearityType = "linear",
    param: Dict[str, Any] | None = None,
):
    param = param or {}
    nonlinearity = resolve_nonlinearity(activation)
    gain = resolve_gain(activation, param)

    def initializer_fn(tensor):
        if init_type == "xavier_uniform":
            nn.init.xavier_uniform_(tensor, gain=gain)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(tensor, gain=gain)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(tensor, a=param.get("a", 0), nonlinearity=nonlinearity)
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(tensor, a=param.get("a", 0), nonlinearity=nonlinearity)
        elif init_type == "orthogonal":
            nn.init.orthogonal_(tensor, gain=gain)
        elif init_type == "normal":
            nn.init.normal_(tensor, mean=param.get("mean", 0.0), std=param.get("std", 0.0001))
        elif init_type == "uniform":
            nn.init.uniform_(tensor, a=param.get("a", -0.05), b=param.get("b", 0.05))
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        return tensor

    return initializer_fn
