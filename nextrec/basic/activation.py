"""
Activation function definitions.

Date: create on 27/10/2025
Checkpoint: edit on 07/02/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch
import torch.nn as nn


from nextrec.utils.types import ActivationName


class Dice(nn.Module):
    """
    Dice activation function from the paper:
    "Deep Interest Network for Click-Through Rate Prediction" (Zhou et al., 2018)

    Dice(x) = p(x) * x + (1 - p(x)) * alpha * x
    where p(x) = sigmoid((x - E[x]) / sqrt(Var[x] + epsilon))
    """

    def __init__(self, emb_size: int, epsilon: float = 1e-3):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(emb_size))
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon, affine=False)

    def forward(self, x):
        # keep original shape for reshaping back after batch norm
        orig_shape = x.shape  # x: [N, L, emb_size] or [N, emb_size]
        x2 = x.reshape(-1, orig_shape[-1])  # x2:[N*L, emb_size] or [N, emb_size]
        x_norm = self.bn(x2)
        p = torch.sigmoid(x_norm).reshape(orig_shape)
        return x * (self.alpha + (1 - self.alpha) * p)


def activation_layer(
    activation: ActivationName = "none",
    emb_size: int | None = None,
):
    """Create an activation layer based on the given activation name."""
    if activation == "dice":
        if emb_size is None:
            raise ValueError("[ActivationLayer Error]: emb_size is required for Dice activation")
        return Dice(emb_size)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "relu6":
        return nn.ReLU6()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "softsign":
        return nn.Softsign()
    elif activation == "hardswish":
        return nn.Hardswish()
    elif activation == "mish":
        return nn.Mish()
    elif activation in ["silu", "swish"]:
        return nn.SiLU()
    elif activation == "hardsigmoid":
        return nn.Hardsigmoid()
    elif activation == "tanhshrink":
        return nn.Tanhshrink()
    elif activation == "softshrink":
        return nn.Softshrink()
    elif activation in ["none", "linear", "identity"]:
        return nn.Identity()
    else:
        raise ValueError(f"[ActivationLayer Error]: Unsupported activation function: {activation}")
