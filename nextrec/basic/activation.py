"""
Activation function definitions

Date: create on 27/10/2025
Checkpoint: edit on 29/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch
import torch.nn as nn


class Dice(nn.Module):
    """
    Dice activation function from the paper:
    "Deep Interest Network for Click-Through Rate Prediction" (Zhou et al., 2018)

    Dice(x) = p(x) * x + (1 - p(x)) * alpha * x
    where p(x) = sigmoid((x - E[x]) / sqrt(Var[x] + epsilon))
    """

    def __init__(self, emb_size: int, epsilon: float = 1e-9):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.zeros(emb_size))
        self.bn = nn.BatchNorm1d(emb_size)

    def forward(self, x):
        # x shape: (batch_size, emb_size) or (batch_size, seq_len, emb_size)
        original_shape = x.shape

        if x.dim() == 3:
            # For 3D input (batch_size, seq_len, emb_size), reshape to 2D
            batch_size, seq_len, emb_size = x.shape
            x = x.view(-1, emb_size)
        x_norm = self.bn(x)
        p = torch.sigmoid(x_norm)
        output = p * x + (1 - p) * self.alpha * x
        if len(original_shape) == 3:
            output = output.view(original_shape)
        return output


def activation_layer(activation: str, emb_size: int | None = None):
    """Create an activation layer based on the given activation name."""
    activation = activation.lower()
    if activation == "dice":
        if emb_size is None:
            raise ValueError(
                "[ActivationLayer Error]: emb_size is required for Dice activation"
            )
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
        raise ValueError(
            f"[ActivationLayer Error]: Unsupported activation function: {activation}"
        )
