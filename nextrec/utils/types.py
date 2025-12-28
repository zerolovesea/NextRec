"""
Shared type aliases for NextRec.

Keep Literal-based public string options centralized to avoid drift.
"""

from typing import Literal

OptimizerName = Literal["adam", "sgd", "adamw", "adagrad", "rmsprop"]

SchedulerName = Literal["step", "cosine"]

LossName = Literal[
    "bce",
    "binary_crossentropy",
    "weighted_bce",
    "focal",
    "focal_loss",
    "cb_focal",
    "class_balanced_focal",
    "crossentropy",
    "ce",
    "mse",
    "mae",
    "bpr",
    "hinge",
    "triplet",
    "sampled_softmax",
    "softmax",
    "infonce",
    "listnet",
    "listmle",
    "approx_ndcg",
]

ActivationName = Literal[
        "dice",
        "relu",
        "relu6",
        "elu",
        "selu",
        "leaky_relu",
        "prelu",
        "gelu",
        "sigmoid",
        "tanh",
        "softplus",
        "softsign",
        "hardswish",
        "mish",
        "silu",
        "swish",
        "hardsigmoid",
        "tanhshrink",
        "softshrink",
        "none",
        "linear",
        "identity",
    ] 