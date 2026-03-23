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

TrainingModeName = Literal["pointwise", "pairwise", "listwise"]

TaskTypeName = Literal["binary", "regression", "generative"]

TaskTypeInput = TaskTypeName | str

EmbeddingInitType = Literal[
    "normal",
    "uniform",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
]

SequenceCombinerType = Literal[
    "mean",
    "sum",
    "concat",
    "dot_attention",
    "self_attention",
]

InitializerActivationType = Literal[
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
]

MetricsName = Literal[
    "auc",
    "gauc",
    "ks",
    "logloss",
    "accuracy",
    "acc",
    "precision",
    "recall",
    "f1",
    "micro_f1",
    "macro_f1",
    "mse",
    "mae",
    "rmse",
    "r2",
    "mape",
    "msle",
    "auc",
    "gauc",
    "precision@10",
    "hitrate@10",
    "map@10",
    "cosine",
    "recall@5",
    "recall@10",
    "recall@20",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "mrr@5",
    "mrr@10",
    "mrr@20",
    "topk_recall@5",
    "topk_recall@10",
    "topk_recall@20",
    "topk_precision@5",
    "topk_precision@10",
    "topk_precision@20",
    "lift@5",
    "lift@10",
    "lift@20",
]
