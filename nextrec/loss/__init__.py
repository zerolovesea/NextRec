from nextrec.loss.listwise import (
    ApproxNDCGLoss,
    InfoNCELoss,
    ListMLELoss,
    ListNetLoss,
    SampledSoftmaxLoss,
)
from nextrec.loss.grad_norm import GradNormLossWeighting
from nextrec.loss.loss_utils import VALID_TASK_TYPES, get_loss_fn, get_loss_kwargs
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss
from nextrec.loss.pointwise import (
    ClassBalancedFocalLoss,
    CosineContrastiveLoss,
    FocalLoss,
    WeightedBCELoss,
)

__all__ = [
    # Pointwise
    "CosineContrastiveLoss",
    "WeightedBCELoss",
    "FocalLoss",
    "ClassBalancedFocalLoss",
    # Pairwise
    "BPRLoss",
    "HingeLoss",
    "TripletLoss",
    # Listwise
    "SampledSoftmaxLoss",
    "InfoNCELoss",
    "ListNetLoss",
    "ListMLELoss",
    "ApproxNDCGLoss",
    # Multi-task weighting
    "GradNormLossWeighting",
    # Utilities
    "get_loss_fn",
    "get_loss_kwargs",
    "VALID_TASK_TYPES",
]
