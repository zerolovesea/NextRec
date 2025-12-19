import pytest
import torch
import torch.nn as nn

from nextrec.basic.model import BaseModel
from nextrec.loss import (
    BPRLoss,
    ClassBalancedFocalLoss,
    FocalLoss,
    HingeLoss,
    ListNetLoss,
    SampledSoftmaxLoss,
    TripletLoss,
    WeightedBCELoss,
    get_loss_fn,
)


def test_focal_loss_binary_prefers_confident_logits():
    labels = torch.tensor([1.0, 0.0])
    neutral_probs = torch.full_like(labels, 0.5)
    strong_probs = torch.tensor([0.9, 0.1])

    loss_fn = FocalLoss(gamma=2.0)
    neutral_loss = loss_fn(neutral_probs, labels)
    strong_loss = loss_fn(strong_probs, labels)

    assert strong_loss.item() < neutral_loss.item()


def test_weighted_bce_adjusts_positive_penalty():
    probs = torch.full((4,), 0.2)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    high_weight = WeightedBCELoss(pos_weight=3.0)
    low_weight = WeightedBCELoss(pos_weight=0.5)

    assert high_weight(probs, labels).item() > low_weight(probs, labels).item()


def test_weighted_bce_auto_balance_matches_manual_ratio():
    probs = torch.full((6,), 0.2)
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    manual_ratio = (labels.numel() - labels.sum()) / labels.sum()

    auto_loss = WeightedBCELoss(pos_weight=None, auto_balance=True)
    manual_loss = WeightedBCELoss(pos_weight=manual_ratio)

    assert torch.allclose(
        auto_loss(probs, labels), manual_loss(probs, labels), atol=1e-6
    )


def test_class_balanced_focal_loss_reduces_meaningfully():
    logits = torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss_fn = ClassBalancedFocalLoss(class_counts=[50, 5], gamma=1.5)
    loss = loss_fn(logits, targets)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None


def test_class_balanced_focal_requires_counts():
    with pytest.raises(ValueError):
        get_loss_fn(task_type="binary", loss="class_balanced_focal")


def test_bpr_loss_supports_multiple_negatives():
    pos = torch.tensor([3.0, 2.0])
    neg = torch.tensor([[1.0, 0.5], [0.0, -1.0]])
    loss = BPRLoss()(pos, neg)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_listnet_loss_runs_backward():
    scores = torch.tensor([[0.2, 0.5, 1.0]], requires_grad=True)
    labels = torch.tensor([[0.0, 1.0, 2.0]])
    loss = ListNetLoss()(scores, labels)
    loss.backward()
    assert scores.grad is not None


def test_sampled_softmax_shapes():
    pos_logits = torch.tensor([1.5, 0.8])
    neg_logits = torch.tensor([[0.2, -0.3], [-0.1, 0.4]])
    loss = SampledSoftmaxLoss()(pos_logits, neg_logits)
    assert loss.item() > 0


def test_get_loss_fn_routes_pairwise():
    loss_fn = get_loss_fn(loss="bpr")
    assert isinstance(loss_fn, BPRLoss)


def test_get_loss_fn_routes_hinge_and_triplet():
    hinge = get_loss_fn(loss="hinge")
    triplet = get_loss_fn(loss="triplet", margin=0.3)

    assert isinstance(hinge, HingeLoss)
    assert isinstance(triplet, TripletLoss)
    assert triplet.margin == 0.3


class _DummyBinaryModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummyBinary"

    @property
    def default_task(self) -> str:
        return "binary"

    def forward(self, X_input):
        return torch.zeros(1, 1)

    def __init__(self):
        super().__init__(
            dense_features=[],
            sparse_features=[],
            sequence_features=[],
            target=["y"],
            task="binary",
        )
        self.dummy = nn.Parameter(torch.zeros(1))


def test_compile_acceptsloss_params():
    model = _DummyBinaryModel()
    model.compile(loss="focal", loss_params={"gamma": 1.5})
    assert isinstance(model.loss_fn[0], FocalLoss)
    assert model.loss_fn[0].gamma == 1.5
