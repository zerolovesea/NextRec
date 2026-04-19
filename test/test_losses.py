import pytest
import torch
import torch.nn as nn

from nextrec.engine.model import Model as BaseModel
from nextrec.loss.listwise import ListNetLoss, SampledSoftmaxLoss
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss
from nextrec.loss.pointwise import (
    ClassBalancedFocalLoss,
    FocalLoss,
    WeightedBCELoss,
)
from nextrec.utils.loss import get_loss_fn, get_loss_list, scale_task_loss
from nextrec.utils.model import compute_ranking_loss


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

    assert torch.allclose(auto_loss(probs, labels), manual_loss(probs, labels), atol=1e-6)


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


def test_scale_task_loss_uses_valid_ratio():
    task_loss = torch.tensor(4.0)
    scaled = scale_task_loss(task_loss, valid_count=3, total_count=6)
    assert torch.allclose(scaled, torch.tensor(2.0))


def test_get_loss_fn_routes_pairwise():
    loss_fn = get_loss_fn(loss="bpr")
    assert isinstance(loss_fn, BPRLoss)


def test_get_loss_fn_routes_hinge_and_triplet():
    hinge = get_loss_fn(loss="hinge")
    triplet = get_loss_fn(loss="triplet", margin=0.3)

    assert isinstance(hinge, HingeLoss)
    assert isinstance(triplet, TripletLoss)
    assert triplet.margin == 0.3


def test_compile_requires_explicit_loss():
    model = _DummyBinaryModel()
    with pytest.raises(ValueError, match="provided explicitly"):
        model.compile(loss=None)


class _DummyBinaryModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummyBinary"

    @property
    def model_family(self) -> str:
        return "ranking"

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


class _DummyMultiTaskModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummyMultiTask"

    @property
    def model_family(self) -> str:
        return "multitask"

    @property
    def default_task(self) -> list[str]:
        return ["binary", "binary"]

    def forward(self, X_input):
        return torch.zeros(1, 2)

    def __init__(self):
        super().__init__(
            dense_features=[],
            sparse_features=[],
            sequence_features=[],
            target=["y1", "y2"],
            task=["binary", "binary"],
        )
        self.dummy = nn.Parameter(torch.zeros(1))


class _DummyPairwiseModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummyPairwise"

    @property
    def model_family(self) -> str:
        return "ranking"

    @property
    def default_task(self) -> str:
        return "binary"

    def forward(self, X_input):
        return torch.zeros(1, 2)

    def __init__(self):
        super().__init__(
            dense_features=[],
            sparse_features=[],
            sequence_features=[],
            target=["y"],
            task="binary",
            training_mode="pairwise",
        )
        self.dummy = nn.Parameter(torch.zeros(1))


class _DummyGenerativeModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummyGenerative"

    @property
    def model_family(self) -> str:
        return "ranking"

    @property
    def default_task(self) -> str:
        return "generative"

    def forward(self, X_input):
        return torch.zeros(1, 4)

    def __init__(self):
        self.vocab_size = 4
        super().__init__(
            dense_features=[],
            sparse_features=[],
            sequence_features=[],
            target=["y"],
            task="generative",
        )
        self.dummy = nn.Parameter(torch.zeros(1))


def test_compile_acceptsloss_params():
    model = _DummyBinaryModel()
    model.compile(loss="focal", loss_params={"gamma": 1.5})
    assert isinstance(model.loss_fn[0], FocalLoss)
    assert model.loss_fn[0].gamma == 1.5


def test_multitask_loss_ignores_negative_labels():
    model = _DummyMultiTaskModel()
    model.compile(loss="bce")

    y_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.4]])
    y_true = torch.tensor([[1.0, -1.0], [0.0, 1.0], [-1.0, -1.0]])

    loss = model.compute_loss(y_pred, y_true)

    task1_mask = y_true[:, 0] != -1
    task2_mask = y_true[:, 1] != -1
    task1_loss = model.loss_fn[0](y_pred[task1_mask, 0:1], y_true[task1_mask, 0:1])
    task2_loss = model.loss_fn[1](y_pred[task2_mask, 1:2], y_true[task2_mask, 1:2])
    expected = task1_loss + task2_loss
    assert torch.allclose(loss, expected)


def test_multitask_loss_rejects_fully_missing_task():
    model = _DummyMultiTaskModel()
    model.compile(loss="bce")

    y_pred = torch.tensor([[0.6, 0.3], [0.2, 0.9]])
    y_true = torch.tensor([[1.0, -1.0], [0.0, -1.0]])

    with pytest.raises(ValueError, match="Task 1 has no valid labels after ignore_label filtering"):
        model.compute_loss(y_pred, y_true)


def test_single_task_pairwise_loss_uses_full_candidate_list():
    model = _DummyPairwiseModel()
    model.compile(loss="bpr")

    y_pred = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.3],
            [0.2, 0.4, 0.8, 0.1],
        ]
    )
    y_true = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    loss = model.compute_loss(y_pred, y_true)
    expected = compute_ranking_loss(
        training_mode="pairwise",
        loss_fn=model.loss_fn[0],
        y_pred=y_pred,
        y_true=y_true,
    )

    assert torch.allclose(loss, expected)


def test_generative_matching_head_accepts_vocab_logits_in_adapt_output():
    model = _DummyGenerativeModel()
    logits = torch.tensor([[2.0, 0.5, -1.0, 1.2]])

    output = model.training_adapter.adapt_output(model, logits)

    assert torch.equal(output, logits)


def test_single_task_generative_cross_entropy_uses_class_ids():
    model = _DummyGenerativeModel()
    model.compile(loss="ce")

    y_pred = torch.tensor(
        [
            [2.1, 0.3, -1.2, 4.0],
            [0.1, 3.2, 1.1, 0.4],
        ]
    )
    y_true = torch.tensor([3, 1])

    loss = model.compute_loss(y_pred, y_true)
    expected = nn.CrossEntropyLoss()(y_pred, y_true.long())

    assert torch.allclose(loss, expected)


def test_get_loss_list_requires_explicit_loss():
    with pytest.raises(ValueError, match="provided explicitly"):
        get_loss_list(
            loss=None,
            training_mode="pointwise",
            num_tasks=3,
        )


def test_get_loss_list_rejects_bce_for_ranking_modes():
    with pytest.raises(ValueError, match="not valid for training_mode='pairwise'"):
        get_loss_list(
            loss="bce",
            training_mode="pairwise",
            num_tasks=2,
        )


def test_get_loss_list_rejects_short_loss_lists():
    with pytest.raises(ValueError, match="must match num_tasks"):
        get_loss_list(
            loss=["bce", "bpr"],
            training_mode="pointwise",
            num_tasks=3,
        )


def test_fit_requires_compile_with_explicit_loss():
    model = _DummyBinaryModel()
    data = {"y": torch.tensor([1.0, 0.0])}

    with pytest.raises(ValueError, match="compiled before fit"):
        model.fit(
            train_data=data,
            epochs=1,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            use_tensorboard=False,
        )
