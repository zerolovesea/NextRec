import pandas as pd
import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature
from nextrec.basic.model import BaseModel


class _DummyGroupedEvalModel(BaseModel):  # type: ignore[misc]
    @property
    def model_name(self) -> str:
        return "DummyGroupedEval"

    @property
    def default_task(self) -> list[str]:
        return ["binary", "binary"]

    def forward(self, X_input):
        score = X_input["score"].reshape(-1, 1)
        return torch.cat([score, score], dim=1)

    def __init__(self):
        super().__init__(
            dense_features=[DenseFeature(name="score", proj_dim=1)],
            sparse_features=[],
            sequence_features=[],
            target=["task1", "task2"],
            task=["binary", "binary"],
            id_columns=["uid"],
        )
        self.bias = nn.Parameter(torch.zeros(1))


def test_evaluate_grouped_by_column(tmp_path):
    model = _DummyGroupedEvalModel()
    data = pd.DataFrame(
        {
            "uid": ["u1", "u2", "u3", "u4"],
            "product": ["p1", "p1", "p2", "p2"],
            "score": [0.9, 0.1, 0.8, 0.2],
            "task1": [1, 0, 1, 0],
            "task2": [1, 0, 1, 0],
        }
    )
    result = model.evaluate(
        data=data,
        metrics=["precision", "recall"],
        batch_size=2,
        user_id_column="uid",
        group_by="product",
        thresholds=0.5,
    )
    overall_metrics = result["overall"]
    grouped_rows = result["grouped"]

    assert overall_metrics["precision_task1"] == 0.5
    assert overall_metrics["recall_task2"] == 1.0
    assert len(grouped_rows) == 2
    assert grouped_rows[0]["product"] == "p1"
    assert grouped_rows[0]["samples"] == 2
    assert grouped_rows[0]["precision_task1"] == 0.5
    assert grouped_rows[1]["product"] == "p2"
