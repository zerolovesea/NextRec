import torch.nn as nn

from nextrec.basic.model import BaseModel


class _DummyModel(BaseModel):
    @property
    def model_name(self) -> str:
        return "Dummy"

    @property
    def default_task(self) -> str:
        return "binary"

    def __init__(self, device: str = "cpu"):
        super().__init__(target=None, device=device)
        self.token_embedding = nn.Embedding(10, 4)
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def test_register_regularization_weights_supports_nn_embedding_and_dedup():
    model = _DummyModel(device="cpu")

    model.register_regularization_weights(
        embedding_attr="token_embedding", include_modules=["mlp"]
    )

    assert any(
        param is model.token_embedding.weight for param in model.embedding_params
    ), "nn.Embedding.weight should be included in embedding regularization params"

    assert len(model.regularization_weights) == 2
    assert any(param is model.mlp[0].weight for param in model.regularization_weights)
    assert any(param is model.mlp[2].weight for param in model.regularization_weights)

    model.register_regularization_weights(
        embedding_attr="token_embedding", include_modules=["mlp"]
    )

    assert len(model.embedding_params) == 1
    assert len(model.regularization_weights) == 2
