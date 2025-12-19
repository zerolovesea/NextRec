import types

import torch

from nextrec.utils import embedding as embedding_utils


def test_get_auto_embedding_dim():
    assert embedding_utils.get_auto_embedding_dim(1) == 6
    assert embedding_utils.get_auto_embedding_dim(16) == 12
    assert embedding_utils.get_auto_embedding_dim(10000) == 60


def test_encode_multimodel_content_batches(monkeypatch):
    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def __call__(
            self,
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ):
            batch_size = len(batch)
            seq_len = 3
            input_ids = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeAutoModel:
        def __init__(self, hidden_dim=4):
            self.hidden_dim = hidden_dim
            self.call_count = 0

        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            input_ids = kwargs["input_ids"]
            batch_size, seq_len = input_ids.shape
            offset = self.call_count * 100
            self.call_count += 1
            base = torch.arange(
                batch_size * seq_len * self.hidden_dim, dtype=torch.float32
            )
            last_hidden_state = (
                base.reshape(batch_size, seq_len, self.hidden_dim) + offset
            )
            return types.SimpleNamespace(last_hidden_state=last_hidden_state)

    monkeypatch.setattr(embedding_utils, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(embedding_utils, "AutoModel", FakeAutoModel)

    texts = ["t1", "t2", "t3", "t4", "t5"]
    embeddings = embedding_utils.encode_multimodel_content(
        texts=texts, model_name="fake", device="cpu", batch_size=2
    )

    assert embeddings.shape == (5, 4)
    assert torch.allclose(embeddings[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(embeddings[2], torch.tensor([100.0, 101.0, 102.0, 103.0]))
