"""
Unit tests for sequential recommendation models.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import torch

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.adapters import SequentialAdapter
from nextrec.models.sequential.sasrec import SASRec
from test.helpers import assert_no_nan_or_inf


def _parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("item_history",):
        df[col] = df[col].apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)
    return df


def _build_sasrec_model(vocab_size: int, max_len: int = 5, sequence_mode: str = "autoregressive") -> SASRec:
    sequence_features = [
        SequenceFeature(
            name="item_history",
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_dim=8,
            padding_idx=0,
        )
    ]
    model = SASRec(
        sequence_features=sequence_features,
        item_history_name="item_history",
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        sequence_mode=sequence_mode,
        max_seq_len=max_len,
        dropout_rate=0.0,
        target=["next_item"],
        task="sequential",
        device="cpu",
        session_id="sasrec_test",
    )
    model.compile(loss="ce")
    return model


def _build_sasrec_model_with_context(
    vocab_size: int, city_vocab_size: int, max_len: int = 5, sequence_mode: str = "autoregressive"
) -> SASRec:
    sequence_features = [
        SequenceFeature(
            name="item_history",
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_dim=8,
            padding_idx=0,
        )
    ]
    dense_features = [
        DenseFeature(name="user_age", input_dim=1, proj_dim=4, use_projection=True),
    ]
    sparse_features = [
        SparseFeature(name="city_id", vocab_size=city_vocab_size, embedding_dim=4),
    ]
    model = SASRec(
        sequence_features=sequence_features,
        dense_features=dense_features,
        sparse_features=sparse_features,
        item_history_name="item_history",
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        sequence_mode=sequence_mode,
        max_seq_len=max_len,
        dropout_rate=0.0,
        target=["next_item"],
        task="sequential",
        device="cpu",
        session_id="sasrec_test_context",
    )
    model.compile(loss="ce")
    return model


def test_sasrec_forward_and_loss():
    model = _build_sasrec_model(vocab_size=16, max_len=5)
    assert isinstance(model.training_adapter, SequentialAdapter)

    x = {
        "item_history": torch.tensor(
            [
                [1, 2, 3, 4, 0],
                [2, 3, 4, 5, 0],
            ],
            dtype=torch.long,
        )
    }
    y_true = torch.tensor(
        [
            [2, 3, 4, 5, 0],
            [3, 4, 5, 6, 0],
        ],
        dtype=torch.long,
    )

    y_pred = model.forward(x)
    assert y_pred.shape == (2, 5, 16)
    assert_no_nan_or_inf(y_pred, "sasrec_logits")

    last_logits = model.predict_last(x)
    assert last_logits.shape == (2, 16)
    assert_no_nan_or_inf(last_logits, "sasrec_last_logits")

    loss = model.compute_loss(y_pred, y_true)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_sasrec_train_epoch_on_sample_dataset():
    dataset_path = Path("dataset/sasrec_task.csv")
    assert dataset_path.exists()

    df = _parse_list_columns(pd.read_csv(dataset_path))
    vocab_size = max(max(seq) for seq in df["item_history"]) + 1
    model = _build_sasrec_model(vocab_size=vocab_size, max_len=5)
    model.epoch_index = 0
    model.needs_user_ids = False

    train_loader = model.prepare_data_loader(df, batch_size=2, shuffle=False, num_workers=0)
    avg_loss = model.train_epoch(train_loader)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0.0


def test_sasrec_with_context_features_forward_and_train_epoch():
    df = pd.DataFrame(
        {
            "user_age": [23.0, 31.0, 27.0, 42.0],
            "city_id": [1, 2, 3, 1],
            "item_history": [
                [1, 2, 3, 4, 0],
                [2, 3, 4, 5, 0],
                [3, 4, 5, 0, 0],
                [4, 5, 6, 7, 0],
            ],
        }
    )
    vocab_size = max(max(seq) for seq in df["item_history"]) + 1
    model = _build_sasrec_model_with_context(vocab_size=vocab_size, city_vocab_size=4, max_len=5)

    x = {
        "item_history": torch.tensor(df.loc[:1, "item_history"].tolist(), dtype=torch.long),
        "user_age": torch.tensor(df.loc[:1, "user_age"].to_numpy().reshape(-1, 1), dtype=torch.float32),
        "city_id": torch.tensor(df.loc[:1, "city_id"].to_numpy(), dtype=torch.long),
    }
    y_true = torch.tensor(
        [
            [2, 3, 4, 5, 0],
            [3, 4, 5, 6, 0],
        ],
        dtype=torch.long,
    )

    y_pred = model.forward(x)
    assert y_pred.shape == (2, 5, vocab_size)
    assert_no_nan_or_inf(y_pred, "sasrec_context_logits")

    loss = model.compute_loss(y_pred, y_true)
    assert torch.isfinite(loss)

    model.epoch_index = 0
    model.needs_user_ids = False
    train_loader = model.prepare_data_loader(df, batch_size=2, shuffle=False, num_workers=0)
    avg_loss = model.train_epoch(train_loader)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0.0


def test_sasrec_masked_mode_disables_causal_attention_mask():
    model = _build_sasrec_model(vocab_size=16, max_len=5, sequence_mode="masked")
    mask = model.build_sequence_attention_mask(seq_len=5, device=torch.device("cpu"))
    assert mask is None

    x = {
        "item_history": torch.tensor(
            [
                [1, 2, 3, 4, 0],
                [2, 3, 4, 5, 0],
            ],
            dtype=torch.long,
        )
    }
    y_true = torch.tensor(
        [
            [2, 3, 4, 5, 0],
            [3, 4, 5, 6, 0],
        ],
        dtype=torch.long,
    )
    y_pred = model.forward(x)
    assert y_pred.shape == (2, 5, 16)
    loss = model.compute_loss(y_pred, y_true)
    assert torch.isfinite(loss)
