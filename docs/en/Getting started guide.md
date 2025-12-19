# NextRec Quickstart

This short guide shows how to train and launch a recommender model with NextRec using the bundled `dataset/movielens_100k.csv`. For full scripts, see the `tutorials/` directory.

## 1. Environment & Installation

- Requirements: Python 3.10+, PyTorch 1.10+ (GPU/MPS optional).
- Stable install: `pip install nextrec`
- Dev mode: from repo root run `pip install -e .`

## 2. Data & Feature Primer

Recommender models usually mix multiple signal types and convert them into vectors:

- Dense features: continuous or ordered values (age, price, duration, etc.), typically standardized/normalized or log-transformed.
- Sparse features: high-cardinality discrete fields (user ID, item ID, gender, occupation, device type, etc.), indexed then embedded via an embedding lookup.
- Sequence features: variable-length behavior histories (browse/click/purchase lists), truncated/padded and aggregated (mean/sum/attention) to fixed-length vectors.
- Context & multi-modal features: time, geography, slot position, or vectors from text/image/video encoders that can interact with the main features.

In `movielens_100k.csv`, you will see `user_id`, `item_id`, `gender`, `occupation`, dense fields like `age`, and the supervision label `label` (click/high rating).

## 3. Train a Ranking Model in 5 Minutes (DeepFM)

The example below runs end-to-end feature definition, training, and evaluation with DeepFM on MovieLens. It relies on NextRec’s built-in data loading—no custom DataLoader code needed.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM

# 1) Load data
df = pd.read_csv("dataset/movielens_100k.csv")

# 2) Define features
dense_features = [DenseFeature("age")]
sparse_features = [
    SparseFeature("user_id", vocab_size=df["user_id"].max() + 1, embedding_dim=16),
    SparseFeature("item_id", vocab_size=df["item_id"].max() + 1, embedding_dim=16),
    SparseFeature("gender", vocab_size=df["gender"].max() + 1, embedding_dim=4),
    SparseFeature("occupation", vocab_size=df["occupation"].max() + 1, embedding_dim=8),
]

# 3) Split train/valid
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# 4) Instantiate and compile
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.2},
    target="label",
    device="cpu",
    session_id="movielens_deepfm",   # manages logs and checkpoints
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="binary_crossentropy",
)

# 5) Train
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "recall", "precision"],
    epochs=2,
    batch_size=512,
    shuffle=True,
)
```

- `metrics` supports `auc`/`logloss`/`accuracy`/`gauc`; when using GAUC, pass `user_id_column="user_id"`.
- Training uses early stopping and saves best weights/config/logs under `nextrec_logs/movielens_deepfm`.

## 4. Inference & Evaluation

After training, run batch prediction or evaluation:

```python
# Batch prediction
preds = model.predict(valid_df, batch_size=512)

# Metrics
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "gauc", "logloss"],
    batch_size=512,
    user_id_column="user_id",  # only needed for GAUC
)
```

- Save predictions: `model.predict(..., save_path="outputs/preds", save_format="csv")`.
- Load a checkpoint: `model.load("path/to/checkpoint")` to resume training or deploy.

## 5. More Examples & Notebooks

- Ranking: `tutorials/example_ranking_din.py` (e-commerce DIN), `tutorials/movielen_ranking_deepfm.py`
- Retrieval: `tutorials/movielen_match_dssm.py`
- Multi-task: `tutorials/example_multitask.py`
- Notebooks: `tutorials/notebooks/zh/Hands on nextrec.ipynb`, `tutorials/notebooks/zh/Hands on dataprocessor.ipynb`

For large offline features or streaming loads, use `DataProcessor` and `RecDataLoader` to configure CSV/Parquet paths and streaming (`streaming=True`) without changing model code.
