# NextRec Documentation

NextRec is a unified recommendation framework built on PyTorch. It offers modular feature definitions, a reproducible data processing pipeline, and a standard training engine that already powers ranking, retrieval, multi-task, and emerging generative recommendation models.

## What you get
- Unified interface for ranking, retrieval, multi-task, and early generative recommenders (TIGER, HSTU in progress).
- Ready-to-use feature abstractions: `DenseFeature`, `SparseFeature`, `SequenceFeature`.
- End-to-end training loop with `compile`, `fit`, `evaluate`, `predict`, checkpoints, metrics, and early stopping.
- DataProcessor for repeatable numeric/sparse/sequence/target handling with save/load support.
- GPU/MPS ready; tutorials and runnable scripts under `tutorials/`.

## Installation
Using uv (recommended):

```bash
git clone https://github.com/zerolovesea/NextRec.git
cd NextRec
pip install uv
uv sync
source .venv/bin/activate
uv pip install -e .
```

Using pip:

```bash
git clone https://github.com/zerolovesea/NextRec.git
cd NextRec
pip install -r requirements.txt
pip install -r test_requirements.txt
pip install -e .
```

## 5-minute quick start (DeepFM)
Train and predict on MovieLens-style data:

```python
import pandas as pd
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.basic.features import DenseFeature, SparseFeature

df = pd.read_csv("dataset/movielens_100k.csv")

dense_features = [DenseFeature("age")]
sparse_features = [
    SparseFeature("user_id", vocab_size=df["user_id"].max() + 1, embedding_dim=4),
    SparseFeature("item_id", vocab_size=df["item_id"].max() + 1, embedding_dim=4),
    SparseFeature("gender", vocab_size=df["gender"].max() + 1, embedding_dim=4),
    SparseFeature("occupation", vocab_size=df["occupation"].max() + 1, embedding_dim=4),
]

model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    target="label",
    device="cpu",
    session_id="deepfm_demo",
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="bce",
)

model.fit(
    train_data=df,
    metrics=["auc", "recall", "precision"],
    epochs=5,
    batch_size=512,
    shuffle=True,
    verbose=1,
    validation_split=0.1,
)

preds = model.predict(df)
print(preds[:5])
```

## Core API guide
Feature definitions (`nextrec.basic.features`):

- `DenseFeature(name, embedding_dim=1)` for continuous values.
- `SparseFeature(name, vocab_size, embedding_dim=auto, padding_idx=None, l1_reg=0.0, l2_reg=1e-5, trainable=True)` for categorical ids.
- `SequenceFeature(name, vocab_size, max_len=20, combiner="mean", padding_idx=None, l1_reg=0.0, l2_reg=1e-5, trainable=True)` for histories with pooling.

Data processing (`nextrec.data.preprocessor.DataProcessor`):

```python
from nextrec.data.preprocessor import DataProcessor

processor = DataProcessor()
processor.add_numeric_feature("age", scaler="standard")
processor.add_sparse_feature("user_id", encode_method="label")
processor.add_sequence_feature("item_history", encode_method="hash", hash_size=5000, max_len=50, pad_value=0)
processor.add_target("label", target_type="binary")

processor.fit(train_df)                       # learns scalers/encoders
train_arr = processor.transform(train_df)     # dict -> numpy arrays
vocab_sizes = processor.get_vocab_sizes()     # useful for embedding dims
processor.save("processor.pkl")               # persist for serving
processor = DataProcessor.load("processor.pkl")
```

## Training workflow (`nextrec.basic.model.BaseModel` interface)

```python
model.compile(
    optimizer="adam",                          # str, class, or instance
    optimizer_params={"lr": 1e-3},
    scheduler="steplr",                        # optional torch scheduler name/class/instance
    scheduler_params={"step_size": 3, "gamma": 0.5},
    loss="bce",                                # per-task loss or list
)

model.fit(
    train_data=train_df_or_loader,             # dict, DataFrame, or DataLoader
    valid_data=valid_df_or_loader,             # optional validation split
    metrics=["auc", "logloss"],                # or {"label": ["auc", "logloss"]}
    epochs=10,
    batch_size=256,
    shuffle=True,
    verbose=1,
    validation_split=0.1,                      # auto split when valid_data is None
)

scores = model.evaluate(valid_df_or_loader)    # returns metric dict
preds = model.predict(test_df_or_loader)       # numpy array or dict
model.save_model("checkpoint.model")              # default no timestamp (overwrite)
model.load_model("checkpoint.model", map_location="cpu")
```

## Model zoo (`nextrec.models`)
- Ranking: FM, AFM, DeepFM, Wide&Deep, xDeepFM, FiBiNET, PNN, AutoInt, DCN, DIN, DIEN, MaskNet.
- Retrieval: DSSM, DSSM v2 (pairwise), YouTube DNN, MIND, SDM.
- Multi-task: MMOE, PLE, ESMM, ShareBottom.
- Generative (in progress): TIGER, HSTU.

## Tutorials and scripts
- Ready-to-run examples live in `tutorials/` (e.g., `movielen_ranking_deepfm.py`, `example_multitask.py`).
- Datasets used in samples live in `dataset/`. Check `README.md` and `README_zh.md` for dataset prep and more examples.

## Contents

```{toctree}
:maxdepth: 2
:caption: Contents

modules
```

## API reference stub

```{automodule} nextrec
:members:
:noindex:
```
