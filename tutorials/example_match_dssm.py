"""
DSSM (Deep Structured Semantic Model) Example with GAUC metric
Uses match_task.csv generated data
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from nextrec.utils.device import resolve_device
from nextrec.models.match.dssm import DSSM
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

# Evaluate
from sklearn.metrics import roc_auc_score, log_loss
from nextrec.basic.metrics import compute_gauc

# Load generated data
df = pd.read_csv("dataset/match_task.csv")

for col in df.columns:
    if "sequence" in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

print(f"Dataset loaded: {len(df)} samples")
print(f"Users: {df['user_id'].nunique()}")
print(f"Items: {df['item_id'].nunique()}")
print(f"Positive ratio: {df['label'].mean():.4f}")

# Split data
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# User features
user_dense_features = [DenseFeature(f"user_dense_{i}") for i in range(3)]

# User sparse features (including user_id)
user_sparse_features = [
    SparseFeature("user_id", vocab_size=int(df["user_id"].max() + 1), embedding_dim=32)
]
user_sparse_features.extend(
    [
        SparseFeature(
            f"user_sparse_{i}",
            vocab_size=int(df[f"user_sparse_{i}"].max() + 1),
            embedding_dim=16,
        )
        for i in range(5)
    ]
)

# User behavior sequence
user_sequence_features = [
    SequenceFeature(
        "user_sequence_0",
        vocab_size=int(df["user_sequence_0"].apply(lambda x: max(x)).max() + 1),
        embedding_dim=32,
        combiner="mean",
        padding_idx=0,
    )
]

# Item features
item_dense_features = [DenseFeature(f"item_dense_{i}") for i in range(2)]

# Item sparse features (including item_id)
item_sparse_features = [
    SparseFeature("item_id", vocab_size=int(df["item_id"].max() + 1), embedding_dim=32)
]
item_sparse_features.extend(
    [
        SparseFeature(
            f"item_sparse_{i}",
            vocab_size=int(df[f"item_sparse_{i}"].max() + 1),
            embedding_dim=16,
        )
        for i in range(4)
    ]
)

device = resolve_device()

model = DSSM(
    user_dense_features=user_dense_features,
    user_sparse_features=user_sparse_features,
    user_sequence_features=user_sequence_features,
    item_dense_features=item_dense_features,
    item_sparse_features=item_sparse_features,
    item_sequence_features=[],
    user_dnn_hidden_units=[128, 64],
    item_dnn_hidden_units=[128, 64],
    embedding_dim=32,
    dnn_activation="relu",
    dnn_dropout=0.3,
    training_mode="pointwise",
    similarity_metric="cosine",
    temperature=0.05,
    device=device,
    session_id="dssm_tutorial",
)

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "gauc", "logloss"],  # Added GAUC metric
    epochs=1,
    batch_size=512,
    shuffle=True,
    user_id_column="user_id",  # Specify user_id column for GAUC
)

print("Training Complete!")

# Predict
print("Prediction")

predictions = model.predict(valid_df, batch_size=512)
print(f"Prediction shape: {predictions.shape}")
print(f"Prediction sample: {predictions[:10]}")


auc = roc_auc_score(valid_df["label"].values, predictions)
logloss = log_loss(valid_df["label"].values, predictions)
gauc = compute_gauc(valid_df["label"].values, predictions, valid_df["user_id"].values)

print(f"Valid AUC: {auc:.6f}")
print(f"Valid GAUC: {gauc:.6f}")
print(f"Valid LogLoss: {logloss:.6f}")

print(" ")
print("DSSM Example Complete!")
print(" ")
