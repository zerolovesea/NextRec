"""
Multi-task Learning Example with GAUC metric
Uses multitask_task.csv generated data
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from nextrec.models.multi_task.esmm import ESMM
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

# Load generated data
df = pd.read_csv("dataset/multitask_task.csv")

# Parse sequence features from string to list
for col in df.columns:
    if "sequence" in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

task_labels = ["click", "conversion"]

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

num_dense = len([col for col in df.columns if col.startswith("dense_")])
dense_features = [DenseFeature(f"dense_{i}") for i in range(num_dense)]

sparse_features = [
    SparseFeature("user_id", vocab_size=int(df["user_id"].max() + 1), embedding_dim=32),
    SparseFeature("item_id", vocab_size=int(df["item_id"].max() + 1), embedding_dim=32),
]

num_sparse = len([col for col in df.columns if col.startswith("sparse_")])
sparse_features.extend(
    [
        SparseFeature(
            f"sparse_{i}", vocab_size=int(df[f"sparse_{i}"].max() + 1), embedding_dim=16
        )
        for i in range(num_sparse)
    ]
)

sequence_cols = [col for col in df.columns if col.startswith("sequence_")]
sequence_features = [
    SequenceFeature(
        col,
        vocab_size=int(df[col].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1),
        embedding_dim=32,
        padding_idx=0,
    )
    for col in sequence_cols
]

print(f"Dense features: {len(dense_features)}")
print(f"Sparse features: {len(sparse_features)} (including user_id and item_id)")
print(f"Sequence features: {len(sequence_features)}")

# ESMM model
ctr_params = {"dims": [64, 32], "activation": "relu", "dropout": 0.4}

cvr_params = {"dims": [64, 32], "activation": "relu", "dropout": 0.4}

model = ESMM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    ctr_params=ctr_params,
    cvr_params=cvr_params,
    target=task_labels,
    task=["binary", "binary"],
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
    loss=["bce", "bce"],
    device="mps",
    embedding_l1_reg=1e-5,
    embedding_l2_reg=1e-4,
    dense_l1_reg=1e-4,
    dense_l2_reg=1e-3,
    session_id="esmm_tutorial",
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
    loss=["bce", "bce"],
    loss_weights=[0.3, 0.7],
)


# model.compile(
#     optimizer="adam",
#     optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
#     loss=["bce", "bce"],
#     loss_weights={"method": "grad_norm", "alpha": 1.5, "lr": 0.025},
# )

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
preview = predictions.head(5)

print(f"Prediction shape: {predictions.shape}")
print(f"Prediction sample: {predictions[:10]}")

# Evaluation
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "gauc", "logloss"],
    batch_size=512,
    user_id_column="user_id",
)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("")
print("Multi-task Example Complete!")
print("")
