import pandas as pd

from sklearn.model_selection import train_test_split
from nextrec.models.ranking.din import DIN
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

df = pd.read_csv("dataset/ranking_task.csv")

# Parse sequence features from string to list
for col in df.columns:
    if "sequence" in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)


train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

dense_features = [DenseFeature(name=f"dense_{i}", input_dim=1) for i in range(8)]
sparse_features = [
    SparseFeature(
        name="user_id",
        embedding_name="user_emb",
        vocab_size=int(df["user_id"].max() + 1),
        embedding_dim=32,
    ),
    SparseFeature(
        name="item_id",
        embedding_name="item_emb",
        vocab_size=int(df["item_id"].max() + 1),
        embedding_dim=32,
    ),
]
sparse_features.extend(
    [
        SparseFeature(
            name=f"sparse_{i}",
            embedding_name=f"sparse_{i}_emb",
            vocab_size=int(df[f"sparse_{i}"].max() + 1),
            embedding_dim=32,
        )
        for i in range(10)
    ]
)
sequence_features = [
    SequenceFeature(
        name="sequence_0",
        vocab_size=int(df["sequence_0"].apply(lambda x: max(x)).max() + 1),
        embedding_dim=32,
        padding_idx=0,
        embedding_name="item_emb",
    ),
    SequenceFeature(
        name="sequence_1",
        vocab_size=int(df["sequence_1"].apply(lambda x: max(x)).max() + 1),
        embedding_dim=16,
        padding_idx=0,
        embedding_name="sparse_0_emb",
    ),
]


mlp_params = {
    "dims": [256, 128, 64],
    "activation": "relu",
    "dropout": 0.3,
}

model = DIN(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    mlp_params=mlp_params,
    attention_hidden_units=[80, 40],
    attention_activation="sigmoid",
    attention_use_softmax=True,
    target=["label"],
    device="mps",
    embedding_l1_reg=1e-6,
    embedding_l2_reg=1e-5,
    dense_l1_reg=1e-5,
    dense_l2_reg=1e-4,
    session_id="din_tutorial",
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    scheduler="step",
    scheduler_params={"step_size": 3, "gamma": 0.5},
    loss="focal",
    loss_params={"gamma": 2.0, "alpha": 0.25},
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

predictions: pd.DataFrame = model.predict(
    valid_df, batch_size=512, return_dataframe=True
)

print(f"Prediction shape: {predictions.shape}")
print(f"Prediction sample: {predictions[:10]}")

metrics = model.evaluate(
    valid_df,
    metrics=["auc", "gauc", "logloss"],
    batch_size=512,
    user_id_column="user_id",
)
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print("")
print("DIN Example Complete!")
print("")
