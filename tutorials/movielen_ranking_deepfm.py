import pandas as pd

from sklearn.model_selection import train_test_split

from nextrec.data.preprocessor import DataProcessor
from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM


df = pd.read_csv("dataset/movielens_100k.csv")

processor = DataProcessor()
processor.add_sparse_feature("movie_title", encode_method="hash", hash_size=1000)
processor.fit(df)

df = processor.transform(df, return_dict=False)
processor.save(save_path="./")
print("Sample training data:")
print(df.head())

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

dense_features = [DenseFeature("age")]
sparse_features = [
    SparseFeature("user_id", vocab_size=df["user_id"].max() + 1, embedding_dim=4),
    SparseFeature("item_id", vocab_size=df["item_id"].max() + 1, embedding_dim=4),
]

sparse_features.append(
    SparseFeature("gender", vocab_size=df["gender"].max() + 1, embedding_dim=4)
)
sparse_features.append(
    SparseFeature("occupation", vocab_size=df["occupation"].max() + 1, embedding_dim=4)
)
sparse_features.append(
    SparseFeature(
        "movie_title", vocab_size=df["movie_title"].max() + 1, embedding_dim=4
    )
)

model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.2},
    target="label",
    device="cpu",
    session_id="movielens_deepfm_tutorial",
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="binary_crossentropy",
)

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "recall", "precision"],
    epochs=1,
    batch_size=512,
    shuffle=True,
)

predictions = model.predict(valid_df, batch_size=512)
print(f"\nPredictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:10]}")

print("")
print("DeepFM Example Complete!")
print("")
