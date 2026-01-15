import pandas as pd

from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.data.preprocessor import DataProcessor
from nextrec.models.tree_base.xgboost import Xgboost

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
    SparseFeature("user_id", vocab_size=df["user_id"].max() + 1),
    SparseFeature("item_id", vocab_size=df["item_id"].max() + 1),
    SparseFeature("gender", vocab_size=df["gender"].max() + 1),
    SparseFeature("occupation", vocab_size=df["occupation"].max() + 1),
    SparseFeature("movie_title", vocab_size=1000),
]

model = Xgboost(
    dense_features=dense_features,
    sparse_features=sparse_features,
    target="label",
    session_id="movielens_xgboost_tutorial",
    model_params={
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "auc",
    },
)

# model = Catboost(
#     dense_features=dense_features,
#     sparse_features=sparse_features,
#     target="label",
#     session_id="movielens_catboost_tutorial",
#     model_params={
#         "depth": 6,
#         "learning_rate": 0.1,
#         "eval_metric": "AUC",
#         "verbose": False,
#     },
# )

# model = Lightgbm(
#     dense_features=dense_features,
#     sparse_features=sparse_features,
#     target="label",
#     session_id="movielens_lightgbm_tutorial",
#     model_params={
#         "max_depth": 6,
#         "learning_rate": 0.1,
#         "subsample": 0.9,
#         "colsample_bytree": 0.9,
#         "metric": "auc",
#     },
# )

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "ks", "recall"],
    epochs=200,
)

predictions = model.predict(valid_df, batch_size=512)
print(f"\nPredictions shape: {predictions.shape}")
print(f"Sample predictions:\n{predictions.head(10)}")

model_path = model.save_model()
print(f"\nModel saved to: {model_path}")

loaded_model = Xgboost(session_id="movielens_tree_tutorial")
# loaded_model = Catboost(session_id="movielens_tree_tutorial")
# loaded_model = Lightgbm(session_id="movielens_tree_tutorial")

loaded_model.load_model(model_path)
loaded_predictions = loaded_model.predict(valid_df, batch_size=512)
print(f"\nLoaded model predictions shape: {loaded_predictions.shape}")
print(f"Loaded model sample predictions:\n{loaded_predictions.head(10)}")

print("")
print("Tree Example Complete!")
print("")
