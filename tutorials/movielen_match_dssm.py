import numpy as np
import pandas as pd

from nextrec.data.preprocessor import DataProcessor
from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.retrieval.dssm import DSSM
from nextrec.utils import compute_pair_scores

from nextrec.data import build_eval_candidates

df = pd.read_csv("dataset/movielens_100k.csv")

base_cols = ["label"]
user_sparse_cols = ["user_id", "gender", "occupation", "zip_code"]
user_dense_cols = ["age"]
item_sparse_cols = [
    "item_id",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

df = df[base_cols + user_sparse_cols + user_dense_cols + item_sparse_cols]
print(f"Using columns: {df.columns.tolist()}")

processor = DataProcessor()
for col in user_sparse_cols + item_sparse_cols:
    processor.add_sparse_feature(col, encode_method="label")
for col in user_dense_cols:
    processor.add_numeric_feature(col, scaler="minmax")

print("fit transform will cost some time...")
df = processor.fit_transform(df, return_dict=False)

print("\nTransformed head:")
print(df.head())

rng = np.random.default_rng(2025)
all_users = df["user_id"].unique()
rng.shuffle(all_users)

n_users = len(all_users)
n_train = int(n_users * 0.7)
n_valid = int(n_users * 0.3)

train_users = set(all_users[:n_train])
valid_users = set(all_users[n_train : n_train + n_valid])

df_train_all = df[df["user_id"].isin(train_users)].reset_index(drop=True)
df_valid_all = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

print(f"\nTrain users: {len(train_users)}, Valid users: {len(valid_users)}")
print(
    f"Train interactions: {len(df_train_all)}, Valid interactions: {len(df_valid_all)}"
)

# only use positive samples for in-batch InfoNCE training
train_df = df_train_all[df_train_all["label"] == 1].reset_index(drop=True)
print(f"Train positives for contrastive learning: {len(train_df)}")

user_feature_cols = (
    ["user_id"] + [c for c in user_sparse_cols if c != "user_id"] + user_dense_cols
)
item_feature_cols = ["item_id"] + [c for c in item_sparse_cols if c != "item_id"]

user_features = df[user_feature_cols].drop_duplicates("user_id")
item_features = df[item_feature_cols].drop_duplicates("item_id")

# build evaluation candidates
valid_df = build_eval_candidates(
    df_all=df_valid_all,
    user_col="user_id",
    item_col="item_id",
    label_col="label",
    user_features=user_features,
    item_features=item_features,
    num_pos_per_user=5,
    num_neg_per_pos=50,
)

user_dense_features = [DenseFeature(col) for col in user_dense_cols]

# user sparse
user_sparse_features = [
    SparseFeature(col, vocab_size=int(df[col].max()) + 1, embedding_dim=4)
    for col in user_sparse_cols
    if col != "user_id"
]
user_sparse_features.append(
    SparseFeature("user_id", vocab_size=int(df["user_id"].max()) + 1, embedding_dim=32)
)

# item sparse
item_sparse_features = [
    SparseFeature(col, vocab_size=int(df[col].max()) + 1, embedding_dim=4)
    for col in item_sparse_cols
    if col != "item_id"
]
# item_id
item_sparse_features.append(
    SparseFeature("item_id", vocab_size=int(df["item_id"].max()) + 1, embedding_dim=32)
)

model = DSSM(
    user_dense_features=user_dense_features,
    user_sparse_features=user_sparse_features,
    user_sequence_features=[],
    item_dense_features=[],
    item_sparse_features=item_sparse_features,
    item_sequence_features=[],
    embedding_dim=64,
    temperature=0.05,
    user_dnn_hidden_units=[256, 128],
    item_dnn_hidden_units=[256, 128],
    training_mode="pairwise",
    device="cpu",
    session_id="movielens_dssm_tutorial",
)

# Pairwise training requires a pairwise loss such as BPR
model.compile(loss="bpr")

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "gauc", "recall@5", "hitrate@5", "mrr@5", "ndcg@5"],
    epochs=1,
    batch_size=256,
    shuffle=True,
    user_id_column="user_id",
)
# Evaluate
predictions = compute_pair_scores(model, valid_df, batch_size=512)
print(f"\nPredictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:10]}")

print("")
print("DSSM Example Complete!")
print("")
