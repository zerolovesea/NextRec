import ast

import numpy as np
import pandas as pd

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.data import build_eval_candidates
from nextrec.data.dataloader import RecDataLoader
from nextrec.data.preprocessor import DataProcessor
from nextrec.models.retrieval.dssm import DSSM
from nextrec.utils import compute_pair_scores


df = pd.read_csv("dataset/match_task.csv")
df["user_sequence_0"] = df["user_sequence_0"].apply(ast.literal_eval)

user_dense_cols = ["user_dense_0", "user_dense_1", "user_dense_2"]
user_sparse_cols = [
    "user_id",
    "user_sparse_0",
    "user_sparse_1",
    "user_sparse_2",
    "user_sparse_3",
    "user_sparse_4",
]
user_sequence_cols = ["user_sequence_0"]

item_dense_cols = ["item_dense_0", "item_dense_1"]
item_sparse_cols = [
    "item_id",
    "item_sparse_0",
    "item_sparse_1",
    "item_sparse_2",
    "item_sparse_3",
]

processor = DataProcessor()
for col in user_sparse_cols + item_sparse_cols:
    processor.add_sparse_feature(col, encode_method="label")
for col in user_dense_cols + item_dense_cols:
    processor.add_numeric_feature(col, scaler="minmax")
for col in user_sequence_cols:
    processor.add_sequence_feature(col, encode_method="label", max_len=20)
processor.fit(df)

vocab_sizes = processor.get_vocab_sizes()

user_dense_features = [DenseFeature(col) for col in user_dense_cols]
user_sparse_features = [
    SparseFeature(col, vocab_size=vocab_sizes[col], embedding_dim=4)
    for col in user_sparse_cols
    if col != "user_id"
]
user_sparse_features.append(
    SparseFeature("user_id", vocab_size=vocab_sizes["user_id"], embedding_dim=32)
)
user_sequence_features = [
    SequenceFeature(
        "user_sequence_0",
        vocab_size=vocab_sizes["user_sequence_0"],
        max_len=20,
        embedding_dim=8,
        padding_idx=0,
    )
]

item_dense_features = [DenseFeature(col) for col in item_dense_cols]
item_sparse_features = [
    SparseFeature(col, vocab_size=vocab_sizes[col], embedding_dim=4)
    for col in item_sparse_cols
    if col != "item_id"
]
item_sparse_features.append(
    SparseFeature("item_id", vocab_size=vocab_sizes["item_id"], embedding_dim=32)
)

rec_loader = RecDataLoader(
    dense_features=user_dense_features + item_dense_features,
    sparse_features=user_sparse_features + item_sparse_features,
    sequence_features=user_sequence_features,
    target="label",
    id_columns=["user_id"],
    processor=processor,
)

rng = np.random.default_rng(2025)
perm = rng.permutation(len(df))
cut = int(len(df) * 0.8)
train_df = df.iloc[perm[:cut]].reset_index(drop=True)
valid_df = df.iloc[perm[cut:]].reset_index(drop=True)

# Pairwise in-batch training uses positive pairs only.
train_df = train_df[train_df["label"] == 1].reset_index(drop=True)

train_loader = rec_loader.create_dataloader(
    train_df, batch_size=256, shuffle=True, num_workers=0
)
valid_loader = rec_loader.create_dataloader(
    valid_df, batch_size=256, shuffle=False, num_workers=0
)

df_encoded = processor.transform(df, return_dict=False)
train_df_encoded = df_encoded.iloc[perm[:cut]].reset_index(drop=True)
valid_df_encoded = df_encoded.iloc[perm[cut:]].reset_index(drop=True)

model = DSSM(
    user_dense_features=user_dense_features,
    user_sparse_features=user_sparse_features,
    user_sequence_features=user_sequence_features,
    item_dense_features=item_dense_features,
    item_sparse_features=item_sparse_features,
    item_sequence_features=[],
    embedding_dim=64,
    temperature=0.05,
    user_dnn_hidden_units=[256, 128],
    item_dnn_hidden_units=[256, 128],
    training_mode="pairwise",
    device="cpu",
    session_id="match_task_pairwise_tutorial",
)

model.compile(loss="bpr")
model.fit(
    train_data=train_loader,
    valid_data=valid_loader,
    metrics=["auc"],
    epochs=1,
    batch_size=256,
    shuffle=True,
    user_id_column="user_id",
)

user_features = (
    df_encoded[user_dense_cols + user_sparse_cols + user_sequence_cols]
    .drop_duplicates("user_id")
    .reset_index(drop=True)
)
item_features = (
    df_encoded[item_dense_cols + item_sparse_cols]
    .drop_duplicates("item_id")
    .reset_index(drop=True)
)

valid_candidates = build_eval_candidates(
    df_all=valid_df_encoded,
    user_col="user_id",
    item_col="item_id",
    label_col="label",
    user_features=user_features,
    item_features=item_features,
    num_pos_per_user=5,
    num_neg_per_pos=50,
)

scores = compute_pair_scores(model, valid_candidates, batch_size=512)
valid_candidates = valid_candidates.assign(score=scores)
topk = (
    valid_candidates.sort_values(["user_id", "score"], ascending=[True, False])
    .groupby("user_id")
    .head(10)
)
print(topk[["user_id", "item_id", "label", "score"]].head(20))

print("Example complete.")
