"""
SASRec 序列推荐示例

本示例演示如何使用 NextRec 训练一个 SASRec 序列推荐模型,
并直接使用框架内置的 sequential top-k 指标进行验证。

主要功能:
    - 数据加载与预处理
    - 特征定义(稠密特征、稀疏特征、序列特征)
    - SASRec 模型构建与训练
    - 验证集指标评估
    - 下一物品 Top-K 预测

使用方法:
    直接运行此脚本:
        python tutorials/example_sequential_sasrec.py

数据格式:
    - user_id: 用户ID
    - user_age: 用户年龄
    - city_id: 城市ID
    - item_history: 用户历史行为序列, 已 padding 到固定长度

作者: NextRec Team
"""

from __future__ import annotations

import ast

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.models.sequential.sasrec import SASRec


def parse_list_column(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)


# ==============================================================================
# 1. 数据加载和预处理
# ==============================================================================
df = pd.read_csv("dataset/sasrec_task.csv")
df["user_age"] = df["user_age"].astype(float)
df["item_history"] = parse_list_column(df["item_history"])

# 训练集 / 验证集划分
train_df, valid_df = train_test_split(df, test_size=0.25, random_state=42)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")

# 序列特征的词表大小由全量数据确定
max_item_id = max(max(seq) for seq in df["item_history"])
max_seq_len = len(train_df.iloc[0]["item_history"])

# ==============================================================================
# 2. 特征定义
# ==============================================================================
sequence_features = [
    SequenceFeature(
        name="item_history",
        vocab_size=max_item_id + 1,
        max_len=max_seq_len,
        embedding_dim=16,
        padding_idx=0,
    )
]
dense_features = [
    DenseFeature(name="user_age", input_dim=1, proj_dim=4, use_projection=True),
]
sparse_features = [
    SparseFeature(name="city_id", vocab_size=int(df["city_id"].max()) + 1, embedding_dim=4),
]

# ==============================================================================
# 3. 模型构建
# ==============================================================================
model = SASRec(
    sequence_features=sequence_features,
    dense_features=dense_features,
    sparse_features=sparse_features,
    item_history_name="item_history",
    hidden_dim=16,
    num_heads=2,
    num_blocks=1,
    max_seq_len=max_seq_len,
    dropout_rate=0.1,
    target=["next_item"],
    task="sequential",
    id_columns=["user_id"],
    device="cpu",
    session_id="sasrec_tutorial",
)

# 使用交叉熵训练 next-item prediction
model.compile(loss="ce")

# ==============================================================================
# 4. 模型训练与验证
# ==============================================================================
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    epochs=3,
    batch_size=2,
    shuffle=True,
    use_tensorboard=False,
    early_stop_patience=0,
    num_workers=0,
    user_id_column="user_id",
)

# ==============================================================================
# 5. 推理示例
# ==============================================================================
example_input = {
    "item_history": torch.tensor([train_df.loc[0, "item_history"]], dtype=torch.long),
    "user_age": torch.tensor([[train_df.loc[0, "user_age"]]], dtype=torch.float32),
    "city_id": torch.tensor([int(train_df.loc[0, "city_id"])], dtype=torch.long),
}
logits = model.predict_last(example_input)
top5 = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()
print(f"Top-5 next-item predictions for sample 0: {top5}")
print("SASRec tutorial finished.")
