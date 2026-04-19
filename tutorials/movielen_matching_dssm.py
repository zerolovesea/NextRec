"""
MovieLens 数据集召回模型示例 - DSSM 双塔模型

文件说明:
    本示例演示如何使用真实的 MovieLens 100K 数据集训练 DSSM (Deep Structured Semantic Model)
    双塔召回模型。示例包含完整的数据预处理、用户特征提取、物品特征提取、模型训练和召回评估流程。

主要功能:
    - MovieLens 数据加载与预处理
    - 数据编码与转换
    - 用户特征和物品特征定义
    - 基于用户的数据集划分
    - 评估候选集构建
    - DSSM 模型训练
    - 召回效果评估

使用方法:
    直接运行此脚本:
        python tutorials/movielen_matching_dssm.py

测试数据格式:
    - user_id: 用户ID
    - item_id: 电影ID
    - age: 用户年龄
    - gender: 用户性别
    - occupation: 用户职业
    - zip_code: 用户邮编
    - unknown, Action, Adventure, ...: 电影类型标签(多个)
    - label: 标签 (1表示用户喜欢该电影)

模型架构:
    使用 DSSM 双塔模型:
        - 用户塔: 编码用户特征(ID、性别、职业、邮编、年龄)
        - 物品塔: 编码物品特征(ID、电影类型)
        - 训练模式: pairwise (使用 BPR 损失)
        - 相似度度量: 内积/余弦相似度

输出:
    - 训练好的模型
    - 评估候选集的预测分数
    - 召回评估指标(AUC、GAUC、Recall、HitRate、MRR、NDCG)

作者: NextRec Team
创建日期: 2026
最后更新: 2026-01-28
"""

import numpy as np
import pandas as pd

from nextrec.data.preprocessor import DataProcessor
from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.matching.dssm import DSSM
from nextrec.utils.model import compute_pair_scores
from nextrec.data.data_processing import build_eval_candidates

# ==============================================================================
# 1. 数据加载
# ==============================================================================

# 加载 MovieLens 100K 数据集
df = pd.read_csv("dataset/movielens_100k.csv")

# 定义特征列名
base_cols = ["label"]  # 标签列
user_sparse_cols = ["user_id", "gender", "occupation", "zip_code"]  # 用户稀疏特征
user_dense_cols = ["age"]  # 用户稠密特征
item_sparse_cols = [  # 物品稀疏特征(电影类型标签)
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
user_sparse_feature_cols = [f"{col}_label" for col in user_sparse_cols]
user_dense_feature_cols = [f"{col}_minmax" for col in user_dense_cols]
item_sparse_feature_cols = [f"{col}_label" for col in item_sparse_cols]
user_id_col = "user_id_label"
item_id_col = "item_id_label"

# 只保留需要的列
df = df[base_cols + user_sparse_cols + user_dense_cols + item_sparse_cols]
print(f"Using columns: {df.columns.tolist()}")

# ==============================================================================
# 2. 数据预处理
# ==============================================================================

# 创建数据处理器
processor = DataProcessor()

# 稀疏特征使用 label encoding
for col in user_sparse_cols + item_sparse_cols:
    processor.add_sparse_feature(col, encode_method="label")

# 稠密特征使用 minmax 归一化
for col in user_dense_cols:
    processor.add_numeric_feature(col, scaler="minmax")

print("fit transform will cost some time...")
# 在数据上拟合处理器,并获取各特征词表大小
processor.fit(df)
vocab_sizes = processor.get_vocab_sizes()

# 转换数据后,直接使用新增的编码/归一化列进行训练
df_transformed = processor.transform(df, return_dict=False).to_pandas()
selected_cols = base_cols + user_sparse_feature_cols + user_dense_feature_cols + item_sparse_feature_cols
df = df_transformed[selected_cols]

print("\nTransformed head:")
print(df.head())

# ==============================================================================
# 3. 基于用户的数据集划分
# ==============================================================================

# 使用随机数生成器
rng = np.random.default_rng(2025)
all_users = df[user_id_col].unique()
rng.shuffle(all_users)  # 打乱用户顺序

# 计算训练集和验证集的用户数量
n_users = len(all_users)
n_train = int(n_users * 0.7)  # 70% 用户用于训练
n_valid = int(n_users * 0.3)  # 30% 用户用于验证

# 划分训练用户和验证用户
train_users = set(all_users[:n_train])
valid_users = set(all_users[n_train : n_train + n_valid])

# 根据用户划分数据集
df_train_all = df[df[user_id_col].isin(train_users)].reset_index(drop=True)
df_valid_all = df[df[user_id_col].isin(valid_users)].reset_index(drop=True)

print(f"\nTrain users: {len(train_users)}, Valid users: {len(valid_users)}")
print(f"Train interactions: {len(df_train_all)}, Valid interactions: {len(df_valid_all)}")

# Pairwise 训练模式仅使用正样本(in-batch negative sampling)
train_df = df_train_all[df_train_all["label"] == 1].reset_index(drop=True)
print(f"Train positives for contrastive learning: {len(train_df)}")

# ==============================================================================
# 4. 提取用户和物品特征
# ==============================================================================

# 定义用户特征列和物品特征列
user_feature_cols = [user_id_col] + [c for c in user_sparse_feature_cols if c != user_id_col] + user_dense_feature_cols
item_feature_cols = [item_id_col] + [c for c in item_sparse_feature_cols if c != item_id_col]

# 提取唯一的用户特征和物品特征
user_features = df[user_feature_cols].drop_duplicates(user_id_col)
item_features = df[item_feature_cols].drop_duplicates(item_id_col)

# ==============================================================================
# 5. 构建评估候选集
# ==============================================================================

# 为每个验证用户生成评估候选集(包含正负样本)
valid_df = build_eval_candidates(
    df_all=df_valid_all,  # 验证集数据
    user_col=user_id_col,
    item_col=item_id_col,
    label_col="label",
    user_features=user_features,  # 用户特征表
    item_features=item_features,  # 物品特征表
    num_pos_per_user=5,  # 每个用户保留5个正样本
    num_neg_per_pos=50,  # 每个正样本对应50个负样本
)

# ==============================================================================
# 6. 特征定义
# ==============================================================================

# 定义用户稠密特征
user_dense_features = [DenseFeature(col) for col in user_dense_feature_cols]

# 定义用户稀疏特征
# 除 user_id 外使用较小的 embedding 维度(4)
user_sparse_features = [
    SparseFeature(col, vocab_size=vocab_sizes[col], embedding_dim=4)
    for col in user_sparse_feature_cols
    if col != user_id_col
]
# user_id 使用较大的 embedding 维度(32)
user_sparse_features.append(SparseFeature(user_id_col, vocab_size=vocab_sizes[user_id_col], embedding_dim=32))

# 定义物品稀疏特征
# 除 item_id 外使用较小的 embedding 维度(4)
item_sparse_features = [
    SparseFeature(col, vocab_size=vocab_sizes[col], embedding_dim=4)
    for col in item_sparse_feature_cols
    if col != item_id_col
]
# item_id 使用较大的 embedding 维度(32)
item_sparse_features.append(SparseFeature(item_id_col, vocab_size=vocab_sizes[item_id_col], embedding_dim=32))

# ==============================================================================
# 7. 模型构建
# ==============================================================================

# 创建 DSSM 模型
model = DSSM(
    user_dense_features=user_dense_features,
    user_sparse_features=user_sparse_features,
    user_sequence_features=[],  # 此示例不使用用户序列特征
    item_dense_features=[],  # 此示例不使用物品稠密特征
    item_sparse_features=item_sparse_features,
    item_sequence_features=[],  # 此示例不使用物品序列特征
    embedding_dim=64,  # 用户和物品向量的维度
    temperature=0.05,  # 对比学习温度参数
    user_mlp_params={"hidden_dims": [256, 128]},  # 用户塔 MLP 参数
    item_mlp_params={"hidden_dims": [256, 128]},  # 物品塔 MLP 参数
    training_mode="pairwise",  # 使用 pairwise 训练模式
    sampling_mode="inbatch",  # 使用 batch 内负采样
    device="cpu",
    session_id="movielens_dssm_tutorial",
)

# ==============================================================================
# 8. 模型编译
# ==============================================================================

# 编译模型: Pairwise 训练需要使用 pairwise 损失函数(如 BPR)
model.compile(loss="bpr")

# ==============================================================================
# 9. 模型训练
# ==============================================================================

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=[
        "auc",
        "gauc",
        "recall@5",
        "hitrate@10",
        "mrr@5",
        "ndcg@5",
    ],  # 召回评估指标
    epochs=1,  # 训练轮数
    batch_size=256,
    shuffle=True,
    group_id=user_id_col,  # 用于分组评估
)

# ==============================================================================
# 10. 模型评估
# ==============================================================================

# 计算用户-物品对的匹配分数
predictions = compute_pair_scores(model, valid_df, batch_size=512)
print(f"\nPredictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:10]}")

print("")
print("DSSM Example Complete!")
print("")
