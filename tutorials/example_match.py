"""
召回模型示例 - DSSM 双塔模型训练

本示例演示如何使用 NextRec 框架训练一个召回(匹配)模型,使用 DSSM (Deep Structured Semantic Model)
双塔架构进行用户-物品匹配。示例包含完整的数据预处理、特征工程、模型训练和评估流程。

主要功能:
    - 数据加载与预处理
    - 特征定义(稠密特征、稀疏特征、序列特征)
    - 数据编码与转换
    - DSSM 模型构建与训练
    - 召回候选生成与评分

使用方法:
    直接运行此脚本:
        python tutorials/example_match.py

测试数据格式:
    - user_id: 用户ID
    - item_id: 物品ID
    - label: 标签 (1表示正样本, 0表示负样本)
    - user_dense_*: 用户稠密特征
    - user_sparse_*: 用户稀疏特征
    - user_sequence_*: 用户序列特征
    - item_dense_*: 物品稠密特征
    - item_sparse_*: 物品稀疏特征

模型架构:
    使用 DSSM 双塔模型:
        - 用户塔: 编码用户特征得到用户向量
        - 物品塔: 编码物品特征得到物品向量
        - 训练模式: pairwise (成对训练)
        - 损失函数: BPR (Bayesian Personalized Ranking)

输出:
    - 训练好的模型
    - Top-K 召回结果
    - 评估指标

作者: NextRec Team
创建日期: 2026
最后更新: 2026-01-28
"""

import ast

import pandas as pd
from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.data.data_processing import build_eval_candidates
from nextrec.data.dataloader import RecDataLoader
from nextrec.data.preprocessor import DataProcessor
from nextrec.models.retrieval.dssm import DSSM
from nextrec.utils.model import compute_pair_scores

# ==============================================================================
# 1. 数据加载和特征定义
# ==============================================================================

# 加载数据集
df = pd.read_csv("dataset/match_task.csv")
# 将字符串形式的序列特征转换为列表
df["user_sequence_0"] = df["user_sequence_0"].apply(ast.literal_eval)

# 定义特征列名
user_dense_cols = ["user_dense_0", "user_dense_1", "user_dense_2"]  # 用户稠密特征
user_sparse_cols = [
    "user_id",
    "user_sparse_0",
    "user_sparse_1",
    "user_sparse_2",
    "user_sparse_3",
    "user_sparse_4",
]  # 用户稀疏特征
user_sequence_cols = ["user_sequence_0"]  # 用户序列特征

item_dense_cols = ["item_dense_0", "item_dense_1"]  # 物品稠密特征
item_sparse_cols = [
    "item_id",
    "item_sparse_0",
    "item_sparse_1",
    "item_sparse_2",
    "item_sparse_3",
]  # 物品稀疏特征

# ==============================================================================
# 2. 数据预处理
# ==============================================================================

# 创建数据处理器并配置特征编码方式
processor = DataProcessor()
# 稀疏特征使用 label encoding
for col in user_sparse_cols + item_sparse_cols:
    processor.add_sparse_feature(col, encode_method="label")
# 稠密特征使用 minmax 归一化
for col in user_dense_cols + item_dense_cols:
    processor.add_numeric_feature(col, scaler="minmax")
# 序列特征使用 label encoding,并设置最大长度为 20
for col in user_sequence_cols:
    processor.add_sequence_feature(col, encode_method="label", max_len=20)
# 在数据上拟合处理器,学习编码映射关系
processor.fit(df)

# 获取各特征的词汇表大小
vocab_sizes = processor.get_vocab_sizes()

# 定义用户侧特征
user_dense_features = [DenseFeature(col) for col in user_dense_cols]
# 用户稀疏特征,除 user_id 外使用较小的 embedding 维度(4)
user_sparse_features = [
    SparseFeature(col, vocab_size=vocab_sizes[col], embedding_dim=4) for col in user_sparse_cols if col != "user_id"
]
# user_id 使用较大的 embedding 维度(32)
user_sparse_features.append(SparseFeature("user_id", vocab_size=vocab_sizes["user_id"], embedding_dim=32))
# 用户序列特征,设置最大长度和 padding 索引
user_sequence_features = [
    SequenceFeature(
        "user_sequence_0",
        vocab_size=vocab_sizes["user_sequence_0"],
        max_len=20,
        embedding_dim=8,
        padding_idx=0,
    )
]

# 定义物品侧特征
item_dense_features = [DenseFeature(col) for col in item_dense_cols]
# 物品稀疏特征,除 item_id 外使用较小的 embedding 维度(4)
item_sparse_features = [
    SparseFeature(col, vocab_size=vocab_sizes[col], embedding_dim=4) for col in item_sparse_cols if col != "item_id"
]
# item_id 使用较大的 embedding 维度(32)
item_sparse_features.append(SparseFeature("item_id", vocab_size=vocab_sizes["item_id"], embedding_dim=32))

# 创建DataLoader
rec_loader = RecDataLoader(
    dense_features=user_dense_features + item_dense_features,
    sparse_features=user_sparse_features + item_sparse_features,
    sequence_features=user_sequence_features,
    target="label",  # 目标列名
    id_columns=["user_id"],  # 用户ID列,用于分组评估
    processor=processor,  # 数据处理器
)

# ==============================================================================
# 3. 数据集划分
# ==============================================================================

# 使用 sklearn 的 train_test_split 划分训练集和验证集
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2025)

# Pairwise 训练模式仅使用正样本(in-batch negative sampling)
train_df = train_df[train_df["label"] == 1].reset_index(drop=True)

# 创建训练数据加载器
train_loader = rec_loader.create_dataloader(train_df, batch_size=256, shuffle=True, num_workers=0)
# 创建验证数据加载器
valid_loader = rec_loader.create_dataloader(valid_df, batch_size=256, shuffle=False, num_workers=0)

# ==============================================================================
# 4. 数据编码转换(用于后续评估)
# ==============================================================================

# 对完整数据集进行编码转换
df_encoded = processor.transform(df, return_dict=False)
# 对训练集和验证集进行编码转换
train_df_encoded = processor.transform(train_df, return_dict=False)
valid_df_encoded = processor.transform(valid_df, return_dict=False)

# ==============================================================================
# 5. 模型构建与训练
# ==============================================================================

# 创建 DSSM 双塔模型
model = DSSM(
    user_dense_features=user_dense_features,
    user_sparse_features=user_sparse_features,
    user_sequence_features=user_sequence_features,
    item_dense_features=item_dense_features,
    item_sparse_features=item_sparse_features,
    item_sequence_features=[],
    embedding_dim=64,
    temperature=0.05,
    user_mlp_params={"hidden_dims": [256, 128]},
    item_mlp_params={"hidden_dims": [256, 128]},
    training_mode="pairwise",
    device="cpu",
    session_id="match_task_pairwise_tutorial",
)

# 编译模型,使用 BPR 损失函数(适合 pairwise 训练)
model.compile(loss="bpr")
# 训练模型
model.fit(
    train_data=train_loader,  # 训练数据加载器
    valid_data=valid_loader,  # 验证数据加载器
    metrics=["auc"],  # 评估指标
    epochs=1,  # 训练轮数
    batch_size=256,  # 批次大小
    shuffle=True,  # 是否打乱数据
    user_id_column="user_id",  # 用户ID列名,用于分组评估
)

# ==============================================================================
# 6. 模型评估与召回
# ==============================================================================

# 提取唯一的用户特征和物品特征,用于召回评估
user_features = (
    df_encoded[user_dense_cols + user_sparse_cols + user_sequence_cols]
    .drop_duplicates("user_id")  # 去重,每个用户只保留一条记录
    .reset_index(drop=True)
)
item_features = (
    df_encoded[item_dense_cols + item_sparse_cols]
    .drop_duplicates("item_id")  # 去重,每个物品只保留一条记录
    .reset_index(drop=True)
)

# 构建评估候选集
# 为每个用户生成正负样本候选,用于评估召回效果
valid_candidates = build_eval_candidates(
    df_all=valid_df_encoded,  # 验证集
    user_col="user_id",  # 用户列名
    item_col="item_id",  # 物品列名
    label_col="label",  # 标签列名
    user_features=user_features,  # 用户特征表
    item_features=item_features,  # 物品特征表
    num_pos_per_user=5,  # 每个用户保留的正样本数
    num_neg_per_pos=50,  # 每个正样本对应的负样本数
)

# 计算用户-物品对的匹配分数
scores = compute_pair_scores(model, valid_candidates, batch_size=512)
valid_candidates = valid_candidates.assign(score=scores)

# 为每个用户选取 Top-10 召回结果
topk = (
    valid_candidates.sort_values(["user_id", "score"], ascending=[True, False])  # 按分数降序排序
    .groupby("user_id")  # 按用户分组
    .head(10)  # 每组取前10条
)
# 打印部分召回结果
print(topk[["user_id", "item_id", "label", "score"]].head(20))

print("召回模型示例运行完成!")
