"""
多任务学习模型示例 - ESMM 模型训练

本示例演示如何使用 NextRec 框架训练一个多任务学习模型 ESMM (Entire Space Multi-Task Model)。
ESMM 模型通过联合建模点击率 (CTR) 和转化率 (CVR) 两个任务,解决传统 CVR 建模中的样本选择偏差问题。
示例包含完整的数据预处理、特征工程、模型训练和评估流程。

主要功能:
    - 数据加载与预处理
    - 特征定义(稠密特征、稀疏特征、序列特征)
    - ESMM 多任务模型构建与训练
    - 支持多种损失函数(BCE、Focal Loss)
    - 支持动态损失权重调整(GradNorm)
    - 模型评估与预测

使用方法:
    直接运行此脚本:
        python tutorials/example_multitask.py

测试数据格式:
    - user_id: 用户ID
    - item_id: 物品ID
    - click: 点击标签 (1表示点击, 0表示未点击)
    - conversion: 转化标签 (1表示转化, 0表示未转化)
    - dense_*: 稠密特征
    - sparse_*: 稀疏特征
    - sequence_*: 序列特征(字符串格式的列表)

模型架构:
    使用 ESMM 模型:
        - CTR 塔: 预测点击率
        - CVR 塔: 预测转化率
        - CTCVR: 点击后转化率 = CTR × CVR
        - 通过全空间建模避免样本选择偏差

输出:
    - 训练好的模型
    - 预测结果
    - 评估指标(AUC、GAUC、LogLoss)

作者: NextRec Team
创建日期: 2026
最后更新: 2026-01-28
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from nextrec.models.multitask.esmm import ESMM
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

# ==============================================================================
# 1. 数据加载和预处理
# ==============================================================================

# 加载生成的多任务数据集
df = pd.read_csv("dataset/multitask_task.csv")

# 将序列特征从字符串格式转换为列表格式
# 数据集中序列特征以字符串形式存储,需要使用 eval 转换为 Python 列表
for col in df.columns:
    if "sequence" in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

# 定义任务标签列名
task_labels = ["click", "conversion"]  # 点击和转化两个任务

# ==============================================================================
# 2. 数据集划分
# ==============================================================================

# 划分训练集和验证集(80% 训练, 20% 验证)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# ==============================================================================
# 3. 特征定义
# ==============================================================================

# 统计稠密特征数量并创建特征对象
num_dense = len([col for col in df.columns if col.startswith("dense_")])
dense_features = [DenseFeature(f"dense_{i}") for i in range(num_dense)]

# 创建稀疏特征对象
# user_id 和 item_id 使用较大的 embedding 维度(32)
sparse_features = [
    SparseFeature("user_id", vocab_size=int(df["user_id"].max() + 1), embedding_dim=32),
    SparseFeature("item_id", vocab_size=int(df["item_id"].max() + 1), embedding_dim=32),
]

# 添加其他稀疏特征,使用较小的 embedding 维度(16)
num_sparse = len([col for col in df.columns if col.startswith("sparse_")])
sparse_features.extend(
    [
        SparseFeature(f"sparse_{i}", vocab_size=int(df[f"sparse_{i}"].max() + 1), embedding_dim=16)
        for i in range(num_sparse)
    ]
)

# 创建序列特征对象
# 序列特征用于建模用户行为序列,如浏览历史、购买历史等
sequence_cols = [col for col in df.columns if col.startswith("sequence_")]
sequence_features = [
    SequenceFeature(
        col,
        vocab_size=int(df[col].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1),
        embedding_dim=32,
        padding_idx=0,  # 使用 0 作为填充索引
    )
    for col in sequence_cols
]

# 打印特征统计信息
print(f"Dense features: {len(dense_features)}")
print(f"Sparse features: {len(sparse_features)} (including user_id and item_id)")
print(f"Sequence features: {len(sequence_features)}")

# ==============================================================================
# 4. 模型构建
# ==============================================================================

# 定义 CTR 塔的 MLP 参数
ctr_params = {"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.4}

# 定义 CVR 塔的 MLP 参数
cvr_params = {"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.4}

# 创建 ESMM 模型
model = ESMM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    ctr_mlp_params=ctr_params,  # CTR 塔的 MLP 参数
    cvr_mlp_params=cvr_params,  # CVR 塔的 MLP 参数
    target=task_labels,  # 任务标签
    task=["binary", "binary"],  # 两个二分类任务
    device="cpu",
    session_id="esmm_tutorial",
)

# ==============================================================================
# 5. 模型编译
# ==============================================================================

# 选项1: 使用固定损失权重的二元交叉熵损失
# model.compile(
#     optimizer="adam",
#     optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
#     loss=["bce", "bce"],
#     loss_weights=[0.3, 0.7],  # 手动设置损失权重
# )

# 选项2: 使用 Focal Loss + GradNorm 动态损失权重调整(当前使用)
# Focal Loss 能够缓解类别不平衡问题
# GradNorm 能够自动平衡多任务学习中的梯度
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
    loss=["focal", "focal"],  # 使用 Focal Loss
    loss_params=[
        {"gamma": 2.0, "alpha": 0.25},  # CTR 任务的 Focal Loss 参数
        {"gamma": 1.0, "alpha": 0.75},  # CVR 任务的 Focal Loss 参数
    ],
    loss_weights={"method": "grad_norm", "alpha": 1.5, "lr": 0.025},  # GradNorm 参数
)

# 选项3: 使用二元交叉熵损失 + GradNorm 动态损失权重调整
# model.compile(
#     optimizer="adam",
#     optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
#     loss=["bce", "bce"],
#     loss_weights={"method": "grad_norm", "alpha": 1.5, "lr": 0.025},
# )

# ==============================================================================
# 6. 模型训练
# ==============================================================================

model.fit(
    train_data=train_df,
    # valid_data=valid_df,  # 可选:传入验证集进行验证
    metrics=["auc", "gauc", "logloss"],  # 评估指标: AUC、GAUC(分组AUC)、对数损失
    epochs=2,  # 训练轮数
    batch_size=512,  # 批次大小
    shuffle=True,  # 是否打乱训练数据
    group_id="user_id",  # 指定分组列,用于计算 GAUC
)

print("Training Complete!")

# ==============================================================================
# 7. 模型预测
# ==============================================================================

print("Prediction")

# 对验证集进行预测
predictions = model.predict(valid_df, batch_size=512)
preview = predictions.head(5)

print(f"Prediction shape: {predictions.shape}")
print(f"Prediction sample: {predictions[:10]}")

# ==============================================================================
# 8. 模型评估(可选)
# ==============================================================================

# 对验证集进行评估
# metrics = model.evaluate(
#     valid_df,
#     metrics=["auc", "gauc", "logloss"],
#     batch_size=512,
#     group_id="user_id",
# )
# for name, value in metrics.items():
#     print(f"{name}: {value:.4f}")

# print("")
# print("Multi-task Example Complete!")
# print("")
