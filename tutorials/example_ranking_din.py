"""
深度兴趣网络(DIN)模型示例

文件说明:
    本示例演示如何使用 NextRec 框架训练 DIN (Deep Interest Network) 排序模型。
    DIN 模型通过注意力机制对用户历史行为序列进行建模,能够捕捉用户对不同商品的兴趣强度,
    在点击率预估等排序任务中表现优异。

主要功能:
    - 数据加载与预处理
    - 特征定义(稠密特征、稀疏特征、序列特征)
    - 行为序列与候选物品的注意力建模
    - DIN 模型构建与训练
    - 模型评估与预测

使用方法:
    直接运行此脚本:
        python tutorials/example_ranking_din.py

测试数据格式:
    - user_id: 用户ID
    - item_id: 候选物品ID
    - label: 标签 (1表示点击, 0表示未点击)
    - dense_*: 稠密特征
    - sparse_*: 稀疏特征
    - sequence_*: 序列特征(用户历史行为序列,字符串格式的列表)

模型架构:
    使用 DIN 模型:
        - 注意力机制: 计算历史行为序列中每个物品与候选物品的相关性
        - 加权池化: 根据注意力权重对历史行为进行加权求和
        - MLP 预测: 将特征和加权后的行为表示输入 MLP 进行点击率预估

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
from nextrec.models.ranking.din import DIN
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

# ==============================================================================
# 1. 数据加载和预处理
# ==============================================================================

# 加载排序任务数据集
df = pd.read_csv("dataset/ranking_task.csv")

# 将序列特征从字符串格式转换为列表格式
# 数据集中序列特征以字符串形式存储,需要使用 eval 转换为 Python 列表
for col in df.columns:
    if "sequence" in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

# ==============================================================================
# 2. 数据集划分
# ==============================================================================

# 划分训练集和验证集(80% 训练, 20% 验证)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# ==============================================================================
# 3. 特征定义
# ==============================================================================

# 定义稠密特征(8个)
dense_features = [DenseFeature(name=f"dense_{i}", input_dim=1) for i in range(8)]

# 定义稀疏特征
# user_id 和 item_id 使用较大的 embedding 维度(32)
sparse_features = [
    SparseFeature(
        name="user_id",
        embedding_name="user_emb",  # embedding 名称,用于权重共享
        vocab_size=int(df["user_id"].max() + 1),
        embedding_dim=32,
    ),
    SparseFeature(
        name="item_id",
        embedding_name="item_emb",  # 与序列特征共享 embedding
        vocab_size=int(df["item_id"].max() + 1),
        embedding_dim=32,
    ),
]

# 添加其他稀疏特征(10个)
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

# 定义序列特征
# sequence_0: 用户历史浏览物品序列,与 item_id 共享 embedding (item_emb)
# sequence_1: 用户历史行为序列,与 sparse_0 共享 embedding (sparse_0_emb)
sequence_features = [
    SequenceFeature(
        name="sequence_0",
        vocab_size=int(df["sequence_0"].apply(lambda x: max(x)).max() + 1),
        embedding_dim=32,
        padding_idx=0,  # 填充索引
        embedding_name="item_emb",  # 与 item_id 共享 embedding
    ),
    SequenceFeature(
        name="sequence_1",
        vocab_size=int(df["sequence_1"].apply(lambda x: max(x)).max() + 1),
        embedding_dim=16,
        padding_idx=0,
        embedding_name="sparse_0_emb",  # 与 sparse_0 共享 embedding
    ),
]

# ==============================================================================
# 4. 模型构建
# ==============================================================================

# 定义 MLP 参数(深度神经网络部分)
mlp_params = {
    "hidden_dims": [256, 128, 64],  # 隐藏层维度
    "activation": "relu",  # 激活函数
    "dropout": 0.3,  # Dropout 比例
}

# 创建 DIN 模型
model = DIN(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    behavior_feature_name="sequence_0",  # 行为序列特征名称(用户历史浏览物品)
    candidate_feature_name="item_id",  # 候选物品特征名称(当前物品)
    mlp_params=mlp_params,  # MLP 参数
    attention_mlp_params={  # 注意力网络参数
        "hidden_dims": [80, 40],  # 注意力 MLP 的隐藏层维度
        "activation": "dice",  # 使用 DICE 激活函数
        "dropout": 0.2,
    },
    attention_use_softmax=True,  # 是否使用 softmax 归一化注意力权重
    target=["label"],  # 目标列名
    device="cpu",
    session_id="din_tutorial",
)

# ==============================================================================
# 5. 模型编译
# ==============================================================================

# 编译模型:配置优化器、学习率调度器和损失函数
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},  # Adam 优化器参数
    scheduler="step",  # 使用 StepLR 学习率调度器
    scheduler_params={"step_size": 3, "gamma": 0.5},  # 每3轮学习率衰减为原来的0.5倍
    loss="focal",  # 使用 Focal Loss 缓解类别不平衡
    loss_params={"gamma": 2.0, "alpha": 0.25},  # Focal Loss 参数
)

# ==============================================================================
# 6. 模型训练
# ==============================================================================

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "gauc", "logloss"],  # 评估指标: AUC、GAUC(分组AUC)、对数损失
    epochs=1,  # 训练轮数
    batch_size=512,  # 批次大小
    shuffle=True,  # 是否打乱训练数据
    user_id_column="user_id",  # 指定用户ID列,用于计算 GAUC
)

print("Training Complete!")

# ==============================================================================
# 7. 模型预测
# ==============================================================================

print("Prediction")

# 对验证集进行预测
predictions = model.predict(valid_df, batch_size=512, return_dataframe=True)

print(f"Prediction shape: {predictions.shape}")
print(f"Prediction sample: {predictions[:10]}")

# ==============================================================================
# 8. 模型评估
# ==============================================================================

# 对验证集进行评估
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "gauc", "logloss"],
    batch_size=512,
    user_id_column="user_id",
)

# 打印评估指标
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print("")
print("DIN Example Complete!")
print("")
