"""
MovieLens 数据集排序模型示例 - DeepFM

文件说明:
    本示例演示如何使用真实的 MovieLens 100K 数据集训练 DeepFM 排序模型。
    DeepFM 结合了因子分解机(FM)和深度神经网络(DNN),能够同时捕捉低阶和高阶特征交互,
    在点击率预估等排序任务中表现优异。

主要功能:
    - MovieLens 数据加载与预处理
    - 特征编码(哈希编码)
    - 特征定义(稠密特征、稀疏特征)
    - DeepFM 模型训练
    - 模型评估与预测
    - 集成实验跟踪(SwanLab)

使用方法:
    直接运行此脚本:
        python tutorials/movielen_ranking_deepfm.py

    前置条件:
        - 确保 dataset/movielens_100k.csv 文件存在
        - 安装所有必要的依赖包
        - 可选: 安装 SwanLab 用于实验跟踪

测试数据格式:
    - user_id: 用户ID
    - item_id: 电影ID
    - age: 用户年龄
    - gender: 用户性别
    - occupation: 用户职业
    - movie_title: 电影标题
    - label: 标签 (1表示用户喜欢该电影)

模型架构:
    使用 DeepFM 模型:
        - FM 部分: 建模一阶和二阶特征交互
        - DNN 部分: 建模高阶特征交互
        - 输出: FM 和 DNN 的输出相加得到最终预测

输出:
    - 训练好的模型
    - 预测结果
    - 评估指标(AUC、Recall、Precision、KS)
    - SwanLab 实验记录(如果启用)

作者: NextRec Team
创建日期: 2026
最后更新: 2026-01-28
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from nextrec.data.preprocessor import DataProcessor
from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM

# ==============================================================================
# 1. 数据加载
# ==============================================================================

# 加载 MovieLens 100K 数据集
df = pd.read_csv("dataset/movielens_100k.csv")

# ==============================================================================
# 2. 数据预处理
# ==============================================================================

# 创建数据预处理器
processor = DataProcessor()

# 对 movie_title 进行哈希编码
# 哈希编码可以将高基数类别特征映射到固定大小的空间
processor.add_sparse_feature("movie_title", encode_method="hash", hash_size=1000)

# 在数据上拟合处理器,学习编码映射关系
processor.fit(df)

# 对数据进行转换
df = processor.transform(df, return_dict=False)

# 保存处理器,以便后续使用
processor.save(save_path="./")

print("Sample training data:")
print(df.head())

# ==============================================================================
# 3. 数据集划分
# ==============================================================================

# 划分训练集和验证集(80% 训练, 20% 验证)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# ==============================================================================
# 4. 特征定义
# ==============================================================================

# 定义稠密特征
dense_features = [DenseFeature("age")]  # 年龄是数值型特征

# 定义稀疏特征
# 所有稀疏特征使用相同的 embedding 维度(4)
sparse_features = [
    SparseFeature("user_id", vocab_size=df["user_id"].max() + 1, embedding_dim=4),
    SparseFeature("item_id", vocab_size=df["item_id"].max() + 1, embedding_dim=4),
]

# 添加其他类别特征
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

# ==============================================================================
# 5. 模型构建
# ==============================================================================

# 创建 DeepFM 模型
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={  # DNN 部分的 MLP 参数
        "hidden_dims": [256, 128],  # 隐藏层维度
        "activation": "relu",  # 激活函数
        "dropout": 0.2,  # Dropout 比例
    },
    target="label",  # 目标列名
    device="cpu",
    session_id="movielens_deepfm_tutorial",
)

# ==============================================================================
# 6. 模型编译
# ==============================================================================

# 编译模型:配置优化器和损失函数
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},  # Adam 优化器参数
    loss="binary_crossentropy",  # 二元交叉熵损失
)

# ==============================================================================
# 7. 模型训练
# ==============================================================================

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "recall", "precision", "ks"],  # 评估指标
    epochs=15,  # 训练轮数
    batch_size=512,  # 批次大小
    shuffle=True,  # 是否打乱训练数据
    # use_swanlab=True,  # 使用 SwanLab 进行实验跟踪
    # swanlab_api="your_swanlab_api_key",  # 替换为
    # swanlab_kwargs={
    #     "project": "NextRec",
    #     "name": "tutorial_movielens_deepfm",
    # },  # SwanLab 配置
)

# ==============================================================================
# 8. 模型预测
# ==============================================================================

# 对验证集进行预测
predictions = model.predict(valid_df, batch_size=512)
print(f"\nPredictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:10]}")

print("")
print("DeepFM Example Complete!")
print("")
