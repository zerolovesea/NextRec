"""
树模型示例 - XGBoost/CatBoost/LightGBM

文件说明:
    本示例演示如何使用 NextRec 框架训练树模型进行推荐任务。NextRec 集成了常用的树模型库
    (XGBoost、CatBoost、LightGBM),支持稠密特征和稀疏特征的自动处理,以及模型的保存与加载。

主要功能:
    - 数据加载与预处理
    - 特征编码(哈希编码、标签编码)
    - 特征定义(稠密特征、稀疏特征)
    - 树模型训练(XGBoost/CatBoost/LightGBM)
    - 模型评估与预测
    - 模型保存与加载

使用方法:
    直接运行此脚本:
        python tutorials/example_tree.py

    前置条件:
        - 安装树模型库: pip install xgboost catboost lightgbm

测试数据格式:
    - user_id: 用户ID
    - item_id: 物品ID
    - age: 用户年龄
    - gender: 用户性别
    - occupation: 用户职业
    - movie_title: 电影标题
    - label: 标签 (1表示喜欢, 0表示不喜欢)

模型说明:
    1. XGBoost: 基于梯度提升决策树,适用于各种回归和分类任务
    2. CatBoost: Yandex 开发的梯度提升库,对类别特征有更好的支持
    3. LightGBM: 微软开发的高效梯度提升框架,训练速度快

输出:
    - 训练好的模型
    - 预测结果
    - 评估指标(AUC、KS、Recall)
    - 保存的模型文件

作者: NextRec Team
创建日期: 2026
最后更新: 2026-01-28
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.data.preprocessor import DataProcessor
from nextrec.models.tree_base.xgboost import Xgboost

# ==============================================================================
# 1. 数据加载和预处理
# ==============================================================================

# 加载 MovieLens 100K 数据集
df = pd.read_csv("dataset/movielens_100k.csv")

# 创建数据预处理器
processor = DataProcessor()

# 对字符串类别特征进行标签编码
processor.add_sparse_feature("gender", encode_method="label")
processor.add_sparse_feature("occupation", encode_method="label")

# 对 movie_title 进行哈希编码,将文本特征转换为固定维度的数值特征
# 哈希编码可以避免词汇表过大的问题
processor.add_sparse_feature("movie_title", encode_method="hash", hash_size=1000)

# 在数据上拟合处理器,学习编码映射关系
processor.fit(df)
vocab_sizes = processor.get_vocab_sizes()

# 对数据进行转换,直接使用新增的处理后列
df_transformed = processor.transform(df, return_dict=False).to_pandas()
selected_cols = [
    "label",
    "user_id",
    "item_id",
    "age",
    "gender_label",
    "occupation_label",
    "movie_title_hash",
]
df = df_transformed[selected_cols]

# 保存处理器,以便后续使用
processor.save(save_path="./")

print("Sample training data:")
print(df.head())

# ==============================================================================
# 2. 数据集划分
# ==============================================================================

# 划分训练集和验证集(80% 训练, 20% 验证)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# ==============================================================================
# 3. 特征定义
# ==============================================================================

# 定义稠密特征
dense_features = [DenseFeature("age")]  # 年龄是数值型特征

# 定义稀疏特征
# 稀疏特征需要指定词汇表大小(vocab_size)
sparse_features = [
    SparseFeature("user_id", vocab_size=int(df["user_id"].max()) + 1),
    SparseFeature("item_id", vocab_size=int(df["item_id"].max()) + 1),
    SparseFeature("gender_label", vocab_size=vocab_sizes["gender_label"]),
    SparseFeature("occupation_label", vocab_size=vocab_sizes["occupation_label"]),
    SparseFeature("movie_title_hash", vocab_size=vocab_sizes["movie_title_hash"]),  # 哈希编码后的固定大小
]

# ==============================================================================
# 4. 模型构建 - XGBoost
# ==============================================================================

# 创建 XGBoost 模型
model = Xgboost(
    dense_features=dense_features,
    sparse_features=sparse_features,
    target="label",  # 目标列名
    session_id="movielens_xgboost_tutorial",
    model_params={  # XGBoost 模型参数
        "max_depth": 6,  # 树的最大深度
        "learning_rate": 0.1,  # 学习率
        "subsample": 0.9,  # 样本采样比例
        "colsample_bytree": 0.9,  # 特征采样比例
        "eval_metric": "auc",  # 评估指标
    },
)

# ==============================================================================
# 4. 模型构建 - CatBoost (可选)
# ==============================================================================

# 如需使用 CatBoost,取消下面代码的注释
# model = Catboost(
#     dense_features=dense_features,
#     sparse_features=sparse_features,
#     target="label",
#     session_id="movielens_catboost_tutorial",
#     model_params={
#         "depth": 6,  # 树的深度
#         "learning_rate": 0.1,  # 学习率
#         "eval_metric": "AUC",  # 评估指标
#         "verbose": False,  # 是否打印训练日志
#     },
# )

# ==============================================================================
# 4. 模型构建 - LightGBM (可选)
# ==============================================================================

# 如需使用 LightGBM,取消下面代码的注释
# model = Lightgbm(
#     dense_features=dense_features,
#     sparse_features=sparse_features,
#     target="label",
#     session_id="movielens_lightgbm_tutorial",
#     model_params={
#         "max_depth": 6,  # 树的最大深度
#         "learning_rate": 0.1,  # 学习率
#         "subsample": 0.9,  # 样本采样比例
#         "colsample_bytree": 0.9,  # 特征采样比例
#         "metric": "auc",  # 评估指标
#     },
# )

# ==============================================================================
# 5. 模型训练
# ==============================================================================

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "ks", "recall"],  # 评估指标: AUC、KS统计量、召回率
    epochs=200,  # 训练轮数(树的数量)
)

# ==============================================================================
# 6. 模型预测
# ==============================================================================

# 对验证集进行预测
predictions = model.predict(valid_df, batch_size=512)
print(f"\nPredictions shape: {predictions.shape}")
print(f"Sample predictions:\n{predictions.head(10)}")

# ==============================================================================
# 7. 模型保存与加载
# ==============================================================================

# 保存模型
model_path = model.save_model()
print(f"\nModel saved to: {model_path}")

# 加载模型
# 注意:需要使用相同的模型类和 session_id
loaded_model = Xgboost(session_id="movielens_tree_tutorial")
# loaded_model = Catboost(session_id="movielens_tree_tutorial")  # CatBoost
# loaded_model = Lightgbm(session_id="movielens_tree_tutorial")  # LightGBM

loaded_model.load_model(model_path)

# 使用加载的模型进行预测
loaded_predictions = loaded_model.predict(valid_df, batch_size=512)
print(f"\nLoaded model predictions shape: {loaded_predictions.shape}")
print(f"Loaded model sample predictions:\n{loaded_predictions.head(10)}")

print("")
print("Tree Example Complete!")
print("")
