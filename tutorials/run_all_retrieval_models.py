"""
批量运行所有召回模型示例

文件说明:
    本示例演示如何批量训练和测试 NextRec 框架支持的所有召回(匹配)模型。
    通过统一的训练接口和合成数据,可以快速验证各个召回模型的功能和性能。

主要功能:
    - 生成合成的召回任务数据
    - 批量训练多个召回模型(DSSM、YoutubeDNN、MIND)
    - 统一的模型训练和评估流程
    - 收集训练结果和错误信息

支持的模型:
    1. DSSM (Deep Structured Semantic Model): 双塔模型,使用余弦相似度
    2. YoutubeDNN: YouTube 推荐系统使用的深度召回模型,支持负采样
    3. MIND (Multi-Interest Network with Dynamic Routing): 多兴趣网络,使用胶囊网络建模用户多样化兴趣

使用方法:
    直接运行此脚本:
        python tutorials/run_all_retrieval_models.py

数据要求:
    使用合成数据,不需要外部数据文件。脚本会自动生成:
        - 用户特征(稠密、稀疏、序列)
        - 物品特征(稠密、稀疏、序列)
        - 用户-物品交互标签

输出:
    - 各模型的训练日志
    - 评估指标
    - 训练成功/失败统计
    - 失败模型列表

作者: Yang Zhou, zyaztec@gmail.com
创建日期: 2025-12-06
最后更新: 2026-01-28
"""

from nextrec.models.retrieval.dssm import DSSM
from nextrec.models.retrieval.youtube_dnn import YoutubeDNN
from nextrec.models.retrieval.mind import MIND

from nextrec.utils.model import compute_pair_scores
from nextrec.utils.data import generate_match_data


def train_model(
    model_class,
    model_name,
    user_dense_features,
    user_sparse_features,
    user_sequence_features,
    item_dense_features,
    item_sparse_features,
    item_sequence_features,
    train_df,
    valid_df,
    device="cpu",
    **kwargs,
):
    """
    训练单个召回模型

    参数:
        model_class: 模型类
        model_name: 模型名称(用于日志输出)
        user_dense_features: 用户稠密特征列表
        user_sparse_features: 用户稀疏特征列表
        user_sequence_features: 用户序列特征列表
        item_dense_features: 物品稠密特征列表
        item_sparse_features: 物品稀疏特征列表
        item_sequence_features: 物品序列特征列表
        train_df: 训练数据
        valid_df: 验证数据
        device: 设备(cpu/cuda)
        **kwargs: 模型特定参数

    返回:
        success: 是否训练成功
        metrics: 评估指标字典
    """
    print("=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    try:
        # ==============================================================================
        # 1. 创建模型
        # ==============================================================================

        model = model_class(
            user_dense_features=user_dense_features,
            user_sparse_features=user_sparse_features,
            user_sequence_features=user_sequence_features,
            item_dense_features=item_dense_features,
            item_sparse_features=item_sparse_features,
            item_sequence_features=item_sequence_features,
            device=device,
            session_id=f"match_{model_name.lower()}_tutorial",
            **kwargs,
        )

        # ==============================================================================
        # 2. 编译模型
        # ==============================================================================

        model.compile(
            optimizer="adam",
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        )

        # ==============================================================================
        # 3. 训练模型
        # ==============================================================================

        model.fit(
            train_data=train_df,
            valid_data=valid_df,
            epochs=1,  # 仅训练1轮用于快速验证
            batch_size=512,
            shuffle=True,
            use_tensorboard=False,  # 不使用 TensorBoard
            user_id_column="user_id",
        )

        # ==============================================================================
        # 4. 评估模型
        # ==============================================================================

        metrics = model.evaluate(
            valid_df,
            batch_size=512,
            user_id_column="user_id",
        )

        # ==============================================================================
        # 5. 计算样本分数
        # ==============================================================================

        sample_scores = compute_pair_scores(model, valid_df.head(2048), batch_size=512)
        print(f"{model_name} sample scores: {sample_scores[:5]}")

        print(f"{model_name} completed successfully")
        return True, metrics

    except Exception as e:
        print(f"{model_name} failed with error: {str(e)}")
        return False, None


def main():
    """
    主函数: 批量运行所有召回模型
    """
    print("=" * 80)
    print("Training all supported match models with synthetic data")
    print("=" * 80)

    device = "cpu"

    # ==============================================================================
    # 1. 生成合成数据
    # ==============================================================================

    (
        df,
        user_dense_features,
        user_sparse_features,
        user_sequence_features,
        item_dense_features,
        item_sparse_features,
        item_sequence_features,
    ) = generate_match_data(
        n_samples=10000,  # 样本数量
        user_vocab_size=1000,  # 用户词汇表大小
        item_vocab_size=5000,  # 物品词汇表大小
        category_vocab_size=100,  # 类别词汇表大小
        brand_vocab_size=200,  # 品牌词汇表大小
        city_vocab_size=100,  # 城市词汇表大小
        user_feature_vocab_size=50,  # 用户特征词汇表大小
        item_feature_vocab_size=50,  # 物品特征词汇表大小
        sequence_max_len=50,  # 序列最大长度
        user_embedding_dim=32,  # 用户 embedding 维度
        item_embedding_dim=32,  # 物品 embedding 维度
        seed=42,  # 随机种子
    )

    # ==============================================================================
    # 2. 划分训练集和验证集
    # ==============================================================================

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    valid_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")

    results = {}

    # ==============================================================================
    # 3. 定义要训练的模型列表
    # ==============================================================================

    models_to_train = [
        (
            DSSM,
            "DSSM",
            {
                "user_mlp_params": {  # 用户塔 MLP 参数
                    "hidden_dims": [256, 128, 64],
                    "activation": "relu",
                    "dropout": 0.2,
                },
                "item_mlp_params": {  # 物品塔 MLP 参数
                    "hidden_dims": [256, 128, 64],
                    "activation": "relu",
                    "dropout": 0.2,
                },
                "embedding_dim": 64,  # 用户和物品向量维度
                "similarity_metric": "cosine",  # 相似度度量:余弦相似度
                "training_mode": "pointwise",  # 训练模式:pointwise
            },
        ),
        (
            YoutubeDNN,
            "YoutubeDNN",
            {
                "user_mlp_params": {
                    "hidden_dims": [256, 128, 64],
                    "activation": "relu",
                    "dropout": 0.2,
                },
                "item_mlp_params": {
                    "hidden_dims": [256, 128, 64],
                    "activation": "relu",
                    "dropout": 0.2,
                },
                "embedding_dim": 64,
                "training_mode": "listwise",  # 训练模式:listwise
                "num_negative_samples": 100,  # 负采样数量
            },
        ),
        (
            MIND,
            "MIND",
            {
                "item_mlp_params": {
                    "hidden_dims": [256, 128],
                    "activation": "relu",
                    "dropout": 0.2,
                },
                "embedding_dim": 64,
                "num_interests": 4,  # 用户兴趣数量
                "capsule_bilinear_type": 2,  # 胶囊网络双线性类型
                "routing_times": 3,  # 动态路由迭代次数
                "training_mode": "pointwise",
                "similarity_metric": "dot",  # 相似度度量:内积
            },
        ),
    ]

    # ==============================================================================
    # 4. 批量训练模型
    # ==============================================================================

    successful = 0
    failed = 0
    failed_models = []

    for model_class, model_name, extra_params in models_to_train:
        success, metrics = train_model(
            model_class=model_class,
            model_name=model_name,
            user_dense_features=user_dense_features,
            user_sparse_features=user_sparse_features,
            user_sequence_features=user_sequence_features,
            item_dense_features=item_dense_features,
            item_sparse_features=item_sparse_features,
            item_sequence_features=item_sequence_features,
            train_df=train_df,
            valid_df=valid_df,
            device=device,
            **extra_params,
        )

        if success:
            successful += 1
            results[model_name] = metrics
        else:
            failed += 1
            failed_models.append(model_name)

    # ==============================================================================
    # 5. 打印训练总结
    # ==============================================================================

    print("Test Summary")
    print(f"Total models: {len(models_to_train)}")
    print(f"Successful counts: {successful}")
    print(f"Failed counts: {failed}, Models: {failed_models}")


if __name__ == "__main__":
    main()
