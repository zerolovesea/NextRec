"""
批量运行所有序列模型示例

文件说明:
    本示例演示如何批量训练和测试 NextRec 框架支持的所有序列推荐模型。
    通过统一的训练接口和合成数据,可以快速验证各个序列模型的功能和基础可用性。

主要功能:
    - 生成合成的序列推荐任务数据
    - 批量训练多个序列模型
    - 统一的模型训练和评估流程
    - 收集训练结果和错误信息

支持的模型:
    1. SASRec: 自注意力序列推荐模型
    2. Tiger: 基于 SASRec 接口的 Tiger 占位实现
    3. HSTU: Meta Generative Recommenders 的 HSTU 编码器
    4. GRU4Rec: 基于 GRU 的经典会话推荐模型

使用方法:
    直接运行此脚本:
        python tutorials/run_all_sequential_models.py

数据要求:
    使用合成数据,不需要外部数据文件。脚本会自动生成:
        - 用户历史行为序列
        - 自回归 next-item 标签序列

输出:
    - 各模型的训练日志
    - 评估指标(HitRate@10、NDCG@10、MRR@10)
    - 训练成功/失败统计
    - 失败模型列表

作者: Yang Zhou, zyaztec@gmail.com
创建日期: 2026-04-21
最后更新: 2026-04-21
"""

from __future__ import annotations

from nextrec.models.sequential.hstu import HSTU
from nextrec.models.sequential.sasrec import SASRec
from nextrec.models.sequential.tiger import Tiger
from nextrec.utils.data import generate_sequential_data


def train_model(
    model_class,
    model_name,
    sequence_features,
    train_df,
    valid_df,
    device="cpu",
    **kwargs,
):
    """
    训练单个序列模型。

    返回:
        success: 是否训练成功
        metrics: 评估指标字典
    """
    print("=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    try:
        model = model_class(
            sequence_features=sequence_features,
            item_history_name="item_history",
            target=["next_item"],
            task="generative",
            device=device,
            session_id=f"sequential_{model_name.lower()}_tutorial",
            **kwargs,
        )

        model.compile(
            optimizer="adam",
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            loss="ce",
        )

        model.fit(
            train_data=train_df,
            valid_data=valid_df,
            metrics=["hitrate@10", "ndcg@10", "mrr@10"],
            epochs=1,
            batch_size=256,
            shuffle=True,
            use_tensorboard=False,
        )

        metrics = model.evaluate(
            valid_df,
            metrics=["hitrate@10", "ndcg@10", "mrr@10"],
            batch_size=256,
        )

        print(f"{model_name} completed successfully")
        return True, metrics

    except Exception as e:
        print(f"{model_name} failed with error: {str(e)}")
        return False, None


def main():
    """主函数: 批量运行所有序列模型。"""
    print("-" * 80)
    print("Training all supported sequential models with synthetic data")

    device = "cpu"

    df, sequence_features = generate_sequential_data(
        n_samples=4000,
        item_vocab_size=200,
        sequence_max_len=20,
        embedding_dim=16,
        seed=42,
    )

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    valid_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")

    results = {}
    models_to_train = [
        (
            SASRec,
            "SASRec",
            {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 2,
                "ff_hidden_dim": 64,
                "max_seq_len": 20,
                "dropout": 0.1,
            },
        ),
        (
            Tiger,
            "Tiger",
            {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 2,
                "ff_hidden_dim": 64,
                "max_seq_len": 20,
                "dropout": 0.1,
            },
        ),
        (
            HSTU,
            "HSTU",
            {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "ff_hidden_dim": 64,
                "max_seq_len": 20,
                "dropout_rate": 0.1,
                "use_rab_pos": True,
                "use_temporal_bias": False,
                "tie_embeddings": True,
            },
        )
    ]

    successful = 0
    failed = 0
    failed_models = []

    for model_class, model_name, extra_params in models_to_train:
        success, metrics = train_model(
            model_class=model_class,
            model_name=model_name,
            sequence_features=sequence_features,
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

    print("Test Summary")
    print(f"Total models: {len(models_to_train)}")
    print(f"Successful counts: {successful}")
    print(f"Failed counts: {failed}, Models: {failed_models}")


if __name__ == "__main__":
    main()
