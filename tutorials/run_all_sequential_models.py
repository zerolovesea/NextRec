"""
批量运行所有序列推荐模型示例

文件说明:
    本示例演示如何批量训练和测试 NextRec 框架支持的所有序列推荐模型。
    通过统一的训练接口和示例数据,可以快速验证各个 sequential 模型的功能和可用性。

主要功能:
    - 加载框架内置的序列推荐数据
    - 批量训练多个序列推荐模型
    - 统一的模型训练和评估流程
    - 收集训练结果和错误信息

支持的模型:
    1. SASRec: 自注意力序列推荐模型
    2. BERT4Rec: 双向 Transformer 掩码序列推荐模型
    3. GRU4Rec: 基于 GRU 的序列推荐模型
    4. CL4SRec: 对比学习增强的 SASRec
    5. S3Rec: 自监督序列推荐预训练模型
    6. HSTU: Meta Generative Recommenders 中的 HSTU 编码器

使用方法:
    直接运行此脚本:
        python tutorials/run_all_sequential_models.py

数据要求:
    使用仓库内置的示例数据,不需要外部数据文件。脚本会自动加载:
        - 用户ID
        - 物品历史行为序列
        - next-item 标签序列

输出:
    - 各模型的训练日志
    - 评估指标(HitRate@5、NDCG@5)
    - 训练成功/失败统计
    - 失败模型列表

作者: Yang Zhou, zyaztec@gmail.com
创建日期: 2026-03-31
最后更新: 2026-03-31
"""

from __future__ import annotations

import ast

import pandas as pd

from nextrec.basic.features import SequenceFeature
from nextrec.models.sequential.bert4rec import BERT4Rec
from nextrec.models.sequential.cl4srec import CL4SRec
from nextrec.models.sequential.gru4rec import GRU4Rec
from nextrec.models.sequential.hstu import HSTU
from nextrec.models.sequential.s3rec import S3Rec
from nextrec.models.sequential.sasrec import SASRec


def parse_list_column(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)


def build_next_item_labels(sequence: list[int], padding_idx: int = 0) -> list[int]:
    """
    根据 item_history 构造 next-item 标签序列
    """
    if not sequence:
        return []
    return list(sequence[1:]) + [padding_idx]


def load_sequential_data():
    """
    加载序列推荐示例数据并构造特征定义
    """
    df = pd.read_csv("dataset/sasrec_task.csv")
    df["item_history"] = parse_list_column(df["item_history"])
    df["next_item"] = df["item_history"].apply(build_next_item_labels)

    max_item_id = max(max(seq) for seq in df["item_history"])
    max_seq_len = len(df.iloc[0]["item_history"])

    sequence_features = [
        SequenceFeature(
            name="item_history",
            vocab_size=max_item_id + 1,
            max_len=max_seq_len,
            embedding_dim=16,
            padding_idx=0,
        )
    ]

    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    valid_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, valid_df, sequence_features, max_seq_len


def train_model(
    model_class,
    model_name,
    sequence_features,
    train_df,
    valid_df,
    max_seq_len,
    device="cpu",
    **kwargs,
):
    """
    训练单个序列推荐模型

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
            sequence_features=sequence_features,
            item_history_name="item_history",
            max_seq_len=max_seq_len,
            target=["next_item"],
            task="sequential",
            id_columns=["user_id"],
            device=device,
            session_id=f"sequential_{model_name.lower()}_tutorial",
            **kwargs,
        )

        # ==============================================================================
        # 2. 编译模型
        # ==============================================================================

        model.compile(
            optimizer="adam",
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            loss="ce",
        )

        # ==============================================================================
        # 3. 训练模型
        # ==============================================================================

        model.fit(
            train_data=train_df,
            valid_data=valid_df,
            metrics=["hitrate@5", "ndcg@5"],
            epochs=1,
            batch_size=2,
            shuffle=True,
            use_tensorboard=False,
            early_stop_patience=0,
            num_workers=0,
            user_id_column="user_id",
        )

        # ==============================================================================
        # 4. 评估模型
        # ==============================================================================

        metrics = model.evaluate(
            valid_df,
            metrics=["hitrate@5", "ndcg@5"],
            batch_size=2,
            num_workers=0,
        )

        print(f"{model_name} completed successfully")
        return True, metrics

    except Exception as e:
        print(f"{model_name} failed with error: {str(e)}")
        return False, None


def main():
    """
    主函数: 批量运行所有序列推荐模型
    """
    print("=" * 80)
    print("Training all supported sequential models with sample data")
    print("=" * 80)

    device = "cpu"

    # ==============================================================================
    # 1. 加载示例数据
    # ==============================================================================

    train_df, valid_df, sequence_features, max_seq_len = load_sequential_data()
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")

    # ==============================================================================
    # 2. 定义模型参数
    # ==============================================================================

    results = {}

    # ==============================================================================
    # 3. 定义要训练的模型列表
    # ==============================================================================

    models_to_train = [
        (
            SASRec,
            "SASRec",
            {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "dropout_rate": 0.0,
                "sequence_mode": "autoregressive",
            },
        ),
        (
            BERT4Rec,
            "BERT4Rec",
            {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "dropout_rate": 0.0,
                "mask_ratio": 0.4,
            },
        ),
        (
            GRU4Rec,
            "GRU4Rec",
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout_rate": 0.0,
            },
        ),
        (
            CL4SRec,
            "CL4SRec",
            {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "dropout_rate": 0.0,
                "cl_weight": 0.1,
                "temperature": 0.2,
            },
        ),
        (
            S3Rec,
            "S3Rec",
            {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "dropout_rate": 0.0,
                "mask_ratio": 0.4,
                "mip_weight": 1.0,
                "sp_weight": 0.1,
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
                "dropout_rate": 0.0,
                "use_rab_pos": True,
                "use_temporal_bias": False,
                "tie_embeddings": True,
            },
        ),
    ]

    successful = 0
    failed = 0
    failed_models = []

    # ==============================================================================
    # 4. 批量训练模型
    # ==============================================================================

    for model_class, model_name, extra_params in models_to_train:
        success, metrics = train_model(
            model_class=model_class,
            model_name=model_name,
            sequence_features=sequence_features,
            train_df=train_df,
            valid_df=valid_df,
            max_seq_len=max_seq_len,
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
