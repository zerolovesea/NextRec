---
title: 分布式训练
description: 分布式训练配置与使用说明
---

# 分布式训练

NextRec 封装了基于Pytorch DDP的分布式训练方法，支持单机多卡训练。

## 示例代码

NextRec的Github仓库内提供了一个简单的示例代码，通过`torchrun --nproc_per_node=2 example_distributed_training.py`进行执行验证。

```python
import os
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split

from nextrec.utils.data import generate_distributed_ranking_data
from nextrec.models.ranking.deepfm import DeepFM


def main():
    """
    主函数: 初始化分布式环境并进行训练
    """
    # 通过环境变量判断是否在分布式模式下运行
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_distributed:
        # 获取分布式训练的关键参数
        rank = int(os.environ["RANK"])  # 当前进程的全局排名
        world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前节点内的本地排名

        print(f"[Rank {rank}/{world_size}] Initializing distributed training...")
        print(f"[Rank {rank}/{world_size}] Local rank: {local_rank}")

        # 设置设备
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)  # 设置当前进程使用的 GPU
    else:
        # 非分布式模式
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Warning: Not in distributed mode. Training on single device.")

    # 生成合成数据
    df, dense_features, sparse_features, sequence_features = generate_distributed_ranking_data(
        num_samples=100000,  # 样本总数
        num_users=10000,  # 用户数量
        num_items=5000,  # 物品数量
        num_categories=20,  # 类别数量
        num_cities=100,  # 城市数量
        max_seq_len=50,  # 序列最大长度
        embedding_dim=32,  # embedding 维度
        seed=42,  # 随机种子
    )

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

    if rank == 0:
        print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
        print("Each GPU will process different batches from this dataset")

    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        mlp_params={  # MLP 参数
            "dims": [256, 128, 64],
            "activation": "relu",
            "dropout": 0.3,
        },
        target="label",
        device=device,
        distributed=is_distributed,  # 启用分布式训练
        rank=rank,  # 全局排名
        world_size=world_size,  # 总进程数
        local_rank=local_rank,  # 本地排名
        embedding_l1_reg=1e-6,  # Embedding L1 正则化
        embedding_l2_reg=1e-5,  # Embedding L2 正则化
        dense_l1_reg=1e-6,  # 稠密层 L1 正则化
        dense_l2_reg=1e-5,  # 稠密层 L2 正则化
        session_id="distributed_deepfm_tutorial",
    )

    model.compile(
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        loss="bce",  # 二元交叉熵损失
        scheduler="cosine",  # 余弦退火学习率调度器
        scheduler_params={"T_max": 10, "eta_min": 1e-6},  # 10轮后降到最小学习率
    )

    if rank == 0:
        print("\nStart Training")

    # 使用分布式数据并行进行训练
    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        epochs=10,  # 训练轮数
        batch_size=512,  # 每个 GPU 的批次大小(总批次大小 = batch_size × world_size)
        shuffle=True,  # 是否打乱数据
        metrics=["auc", "logloss"],  # 评估指标
        num_workers=4,  # 每个进程的 DataLoader 工作线程数
        use_tensorboard=True,  # 使用 TensorBoard 记录训练过程
    )

    # 在分布式训练中,确保所有进程都完成训练后再继续
    if is_distributed and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("Training Complete")
        print(f"Best model saved to: {model.best_checkpoint_path}")


    # 重要: evaluate() 使用分布式 all_gather 操作
    # 所有进程必须一起调用 evaluate(),即使只有 rank 0 打印结果
    if rank == 0:
        print("Final Evaluation")

    eval_metrics = model.evaluate(
        data=valid_df,
        batch_size=1024,
        metrics=["auc", "logloss", "accuracy"],
    )

    if rank == 0:
        print("Validation Metrics:")
        for metric_name, metric_value in eval_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # predict() 不使用分布式操作,可以只在 rank 0 调用
        print("Prediction Example")
        sample_df = valid_df.head(10)
        predictions = model.predict(data=sample_df, batch_size=10, return_dataframe=True)
        print(predictions)

    # 最终同步并销毁进程组
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            print("\nDistributed training cleaned up successfully.")


if __name__ == "__main__":
    main()


```