"""
分布式训练示例 - 单机多卡

文件说明:
    本示例演示如何使用 NextRec 框架进行分布式训练(单机多GPU)。通过 PyTorch 的
    DistributedDataParallel (DDP) 实现数据并行训练,提高训练速度和效率。

主要功能:
    - 分布式训练环境初始化
    - 数据并行训练(每个 GPU 处理不同的批次)
    - 梯度同步和模型参数同步
    - 学习率调度
    - 分布式评估和预测

使用方法:
    方法1: 使用 torchrun (推荐)
        torchrun --nproc_per_node=2 example_distributed_training.py

    方法2: 使用 torch.distributed.launch
        python -m torch.distributed.launch --nproc_per_node=2 example_distributed_training.py

    方法3: 指定特定 GPU
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 example_distributed_training.py

参数说明:
    --nproc_per_node: 每个节点(机器)使用的 GPU 数量

数据要求:
    使用合成数据,不需要外部数据文件。脚本会自动生成:
        - 用户特征(稀疏、稠密、序列)
        - 物品特征
        - 用户-物品交互标签

模型说明:
    使用 DeepFM 作为示例模型,支持:
        - L1/L2 正则化
        - 学习率余弦退火
        - TensorBoard 日志记录
        - 模型检查点保存

输出:
    - 训练好的模型(最佳检查点)
    - 训练日志
    - 评估指标(AUC、LogLoss、Accuracy)
    - TensorBoard 日志文件

作者: Yang Zhou, zyaztec@gmail.com
创建日期: 2025-12-04
最后更新: 2026-01-28
"""

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
    # ==============================================================================
    # 1. 检查是否处于分布式模式
    # ==============================================================================

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

    # ==============================================================================
    # 2. 生成合成数据
    # ==============================================================================

    df, dense_features, sparse_features, sequence_features = (
        generate_distributed_ranking_data(
            num_samples=100000,  # 样本总数
            num_users=10000,  # 用户数量
            num_items=5000,  # 物品数量
            num_categories=20,  # 类别数量
            num_cities=100,  # 城市数量
            max_seq_len=50,  # 序列最大长度
            embedding_dim=32,  # embedding 维度
            seed=42,  # 随机种子
        )
    )

    # ==============================================================================
    # 3. 划分训练集和验证集
    # ==============================================================================

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

    if rank == 0:
        print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
        print("Each GPU will process different batches from this dataset")

    # ==============================================================================
    # 4. 构建 DeepFM 模型
    # ==============================================================================

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

    # ==============================================================================
    # 5. 编译模型
    # ==============================================================================

    model.compile(
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        loss="bce",  # 二元交叉熵损失
        scheduler="cosine",  # 余弦退火学习率调度器
        scheduler_params={"T_max": 10, "eta_min": 1e-6},  # 10轮后降到最小学习率
    )

    if rank == 0:
        print("\nStart Training")

    # ==============================================================================
    # 6. 训练模型
    # ==============================================================================

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

    # ==============================================================================
    # 7. 同步所有进程
    # ==============================================================================

    # 在分布式训练中,确保所有进程都完成训练后再继续
    if is_distributed and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("Training Complete")
        print(f"Best model saved to: {model.best_checkpoint_path}")

    # ==============================================================================
    # 8. 模型评估
    # ==============================================================================

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

        # ==============================================================================
        # 9. 模型预测(仅在主进程)
        # ==============================================================================

        # predict() 不使用分布式操作,可以只在 rank 0 调用
        print("Prediction Example")
        sample_df = valid_df.head(10)
        predictions = model.predict(
            data=sample_df, batch_size=10, return_dataframe=True
        )
        print(predictions)

    # ==============================================================================
    # 10. 清理分布式环境
    # ==============================================================================

    # 最终同步并销毁进程组
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            print("\nDistributed training cleaned up successfully.")


if __name__ == "__main__":
    main()
