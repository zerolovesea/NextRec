"""
分布式训练示例 - 大数据集分片加载

文件说明:
    本示例演示如何使用 NextRec 框架进行大规模数据集的分布式训练。通过数据分片技术,
    每个 GPU 只加载部分数据到内存,有效解决大数据集无法完全加载到内存的问题。

主要功能:
    - 大数据集分片加载(每个 GPU 加载不同的数据分片)
    - 分布式训练环境初始化
    - 内存优化的数据加载策略
    - 模型训练与评估

使用方法:
    torchrun --nproc_per_node=2 example_distributed_training_large_dataset.py

数据准备:
    数据应预先分片为 Parquet 格式文件:
        训练集: train_part_0.parquet, train_part_1.parquet, ...
        验证集: valid_part_0.parquet, valid_part_1.parquet, ...

    每个 GPU 将根据其 rank 加载对应的数据分片,实现内存分布式存储。

分片策略:
    - 总数据量 / GPU 数量 = 每个 GPU 的数据量
    - 例如: 10TB 数据 / 8 GPUs = 每个 GPU 加载 1.25TB
    - 支持任意数量的分片文件

模型说明:
    使用 DeepFM 作为示例模型,支持大规模稀疏特征的高效训练。

注意事项:
    1. 确保数据已经预先分片并保存为 Parquet 格式
    2. 修改 data_dir 变量指向实际的数据目录
    3. 根据实际特征调整 dense_features 和 sparse_features
    4. 设置 auto_ddp_sampler=False,因为数据已按 rank 分片

输出:
    - 训练好的模型
    - 训练日志
    - 评估指标
    - TensorBoard 日志文件

作者: Yang Zhou, zyaztec@gmail.com
创建日期: 2025-12-04
最后更新: 2026-01-28
"""

import os
import glob
import pandas as pd
import torch
import torch.distributed as dist

from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM


def load_dataset_sharded(data_dir: str, rank: int, world_size: int, split: str = "train"):
    """
    加载数据分片

    说明:
        加载预分片的训练/验证数据,用于分布式训练。数据应预先划分为训练集和验证集,
        并进一步分片为多个文件:
            - train_part_0.parquet, train_part_1.parquet, ...
            - valid_part_0.parquet, valid_part_1.parquet, ...

        每个设备根据其 rank 加载对应的分片,实现内存分布式加载。
        内存使用量 = 总数据大小 / 设备数量

    参数:
        data_dir: 包含分片 Parquet 文件的目录
        rank: 当前进程的全局排名
        world_size: 总进程数
        split: 数据集类型,'train' 或 'valid'

    返回:
        df: 当前进程加载的数据 DataFrame

    示例:
        假设有 8 个分片文件和 2 个 GPU:
            - Rank 0 加载: part_0, part_2, part_4, part_6
            - Rank 1 加载: part_1, part_3, part_5, part_7
    """
    # 构建分片文件匹配模式
    shard_pattern = f"{split}_part_*.parquet"
    shard_files = sorted(glob.glob(os.path.join(data_dir, shard_pattern)))

    # 检查是否找到分片文件
    if not shard_files:
        raise FileNotFoundError(f"No {split} shards found at {data_dir}/{shard_pattern}")

    if rank == 0:
        print(f"[Rank 0] Found {len(shard_files)} {split} shards")

    # ==============================================================================
    # 分配分片: 每个 GPU 加载序号为 rank, rank+world_size, rank+2*world_size, ... 的分片
    # ==============================================================================

    my_shards = [f for i, f in enumerate(shard_files) if i % world_size == rank]
    print(f"[Rank {rank}] Loading {len(my_shards)} {split} shards: {[os.path.basename(f) for f in my_shards]}")

    # ==============================================================================
    # 加载并合并分片
    # ==============================================================================

    dfs = []
    for shard_file in my_shards:
        df_shard = pd.read_parquet(shard_file)
        dfs.append(df_shard)

    # 将多个分片合并为一个 DataFrame
    df = pd.concat(dfs, ignore_index=True)
    print(f"[Rank {rank}] Loaded {len(df)} {split} rows")
    return df


def main():
    """
    主函数: 初始化分布式环境并进行大数据集训练
    """
    # ==============================================================================
    # 1. 初始化分布式训练环境
    # ==============================================================================

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_distributed:
        rank = int(os.environ["RANK"])  # 当前进程的全局排名
        world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前节点内的本地排名

        print(f"[Rank {rank}/{world_size}] Initializing distributed training...")
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================================================================
    # 2. 加载分片数据
    # ==============================================================================

    # 数据目录(需要修改为实际路径)
    data_dir = "/path/to/your/sharded_data/"

    # 加载训练集和验证集的分片
    train_df = load_dataset_sharded(data_dir, rank, world_size, split="train")
    valid_df = load_dataset_sharded(data_dir, rank, world_size, split="valid")

    if rank == 0:
        print(f"\n[Main process] Train samples (this GPU): {len(train_df)}, Valid samples (this GPU): {len(valid_df)}")

    # ==============================================================================
    # 3. 定义特征
    # ==============================================================================

    # 定义稠密特征(5个数值型特征)
    dense_features = [DenseFeature(name=f"dense_{i}", input_dim=1) for i in range(5)]

    # 定义稀疏特征的参数
    embedding_dim = 32  # embedding 维度
    user_id_vocab_size = 5000  # 用户ID词汇表大小
    item_id_vocab_size = 20000  # 物品ID词汇表大小

    # 定义稀疏特征
    sparse_features = [
        SparseFeature(
            name="user_id",
            embedding_name="user_emb",  # embedding 名称,用于权重共享
            vocab_size=user_id_vocab_size,
            embedding_dim=embedding_dim,
        ),
        SparseFeature(
            name="item_id",
            embedding_name="item_emb",
            vocab_size=item_id_vocab_size,
            embedding_dim=embedding_dim,
        ),
    ]

    # ==============================================================================
    # 4. 构建 DeepFM 模型
    # ==============================================================================

    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=None,  # 此示例不使用序列特征
        mlp_params={"dims": [256, 128, 64], "activation": "relu", "dropout": 0.3},
        target="label",
        device=device,
        distributed=is_distributed,  # 启用分布式训练
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        session_id="distributed_large_dataset",
    )

    # ==============================================================================
    # 5. 编译模型
    # ==============================================================================

    model.compile(
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        loss="bce",  # 二元交叉熵损失
    )

    # ==============================================================================
    # 6. 训练模型
    # ==============================================================================

    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        epochs=10,  # 训练轮数
        batch_size=512,  # 每个 GPU 的批次大小
        shuffle=True,  # 是否打乱数据
        metrics=["auc", "logloss"],  # 评估指标
        num_workers=4,  # 每个进程的 DataLoader 工作线程数
        use_tensorboard=True,  # 使用 TensorBoard 记录训练过程
        auto_ddp_sampler=False,  # 不使用 DDP 采样器,因为数据已按 rank 分片
    )

    # ==============================================================================
    # 7. 清理分布式环境
    # ==============================================================================

    if is_distributed and dist.is_initialized():
        dist.barrier()  # 同步所有进程
        dist.destroy_process_group()  # 销毁进程组

    if rank == 0:
        print("Training completed successfully!")


if __name__ == "__main__":
    main()
