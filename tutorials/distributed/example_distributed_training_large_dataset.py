"""
Distributed Training with NextRec - Large Dataset

Date: create on 04/12/2025
Checkpoint: edit on 04/12/2025
Author: Yang Zhou,zyaztec@gmail.com

Usage:
torchrun --nproc_per_node=2 example_distributed_training_large_dataset.py
"""

import os
import glob
import pandas as pd
import torch
import torch.distributed as dist

from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM


def load_dataset_sharded(
    data_dir: str, rank: int, world_size: int, split: str = "train"
):
    """
    load pre-split train/valid data shards for distributed training.

    data should be pre-split into train/valid sets and sharded:
    - train_part_0.parquet, train_part_1.parquet, ...
    - valid_part_0.parquet, valid_part_1.parquet, ...

    each device loads its assigned shards based on rank.
    memory usage = total data size / number of devices

    Args:
        data_dir: Directory containing sharded parquet files
        rank: Current process rank
        world_size: Total number of processes
        split: 'train' or 'valid'
    """
    shard_pattern = f"{split}_part_*.parquet"
    shard_files = sorted(glob.glob(os.path.join(data_dir, shard_pattern)))

    if not shard_files:
        raise FileNotFoundError(
            f"No {split} shards found at {data_dir}/{shard_pattern}"
        )

    if rank == 0:
        print(f"[Rank 0] Found {len(shard_files)} {split} shards")

    # Each GPU loads every N-th shard
    my_shards = [f for i, f in enumerate(shard_files) if i % world_size == rank]
    print(
        f"[Rank {rank}] Loading {len(my_shards)} {split} shards: {[os.path.basename(f) for f in my_shards]}"
    )

    # Merge shards
    dfs = []
    for shard_file in my_shards:
        df_shard = pd.read_parquet(shard_file)
        dfs.append(df_shard)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[Rank {rank}] Loaded {len(df)} {split} rows")
    return df


def main():
    # init distributed training
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(f"[Rank {rank}/{world_size}] Initializing distributed training...")
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "/path/to/your/sharded_data/"

    train_df = load_dataset_sharded(data_dir, rank, world_size, split="train")
    valid_df = load_dataset_sharded(data_dir, rank, world_size, split="valid")

    if rank == 0:
        print(
            f"\n[Main process] Train samples (this GPU): {len(train_df)}, Valid samples (this GPU): {len(valid_df)}"
        )

    dense_features = [DenseFeature(name=f"dense_{i}", input_dim=1) for i in range(5)]

    embedding_dim = 32
    user_id_vocab_size = 5000
    item_id_vocab_size = 20000
    sparse_features = [
        SparseFeature(
            name="user_id",
            embedding_name="user_emb",
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

    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=None,
        mlp_params={"dims": [256, 128, 64], "activation": "relu", "dropout": 0.3},
        target="label",
        device=device,
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        session_id="distributed_large_dataset",
    )

    model.compile(
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        loss="bce",
    )

    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        epochs=10,
        batch_size=512,  # Per-GPU batch size
        shuffle=True,
        metrics=["auc", "logloss"],
        num_workers=4,
        use_tensorboard=True,
        auto_ddp_sampler=False,  # data already sharded per rank
    )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print("Training completed successfully!")


if __name__ == "__main__":
    main()
