"""
Distributed Training with NextRec (Single Machine, Dual GPU)

Date: create on 04/12/2025
Checkpoint: edit on 04/12/2025
Author: Yang Zhou,zyaztec@gmail.com

Usage:
# method 1: Using torchrun (recommended)
torchrun --nproc_per_node=2 example_distributed_training.py

# method 2: Using python -m torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=2 example_distributed_training.py

# method 3: Specifying GPUs with CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 example_distributed_training.py
"""

import os
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split

from nextrec.utils import generate_distributed_ranking_data
from nextrec.models.ranking.deepfm import DeepFM


def main():
    # check if in distributed mode
    # only print logs on main process (rank 0)
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(f"[Rank {rank}/{world_size}] Initializing distributed training...")
        print(f"[Rank {rank}/{world_size}] Local rank: {local_rank}")

        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Warning: Not in distributed mode. Training on single device.")

    # Generate synthetic data with feature definitions
    df, dense_features, sparse_features, sequence_features = (
        generate_distributed_ranking_data(
            num_samples=100000,
            num_users=10000,
            num_items=5000,
            num_categories=20,
            num_cities=100,
            max_seq_len=50,
            embedding_dim=32,
            seed=42,
        )
    )

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

    if rank == 0:
        print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
        print("Each GPU will process different batches from this dataset")

    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        mlp_params={
            "dims": [256, 128, 64],
            "activation": "relu",
            "dropout": 0.3,
        },
        target="label",
        device=device,
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        embedding_l1_reg=1e-6,
        embedding_l2_reg=1e-5,
        dense_l1_reg=1e-6,
        dense_l2_reg=1e-5,
        session_id="distributed_deepfm_tutorial",
    )

    # Compile model
    model.compile(
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        loss="bce",
        scheduler="cosine",
        scheduler_params={"T_max": 10, "eta_min": 1e-6},
    )

    if rank == 0:
        print("\nStart Training")

    # Train model with distributed data parallelism
    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        epochs=10,
        batch_size=512,  # Per-GPU batch size
        shuffle=True,
        metrics=["auc", "logloss"],
        num_workers=4,  # DataLoader workers per process
        use_tensorboard=True,
    )

    # Synchronize all processes after training
    if is_distributed and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("Training Complete")
        print(f"Best model saved to: {model.best_checkpoint_path}")

    # IMPORTANT: evaluate() uses distributed all_gather operations
    # all processes must call evaluate() together, even if only rank 0 prints
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

        # predict() doesn't use distributed operations, safe to call only on rank 0
        print("Prediction Example")
        sample_df = valid_df.head(10)
        predictions = model.predict(
            data=sample_df, batch_size=10, return_dataframe=True
        )
        print(predictions)

    # Final synchronization and cleanup
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            print("\nDistributed training cleaned up successfully.")


if __name__ == "__main__":
    main()
