"""
Distributed Training with NextRec (Single Machine, Dual GPU)

Usage:
    # Method 1: Using torchrun (recommended)
    torchrun --nproc_per_node=2 example_distributed_training.py
    
    # Method 2: Using python -m torch.distributed.launch
    python -m torch.distributed.launch --nproc_per_node=2 example_distributed_training.py

Date: create on 04/12/2025
Checkpoint: edit on 04/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature
from nextrec.models.ranking.deepfm import DeepFM

def generate_synthetic_dataset(num_samples=100000, num_users=10000, num_items=5000):

    print(f"Generating synthetic dataset with {num_samples} samples...")
    np.random.seed(42)
    user_ids = np.random.randint(1, num_users + 1, size=num_samples)
    item_ids = np.random.randint(1, num_items + 1, size=num_samples)
    
    dense_features = {}
    for i in range(5):
        dense_features[f'dense_{i}'] = np.random.randn(num_samples).astype(np.float32)

    sparse_features = {}
    sparse_features['gender'] = np.random.randint(0, 2, size=num_samples)  
    sparse_features['age_group'] = np.random.randint(0, 7, size=num_samples)  
    sparse_features['category'] = np.random.randint(0, 20, size=num_samples)  
    sparse_features['city'] = np.random.randint(0, 100, size=num_samples) 
    
    max_seq_len = 50
    sequence_features = {}
    
    hist_items = []
    for _ in range(num_samples):
        seq_len = np.random.randint(5, max_seq_len + 1)
        hist_item_seq = np.random.randint(1, num_items + 1, size=seq_len).tolist()
        hist_items.append(hist_item_seq)
    sequence_features['hist_items'] = hist_items
    
    hist_categories = []
    for _ in range(num_samples):
        seq_len = np.random.randint(5, max_seq_len + 1)
        hist_cat_seq = np.random.randint(0, 20, size=seq_len).tolist()
        hist_categories.append(hist_cat_seq)
    sequence_features['hist_categories'] = hist_categories
    
    label_probs = 1 / (1 + np.exp(-(
        dense_features['dense_0'] * 0.3 +
        dense_features['dense_1'] * 0.2 +
        (sparse_features['gender'] - 0.5) * 0.5 +
        np.random.randn(num_samples) * 0.1
    )))
    labels = (label_probs > 0.5).astype(np.float32)
    
    data = {
        'user_id': user_ids,
        'item_id': item_ids,
        'label': labels,
        **dense_features,
        **sparse_features,
        **sequence_features,
    }
    
    df = pd.DataFrame(data)
    print(f"Dataset generated: {len(df)} samples, {len(df.columns)} columns")
    print(f"Positive rate: {labels.mean():.4f}")
    
    return df


def main():
    # Check if we're in distributed mode
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if is_distributed:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"[Rank {rank}/{world_size}] Initializing distributed training...")
        print(f"[Rank {rank}/{world_size}] Local rank: {local_rank}")
        
        # Set device for this process
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Warning: Not in distributed mode. Training on single device.")
    
    # Generate synthetic dataset (all processes generate the same data)
    # This ensures consistent data across all processes
    df = generate_synthetic_dataset(num_samples=100000, num_users=10000, num_items=5000)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)
    
    if rank == 0:
        print(f"[Main process] Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
    
    if rank == 0:
        print("\n=== Feature Configuration ===")
    
    dense_features = [DenseFeature(name=f'dense_{i}', input_dim=1)  for i in range(5)]
    
    embedding_dim = 32  
    sparse_features = [SparseFeature(name='user_id', embedding_name='user_emb', vocab_size=int(train_df['user_id'].max() + 1), embedding_dim=embedding_dim),
                       SparseFeature(name='item_id', embedding_name='item_emb', vocab_size=int(train_df['item_id'].max() + 1), embedding_dim=embedding_dim),
                       SparseFeature(name='gender', embedding_name='gender_emb', vocab_size=2, embedding_dim=embedding_dim),    
                       SparseFeature(name='age_group', embedding_name='age_group_emb', vocab_size=7, embedding_dim=embedding_dim),
                       SparseFeature(name='category', embedding_name='category_emb', vocab_size=20, embedding_dim=embedding_dim),
                       SparseFeature(name='city', embedding_name='city_emb', vocab_size=100, embedding_dim=embedding_dim)]
    
    # Define sequence features
    sequence_features = [SequenceFeature(name='hist_items', vocab_size=int(train_df['item_id'].max() + 1), embedding_dim=embedding_dim, max_len=50, padding_idx=0, embedding_name='item_emb'),
                         SequenceFeature(name='hist_categories', vocab_size=20, embedding_dim=embedding_dim, max_len=50, padding_idx=0, embedding_name='category_emb'),]
    
    if rank == 0:
        print(f"Dense features: {len(dense_features)}")
        print(f"Sparse features: {len(sparse_features)}")
        print(f"Sequence features: {len(sequence_features)}")
    
    if rank == 0:
        print("\n=== Model Configuration ===")
    
    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        mlp_params={
        "dims": [256, 128, 64],
        "activation": "relu",
        "dropout": 0.3,
    },
        target='label',
        device=device,
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        embedding_l1_reg=1e-6,
        embedding_l2_reg=1e-5,
        dense_l1_reg=1e-6,
        dense_l2_reg=1e-5,
        session_id=f"distributed_deepfm_tutorial",
    )
    
    # Compile model
    model.compile(
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        loss='bce',
        scheduler="cosine",
        scheduler_params={"T_max": 10, "eta_min": 1e-6},
    )
    
    if rank == 0:
        print("\n=== Start Training ===")
    
    # Train model with distributed data parallelism
    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        epochs=10,
        batch_size=512,  # Per-GPU batch size
        shuffle=True,
        metrics=['auc', 'logloss'],
        num_workers=4,  # DataLoader workers per process
        tensorboard=True,
    )
    
    # Synchronize all processes after training
    if is_distributed and dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        print("\n=== Training Complete ===")
        print(f"Best model saved to: {model.best_checkpoint_path}")
    
    # IMPORTANT: evaluate() uses distributed all_gather operations
    # all processes must call evaluate() together, even if only rank 0 prints
    if rank == 0:
        print("\n=== Final Evaluation ===")
    
    # All processes evaluate together (required for distributed gather)
    eval_metrics = model.evaluate(
        data=valid_df,
        batch_size=1024,
        metrics=['auc', 'logloss', 'accuracy'],
    )
    
    # Only rank 0 prints results
    if rank == 0:
        print("Validation Metrics:")
        for metric_name, metric_value in eval_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # predict() doesn't use distributed operations, safe to call only on rank 0
        print("\n=== Prediction Example ===")
        sample_df = valid_df.head(10)
        predictions = model.predict(data=sample_df, batch_size=10, return_dataframe=True)
        print(predictions)
    
    # Final synchronization and cleanup
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            print("\n[Main process] Distributed training cleaned up successfully.")


if __name__ == '__main__':
    """
    Entry point for distributed training.
    
    To run this script with 2 GPUs on a single machine:
    
    Method 1 (Recommended - PyTorch >= 1.10):
        torchrun --nproc_per_node=2 example_distributed_training.py
    
    Method 2 (Legacy):
        python -m torch.distributed.launch --nproc_per_node=2 example_distributed_training.py
    
    Method 3 (Using CUDA_VISIBLE_DEVICES to control specific GPUs):
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 example_distributed_training.py
    
    For single GPU testing:
        python example_distributed_training.py
    
    Notes:
    - Each GPU will handle a portion of the batch
    - Gradients are synchronized across GPUs using All-Reduce
    - Model checkpoints are saved only on the main process (rank 0)
    - All processes participate in training but only rank 0 handles I/O
    """
    main()
