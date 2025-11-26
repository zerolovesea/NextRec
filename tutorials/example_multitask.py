"""
Multi-task Learning Example with GAUC metric
Uses multitask_task.csv generated data
"""

import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from nextrec.models.multi_task.esmm import ESMM
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature
from nextrec.basic.metrics import compute_gauc

# Load generated data
df = pd.read_csv('dataset/multitask_task.csv')

# Parse sequence features from string to list
for col in df.columns:
    if 'sequence' in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

print(f"Dataset loaded: {len(df)} samples")
print(f"Users: {df['user_id'].nunique()}")
print(f"Items: {df['item_id'].nunique()}")

print(f"\nData sample:")
print(df.head(2))

# Check task label distribution
task_labels = ['click', 'conversion']
print(f"\nTask label distribution:")
for label in task_labels:
    print(f"  {label}: {df[label].mean():.4f}")

print(f"\nSequence sample:")
print(f"sequence_0[0]: {df['sequence_0'].iloc[0]}")

# Train/valid split
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# Dense features
num_dense = len([col for col in df.columns if col.startswith('dense_')])
dense_features = [
    DenseFeature(f'dense_{i}')
    for i in range(num_dense)
]

# Sparse features (including user_id and item_id)
sparse_features = [
    SparseFeature('user_id', vocab_size=int(df['user_id'].max() + 1), embedding_dim=32),
    SparseFeature('item_id', vocab_size=int(df['item_id'].max() + 1), embedding_dim=32),
]

# Add other sparse features
num_sparse = len([col for col in df.columns if col.startswith('sparse_')])
sparse_features.extend([
    SparseFeature(
        f'sparse_{i}',
        vocab_size=int(df[f'sparse_{i}'].max() + 1),
        embedding_dim=16
    )
    for i in range(num_sparse)
])

# Sequence features
sequence_cols = [col for col in df.columns if col.startswith('sequence_')]
sequence_features = [
    SequenceFeature(
        col,
        vocab_size=int(df[col].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1),
        embedding_dim=32,
        padding_idx=0
    )
    for col in sequence_cols
]

print(f"Dense features: {len(dense_features)}")
print(f"Sparse features: {len(sparse_features)} (including user_id and item_id)")
print(f"Sequence features: {len(sequence_features)}")

# ESMM model
ctr_params = {
    "dims": [64, 32],  
    "activation": "relu",
    "dropout": 0.4  
}

cvr_params = {
    "dims": [64, 32],  
    "activation": "relu",
    "dropout": 0.4  
}

model = ESMM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    ctr_params=ctr_params,
    cvr_params=cvr_params,
    target=task_labels,
    task=['binary', 'binary'],
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},  
    loss=['bce', 'bce'],
    device='mps',
    embedding_l1_reg=1e-5,  
    embedding_l2_reg=1e-4,  
    dense_l1_reg=1e-4, 
    dense_l2_reg=1e-3, 
    session_id="esmm_tutorial",
)

print(f"\nModel: {model.model_name}")
print(f"Tasks: {len(task_labels)}")

print("\n" + "=" * 60)
print("Start Training with GAUC metric")
print("=" * 60)
print(f"Train size: {len(train_df)}")
print(f"Valid size: {len(valid_df)}")

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=['auc', 'gauc', 'logloss'],  # Added GAUC metric
    epochs=3,
    batch_size=512,
    shuffle=True,
    user_id_column='user_id'  # Specify user_id column for GAUC
)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)

# Predict
print("\n" + "=" * 60)
print("Model Prediction")
print("=" * 60)

predictions = model.predict(valid_df, batch_size=512)
print(f"Prediction shape: {predictions.shape}  # (samples, tasks)")
print(f"\nTask predictions samples:")
for i, task_name in enumerate(task_labels):
    print(f"  {task_name}: {predictions[:5, i]}")

# Evaluation
print("\n" + "=" * 60)
print("Evaluation with GAUC")
print("=" * 60)


for i, task_name in enumerate(task_labels):
    y_true = valid_df[task_name].values
    y_pred = predictions[:, i]
    
    auc = roc_auc_score(y_true, y_pred)
    gauc = compute_gauc(y_true, y_pred, valid_df['user_id'].values)
    logloss = log_loss(y_true, y_pred)
    
    print(f"\nTask: {task_name}")
    print(f"  AUC: {auc:.4f}")
    print(f"  GAUC: {gauc:.4f}")
    print(f"  LogLoss: {logloss:.4f}")

print("\n" + "=" * 60)
print("Multi-task Example Complete!")
print("=" * 60)
