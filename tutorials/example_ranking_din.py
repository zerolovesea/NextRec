"""
DIN (Deep Interest Network) Ranking Model Example with GAUC metric
Uses ranking_task.csv generated data
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from nextrec.models.ranking.din import DIN
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

# Load generated data
df = pd.read_csv('dataset/ranking_task.csv')

# Parse sequence features from string to list
for col in df.columns:
    if 'sequence' in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

print(f"Dataset loaded: {len(df)} samples")
print(f"Users: {df['user_id'].nunique()}")
print(f"Items: {df['item_id'].nunique()}")
print(f"Positive ratio: {df['label'].mean():.4f}")

print(f"\nData sample:")
print(df.head(2))

# Check sequence features
print(f"\nSequence samples:")
print(f"sequence_0[0]: {df['sequence_0'].iloc[0]}")
print(f"  actual length: {len([x for x in df['sequence_0'].iloc[0] if x != 0])}")
print(f"sequence_1[0]: {df['sequence_1'].iloc[0]}")
print(f"  actual length: {len([x for x in df['sequence_1'].iloc[0] if x != 0])}")

# Train/valid split
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# Dense features
dense_features = [DenseFeature(name=f'dense_{i}', input_dim=1) for i in range(8)]

# Sparse features (including user_id and item_id)
sparse_features = [SparseFeature(name='user_id', embedding_name='user_emb', vocab_size=int(df['user_id'].max() + 1), embedding_dim=32), SparseFeature(name='item_id', embedding_name='item_emb', vocab_size=int(df['item_id'].max() + 1), embedding_dim=32),]

# Add other sparse features
sparse_features.extend([SparseFeature(name=f'sparse_{i}', embedding_name=f'sparse_{i}_emb', vocab_size=int(df[f'sparse_{i}'].max() + 1), embedding_dim=32) for i in range(10)])

# Sequence features
sequence_features = [
    SequenceFeature(name='sequence_0', vocab_size=int(df['sequence_0'].apply(lambda x: max(x)).max() + 1), embedding_dim=32, padding_idx=0, embedding_name='item_emb'),
    SequenceFeature(name='sequence_1', vocab_size=int(df['sequence_1'].apply(lambda x: max(x)).max() + 1), embedding_dim=16, padding_idx=0, embedding_name='sparse_0_emb'),]

print(f"\nDense features: {len(dense_features)}")
print(f"Sparse features: {len(sparse_features)} (including user_id and item_id)")
print(f"Sequence features: {len(sequence_features)}")

mlp_params = {
    "dims": [256, 128, 64],
    "activation": "relu",
    "dropout": 0.3,
}

model = DIN(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    mlp_params=mlp_params,
    attention_hidden_units=[80, 40],
    attention_activation='sigmoid',
    attention_use_softmax=True,
    target=['label'],
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="focal",  # imbalance-friendly loss, see nextrec.loss.pointwise for options
    loss_params={"gamma": 2.0, "alpha": 0.25},  # Example focal loss parameters
    device='mps',
    embedding_l1_reg=1e-6,
    embedding_l2_reg=1e-5,
    dense_l1_reg=1e-5,
    dense_l2_reg=1e-4,
    session_id="din_tutorial",
)

model.compile(
            optimizer = "adam",
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5},
            loss = "focal",
            loss_params={"gamma": 2.0, "alpha": 0.25},
        )

print(f"\nModel: {model.model_name}")
print(f"Attention: compute relevance between history and candidate item")
print(f"MLP: {mlp_params['dims']}")

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


# Predict
print(" ")
print("Model Prediction")
print(" ")


pred_df = model.predict(valid_df, batch_size=512)
preview = pred_df.head(5)
print(f"\nPrediction sample (first 5 rows):\n{preview}")

# Evaluate
metrics = model.evaluate(
    valid_df,
    metrics=['auc', 'gauc', 'logloss'],
    batch_size=512,
    user_id_column='user_id'
)
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print(" ")
print("DIN Example Complete!")
print(" ")
