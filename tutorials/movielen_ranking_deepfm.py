import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from sklearn.model_selection import train_test_split

from nextrec.data.preprocessor import DataProcessor
from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM

def example_movielens_100k_deepfm():
    """Example: MovieLens 100K for rating prediction."""
    df = pd.read_csv("dataset/movielens_100k.csv")

    processor = DataProcessor(session_id="movielens_deepfm")
    processor.add_sparse_feature('movie_title', encode_method='hash', hash_size=1000)
    processor.fit(df)

    df = processor.transform(df, return_dict=False)
    processor.save(save_path="movielens_deepfm_processor")
    print(f"\nSample training data:")
    print(df.head())

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=2024)

    dense_features = [DenseFeature('age')]
    sparse_features = [
        SparseFeature('user_id', vocab_size=df['user_id'].max()+1, embedding_dim=4),
        SparseFeature('item_id', vocab_size=df['item_id'].max()+1, embedding_dim=4),
    ]
    
    sparse_features.append(SparseFeature('gender', vocab_size=df['gender'].max()+1, embedding_dim=4))
    sparse_features.append(SparseFeature('occupation', vocab_size=df['occupation'].max()+1, embedding_dim=4))
    sparse_features.append(SparseFeature('movie_title', vocab_size=df['movie_title'].max()+1, embedding_dim=4))

    print("\n--- Training DeepFM Model ---")
    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.2},
        target='label',
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        device='cpu',
        session_id="/Users/zyaztec/Downloads/movielens_deepfm"
    )
    
    model.fit(
        train_data=train_df,
        valid_data=test_df,
        metrics=['auc', 'recall','precision'],
        epochs=1,  
        batch_size=512,
        shuffle=True
    )
    
    # Evaluate
    predictions = model.predict(test_df, batch_size=512)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:10]}")


if __name__ == "__main__":
    example_movielens_100k_deepfm()
