"""
Run All Match Models Tutorial

Date: create on 06/12/2025
Checkpoint: edit on 06/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

from nextrec.models.retrieval.dssm import DSSM
from nextrec.models.retrieval.youtube_dnn import YoutubeDNN
from nextrec.models.retrieval.mind import MIND

from nextrec.utils import compute_pair_scores, generate_match_data


def train_model(
    model_class,
    model_name,
    user_dense_features,
    user_sparse_features,
    user_sequence_features,
    item_dense_features,
    item_sparse_features,
    item_sequence_features,
    train_df,
    valid_df,
    device="cpu",
    **kwargs,
):

    print("=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    try:
        model = model_class(
            user_dense_features=user_dense_features,
            user_sparse_features=user_sparse_features,
            user_sequence_features=user_sequence_features,
            item_dense_features=item_dense_features,
            item_sparse_features=item_sparse_features,
            item_sequence_features=item_sequence_features,
            device=device,
            session_id=f"match_{model_name.lower()}_tutorial",
            **kwargs,
        )

        model.compile(
            optimizer="adam",
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        )

        model.fit(
            train_data=train_df,
            valid_data=valid_df,
            epochs=1,
            batch_size=512,
            shuffle=True,
            use_tensorboard=False,
            user_id_column="user_id",
        )

        metrics = model.evaluate(
            valid_df,
            batch_size=512,
            user_id_column="user_id",
        )

        sample_scores = compute_pair_scores(model, valid_df.head(2048), batch_size=512)
        print(f"{model_name} sample scores: {sample_scores[:5]}")

        print(f"{model_name} completed successfully")
        return True, metrics

    except Exception as e:
        print(f"{model_name} failed with error: {str(e)}")
        return False, None


def main():
    """Main function to run all match models"""
    print("=" * 80)
    print("Training all supported match models with synthetic data")
    print("=" * 80)

    device = "cpu"

    (
        df,
        user_dense_features,
        user_sparse_features,
        user_sequence_features,
        item_dense_features,
        item_sparse_features,
        item_sequence_features,
    ) = generate_match_data(
        n_samples=10000,
        user_vocab_size=1000,
        item_vocab_size=5000,
        category_vocab_size=100,
        brand_vocab_size=200,
        city_vocab_size=100,
        user_feature_vocab_size=50,
        item_feature_vocab_size=50,
        sequence_max_len=50,
        user_embedding_dim=32,
        item_embedding_dim=32,
        seed=42,
    )

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    valid_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")

    results = {}

    models_to_train = [
        (
            DSSM,
            "DSSM",
            {
                "user_dnn_hidden_units": [256, 128, 64],
                "item_dnn_hidden_units": [256, 128, 64],
                "embedding_dim": 64,
                "dnn_activation": "relu",
                "dnn_dropout": 0.2,
                "similarity_metric": "cosine",
                "training_mode": "pointwise",
            },
        ),
        (
            YoutubeDNN,
            "YoutubeDNN",
            {
                "user_dnn_hidden_units": [256, 128, 64],
                "item_dnn_hidden_units": [256, 128, 64],
                "embedding_dim": 64,
                "dnn_activation": "relu",
                "dnn_dropout": 0.2,
                "training_mode": "listwise",
                "num_negative_samples": 100,
            },
        ),
        (
            MIND,
            "MIND",
            {
                "item_dnn_hidden_units": [256, 128],
                "embedding_dim": 64,
                "num_interests": 4,
                "capsule_bilinear_type": 2,
                "routing_times": 3,
                "dnn_activation": "relu",
                "dnn_dropout": 0.2,
                "training_mode": "pointwise",
                "similarity_metric": "dot",
            },
        ),
    ]

    successful = 0
    failed = 0
    failed_models = []

    for model_class, model_name, extra_params in models_to_train:
        success, metrics = train_model(
            model_class=model_class,
            model_name=model_name,
            user_dense_features=user_dense_features,
            user_sparse_features=user_sparse_features,
            user_sequence_features=user_sequence_features,
            item_dense_features=item_dense_features,
            item_sparse_features=item_sparse_features,
            item_sequence_features=item_sequence_features,
            train_df=train_df,
            valid_df=valid_df,
            device=device,
            **extra_params,
        )

        if success:
            successful += 1
            results[model_name] = metrics
        else:
            failed += 1
            failed_models.append(model_name)

    print("Test Summary")
    print(f"Total models: {len(models_to_train)}")
    print(f"Successful counts: {successful}")
    print(f"Failed counts: {failed}, Models: {failed_models}")


if __name__ == "__main__":
    main()
