"""
Run All Multi-Task Models Tutorial

Date: create on 06/12/2025
Checkpoint: edit on 06/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

from nextrec.models.multi_task.esmm import ESMM
from nextrec.models.multi_task.mmoe import MMOE
from nextrec.models.multi_task.ple import PLE
from nextrec.models.multi_task.share_bottom import ShareBottom

from nextrec.utils import generate_multitask_data


def train_model(
    model_class,
    model_name,
    dense_features,
    sparse_features,
    sequence_features,
    train_df,
    valid_df,
    device="cpu",
    **kwargs,
):

    print("=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    try:
        # Create model
        model = model_class(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            device=device,
            session_id=f"multitask_{model_name.lower()}_tutorial",
            **kwargs,
        )

        model.compile(
            optimizer="adam",
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            loss=["bce"] * len(kwargs.get("target", ["task1", "task2"])),
            loss_weights={"method": "grad_norm", "alpha": 1.5, "lr": 0.025},
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

        print(f"{model_name} completed successfully")
        return True, metrics

    except Exception as e:
        print(f"{model_name} failed with error: {str(e)}")
        return False, None


def main():
    """Main function to run all multi-task models"""
    print("=" * 80)
    print("Training all supported multi-task models with synthetic data")
    print("=" * 80)

    device = "cpu"

    df, dense_features, sparse_features, sequence_features = generate_multitask_data(
        n_samples=10000,
        n_dense=5,
        n_sparse=8,
        n_sequences=2,
        user_vocab_size=1000,
        item_vocab_size=500,
        sparse_vocab_size=50,
        sequence_max_len=20,
        embedding_dim=16,
        seed=42,
    )

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    valid_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")

    tower_params = {"dims": [256, 128, 64], "activation": "relu", "dropout": 0.2}
    results = {}

    models_to_train = [
        (
            ESMM,
            "ESMM",
            {
                "ctr_params": tower_params,
                "cvr_params": tower_params,
                "target": ["click", "ctcvr"],
            },
        ),
        (
            MMOE,
            "MMOE",
            {
                "expert_params": tower_params,
                "tower_params_list": [tower_params, tower_params],
                "num_experts": 4,
                "target": ["click", "conversion"],
            },
        ),
        (
            PLE,
            "PLE",
            {
                "shared_expert_params": tower_params,
                "specific_expert_params": tower_params,
                "tower_params_list": [tower_params, tower_params],
                "num_shared_experts": 2,
                "num_specific_experts": 2,
                "num_levels": 2,
                "target": ["click", "conversion"],
            },
        ),
        (
            ShareBottom,
            "ShareBottom",
            {
                "bottom_params": tower_params,
                "tower_params_list": [tower_params, tower_params],
                "target": ["click", "conversion"],
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
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
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
