"""
Run All Ranking Models Tutorial

Date: create on 06/12/2025
Checkpoint: edit on 06/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

from nextrec.models.ranking.fm import FM
from nextrec.models.ranking.lr import LR
from nextrec.models.ranking.eulernet import EulerNet
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.models.ranking.din import DIN
from nextrec.models.ranking.dien import DIEN
from nextrec.models.ranking.dcn import DCN
from nextrec.models.ranking.autoint import AutoInt
from nextrec.models.ranking.widedeep import WideDeep
from nextrec.models.ranking.xdeepfm import xDeepFM
from nextrec.models.ranking.fibinet import FiBiNET
from nextrec.models.ranking.afm import AFM
from nextrec.models.ranking.ffm import FFM
from nextrec.models.ranking.pnn import PNN
from nextrec.models.ranking.masknet import MaskNet

from nextrec.utils import generate_ranking_data


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
        # Determine if model needs sequence features
        # DIN and DIEN require sequence features
        if model_name in ["DIN", "DIEN"]:
            seq_feats = sequence_features
        else:
            seq_feats = []

        # MaskNet requires all features to have the same embedding_dim
        # Set dense features' embedding_dim to match sparse features for MaskNet
        if model_name == "MaskNet":
            from nextrec.basic.features import DenseFeature

            embedding_dim = sparse_features[0].embedding_dim if sparse_features else 16
            adjusted_dense_features = [
                DenseFeature(
                    name=f.name,
                    embedding_dim=embedding_dim,
                    input_dim=f.input_dim,
                    use_embedding=True,
                )
                for f in dense_features
            ]
        elif model_name == "PNN":
            from nextrec.basic.features import DenseFeature

            embedding_dim = sparse_features[0].embedding_dim if sparse_features else 16
            adjusted_dense_features = [
                DenseFeature(
                    name=f.name,
                    embedding_dim=embedding_dim,
                    input_dim=f.input_dim,
                    use_embedding=True,
                )
                for f in dense_features
            ]
        else:
            adjusted_dense_features = dense_features

        model = model_class(
            dense_features=adjusted_dense_features,
            sparse_features=sparse_features,
            sequence_features=seq_feats,
            target=["label"],
            device=device,
            session_id=f"ranking_{model_name.lower()}_tutorial",
            **kwargs,
        )

        model.compile(
            optimizer="adam",
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            loss="binary_crossentropy",
        )

        model.fit(
            train_data=train_df,
            valid_data=valid_df,
            metrics=["auc", "logloss"],
            epochs=1,
            batch_size=512,
            shuffle=True,
            use_tensorboard=False,
        )

        metrics = model.evaluate(valid_df, metrics=["auc", "logloss"], batch_size=512)

        print(f"{model_name} completed successfully")
        return True, metrics

    except Exception as e:
        print(f"{model_name} failed with error: {str(e)}")
        return False, None


def main():
    print("=" * 80)
    print("Training all supported ranking models with synthetic data")
    print("=" * 80)

    device = "cpu"

    df, dense_features, sparse_features, sequence_features = generate_ranking_data(
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

    mlp_params = {
        "dims": [256, 128, 64],
        "activation": "relu",
        "dropout": 0.2,
    }
    results = {}

    behavior_feature_name = sequence_features[0].name if sequence_features else None
    candidate_feature_name = "item_id"

    models_to_train = [
        (LR, "LR", {}),
        (FM, "FM", {}),
        (FFM, "FFM", {}),
        (EulerNet, "EulerNet", {"num_layers": 2, "num_orders": 8}),
        (DeepFM, "DeepFM", {"mlp_params": mlp_params}),
        (WideDeep, "WideDeep", {"mlp_params": mlp_params}),
        (DCN, "DCN", {"mlp_params": mlp_params, "cross_num": 3}),
        (xDeepFM, "xDeepFM", {"mlp_params": mlp_params, "cin_size": [128, 128]}),
        (
            AutoInt,
            "AutoInt",
            {
                "att_layer_num": 3,
                "att_embedding_dim": 16,
                "att_head_num": 2,
                "att_dropout": 0.2,
            },
        ),
        (AFM, "AFM", {"attention_dim": 64, "attention_dropout": 0.2}),
        (
            PNN,
            "PNN",
            {
                "mlp_params": mlp_params,
                "product_type": "inner",  # set to "outer" to use outer-product kernel
                "outer_product_dim": 64,
            },
        ),
        (
            FiBiNET,
            "FiBiNET",
            {
                "mlp_params": mlp_params,
                "bilinear_type": "field_interaction",
                "senet_reduction": 3,
            },
        ),
        (
            DIN,
            "DIN",
            {
                "mlp_params": mlp_params,
                "attention_hidden_units": [80, 40],
                "attention_activation": "sigmoid",
                "behavior_feature_name": behavior_feature_name,
                "candidate_feature_name": candidate_feature_name,
            },
        ),
        (
            DIEN,
            "DIEN",
            {
                "mlp_params": mlp_params,
                "gru_hidden_size": 32,
                "attention_hidden_units": [80, 40],
                # DIEN-specific required args match synthetic data columns
                "behavior_feature_name": behavior_feature_name,  # first generated sequence
                "candidate_feature_name": candidate_feature_name,  # candidate item id
                # optional negative behavior sequence for auxiliary loss
                "use_negsampling": True,
                "neg_behavior_feature_name": "sequence_1",  # second generated sequence as negatives
            },
        ),
        (
            MaskNet,
            "MaskNet",
            {
                "mlp_params": mlp_params,
                "architecture": "parallel",
                "num_blocks": 3,
                "mask_hidden_dim": 64,
                "block_hidden_dim": 256,
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
