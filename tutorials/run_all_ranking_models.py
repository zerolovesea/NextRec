"""
批量运行所有排序模型示例

文件说明:
    本示例演示如何批量训练和测试 NextRec 框架支持的所有排序(CTR预估)模型。
    通过统一的训练接口和合成数据,可以快速验证各个排序模型的功能和性能。

主要功能:
    - 生成合成的排序任务数据
    - 批量训练多个排序模型
    - 统一的模型训练和评估流程
    - 收集训练结果和错误信息

支持的模型:
    1. LR (Logistic Regression): 逻辑回归
    2. FM (Factorization Machines): 因子分解机
    3. FFM (Field-aware Factorization Machines): 域感知因子分解机
    4. EulerNet: 基于欧拉公式的神经网络
    5. DeepFM: FM + DNN
    6. WideDeep: Wide & Deep
    7. DCN (Deep & Cross Network): 深度交叉网络
    8. xDeepFM: 极深因子分解机
    9. AutoInt: 自动特征交互
    10. AFM (Attentional Factorization Machines): 注意力因子分解机
    11. PNN (Product-based Neural Networks): 基于乘积的神经网络
    12. FiBiNET: 特征重要性和双线性特征交互网络
    13. DIN (Deep Interest Network): 深度兴趣网络
    14. DIEN (Deep Interest Evolution Network): 深度兴趣演化网络
    15. MaskNet: 掩码网络

使用方法:
    直接运行此脚本:
        python tutorials/run_all_ranking_models.py

数据要求:
    使用合成数据,不需要外部数据文件。脚本会自动生成:
        - 稠密特征
        - 稀疏特征
        - 序列特征
        - 二分类标签

输出:
    - 各模型的训练日志
    - 评估指标(AUC、LogLoss)
    - 训练成功/失败统计
    - 失败模型列表

作者: Yang Zhou, zyaztec@gmail.com
创建日期: 2025-12-06
最后更新: 2026-01-28
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

from nextrec.utils.data import generate_ranking_data


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
    """
    训练单个排序模型

    参数:
        model_class: 模型类
        model_name: 模型名称(用于日志输出)
        dense_features: 稠密特征列表
        sparse_features: 稀疏特征列表
        sequence_features: 序列特征列表
        train_df: 训练数据
        valid_df: 验证数据
        device: 设备(cpu/cuda)
        **kwargs: 模型特定参数

    返回:
        success: 是否训练成功
        metrics: 评估指标字典
    """
    print("=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    try:
        # ==============================================================================
        # 1. 确定是否使用序列特征
        # ==============================================================================

        # DIN 和 DIEN 需要序列特征
        if model_name in ["DIN", "DIEN"]:
            seq_feats = sequence_features
        else:
            seq_feats = []

        # ==============================================================================
        # 2. 处理特殊模型的特征要求
        # ==============================================================================

        # MaskNet 要求所有特征具有相同的 proj_dim
        # 为稠密特征设置投影层以匹配稀疏特征的 embedding 维度
        if model_name == "MaskNet":
            from nextrec.basic.features import DenseFeature

            embedding_dim = sparse_features[0].embedding_dim if sparse_features else 16
            adjusted_dense_features = [
                DenseFeature(
                    name=f.name,
                    proj_dim=embedding_dim,  # 投影维度
                    input_dim=f.input_dim,
                    use_projection=True,  # 启用投影层
                )
                for f in dense_features
            ]
        # PNN 也需要统一的特征维度
        elif model_name == "PNN":
            from nextrec.basic.features import DenseFeature

            embedding_dim = sparse_features[0].embedding_dim if sparse_features else 16
            adjusted_dense_features = [
                DenseFeature(
                    name=f.name,
                    proj_dim=embedding_dim,
                    input_dim=f.input_dim,
                    use_projection=True,
                )
                for f in dense_features
            ]
        else:
            adjusted_dense_features = dense_features

        # ==============================================================================
        # 3. 创建模型
        # ==============================================================================

        model = model_class(
            dense_features=adjusted_dense_features,
            sparse_features=sparse_features,
            sequence_features=seq_feats,
            target=["label"],
            device=device,
            session_id=f"ranking_{model_name.lower()}_tutorial",
            **kwargs,
        )

        # ==============================================================================
        # 4. 编译模型
        # ==============================================================================

        model.compile(
            optimizer="adam",
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            loss="binary_crossentropy",  # 二元交叉熵损失
        )

        # ==============================================================================
        # 5. 训练模型
        # ==============================================================================

        model.fit(
            train_data=train_df,
            valid_data=valid_df,
            metrics=["auc", "logloss"],  # 评估指标: AUC 和对数损失
            epochs=1,  # 仅训练1轮用于快速验证
            batch_size=512,
            shuffle=True,
            use_tensorboard=False,  # 不使用 TensorBoard
        )

        # ==============================================================================
        # 6. 评估模型
        # ==============================================================================

        metrics = model.evaluate(valid_df, metrics=["auc", "logloss"], batch_size=512)

        print(f"{model_name} completed successfully")
        return True, metrics

    except Exception as e:
        print(f"{model_name} failed with error: {str(e)}")
        return False, None


def main():
    """
    主函数: 批量运行所有排序模型
    """
    print("=" * 80)
    print("Training all supported ranking models with synthetic data")
    print("=" * 80)

    device = "cpu"

    # ==============================================================================
    # 1. 生成合成数据
    # ==============================================================================

    df, dense_features, sparse_features, sequence_features = generate_ranking_data(
        n_samples=10000,  # 样本数量
        n_dense=5,  # 稠密特征数量
        n_sparse=8,  # 稀疏特征数量
        n_sequences=2,  # 序列特征数量
        user_vocab_size=1000,  # 用户词汇表大小
        item_vocab_size=500,  # 物品词汇表大小
        sparse_vocab_size=50,  # 稀疏特征词汇表大小
        sequence_max_len=20,  # 序列最大长度
        embedding_dim=16,  # embedding 维度
        seed=42,  # 随机种子
    )

    # ==============================================================================
    # 2. 划分训练集和验证集
    # ==============================================================================

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    valid_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")

    # ==============================================================================
    # 3. 定义模型参数
    # ==============================================================================

    # MLP 参数(用于深度网络部分)
    mlp_params = {
        "hidden_dims": [256, 128, 64],  # 隐藏层维度
        "activation": "relu",  # 激活函数
        "dropout": 0.2,  # Dropout 比例
    }
    results = {}

    # 为 DIN 和 DIEN 准备行为特征和候选特征名称
    behavior_feature_name = sequence_features[0].name if sequence_features else None
    candidate_feature_name = "item_id"

    # ==============================================================================
    # 4. 定义要训练的模型列表
    # ==============================================================================

    models_to_train = [
        # 基础模型
        (LR, "LR", {}),  # 逻辑回归,无额外参数
        (FM, "FM", {}),  # 因子分解机,无额外参数
        (FFM, "FFM", {}),  # 域感知因子分解机,无额外参数
        # 欧拉网络
        (EulerNet, "EulerNet", {"num_layers": 2, "num_orders": 8}),  # 欧拉网络,2层,8阶
        # 深度模型
        (DeepFM, "DeepFM", {"mlp_params": mlp_params}),  # DeepFM
        (WideDeep, "WideDeep", {"mlp_params": mlp_params}),  # Wide & Deep
        (DCN, "DCN", {"mlp_params": mlp_params, "cross_num": 3}),  # DCN,3层交叉网络
        (
            xDeepFM,
            "xDeepFM",
            {"mlp_params": mlp_params, "cin_size": [128, 128]},
        ),  # xDeepFM,CIN大小
        # 注意力模型
        (
            AutoInt,
            "AutoInt",
            {
                "att_layer_num": 3,  # 注意力层数
                "att_embedding_dim": 16,  # 注意力 embedding 维度
                "att_head_num": 2,  # 注意力头数
                "att_dropout": 0.2,  # 注意力 Dropout
            },
        ),
        (AFM, "AFM", {"attention_dim": 64, "attention_dropout": 0.2}),  # AFM,注意力维度
        # 乘积神经网络
        (
            PNN,
            "PNN",
            {
                "mlp_params": mlp_params,
                "product_type": "inner",  # 内积类型,可设置为 "outer"
                "outer_product_dim": 64,  # 外积维度
            },
        ),
        # 双线性交互网络
        (
            FiBiNET,
            "FiBiNET",
            {
                "mlp_params": mlp_params,
                "bilinear_type": "field_interaction",  # 双线性类型
                "senet_reduction": 3,  # SENet 压缩比例
            },
        ),
        # 兴趣建模
        (
            DIN,
            "DIN",
            {
                "mlp_params": mlp_params,
                "attention_mlp_params": {  # 注意力 MLP 参数
                    "hidden_dims": [80, 40],
                    "activation": "sigmoid",
                },
                "behavior_feature_name": behavior_feature_name,  # 行为序列特征名
                "candidate_feature_name": candidate_feature_name,  # 候选物品特征名
            },
        ),
        (
            DIEN,
            "DIEN",
            {
                "mlp_params": mlp_params,
                "gru_hidden_size": 32,  # GRU 隐藏层大小
                "attention_mlp_params": {"hidden_dims": [80, 40]},  # 注意力 MLP 参数
                "behavior_feature_name": behavior_feature_name,  # 行为序列特征名
                "candidate_feature_name": candidate_feature_name,  # 候选物品特征名
                "use_negsampling": True,  # 使用负采样
                "neg_behavior_feature_name": "sequence_1",  # 负采样序列特征名
            },
        ),
        # 掩码网络
        (
            MaskNet,
            "MaskNet",
            {
                "mlp_params": mlp_params,
                "architecture": "parallel",  # 架构类型: parallel
                "num_blocks": 3,  # 块数量
                "mask_hidden_dim": 64,  # 掩码隐藏层维度
                "block_hidden_dim": 256,  # 块隐藏层维度
            },
        ),
    ]

    # ==============================================================================
    # 5. 批量训练模型
    # ==============================================================================

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

    # ==============================================================================
    # 6. 打印训练总结
    # ==============================================================================

    print("Test Summary")
    print(f"Total models: {len(models_to_train)}")
    print(f"Successful counts: {successful}")
    print(f"Failed counts: {failed}, Models: {failed_models}")


if __name__ == "__main__":
    main()
