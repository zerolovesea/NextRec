"""
批量运行所有多任务学习模型示例

文件说明:
    本示例演示如何批量训练和测试 NextRec 框架支持的所有多任务学习模型。
    通过统一的训练接口和合成数据,可以快速验证各个多任务模型的功能和性能。

主要功能:
    - 生成合成的多任务学习数据
    - 批量训练多个多任务模型
    - 统一的模型训练和评估流程
    - 收集训练结果和错误信息

支持的模型:
    1. APG (Adaptive Parameter Generation): 自适应参数生成网络
    2. CrossStitch: 交叉缝合网络,任务间软参数共享
    3. ESCM (Entire Space Cross-Task Model): 全空间跨任务模型
    4. ESMM (Entire Space Multi-Task Model): 全空间多任务模型
    5. HMOE (Hierarchical Mixture of Experts): 层次化专家混合
    6. MMOE (Multi-gate Mixture of Experts): 多门控专家混合
    7. PEPNet (Parameter and Embedding Personalized Network): 参数和嵌入个性化网络
    8. PLE (Progressive Layered Extraction): 渐进式分层提取
    9. POSO (Partially Ordered Set Optimization): 偏序集优化
    10. ShareBottom: 共享底层网络

使用方法:
    直接运行此脚本:
        python tutorials/run_all_multitask_models.py

数据要求:
    使用合成数据,不需要外部数据文件。脚本会自动生成:
        - 稠密特征
        - 稀疏特征
        - 序列特征
        - 多个任务标签(点击、转化)

输出:
    - 各模型的训练日志
    - 评估指标
    - 训练成功/失败统计
    - 失败模型列表

作者: Yang Zhou, zyaztec@gmail.com
创建日期: 2025-12-06
最后更新: 2026-01-28
"""

from nextrec.models.multi_task.apg import APG
from nextrec.models.multi_task.cross_stitch import CrossStitch
from nextrec.models.multi_task.escm import ESCM
from nextrec.models.multi_task.esmm import ESMM
from nextrec.models.multi_task.hmoe import HMOE
from nextrec.models.multi_task.mmoe import MMOE
from nextrec.models.multi_task.pepnet import PEPNet
from nextrec.models.multi_task.ple import PLE
from nextrec.models.multi_task.poso import POSO
from nextrec.models.multi_task.share_bottom import ShareBottom

from nextrec.utils.data import generate_multitask_data


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
    训练单个多任务学习模型

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
        # 1. 创建模型
        # ==============================================================================

        model = model_class(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            device=device,
            session_id=f"multitask_{model_name.lower()}_tutorial",
            **kwargs,
        )

        # ==============================================================================
        # 2. 编译模型
        # ==============================================================================

        # 使用 GradNorm 动态损失权重调整
        model.compile(
            optimizer="adam",
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            loss=["bce"] * len(kwargs.get("target", ["task1", "task2"])),  # 所有任务使用BCE损失
            loss_weights={
                "method": "grad_norm",
                "alpha": 1.5,
                "lr": 0.025,
            },  # GradNorm参数
        )

        # ==============================================================================
        # 3. 训练模型
        # ==============================================================================

        model.fit(
            train_data=train_df,
            valid_data=valid_df,
            epochs=1,  # 仅训练1轮用于快速验证
            batch_size=512,
            shuffle=True,
            use_tensorboard=False,  # 不使用 TensorBoard
            user_id_column="user_id",
        )

        # ==============================================================================
        # 4. 评估模型
        # ==============================================================================

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
    """
    主函数: 批量运行所有多任务学习模型
    """
    print("=" * 80)
    print("Training all supported multi-task models with synthetic data")
    print("=" * 80)

    device = "cpu"

    # ==============================================================================
    # 1. 生成合成数据
    # ==============================================================================

    df, dense_features, sparse_features, sequence_features = generate_multitask_data(
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

    # 塔网络参数(较大的网络)
    tower_params = {"hidden_dims": [256, 128, 64], "activation": "relu", "dropout": 0.2}
    # 共享层参数
    shared_mlp_params = {"hidden_dims": [128], "activation": "relu", "dropout": 0.1}
    # 任务特定层参数
    task_mlp_params = {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.1}
    # 小型塔网络参数
    small_tower_params = {
        "hidden_dims": [128, 64],
        "activation": "relu",
        "dropout": 0.1,
    }
    # 专家网络参数
    expert_mlp_params = {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.1}
    # 门控网络参数
    gate_mlp_params = {"hidden_dims": [64], "activation": "relu", "dropout": 0.1}
    # 任务权重网络参数
    task_weight_params = {"hidden_dims": [64], "activation": "relu", "dropout": 0.1}

    results = {}

    # ==============================================================================
    # 4. 定义要训练的模型列表
    # ==============================================================================

    models_to_train = [
        (
            APG,
            "APG",
            {
                "mlp_params": {"hidden_dims": [128, 64], "activation": "relu"},
                "scene_features": ["sparse_0"],  # 场景特征
                "target": ["click", "conversion"],
            },
        ),
        (
            CrossStitch,
            "CrossStitch",
            {
                "shared_mlp_params": shared_mlp_params,  # 共享层参数
                "task_mlp_params": task_mlp_params,  # 任务特定层参数
                "tower_mlp_params": {
                    "hidden_dims": [64],
                    "activation": "relu",
                },  # 塔参数
                "target": ["click", "conversion"],
            },
        ),
        (
            ESCM,
            "ESCM",
            {
                "ctr_mlp_params": small_tower_params,  # CTR塔参数
                "cvr_mlp_params": small_tower_params,  # CVR塔参数
                "target": ["click", "conversion", "ctcvr"],  # 三个任务
            },
        ),
        (
            ESMM,
            "ESMM",
            {
                "ctr_mlp_params": tower_params,  # CTR塔参数
                "cvr_mlp_params": tower_params,  # CVR塔参数
                "target": ["click", "ctcvr"],  # 两个任务
            },
        ),
        (
            HMOE,
            "HMOE",
            {
                "expert_mlp_params": expert_mlp_params,  # 专家网络参数
                "gate_mlp_params": gate_mlp_params,  # 门控网络参数
                "tower_mlp_params_list": [
                    small_tower_params,
                    small_tower_params,
                ],  # 各任务塔参数
                "task_weight_mlp_params": [
                    task_weight_params,
                    task_weight_params,
                ],  # 任务权重参数
                "num_experts": 4,  # 专家数量
                "target": ["click", "conversion"],
            },
        ),
        (
            MMOE,
            "MMOE",
            {
                "expert_mlp_params": tower_params,  # 专家网络参数
                "tower_mlp_params_list": [tower_params, tower_params],  # 各任务塔参数
                "num_experts": 4,  # 专家数量
                "target": ["click", "conversion"],
            },
        ),
        (
            PEPNet,
            "PEPNet",
            {
                "mlp_params": {
                    "hidden_dims": tower_params["hidden_dims"],
                    "activation": tower_params["activation"],
                    "dropout": tower_params["dropout"],
                },
                "domain_features": ["sparse_0"],  # 领域特征
                "user_features": ["user_id"],  # 用户特征
                "item_features": ["item_id"],  # 物品特征
                "target": ["click", "conversion"],
            },
        ),
        (
            PLE,
            "PLE",
            {
                "shared_expert_mlp_params": tower_params,  # 共享专家参数
                "specific_expert_mlp_params": [
                    tower_params,
                    tower_params,
                ],  # 各任务专用专家参数
                "tower_mlp_params_list": [tower_params, tower_params],  # 各任务塔参数
                "num_shared_experts": 2,  # 共享专家数量
                "num_specific_experts": 2,  # 各任务专用专家数量
                "num_levels": 2,  # PLE层数
                "target": ["click", "conversion"],
            },
        ),
        (
            POSO,
            "POSO",
            {
                "main_dense_features": [
                    "dense_0",
                    "dense_1",
                    "dense_2",
                ],  # 主特征(稠密)
                "main_sparse_features": ["user_id", "item_id"],  # 主特征(稀疏)
                "main_sequence_features": [],  # 主特征(序列)
                "pc_dense_features": ["dense_3", "dense_4"],  # 后置特征(稠密)
                "pc_sparse_features": ["sparse_0"],  # 后置特征(稀疏)
                "pc_sequence_features": [],  # 后置特征(序列)
                "tower_mlp_params_list": [
                    small_tower_params,
                    small_tower_params,
                ],  # 塔参数
                "target": ["click", "conversion"],
                "architecture": "mlp",  # 架构类型
            },
        ),
        (
            ShareBottom,
            "ShareBottom",
            {
                "bottom_mlp_params": tower_params,  # 底层网络参数
                "tower_mlp_params_list": [tower_params, tower_params],  # 各任务塔参数
                "target": ["click", "conversion"],
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
