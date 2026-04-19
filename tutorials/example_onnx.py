"""
ONNX 模型导出与推理示例

本示例演示如何使用 NextRec 框架训练深度学习模型并导出为 ONNX 格式,以便在生产环境中进行
高效推理。示例涵盖多任务模型(MMOE、PLE)和排序模型(DeepFM)的训练、ONNX 导出和推理对比。

主要功能:
    - 训练多任务学习模型(MMOE、PLE)
    - 训练排序模型(DeepFM)
    - 将训练好的 PyTorch 模型导出为 ONNX 格式
    - 使用 ONNX Runtime 进行推理
    - 对比 PyTorch 和 ONNX 推理结果

使用方法:
    直接运行此脚本:
        python tutorials/example_onnx.py

    前置条件:
        - 安装 NextRec 的 ONNX 可选依赖: pip install "nextrec[onnx]"

数据要求:
    使用合成数据,不需要外部数据文件。脚本会自动生成:
        - 多任务数据: 包含点击和转化两个任务标签
        - 排序数据: 包含二分类标签

模型说明:
    1. MMOE (Multi-gate Mixture-of-Experts):
       - 使用多个专家网络和门控机制
       - 适用于多任务学习场景

    2. PLE (Progressive Layered Extraction):
       - 渐进式分层提取,分离共享专家和任务专家
       - 更好地平衡任务间的知识共享与任务特异性

    3. DeepFM:
       - 结合 FM 和深度神经网络
       - 适用于点击率预估等排序任务

输出:
    - 导出的 ONNX 模型文件(.onnx)
    - PyTorch 推理结果
    - ONNX 推理结果
    - 结果对比(验证导出正确性)

作者: NextRec Team
创建日期: 2026
最后更新: 2026-01-28
"""

from __future__ import annotations


from sklearn.model_selection import train_test_split

from nextrec.models.multitask.mmoe import MMOE
from nextrec.models.multitask.ple import PLE
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.utils.data import generate_multitask_data, generate_ranking_data


def run_multitask_model(model_class, model_name: str, model_kwargs: dict):
    """
    训练多任务模型并导出为 ONNX 格式

    参数:
        model_class: 模型类(MMOE 或 PLE)
        model_name: 模型名称(用于日志输出)
        model_kwargs: 模型初始化参数
    """
    print("=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    # ==============================================================================
    # 1. 生成合成数据
    # ==============================================================================

    df, dense_features, sparse_features, sequence_features = generate_multitask_data(
        n_samples=2000,  # 样本数量
        n_dense=4,  # 稠密特征数量
        n_sparse=6,  # 稀疏特征数量
        n_sequences=2,  # 序列特征数量
        user_vocab_size=500,  # 用户词汇表大小
        item_vocab_size=300,  # 物品词汇表大小
        sparse_vocab_size=50,  # 稀疏特征词汇表大小
        sequence_max_len=10,  # 序列最大长度
        embedding_dim=8,  # embedding 维度
        seed=42,  # 随机种子
    )

    # 划分训练集和验证集
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

    # ==============================================================================
    # 2. 构建并训练模型
    # ==============================================================================

    model = model_class(
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        target=["click", "conversion"],  # 两个任务:点击和转化
        device="cpu",
        session_id=f"onnx_{model_name.lower()}_example",
        **model_kwargs,
    )

    # 编译模型
    model.compile(
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        loss=["bce", "bce"],  # 两个任务都使用二元交叉熵损失
    )

    # 训练模型
    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        metrics=["auc"],  # 使用 AUC 作为评估指标
        epochs=1,  # 仅训练1轮用于演示
        batch_size=256,
        shuffle=True,
        use_tensorboard=False,  # 不使用 TensorBoard
    )

    # ==============================================================================
    # 3. 导出 ONNX 模型
    # ==============================================================================

    onnx_path = model.export_onnx(batch_size=4)
    print(f"Exported ONNX: {onnx_path}")

    # ==============================================================================
    # 4. 对比 PyTorch 和 ONNX 推理结果
    # ==============================================================================

    # 选取部分验证数据用于推理测试
    sample_df = valid_df.head(32)

    # PyTorch 模型推理
    torch_pred = model.predict(sample_df, batch_size=32, return_dataframe=True)

    # ONNX 模型推理
    onnx_pred = model.predict(
        sample_df,
        batch_size=4,  # ONNX 推理批次大小
        return_dataframe=True,
        inference_model=onnx_path,
    )

    # 打印推理结果
    print("Torch predictions (head):")
    print(torch_pred.head())
    print("ONNX predictions (head):")
    print(onnx_pred.head())


def run_deepfm():
    """
    训练 DeepFM 排序模型并导出为 ONNX 格式
    """
    print("=" * 80)
    print("Training DeepFM")
    print("=" * 80)

    # ==============================================================================
    # 1. 生成合成数据
    # ==============================================================================

    df, dense_features, sparse_features, sequence_features = generate_ranking_data(
        n_samples=2000,  # 样本数量
        n_dense=4,  # 稠密特征数量
        n_sparse=6,  # 稀疏特征数量
        n_sequences=1,  # 序列特征数量
        user_vocab_size=500,
        item_vocab_size=300,
        sparse_vocab_size=50,
        sequence_max_len=10,
        embedding_dim=8,
        seed=123,
    )

    # 划分训练集和验证集
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

    # ==============================================================================
    # 2. 构建并训练 DeepFM 模型
    # ==============================================================================

    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        mlp_params={"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.1},
        target="label",
        device="cpu",
        session_id="onnx_deepfm_example",
    )

    # 编译模型
    model.compile(
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        loss="bce",
    )

    # 训练模型
    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        metrics=["auc"],
        epochs=1,  # 仅训练1轮用于演示
        batch_size=256,
        shuffle=True,
        use_tensorboard=False,
    )

    # ==============================================================================
    # 3. 导出 ONNX 模型
    # ==============================================================================

    onnx_path = model.export_onnx(batch_size=4)
    print(f"Exported ONNX: {onnx_path}")

    # ==============================================================================
    # 4. 对比 PyTorch 和 ONNX 推理结果
    # ==============================================================================

    sample_df = valid_df.head(32)

    # PyTorch 模型推理
    torch_pred = model.predict(sample_df, batch_size=32, return_dataframe=True)

    # ONNX 模型推理
    onnx_pred = model.predict(
        sample_df,
        batch_size=4,
        return_dataframe=True,
        inference_model=onnx_path,
    )

    print("Torch predictions (head):")
    print(torch_pred.head())
    print("ONNX predictions (head):")
    print(onnx_pred.head())


def main():
    """
    主函数: 依次运行多任务模型和排序模型的 ONNX 导出示例
    """
    # ==============================================================================
    # 训练并导出 MMOE 模型
    # ==============================================================================

    run_multitask_model(
        MMOE,
        "MMOE",
        {
            "expert_mlp_params": {"hidden_dims": [64, 32], "activation": "relu"},
            "tower_mlp_params_list": [
                {"hidden_dims": [64, 32], "activation": "relu"},
                {"hidden_dims": [64, 32], "activation": "relu"},
            ],
            "num_experts": 4,  # 4个专家网络
        },
    )

    # ==============================================================================
    # 训练并导出 PLE 模型
    # ==============================================================================

    run_multitask_model(
        PLE,
        "PLE",
        {
            "shared_expert_mlp_params": {"hidden_dims": [64, 32], "activation": "relu"},
            "specific_expert_mlp_params": [
                {"hidden_dims": [64, 32], "activation": "relu"},
                {"hidden_dims": [64, 32], "activation": "relu"},
            ],
            "tower_mlp_params_list": [
                {"hidden_dims": [64, 32], "activation": "relu"},
                {"hidden_dims": [64, 32], "activation": "relu"},
            ],
            "num_shared_experts": 2,  # 2个共享专家
            "num_specific_experts": 2,  # 每个任务2个专用专家
            "num_levels": 2,  # 2层PLE结构
        },
    )

    # ==============================================================================
    # 训练并导出 DeepFM 模型
    # ==============================================================================

    run_deepfm()


if __name__ == "__main__":
    main()
