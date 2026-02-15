"""
Date: create on 09/11/2025
Checkpoint: edit on 15/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Lian J, Zhou X, Zhang F, et al. xdeepfm: Combining explicit and implicit feature interactions for recommender systems[C]//Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018: 1754-1763.
    (https://arxiv.org/abs/1803.05170)

xDeepFM was proposed by Lian et al. in KDD18 to unify explicit and implicit feature interaction learning within a single framework.
Building on the Embedding + MLP architecture, it introduces an additional CIN (Compressed Interaction Network)
that explicitly captures high-order crossed features via convolution over outer products of field embeddings.
The linear part retains first-order information, the deep part (MLP) models implicit non-linear interactions,
and all three branches are trained end-to-end.

Workflows:
- Embedding Layer: transforms sparse/sequence fields into dense vectors
- Linear Part: captures first-order contributions of sparse/sequence fields
- CIN: explicitly builds higher-order feature crosses via convolution over
    outer products of field embeddings, with optional split-half connections
- Deep Part (MLP): models implicit, non-linear interactions across all fields
- Combination: sums outputs from linear, CIN, and deep branches before the
    task-specific prediction layer

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Linear/CIN embedding: sparse + sequence -> input_linear: [Batch, Field_num, Dim_field]
- Linear branch: flatten(input_linear) -> linear_input: [Batch, Field_num * Dim_field] -> y_linear: [Batch, 1]
- CIN branch: input_linear: [Batch, Field_num, Dim_field] -> CIN -> y_cin: [Batch, 1]
- Deep embedding: dense + sparse + sequence -> input_deep: [Batch, Dim_deep]
- Deep branch: input_deep: [Batch, Dim_deep] -> MLP -> y_deep: [Batch, 1]
- Fusion: y_linear + y_cin + y_deep -> y: [Batch, 1]
- Output: y -> prediction layer -> [Batch, 1]

xDeepFM由中科大、北大与微软合作发表于KDD18，将显式与隐式的特征交互学习统一到同一框架。
在延续Embedding + MLP的架构上，额外引入了 CIN（Compressed Interaction Network）
CIN通过对字段嵌入做外积并卷积，显式捕获高阶交叉特征。
线性部分保留一阶信息，深层部分（MLP）建模隐式非线性交互，三者端到端联合训练。

流程：
- 嵌入层：将稀疏/序列特征映射为稠密向量
- 线性部分：建模稀疏/序列特征的一阶贡献
- CIN：通过对字段嵌入做外积并卷积，显式捕获高阶交叉，可选 split-half 以控参
- 深层部分（MLP）：对所有特征进行隐式非线性交互建模
- 融合：线性、CIN、MLP 输出求和后进入任务预测层

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- 线性/CIN embedding：sparse + sequence -> input_linear: [Batch, Field_num, Dim_field]
- 线性分支：flatten(input_linear) -> linear_input: [Batch, Field_num * Dim_field] -> y_linear: [Batch, 1]
- CIN分支：input_linear: [Batch, Field_num, Dim_field] -> CIN -> y_cin: [Batch, 1]
- 深层 embedding：dense + sparse + sequence -> input_deep: [Batch, Dim_deep]
- 深层分支：input_deep: [Batch, Dim_deep] -> MLP -> y_deep: [Batch, 1]
- 融合：y_linear + y_cin + y_deep -> y: [Batch, 1]
- 输出：y -> 预测层 -> [Batch, 1]

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import LR, MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeInput


class CIN(nn.Module):
    """Compressed Interaction Network from xDeepFM (Lian et al., 2018)."""

    def __init__(self, input_dim: int, cin_size: list[int], split_half: bool = True):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("[xDeepFM Error]: CIN input_dim must be > 0.")
        if len(cin_size) == 0:
            raise ValueError("[xDeepFM Error]: cin_size must contain at least one layer size.")
        if any(size <= 0 for size in cin_size):
            raise ValueError(f"[xDeepFM Error]: all cin_size values must be positive, got {cin_size}.")
        if split_half:
            odd_sizes = [size for size in cin_size[:-1] if size % 2 != 0]
            if odd_sizes:
                raise ValueError(
                    "[xDeepFM Error]: split_half=True requires every non-last CIN layer size to be even, "
                    f"got cin_size={cin_size}."
                )

        self.num_layers = len(cin_size)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cin_size[i]
            self.conv_layers.append(
                torch.nn.Conv1d(
                    input_dim * prev_dim,
                    cross_layer_size,
                    1,
                    stride=1,
                    dilation=1,
                    bias=True,
                )
            )
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class xDeepFM(BaseModel):
    @property
    def model_name(self):
        return "xDeepFM"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        mlp_params: dict | None = None,
        cin_size: list[int] | None = None,
        split_half: bool = True,
        **kwargs,
    ):
        """
        Initialize xDeepFM model.
        初始化 xDeepFM 模型。

        Args:
            mlp_params: Parameters for deep branch MLP, e.g.
                {"hidden_dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}.
                深层分支 MLP 参数，例如 {"hidden_dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}。
            cin_size: Hidden layer sizes of CIN, e.g. [128, 128].
                CIN 隐层大小列表，例如 [128, 128]。
                When split_half=True, every non-last layer size must be even.
                当 split_half=True 时，除最后一层外都必须为偶数。
            split_half: Whether to split half of CIN feature maps to hidden path.
                是否在 CIN 中使用 split-half 连接策略。

        Notes:
            - Linear/CIN branches require at least one sparse or sequence feature.
            - All sparse/sequence fields used by CIN must share the same effective output dimension.
              SequenceFeature with combiner='concat' changes effective field dimension and may violate this constraint.
            - 线性/CIN 分支至少需要一个 sparse 或 sequence 特征。
            - CIN 使用的 sparse/sequence 字段必须具有相同的有效输出维度；
              SequenceFeature 的 combiner='concat' 会改变字段有效维度，可能触发维度不一致错误。
        """

        if cin_size is None:
            cin_size = [128, 128]
        elif len(cin_size) == 0:
            raise ValueError("[xDeepFM Error]: cin_size cannot be empty. Use positive layer sizes such as [128, 128].")

        mlp_params = mlp_params or {}

        super(xDeepFM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        # Linear part and CIN part: use sparse and sequence features
        self.linear_features = sparse_features + sequence_features

        # Deep part: use all features
        self.deep_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.deep_features)
        if len(self.linear_features) == 0:
            raise ValueError(
                "[xDeepFM Error]: requires at least one sparse or sequence feature for linear/CIN branches."
            )

        linear_field_dims = [self.embedding.compute_output_dim([feature]) for feature in self.linear_features]
        if len(set(linear_field_dims)) != 1:
            feature_dims = ", ".join(
                f"{feature.name}:{dim}" for feature, dim in zip(self.linear_features, linear_field_dims)
            )
            raise ValueError(
                "[xDeepFM Error]: CIN expects all sparse/sequence fields to share the same effective embedding "
                f"dimension for squeeze_dim=False, but got [{feature_dims}]. "
                "SequenceFeature with combiner='concat' changes field dimension (embedding_dim * max_len) and is "
                "not supported unless all linear/CIN fields have the same effective dimension."
            )

        # Linear part
        linear_dim = self.embedding.compute_output_dim(self.linear_features)
        self.linear = LR(linear_dim)

        # CIN part: Compressed Interaction Network
        num_fields = len(self.linear_features)
        self.cin = CIN(input_dim=num_fields, cin_size=cin_size, split_half=split_half)

        # Deep part: DNN
        self.mlp = MLP(input_dim=self.embedding.compute_output_dim(self.deep_features), **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.task)

        # Register regularization weights
        self.register_regularization_weights(embedding_attr="embedding", include_modules=["linear", "cin", "mlp"])

    def forward(self, x):
        # input_linear: [Batch, Field_num, Dim_field]
        input_linear = self.embedding(x=x, features=self.linear_features, squeeze_dim=False)

        # linear_input: [Batch, Field_num * Dim_field] -> y_linear: [Batch, 1]
        y_linear = self.linear(input_linear.flatten(start_dim=1))

        # CIN branch: [Batch, Field_num, Dim_field] -> y_cin: [Batch, 1]
        y_cin = self.cin(input_linear)  # [B, 1]

        # input_deep: [Batch, Dim_deep]
        input_deep = self.embedding(x=x, features=self.deep_features, squeeze_dim=True)
        # Deep branch: [Batch, Dim_deep] -> y_deep: [Batch, 1]
        y_deep = self.mlp(input_deep)  # [B, 1]

        # Fusion: [Batch, 1] + [Batch, 1] + [Batch, 1] -> [Batch, 1]
        y = y_linear + y_cin + y_deep
        # Output: [Batch, 1] -> task-specific prediction
        return self.prediction_layer(y)
