"""
Date: create on 09/11/2025
Checkpoint: edit on 21/03/2026
Author: Yang Zhou,zyaztec@gmail.com
Reference:
- [1] Ma X, Zhao L, Huang G, Wang Z, Hu Z, Zhu X, Gai K. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate. In: Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’18), 2018, pp. 1137–1140.
URL: https://dl.acm.org/doi/10.1145/3209978.3210007

ESMM was proposed by Alibaba team in SIGIR 2018 to address the sample selection bias and data sparsity issues in CVR (Conversion Rate) prediction.
The core idea is to jointly model two related tasks, CTR (Click-Through Rate) and CVR,
and construct an auxiliary task CTCVR (Click-Through & Conversion Rate) to
enable training on the entire impression space.

Workflow:
- Build a shared embedding representation from all impression features
- Feed shared representation into CTR tower and CVR tower independently
- Convert two logits to probabilities via task head to get CTR and CVR
- Compose CTCVR by probability product: CTCVR = CTR * CVR
- Output [CTR, CTCVR] for joint optimization on impression-level labels

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding: all features -> input_flat: [Batch, Dim_embedding]
- CTR tower: input_flat -> ctr_logit: [Batch, 1]
- CVR tower: input_flat -> cvr_logit: [Batch, 1]
- Task head: cat([ctr_logit, cvr_logit], dim=1) -> [Batch, 2] -> [ctr, cvr]
- CTCVR construction: ctcvr = ctr * cvr -> [Batch, 1]
- Output: cat([ctr, ctcvr], dim=1) -> [Batch, 2]

ESMM 由阿里巴巴团队发表在SIGIR’2018，主要为了解决CVR（转化率）预测中的样本选择偏差与数据稀疏问题。
其核心思想是将CTR（点击率）和CVR（转化率）两个相关任务进行联合建模，
并通过构造一个辅助任务CTCVR（点击且转化率）来实现全空间训练。

流程：
- 对曝光样本的全部特征进行共享 embedding 编码
- 共享表征分别输入 CTR 塔与 CVR 塔，得到两个任务 logits
- 通过任务输出头将 logits 映射为 CTR/CVR 概率
- 按概率分解构造 CTCVR：CTCVR = CTR * CVR
- 输出 [CTR, CTCVR]，在曝光空间进行联合训练

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：所有特征拼接展平 -> input_flat: [Batch, Dim_embedding]
- CTR 塔：input_flat -> ctr_logit: [Batch, 1]
- CVR 塔：input_flat -> cvr_logit: [Batch, 1]
- 任务头：cat([ctr_logit, cvr_logit], dim=1) -> [Batch, 2] -> [ctr, cvr]
- 构造 CTCVR：ctcvr = ctr * cvr -> [Batch, 1]
- 最终输出：cat([ctr, ctcvr], dim=1) -> [Batch, 2]

"""

import torch

from nextrec.basic.adapters import TrainingAdapter
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeInput


class ESMMAdapter(TrainingAdapter):
    """
    Adapter for ESMM-style multi-task probability composition.
    """

    def format_model_output(self, model: BaseModel, raw_output):
        if model.training_modes[0] != "pointwise":
            return raw_output
        preds = super().format_model_output(model, raw_output)
        if not isinstance(preds, torch.Tensor):
            return preds
        ctr, cvr = preds.chunk(2, dim=1)
        ctcvr = ctr * cvr
        return torch.cat([ctr, ctcvr], dim=1)


class ESMM(BaseModel):

    @property
    def model_name(self):
        return "ESMM"

    @property
    def default_task(self):
        return ["binary", "binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        ctr_mlp_params: dict,
        cvr_mlp_params: dict,
        task: list[TaskTypeInput] | None = None,
        target: list[str] | None = None,  # Note: ctcvr = ctr * cvr
        **kwargs,
    ):
        """
        Initialize ESMM model.
        初始化 ESMM 模型。

        Args:
            ctr_mlp_params: CTR tower MLP params, e.g.
                {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.2}.
                CTR 塔 MLP 参数，例如 {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.2}。
            cvr_mlp_params: CVR tower MLP params, e.g.
                {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.2}.
                CVR 塔 MLP 参数，例如 {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.2}。
        """

        target = target or ["ctr", "ctcvr"]

        if len(target) != 2:
            raise ValueError(f"ESMM requires exactly 2 targets (ctr and ctcvr), got {len(target)}")

        self.nums_task = len(target)

        super(ESMM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,  # Both CTR and CTCVR are binary classification
            **kwargs,
        )

        for task_type in self.task:
            if task_type != "binary":
                raise ValueError(f"ESMM task must be binary, got '{task_type}'.")

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        # CTR tower
        self.ctr_tower = MLP(input_dim=input_dim, output_dim=1, **ctr_mlp_params)

        # CVR tower
        self.cvr_tower = MLP(input_dim=input_dim, output_dim=1, **cvr_mlp_params)
        self.grad_norm_shared_modules = ["embedding"]
        # Register regularization weights
        self.register_regularization_weights(embedding_attr="embedding", include_modules=["ctr_tower", "cvr_tower"])

    def set_adapter(self):
        self.training_adapter = ESMMAdapter()

    def forward(self, x):
        # Shared embedding flatten: [Batch, Dim_embedding]
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Two-tower logits:
        # ctr_logit: [Batch, 1]
        # cvr_logit: [Batch, 1]
        ctr_logit = self.ctr_tower(input_flat)
        cvr_logit = self.cvr_tower(input_flat)

        # Task head input and outputs:
        # logits: cat([ctr_logit, cvr_logit], dim=1) -> [Batch, 2]
        # preds: [Batch, 2], split to ctr/cvr each [Batch, 1]
        logits = torch.cat([ctr_logit, cvr_logit], dim=1)
        return logits
