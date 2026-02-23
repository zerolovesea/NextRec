---
title: 快速开始
description: NextRec 快速入门指南
---

# 快速开始

这份文档将帮助开发者在几分钟里快速完成一个推荐模型的训练流程。在此之前需要已经安装Nextrec，开发者可以通过`pip install nextrec`来快速安装。


## 快速训练一个精排模型

下面用 DeepFM 在 MovieLens 数据集上完成从特征定义到训练、评估的全流程。开发者可以直接执行python文件，或在jupyter notebook环境下执行。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM

# 1) 读取数据
df = pd.read_csv("https://raw.githubusercontent.com/zerolovesea/NextRec/main/dataset/movielens_100k.csv")

# 2) 定义特征
dense_features = [DenseFeature("age")]
sparse_features = [
    SparseFeature("user_id", vocab_size=df["user_id"].max() + 1, embedding_dim=16),
    SparseFeature("item_id", vocab_size=df["item_id"].max() + 1, embedding_dim=16),
    SparseFeature("gender", vocab_size=df["gender"].max() + 1, embedding_dim=16),
    SparseFeature("occupation", vocab_size=df["occupation"].max() + 1, embedding_dim=16),
]

# 3) 划分训练/验证集
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# 4) 实例化并编译模型
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.2},
    target="label",
    device="cpu",
    session_id="movielens_deepfm",   # 管理实验日志与检查点
)

# 优化器/损失/学习率调度器统一在 compile 中设置
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="binary_crossentropy",
)

# 5) 训练
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "recall", "precision"],
    epochs=2,
    batch_size=512,
    shuffle=True,
)
```

## 训练日志与指标展示

NextRec提供了系统性且清晰可读的日志管理系统，在开始训练后，你将会看到由多个部分组成的输出，分别包含了：

- **Feature Configuration：特征的定义**
- **Model Parameters：模型参数与架构**
- **Training Configuration：训练参数，包括任务类型，损失函数，优化器参数，评估指标，checkpoint保存路径等**
- **Data Summary：数据分布，包括训练和验证集的分布和比例，以及相应组成的Dataloader的参数**
- **Training：在训练过程中产出的训练和评估指标**

```
Model Summary: DEEPFM


Feature Configuration
--------------------------------------------------------------------------------
Dense Features (1):
  1. age                 

Sparse Features (4):
  #    Name           Vocab Size        Embed Name  Embed Dim
  ---- ------------ ------------ ----------------- ----------
  1    user_id               943           user_id         16
  2    item_id              1349           item_id         16
  3    gender                  2            gender         16
  4    occupation             21        occupation         16

Model Parameters
--------------------------------------------------------------------------------
Model Architecture:
DeepFM(
  (embedding): EmbeddingLayer(
    (embed_dict): ModuleDict(
      (user_id): Embedding(943, 16, padding_idx=0)
      (item_id): Embedding(1349, 16, padding_idx=0)
      (gender): Embedding(2, 16, padding_idx=0)
      (occupation): Embedding(21, 16, padding_idx=0)
    )
    (dense_transforms): ModuleDict()
    (sequence_poolings): ModuleDict()
  )
  (linear): LR(
    (fc): Linear(in_features=64, out_features=1, bias=True)
  )
  (fm): FM()
  (mlp): MLP(
    (mlp): Sequential(
      (0): Linear(in_features=65, out_features=256, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.2, inplace=False)
      (3): Linear(in_features=256, out_features=128, bias=True)
      (4): ReLU()
      (5): Dropout(p=0.2, inplace=False)
      (6): Linear(in_features=128, out_features=1, bias=True)
    )
  )
  (prediction_layer): TaskHead(
    (prediction): PredictionLayer()
  )
)

Total Parameters:        87,027
Trainable Parameters:    87,027
Non-trainable Parameters: 0
Layer-wise Parameters:
  embedding                     : 37,040
  linear                        : 65
  mlp                           : 49,921
  prediction_layer              : 1

Training Configuration
--------------------------------------------------------------------------------
Task Type:               binary
Number of Tasks:         1
Metrics:                 ['auc', 'recall', 'precision']
Target Columns:          ['label']
Device:                  cpu
Optimizer:               adam
  lr                       : 0.001
  weight_decay             : 1e-05
Loss Function:           binary_crossentropy
Loss Params:
  weight                   : None
  reduction                : mean
Loss Weights:            None
GradNorm Enabled:        False
Regularization:
  Embedding L1:          0.0
  Embedding L2:          0.0
  Dense L1:              0.0
  Dense L2:              0.0
Other Settings:
  Early Stop Patience:   20
  Max Gradient Norm:     1.0
  Session ID:            movielens_deepfm
  Features Config Path:  nextrec_logs/movielens_deepfm/features_config.pkl
  Latest Checkpoint:     nextrec_logs/movielens_deepfm/DEEPFM_checkpoint.pt
  Note:                  None

Data Summary
--------------------------------------------------------------------------------
Train Samples:                     79,429
label:
  0:                                 35339 (44.49%)
  1:                                 44090 (55.51%)
Train DataLoader:
  Batch Size:                        512
  Num Workers:                       0
  Pin Memory:                        False
  Persistent Workers:                False
  Sampler:                           RandomSampler

Valid Samples:                     19,858
label:
  0:                                 8783 (44.23%)
  1:                                 11075 (55.77%)
Valid DataLoader:
  Batch Size:                        512
  Num Workers:                       0
  Pin Memory:                        False
  Persistent Workers:                False
  Sampler:                           SequentialSampler

TensorBoard logs saved to: nextrec_logs/movielens_deepfm/tensorboard
To view logs, run:
    tensorboard --logdir nextrec_logs/movielens_deepfm/tensorboard --port 6006

[Training]
--------------------------------------------------------------------------------
Start training:                    2 epochs
Model device:                      cpu

  Epoch 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 156/156 0:00:01 0:00:00

    Epoch 1/2 - Train (loss=0.6150)    
╭───────┬────────┬────────┬───────────╮
│ Task  │    auc │ recall │ precision │
├───────┼────────┼────────┼───────────┤
│ label │ 0.7132 │ 0.8407 │    0.6457 │
╰───────┴────────┴────────┴───────────╯

           Epoch 1/2 - Valid           
╭───────┬────────┬────────┬───────────╮
│ Task  │    auc │ recall │ precision │
├───────┼────────┼────────┼───────────┤
│ label │ 0.7701 │ 0.7916 │    0.7146 │
╰───────┴────────┴────────┴───────────╯
Saved checkpoint to nextrec_logs/movielens_deepfm/DEEPFM_checkpoint.pt
Saved checkpoint to nextrec_logs/movielens_deepfm/DEEPFM_best.pt
Saved best model to:               nextrec_logs/movielens_deepfm/DEEPFM_best.pt with val_auc: 0.770135

  Epoch 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 156/156 0:00:00 0:00:00

    Epoch 2/2 - Train (loss=0.5605)    
╭───────┬────────┬────────┬───────────╮
│ Task  │    auc │ recall │ precision │
├───────┼────────┼────────┼───────────┤
│ label │ 0.7776 │ 0.7911 │    0.7204 │
╰───────┴────────┴────────┴───────────╯

           Epoch 2/2 - Valid           
╭───────┬────────┬────────┬───────────╮
│ Task  │    auc │ recall │ precision │
├───────┼────────┼────────┼───────────┤
│ label │ 0.7750 │ 0.7670 │    0.7275 │
╰───────┴────────┴────────┴───────────╯
Saved checkpoint to nextrec_logs/movielens_deepfm/DEEPFM_checkpoint.pt
Saved checkpoint to nextrec_logs/movielens_deepfm/DEEPFM_best.pt
Saved best model to:               nextrec_logs/movielens_deepfm/DEEPFM_best.pt with val_auc: 0.775013

Restoring model weights from epoch: 2 with best val_auc: 0.775013

Training finished.

Load best model from:             nextrec_logs/movielens_deepfm/DEEPFM_best.pt
```

## 使用训练完的模型进行推理

在训练完成后，我们可以训练完的产物进行推理。在上一部里，我们注意到命令行输出了模型被保存在：`nextrec_logs/movielens_deepfm`这个路径下，这是我们需要加载的完整路径，其中包含了多个文件：

- **DEEPFM_best.pt：验证集上表现最好的模型**
- **DEEPFM_checkpoint.pt：最后一个epoch训练的模型**
- **features_config.pkl：特征定义的保存文件**
- **runs_log.txt & training_metrics.jsonl：训练的日志文件**

```python
import pandas as pd
from nextrec.models.ranking.deepfm import DeepFM

# 1) 读取数据（同训练示例一致）
df = pd.read_csv(
	"https://raw.githubusercontent.com/zerolovesea/NextRec/main/dataset/movielens_100k.csv"
)

# 2) 加载模型
checkpoint_path = "nextrec_logs/movielens_deepfm"  # 需要和训练日志下的模型保存的路径保持一致
loaded_model = DeepFM.from_checkpoint(
	checkpoint_path=checkpoint_path,
    mlp_params={"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.2}, 
	map_location="cpu",
	device="cpu",
)

# 3) 推理
pred_df = loaded_model.predict(df, batch_size=512, return_dataframe=True)
print(pred_df.head())
```

在命令行或jupyter notebook进行执行，你将会看到以下输出：

```
Predicting ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 194/194 0:00:00 0:00:00
      label
0  0.722213
1  0.826596
2  0.115219
3  0.720532
4  0.812802
```

## 常见下一步

- [API 文档](apis/index.md) - NextRec的API文档
- [NextRec CLI](cli/index.md) - 使用命令行工具来训练和推理
