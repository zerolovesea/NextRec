---
title: 日志管理
description: session 目录结构、日志与产物保存路径，以及 CLI 的路径配置
---

# 日志管理

NextRec 用 `session_id` 来区分单次训练，会把一次训练/推理的产物组织到一个session目录里，便于复现实验与线上排查。

用户通过在初始化模型时传入 `session_id` 参数来进行定义。例如：

```python
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.2},
    target="label",
    device="cpu",
    session_id="movielens_deepfm",   # 管理实验日志与检查点
)
```

当开始训练后，会在当前路径下生成`nextrec_logs/`路径，用于存放中间文件，示例结构如下：

```
nextrec_logs/
└── {session_id}/
    ├── {MODEL}_checkpoint.pt    # 中间检查点
    ├── {MODEL}_best.pt          # 最佳模型
    ├── feature_config.pkl       # 特征管理中间文件
    ├── runs_log.txt             # 日志
    └── training_metrics.jsonl   # tensorboard日志文件
```

## NextRec CLI 集成

在命令行工具NextRec CLI中，通过修改训练配置文件中的`session`参数来进行调整日志路径和session id。

```yaml
session:
  id: my_experiment
  artifact_root: ./nextrec_logs
```

## 使用Wandb & SwanLab

Wandb和SwanLab是在线训练日志平台，支持将实时的训练指标和实验记录进行上传记录，能够帮助算法工程师在记录和对比多次实验。NextRec也提供了相应的接口，只需要在使用模型的`fit`方法时配置对应参数即可。

### 参数说明

| 参数 | 说明 |
|-----|------|
| `use_wandb` | 是否使用wandb |
| `use_swanlab` | 是否使用swanlab |
| `wandb_api` | wandb api key |
| `swanlab_api` | swanlab api key |
| `wandb_kwargs` | wandb可选的额外参数，例如`{"project": "nextrec"}` |
| `swanlab_kwargs` | swanlab可选的额外参数，例如`{"project": "nextrec"}` |
| `note` | 本次训练的备注信息记录 |

---

## 下一步

