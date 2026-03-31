---
title: 安装
description: NextRec 安装指南（PyPI/源码安装、可选依赖与常见问题）
---

# 安装

NextRec建议环境为：

- Python：3.10+
- PyTorch：2.0+ （对于需要使用CUDA的开发者，需要安装对应版本的PyTorch）

> 建议在虚拟环境内进行安装，以避免和其他项目的依赖项发生冲突

## 安装方式

### 方式 1：安装 PyPI 版本

```bash
pip install nextrec
```

> 说明：`wandb` 和 `swanlab` 属于可选实验跟踪依赖，默认不会随 `nextrec` 一起安装。
>
> 在部分 Linux 环境中，安装 `wandb` 可能会触发源码构建，并因为缺少 `go` 编译环境而失败。因此默认安装命令保持最小依赖集。
>
> 如果你需要 WandB 或 SwanLab，请额外安装：
>
> ```bash
> pip install "nextrec[tracking]"
> ```

### 方式 2：从源码安装

对于有需求的开发者，可以从源码进行安装：

```bash
git clone https://github.com/zerolovesea/NextRec.git
cd NextRec
pip install -e .
# 安装单元测试的依赖项
pip install -r test_requirements.txt 
```

如果你需要实验跟踪功能，还需要继续执行：

```bash
pip install ".[tracking]"
```

## 快速校验

在安装完成后，可以通过**示例脚本**来验证可用性，脚本将会依次使用示例数据集完整训练所有支持的精排/召回/多目标模型：

```bash
git clone https://github.com/zerolovesea/NextRec.git
cd NextRec

# 训练所有支持的精排模型
python tutorials/run_all_ranking_models.py 
# 训练所有支持的召回模型
python tutorials/run_all_retrieval_models.py 
# 训练所有支持的多任务模型
python tutorials/run_all_multitask_models.py  
```

下一步：阅读 [快速开始](getting-started.md)。
