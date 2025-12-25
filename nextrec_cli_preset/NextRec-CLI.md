# NextRec CLI User Guide

NextRec provides a powerful command-line interface (CLI) that supports model training and prediction through YAML configuration files. This document provides a comprehensive guide to using the CLI and configuration specifications.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
  - [Training Configuration](#training-configuration)
  - [Prediction Configuration](#prediction-configuration)
  - [Feature Configuration](#feature-configuration)
  - [Model Configuration](#model-configuration)
- [Command Line Arguments](#command-line-arguments)
- [Complete Examples](#complete-examples)
- [Advanced Features](#advanced-features)

---

## Quick Start

### Installation

```bash
pip install nextrec
```

### Basic Usage

```bash
# Train a model
nextrec --mode=train --train_config=path/to/train_config.yaml

# Run prediction
nextrec --mode=predict --predict_config=path/to/predict_config.yaml
```

---

## Configuration Files

NextRec CLI uses YAML configuration files to define training and prediction pipelines, Ww provide configuration templates under nextrec_cli_preset. There are four types of configuration files:

1. **Training Configuration** (`train_config.yaml`) - Defines the training pipeline
2. **Prediction Configuration** (`predict_config.yaml`) - Defines the prediction pipeline
3. **Feature Configuration** (`feature_config.yaml`) - Defines feature processing methods
4. **Model Configuration** (`model_config.yaml`) - Defines model architecture and parameters

### Training Configuration

The training configuration file defines the complete training pipeline, including data paths, training parameters, optimizers, etc.

#### Configuration Structure

```yaml
session:
  id: my_experiment_session            # Unique experiment identifier
                                       # Used for logging and checkpoint directory naming
  artifact_root: nextrec_logs          # Root directory for all experiment outputs
                                       # Final path: {artifact_root}/{id}/

data:
  path: /path/to/training/data         # Training data path (file or directory)
  format: parquet                      # Data format: csv, parquet, feather
                                       # Use 'auto' for automatic detection
  target: label                        # Target column(s)
                                       # Single-task: 'label' (string)
                                       # Multi-task: ['label1', 'label2'] (list)
  # id_column: user_id                 # Column for user/sample IDs (optional, for GAUC)
  # user_id_column: user_id            # Alternative key for id column
  valid_ratio: 0.2                     # Auto-split: fraction for validation (0.0-1.0)
  # val_path: /path/to/validation/data # Manual: separate validation dataset path
  # valid_path: /path/to/validation/data
  random_state: 2024                   # Random seed for data splitting
  streaming: false                     # Streaming mode for datasets too large for memory
                                       # false: load full dataset into memory
                                       # true: stream data in chunks

feature_config: path/to/feature_config.yaml  # Feature configuration file path
model_config: path/to/model_config.yaml      # Model configuration file path

dataloader:
  train_batch_size: 512                 # Training batch size
  train_shuffle: true                   # Shuffle training data each epoch
  valid_batch_size: 512                 # Validation batch size
  valid_shuffle: false                  # Shuffle validation data
  num_workers: 4                        # Number of parallel data loading workers
                                        # 0 = single process, >0 = multiprocessing
  # chunk_size: 20000                   # Chunk size when streaming=true

train:
  optimizer: adam                       # Optimizer: adam, sgd, adamw, rmsprop, adagrad
  optimizer_params:
    lr: 0.001                          # Learning rate
    weight_decay: 0.0                  # L2 regularization coefficient
  loss: bce                            # Loss function(s)
                                       # Single-task: bce, weighted_bce, focal_loss, mse
                                       # Multi-task: [bce, weighted_bce, focal_loss]
  # loss_params:                       # Optional, per-task loss params
  # loss_weights:                      # Optional loss weights or GradNorm
  #   - pos_weight: 1.0
  #     logits: false
  metrics:                             # Metrics to compute during training/validation
    - auc
    # - gauc
    # - recall
    # - precision
    # - accuracy
    # - f1
    # - logloss
    # - mse
    # - mae
    # - rmse
  epochs: 10                           # Number of training epochs
  batch_size: 512                      # Overrides dataloader.train_batch_size if set
  shuffle: true                        # Shuffle training data
  device: cpu                          # Device: cpu, cuda, cuda:0, mps
```

#### Parameter Description

##### session Section
- `id`: Session identifier; training artifacts will be saved in `{artifact_root}/{id}/` directory
- `artifact_root`: Root directory for artifacts, default is `nextrec_logs`

##### data Section
- `path`: Training data path, supports:
  - Single file: `/path/to/data.csv` or `/path/to/data.parquet`
  - Directory: `/path/to/data_dir/` (automatically reads all files of the same format)
- `format`: Data format, supports `csv`, `parquet`, `feather`, or `auto`
- `target`: Target column name, can be string or list
  - Single target: `target: label`
  - Multiple targets: `target: [label_apply, label_credit]`
- `id_column`: Column for user/sample IDs (optional, required for GAUC)
- `user_id_column`: Alias for `id_column`
- `valid_ratio`: Validation split ratio (0-1), only effective when `val_path` is not specified
- `val_path`: Independent validation data path (optional)
- `valid_path`: Alias for `val_path`
- `random_state`: Random seed for reproducible data splitting
- `streaming`: Whether to use streaming processing
  - `true`: For large datasets, loads data in chunks
  - `false`: Loads all data into memory at once

##### dataloader Section
- `train_batch_size`: Batch size for training
- `train_shuffle`: Whether to shuffle training data
- `valid_batch_size`: Batch size for validation
- `valid_shuffle`: Whether to shuffle validation data
- `num_workers`: Number of data loading workers
- `chunk_size`: Data chunk size for streaming processing (when `streaming=true`)

##### train Section
- `optimizer`: Optimizer type
  - `adam`: Adam optimizer (recommended)
  - `sgd`: Stochastic Gradient Descent
  - `adamw`: AdamW optimizer
- `optimizer_params`: Optimizer parameters
  - `lr`: Learning rate
  - `weight_decay`: Weight decay (L2 regularization)
- `loss`: Loss function(s)
  - `bce`: Binary Cross Entropy
  - `weighted_bce`: BCE with class weights
  - `focal_loss`: Focal Loss (imbalanced data)
  - `mse`: Mean Squared Error
- `loss_params`: Loss function parameters (optional, per task)
- `loss_weights`: Loss weights (list/number) or GradNorm config
- `metrics`: List of evaluation metrics, supports:
  - `auc`: Area Under ROC Curve
  - `recall`: Recall
  - `precision`: Precision
  - `f1`: F1 Score
  - `gauc`: Group AUC
- `epochs`: Number of training epochs
- `batch_size`: Overrides dataloader batch size if set
- `shuffle`: Whether to shuffle training data
- `device`: Computing device
  - `cpu`: CPU
  - `cuda`: NVIDIA GPU
  - `cuda:0`: Specific NVIDIA GPU index
  - `mps`: Apple Silicon GPU

---

### Prediction Configuration

The prediction configuration file defines the model inference pipeline.

#### Configuration Structure

```yaml
checkpoint_path: /path/to/checkpoint/directory   # Required checkpoint directory
# model_config: /path/to/model_config.yaml       # Optional model config override
# session:
#   id: my_experiment_session                    # Optional session ID

predict:
  data_path: /path/to/prediction/data   # Prediction data path
  source_data_format: parquet           # Input data format: csv, parquet, feather
                                        # Use 'auto' for automatic detection
  id_column: user_id                    # ID column name (optional, for linking predictions)
  name: pred                            # Output filename (without extension)
                                        # Final path: {checkpoint_path}/predictions/{name}.{save_data_format}
  save_data_format: csv                 # Output format: csv, parquet, feather
  preview_rows: 5                       # Number of preview rows (output to log)
  batch_size: 512                       # Prediction batch size
  num_workers: 4                        # Number of data loading threads
  device: cpu                           # Computing device
  streaming: true                       # Whether to stream data from disk
  chunk_size: 20000                     # Chunk size for streaming processing
```

#### Parameter Description

- `checkpoint_path`: Path to trained model checkpoint directory
  - Can be directory (automatically selects latest `.pt` file)
  - Can be specific model file: `path/to/model.pt`
- `model_config`: Model config override (optional, auto-searches in checkpoint directory)
- `predict.data_path`: Path to data for prediction
- `predict.source_data_format`: Input data format or `auto`
- `predict.id_column`: ID column name (optional)
  - If specified, prediction results will include this column
- `predict.name`: Output filename (without extension)
- `predict.save_data_format`: Output format
  - `csv`: CSV file
  - `parquet`: Parquet file
  - `feather`: Feather file
- `predict.batch_size`: Batch size for prediction
- `predict.num_workers`: Number of data loading workers
- `predict.streaming`: Whether to stream data from disk
  - `true`: Streaming processing, for large datasets
  - `false`: Load all data into memory (small datasets)
- `predict.chunk_size`: Chunk size for streaming processing
- `predict.preview_rows`: Number of preview rows
  - Displays first N rows of results in log after prediction

---

### Feature Configuration

The feature configuration file defines how to process and transform input features.

#### Configuration Structure

```yaml
dense:
  age:
    processor_config:
      type: numeric                     # Numeric feature
      scaler: standard                  # Scaling method: standard, minmax, robust
    embedding_config:
      name: age                         # Feature name
      input_dim: 1                      # Input dimension
      embedding_dim: 8                  # Embedding dimension
      use_embedding: true               # Whether to use embedding
  
  income:
    processor_config:
      type: numeric
      scaler: minmax
    embedding_config:
      name: income
      input_dim: 1
      embedding_dim: 8
      use_embedding: false              # Don't use embedding, use raw values

sparse:
  city:
    processor_config:
      type: sparse
      encode_method: label              # Encoding method: label, hash, onehot
      # vocab_size: 1000                # Vocabulary size (optional for label encoding)
    embedding_config:
      name: city
      vocab_size: 1000                  # Embedding vocabulary size
      embedding_dim: 16                 # Embedding dimension
  
  gender:
    processor_config:
      type: sparse
      encode_method: hash               # Hash encoding
      hash_size: 100                    # Hash table size
    embedding_config:
      name: gender
      vocab_size: 100
      embedding_dim: 8

sequence:
  click_history:
    processor_config:
      type: sequence
      encode_method: hash               # Sequence encoding method
      hash_size: 10000                  # Hash table size
      max_len: 50                       # Maximum sequence length
      pad_value: 0                      # Padding value
      truncate: post                    # Truncation mode: post, pre
      separator: ","                    # Separator
    embedding_config:
      name: click_history
      vocab_size: 10000                 # Embedding vocabulary size
      embedding_dim: 32                 # Embedding dimension
      padding_idx: 0                    # Padding index
      combiner: mean                    # Aggregation method: mean, sum, attention

# Feature groups (optional, for specific models)
feature_groups:
  user_features:
    - age
    - gender
    - city
  item_features:
    - item_id
    - category
  context_features:
    - time
    - device
```

#### Parameter Description

##### dense (Numeric Features)
- `processor_config.type`: Must be `numeric`
- `processor_config.scaler`: Scaling method
  - `standard`: Standardization (mean 0, std 1)
  - `minmax`: Min-max normalization (scales to 0-1)
  - `robust`: Robust standardization (uses median and quantiles)
- `embedding_config.use_embedding`: Whether to use embedding for numeric features
  - `true`: Use embedding layer
  - `false`: Use normalized values directly

##### sparse (Categorical Features)
- `processor_config.type`: Must be `sparse`
- `processor_config.encode_method`: Encoding method
  - `label`: Label encoding (recommended)
  - `hash`: Hash encoding (suitable for high-cardinality features)
  - `onehot`: One-hot encoding
- `processor_config.hash_size`: Hash table size (only for hash encoding)
- `embedding_config.vocab_size`: Embedding vocabulary size
- `embedding_config.embedding_dim`: Embedding dimension

##### sequence (Sequence Features)
- `processor_config.type`: Must be `sequence`
- `processor_config.encode_method`: Encoding method (usually `hash` or `label`)
- `processor_config.max_len`: Maximum sequence length
- `processor_config.pad_value`: Padding value (usually 0)
- `processor_config.truncate`: Truncation mode
  - `post`: Truncate from end
  - `pre`: Truncate from beginning
- `processor_config.separator`: Sequence separator
- `embedding_config.padding_idx`: Padding index
- `embedding_config.combiner`: Sequence aggregation method
  - `mean`: Average pooling
  - `sum`: Sum pooling
  - `attention`: Attention mechanism

##### feature_groups (Feature Groups)
Used for models that require feature grouping (such as MaskNet, PLE, etc.).

---

### Model Configuration

The model configuration file defines model architecture and hyperparameters.

#### General Format

```yaml
model: model_name                       # Model name
params:
  # Model-specific parameters
  param1: value1
  param2: value2
```

#### Supported Models

##### Ranking Models

**DeepFM**
```yaml
model: deepfm
params:
  mlp_params:
    dims: [256, 128, 64]               # MLP layer dimensions
    activation: relu                    # Activation function
    dropout: 0.3                        # Dropout ratio
  embedding_l2_reg: 1.0e-5             # Embedding L2 regularization
  dense_l2_reg: 1.0e-4                 # Dense layer L2 regularization
```

**DIN (Deep Interest Network)**
```yaml
model: din
params:
  attention_mlp_dims: [80, 40]         # Attention network dimensions
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**DCN (Deep & Cross Network)**
```yaml
model: dcn
params:
  cross_num: 3                         # Number of cross layers
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**DCN-V2**
```yaml
model: dcn_v2
params:
  cross_num: 3                         # Number of cross layers
  cross_type: matrix                   # Cross type: vector, matrix, mix
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**MaskNet**
```yaml
model: masknet
params:
  architecture: parallel                 # Model type: parallel, serial
  num_blocks: 3                        # Number of blocks
  mask_hidden_dim: 64                  # Mask hidden layer dimension
  block_hidden_dim: 256                # Block hidden layer dimension
  block_dropout: 0.2                   # Dropout ratio
  embedding_l1_reg: 1.0e-6            # Embedding L1 regularization
  dense_l1_reg: 1.0e-5                # Dense L1 regularization
  embedding_l2_reg: 1.0e-5            # Embedding L2 regularization
  dense_l2_reg: 1.0e-4                # Dense L2 regularization
```

**AutoInt**
```yaml
model: autoint
params:
  attention_size: 16                   # Attention dimension
  num_heads: 2                         # Number of multi-head attention heads
  num_layers: 3                        # Number of transformer layers
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

##### Multi-Task Models

**MMOE (Multi-gate Mixture-of-Experts)**
```yaml
model: mmoe
params:
  num_experts: 8                       # Number of expert networks
  expert_dims: [256, 128]              # Expert network dimensions
  gate_dims: [64]                      # Gate network dimensions
  tower_dims: [64, 32]                 # Tower network dimensions
  dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**PLE (Progressive Layered Extraction)**
```yaml
model: ple
params:
  num_levels: 2                        # Number of PLE levels
  num_experts_specific: 4              # Number of task-specific experts
  num_experts_shared: 4                # Number of shared experts
  expert_dims: [256, 128]
  gate_dims: [64]
  tower_dims: [64, 32]
  dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**ESMM (Entire Space Multi-Task Model)**
```yaml
model: esmm
params:
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

---

## Command Line Arguments

### Basic Command

```bash
nextrec --mode=MODE [OPTIONS]
```

### Parameter Description

- `--mode`: Running mode (required)
  - `train`: Training mode
  - `predict`: Prediction mode

- `--train_config`: Training configuration file path (required for training mode)
- `--predict_config`: Prediction configuration file path (required for prediction mode)
- `--config`: General configuration file path (deprecated, use above parameters instead)

### Examples

```bash
# Train a model
nextrec --mode=train --train_config=configs/deepfm_train.yaml

# Run prediction
nextrec --mode=predict --predict_config=configs/deepfm_predict.yaml
```

---

## Complete Examples

### Example 1: Training a DeepFM Model

#### 1. Prepare Data

Assume you have an e-commerce dataset `ecommerce_data.csv`:

```csv
user_id,item_id,age,gender,city,category,price,click_history,label
1,101,25,M,BJ,Electronics,999.0,"[98,99,100]",1
2,102,30,F,SH,Fashion,299.0,"[101,102,103]",0
...
```

#### 2. Create Feature Configuration `feature_config.yaml`

```yaml
dense:
  age:
    processor_config:
      type: numeric
      scaler: standard
    embedding_config:
      name: age
      input_dim: 1
      embedding_dim: 8
      use_embedding: true
  
  price:
    processor_config:
      type: numeric
      scaler: minmax
    embedding_config:
      name: price
      input_dim: 1
      embedding_dim: 8
      use_embedding: true

sparse:
  user_id:
    processor_config:
      type: sparse
      encode_method: label
    embedding_config:
      name: user_id
      vocab_size: 100000
      embedding_dim: 32
  
  item_id:
    processor_config:
      type: sparse
      encode_method: label
    embedding_config:
      name: item_id
      vocab_size: 50000
      embedding_dim: 32
  
  gender:
    processor_config:
      type: sparse
      encode_method: label
    embedding_config:
      name: gender
      vocab_size: 10
      embedding_dim: 8
  
  city:
    processor_config:
      type: sparse
      encode_method: hash
      hash_size: 1000
    embedding_config:
      name: city
      vocab_size: 1000
      embedding_dim: 16
  
  category:
    processor_config:
      type: sparse
      encode_method: label
    embedding_config:
      name: category
      vocab_size: 100
      embedding_dim: 16

sequence:
  click_history:
    processor_config:
      type: sequence
      encode_method: label
      max_len: 50
      pad_value: 0
      truncate: post
      separator: ","
    embedding_config:
      name: click_history
      vocab_size: 50000
      embedding_dim: 32
      padding_idx: 0
      combiner: mean
```

#### 3. Create Model Configuration `model_config.yaml`

```yaml
model: deepfm
params:
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

#### 4. Create Training Configuration `train_config.yaml`

```yaml
session:
  id: deepfm_ecommerce
  artifact_root: nextrec_logs

data:
  path: data/ecommerce_data.csv
  format: csv
  target: label
  valid_ratio: 0.2
  random_state: 2024
  streaming: false

feature_config: feature_config.yaml
model_config: model_config.yaml

dataloader:
  train_batch_size: 512
  train_shuffle: true
  valid_batch_size: 512
  valid_shuffle: false
  num_workers: 4

train:
  optimizer: adam
  optimizer_params:
    lr: 0.001
    weight_decay: 0.00001
  loss: focal
  loss_params:
    alpha: 0.25
    gamma: 2.0
  loss_weights:
    method: grad_norm
    alpha: 1.5
    lr: 0.025
    init_ema_steps: 50
    init_ema_decay: 0.9
  metrics:
    - auc
    - recall
    - precision
  epochs: 10
  batch_size: 512
  shuffle: true
  device: cuda
```

#### 5. Run Training

```bash
nextrec --mode=train --train_config=train_config.yaml
```

#### 6. Create Prediction Configuration `predict_config.yaml`

```yaml
session:
  id: deepfm_ecommerce_predict
  artifact_root: nextrec_logs

checkpoint_path: nextrec_logs/deepfm_ecommerce

predict:
  data_path: data/test_data.csv
  source_data_format: csv
  id_column: user_id
  name: deepfm_predictions
  save_data_format: csv
  batch_size: 1024
  num_workers: 4
  device: cuda
  streaming: true
  chunk_size: 20000
  preview_rows: 10
```

#### 7. Run Prediction

```bash
nextrec --mode=predict --predict_config=predict_config.yaml
```

---

### Example 2: Training Multi-Task Model MMOE

#### 1. Data Format

Multi-task learning requires multiple target columns:

```csv
user_id,item_id,features...,label_click,label_purchase,label_favorite
1,101,...,1,0,1
2,102,...,1,1,0
...
```

#### 2. Training Configuration

```yaml
session:
  id: mmoe_multitask
  artifact_root: nextrec_logs

data:
  path: data/multitask_data.csv
  format: csv
  target: [label_click, label_purchase, label_favorite]  # Multiple targets
  valid_ratio: 0.2
  random_state: 2024
  streaming: false

feature_config: feature_config.yaml
model_config: mmoe_config.yaml

dataloader:
  train_batch_size: 512
  train_shuffle: true
  valid_batch_size: 512
  valid_shuffle: false
  num_workers: 4

train:
  optimizer: adam
  optimizer_params:
    lr: 0.001
    weight_decay: 0.00001
  loss: bce
  loss_weights: [0.3, 0.7]
  metrics:
    - auc
  epochs: 10
  batch_size: 512
  shuffle: true
  device: cuda
```

#### 3. MMOE Model Configuration

```yaml
model: mmoe
params:
  num_experts: 8
  expert_dims: [256, 128]
  gate_dims: [64]
  tower_dims: [64, 32]
  dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

---

## Advanced Features

### Streaming Processing for Large Datasets

When datasets cannot be loaded into memory at once, use streaming processing:

```yaml
data:
  path: /path/to/large_dataset_dir/  # Directory containing multiple data files
  format: parquet
  target: label
  streaming: true                     # Enable streaming processing
  valid_ratio: 0.2

dataloader:
  chunk_size: 50000                   # Read 50000 rows at a time
  train_batch_size: 512
```

### Using Separate Validation Dataset

```yaml
data:
  path: /path/to/train_data.parquet
  val_path: /path/to/valid_data.parquet  # Specify separate validation dataset
  format: parquet
  target: label
```

### Custom Loss Function Parameters

```yaml
train:
  loss: focal
  loss_params:
    alpha: 0.25                        # Class weight
    gamma: 2.0                         # Focusing parameter
    reduction: mean                    # Reduction method
```

### Using Apple Silicon GPU (MPS)

To use Apple Silicon GPU acceleration on macOS:

```yaml
train:
  device: mps
```

### Feature Grouping (for Specific Models)

Some models (like MaskNet) support feature grouping:

```yaml
# feature_config.yaml
feature_groups:
  user_group:
    - user_id
    - age
    - gender
  item_group:
    - item_id
    - category
    - price
  context_group:
    - time
    - device
```

### Multi-Target Prediction

```yaml
# predict_config.yaml
targets: [label_click, label_purchase]  # Override training config targets
```

---

## FAQ

### Q1: How to handle imbalanced data?

Use Focal Loss:

```yaml
train:
  loss: focal
  loss_params:
    alpha: 0.25    # Increase weight for positive samples
    gamma: 2.0     # Increase weight for hard samples
```

### Q2: Where are training artifacts saved?

Training artifacts are saved in `{artifact_root}/{session_id}/` directory:

```
nextrec_logs/
└── my_session/
    ├── processor.pkl              # Data processor
    ├── features_config.pkl        # Feature configuration
    ├── model_epoch_1.pt        # Model checkpoint
    ├── model_epoch_2.pt
    └── runs.log               # Training log
```

### Q3: How to resume training?

The current CLI doesn't support resuming training. Please use the Python API.

### Q4: How to view training logs?

Logs are saved in `{artifact_root}/{session_id}/runs.log` and also output to terminal.

### Q5: Does it support distributed training?

The current CLI version doesn't support distributed training. Please use the Python API.

---

## Path Resolution Rules

Paths in configuration files support the following formats:

1. **Absolute path**: `/absolute/path/to/file`
2. **Relative path**: `relative/path/to/file` (relative to configuration file directory)
3. **Directory**: `/path/to/directory/` (automatically scans files in directory)

---

## Next Steps

- Check [Python API Documentation](https://nextrec.readthedocs.io/) for more advanced features
- Browse [tutorials/](../tutorials/) directory for more examples
- Visit [GitHub Issues](https://github.com/zerolovesea/NextRec/issues) to report issues
