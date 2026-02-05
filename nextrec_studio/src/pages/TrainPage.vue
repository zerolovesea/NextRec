<template>
  <section class="page">
    <div class="card">
      <h2>{{ t('训练配置', 'Training Configuration') }}</h2>
      <p style="color: var(--muted); margin-top: 0;">
        {{ t('用于生成训练相关的标准化 YAML 文件。', 'Generate the standardized YAML for training.') }}
      </p>
    </div>

    <div class="card" v-if="!isIdtank">
      <h2>{{ t('会话配置', 'Session Settings') }}</h2>
      <div class="grid-2">
        <div class="field">
          <label>{{ t('会话 ID', 'Session ID') }}</label>
          <input v-model="form.session.id" placeholder="my_experiment_session" />
        </div>
        <div class="field">
          <label>{{ t('输出根目录', 'Artifact Root') }}</label>
          <input v-model="form.session.artifact_root" placeholder="nextrec_logs" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('数据配置', 'Data Configuration') }}</h2>
      <div class="grid-2">
        <div class="field" v-if="!isIdtank">
          <label>{{ t('训练数据路径', 'Training Data Path') }}</label>
          <input v-model="form.data.path" placeholder="/path/to/training/data" />
        </div>
        <div class="field" v-if="!isIdtank">
          <label>{{ t('数据格式', 'Data Format') }}</label>
          <select v-model="form.data.format">
            <option value="parquet">parquet</option>
            <option value="csv">csv</option>
            <option value="auto">auto</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('验证集比例', 'Validation Ratio') }}</label>
          <input v-model.number="form.data.valid_ratio" type="number" step="0.01" min="0" max="1" />
        </div>
        <div class="field">
          <label>{{ t('随机种子', 'Random Seed') }}</label>
          <input v-model.number="form.data.random_state" type="number" />
        </div>
        <div class="field">
          <label>{{ t('流式训练', 'Streaming') }}</label>
          <select v-model="form.data.streaming">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>
            {{ t('ID 列', 'ID Column') }}
            <span class="help-icon" :data-tip="t('标识用户的唯一列', 'The unique column identifying users')">?</span>
          </label>
          <input v-model="form.data.id_column" :placeholder="isIdtank ? 'phone' : 'user_id'" />
        </div>
      </div>
      <div style="margin-top: 16px;">
        <label style="font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;">
          {{ t('目标列', 'Targets') }}
          <span class="help-icon" :data-tip="t('模型的目标列', 'The target columns for the model')">?</span>
        </label>
        <div class="badge-row" style="margin-top: 8px;">
          <div v-for="(target, index) in form.data.target_list" :key="`target-${index}`" class="field" style="min-width: 160px;">
            <input v-model="form.data.target_list[index]" placeholder="label" />
            <button class="icon-button icon-button-small icon-button-danger" style="margin-top: 6px;" @click="removeItem(form.data.target_list, index)" :disabled="form.data.target_list.length <= 1" aria-label="移除">
              ×
            </button>
          </div>
          <button class="secondary" @click="addTarget">{{ t('添加', 'Add') }}</button>
        </div>
      </div>
      <div class="grid-2" style="margin-top: 12px;" v-if="!isIdtank">
        <div class="field">
          <label>{{ t('特征配置文件', 'Feature Config File') }}</label>
          <input v-model="form.feature_config" placeholder="feature_config.yaml" />
        </div>
        <div class="field">
          <label>{{ t('模型配置文件', 'Model Config File') }}</label>
          <input v-model="form.model_config" placeholder="model_config.yaml" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('Dataloader 配置', 'Dataloader Configuration') }}</h2>
      <div class="grid-2">
        <div class="field">
          <label>{{ t('训练集 Batch Size', 'Train Batch Size') }}</label>
          <input v-model.number="form.dataloader.train_batch_size" type="number" />
        </div>
        <div class="field">
          <label>{{ t('训练集 Shuffle', 'Train Shuffle') }}</label>
          <select v-model="form.dataloader.train_shuffle">
            <option :value="true">true</option>
            <option :value="false">false</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('验证集 Batch Size', 'Valid Batch Size') }}</label>
          <input v-model.number="form.dataloader.valid_batch_size" type="number" />
        </div>
        <div class="field">
          <label>{{ t('验证集 Shuffle', 'Valid Shuffle') }}</label>
          <select v-model="form.dataloader.valid_shuffle">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>
            {{ t('数据加载线程数', 'Num Workers') }}
            <span class="help-icon" :data-tip="t('用于数据加载的线程数量，提高时可以加快数据加载速度', 'The number of threads used for data loading; increasing this can speed up data loading')">?</span>
          </label>
          <input v-model.number="form.dataloader.num_workers" type="number" />
        </div>
        <div class="field">
          <label>
            {{ t('数据预取因子', 'Prefetch Factor') }}
            <span class="help-icon" :data-tip="t('每个工作线程预取的数据批次数量，提高时可以减少数据加载等待', 'The number of batches to prefetch for each worker thread; increasing this can reduce data loading wait time')">?</span>
          </label>
          <input v-model.number="form.dataloader.prefetch_factor" type="number" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('训练参数', 'Training Settings') }}</h2>
      <div class="grid-2">
        <div class="field">
          <label>{{ t('优化器', 'Optimizer') }}</label>
          <select v-model="form.train.optimizer">
            <option value="adam">adam</option>
            <option value="adamw">adamw</option>
            <option value="sgd">sgd</option>
            <option value="rmsprop">rmsprop</option>
            <option value="adagrad">adagrad</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('学习率', 'Learning Rate') }}</label>
          <input v-model.number="form.train.optimizer_params.lr" type="number" step="0.0001" />
        </div>
        <div class="field">
          <label>{{ t('权重衰减', 'Weight Decay') }}</label>
          <input v-model.number="form.train.optimizer_params.weight_decay" type="number" step="0.0001" />
        </div>
        <div class="field">
          <label>{{ t('训练轮数', 'Epochs') }}</label>
          <input v-model.number="form.train.epochs" type="number" />
        </div>
        <div class="field">
          <label>{{ t('训练集 Batch Size', 'Batch Size') }}</label>
          <input v-model.number="form.train.batch_size" type="number" />
        </div>
        <div class="field">
          <label>{{ t('Shuffle', 'Shuffle') }}</label>
          <select v-model="form.train.shuffle">
            <option :value="true">true</option>
            <option :value="false">false</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('日志间隔', 'Log Interval') }}</label>
          <input v-model.number="form.train.log_interval" type="number" />
        </div>
        <div class="field">
          <label>{{ t('设备', 'Device') }}</label>
          <input v-model="form.train.device" placeholder="cpu" />
        </div>
        <div class="field">
          <label>{{ t('备注', 'Note') }}</label>
          <input v-model="form.train.note" placeholder="baseline run" />
        </div>
      </div>

      <div style="margin-top: 16px;">
        <label style="font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;">{{ t('损失函数', 'Loss Function') }}</label>
        <div v-if="lossMismatch" class="alert alert-warn">
          {{ t('Loss 数量多于目标列，请检查。', 'Loss count exceeds targets. Please review.') }}
        </div>
        <div class="badge-row" style="margin-top: 8px; align-items: flex-start;">
          <div v-for="(loss, index) in form.train.loss_list" :key="`loss-${index}`" class="field" style="min-width: 200px;">
            <select v-model="form.train.loss_list[index]">
              <option v-for="lossName in lossOptions" :key="lossName" :value="lossName">{{ displayLossName(lossName) }}</option>
            </select>
            <input
              v-model="form.train.loss_params_list[index]"
              :placeholder="lossParamPlaceholder(index)"
              style="margin-top: 8px; font-size: 11px;"
            />
            <button class="icon-button icon-button-small icon-button-danger" style="margin-top: 6px;" @click="removeLoss(index)" :disabled="form.train.loss_list.length <= 1" aria-label="移除">
              ×
            </button>
          </div>
          <button class="secondary" @click="addLoss">{{ t('添加', 'Add') }}</button>
        </div>
      </div>

      <div style="margin-top: 16px;">
        <label style="font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;">{{ t('损失函数权重', 'Loss Weights') }}</label>
        <div class="field" style="margin-top: 8px;">
          <textarea
            class="textarea-compact"
            v-model="form.train.loss_weights_text"
            :placeholder="t(`例如[0.3, 0.7]或
method: grad_norm
alpha: 1.5
lr: 0.025`, `e.g. [0.3, 0.7] or
method: grad_norm
alpha: 1.5
lr: 0.025`)"
            rows="4"
          ></textarea>
        </div>
      </div>

      <div style="margin-top: 16px;">
        <label style="font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;">{{ t('评估指标', 'Metrics') }}</label>
        <div class="badge-row" style="margin-top: 8px;">
          <div v-for="(metric, index) in form.train.metrics_list" :key="`metric-${index}`" class="field" style="min-width: 200px;">
            <select v-model="form.train.metrics_list[index]">
              <option v-for="metricName in metricsOptions" :key="metricName" :value="metricName">{{ metricName.toUpperCase() }}</option>
            </select>
            <button class="icon-button icon-button-small icon-button-danger" style="margin-top: 6px;" @click="removeItem(form.train.metrics_list, index)" :disabled="form.train.metrics_list.length <= 1" aria-label="移除">
              ×
            </button>
          </div>
          <button class="secondary" @click="addItem(form.train.metrics_list, metricsOptions[0])">{{ t('添加', 'Add') }}</button>
        </div>
      </div>

      <div class="grid-2" style="margin-top: 16px;">
        <div class="field" v-if="!isIdtank">
          <label>{{ t('使用 W&B 管理日志', 'Use W&B') }}</label>
          <select v-model="form.train.use_wandb">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('使用 SwanLab 管理日志', 'Use SwanLab') }}</label>
          <select v-model="form.train.use_swanlab">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
      </div>

      <div v-if="form.train.use_wandb && !isIdtank" class="grid-2" style="margin-top: 16px;">
        <div class="field">
          <label>{{ t('W&B API Key', 'wandb_api') }}</label>
          <input v-model="form.train.wandb_api" :placeholder="t('请输入你的 W&B API Key', 'YOUR_WANDB_API_KEY')" required />
        </div>
        <div class="field">
          <label>{{ t('W&B 项目', 'wandb_kwargs.project') }}</label>
          <input v-model="form.train.wandb_kwargs.project" :placeholder="t('如：NextRec', 'e.g. NextRec')" required />
        </div>
        <div class="field">
          <label>{{ t('W&B 实验名', 'wandb_kwargs.name') }}</label>
          <input v-model="form.train.wandb_kwargs.name" :placeholder="t('如：exp-001', 'e.g. exp-001')" required />
        </div>
      </div>

      <div v-if="form.train.use_swanlab" class="grid-2" style="margin-top: 16px;">
        <div class="field">
          <label>{{ t('SwanLab API Key', 'swanlab_api') }}</label>
          <input v-model="form.train.swanlab_api" :placeholder="t('请输入你的 SwanLab API Key', 'YOUR_SWANLAB_API_KEY')" required />
        </div>
        <div class="field">
          <label>{{ t('SwanLab 项目', 'swanlab_kwargs.project') }}</label>
          <input v-model="form.train.swanlab_kwargs.project" :placeholder="t('如：NextRec', 'e.g. NextRec')" required />
        </div>
        <div class="field">
          <label>{{ t('SwanLab 实验名', 'swanlab_kwargs.name') }}</label>
          <input v-model="form.train.swanlab_kwargs.name" :placeholder="t('如：exp-001', 'e.g. exp-001')" required />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>
        {{ t('模型量化', 'Model Quantization') }}
        <span class="help-icon" :data-tip="t('使用 ONNX 格式导出模型以便于跨平台部署，ONNX 是一种开放模型格式，通过对模型进行量化可以减小模型大小并加快推理速度。', 'Use ONNX format to export the model for cross-platform deployment. ONNX is an open model format, and quantizing the model can reduce its size and speed up inference.')">?</span>
      </h2>
      <div class="grid-2">
        <div class="field">
          <label>{{ t('启用 ONNX 导出', 'Enable ONNX Export') }}</label>
          <select v-model="form.export_onnx.enable">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>
            {{ t('导出 Batch Size', 'Export Batch Size') }}
            <span class="help-icon" :data-tip="t('导出模型时使用的批次大小，ONNX要求固定批次大小来保证模型推理时的正确性。', 'The batch size used when exporting the model; ONNX requires a fixed batch size to ensure correct inference.')">?</span>
          </label>
          <input v-model.number="form.export_onnx.batch_size" type="number" />
        </div>
        <div class="field">
          <label>
            {{ t('ONNX Opset 版本', 'ONNX Opset Version') }}
            <span class="help-icon" :data-tip="t('导出模型时使用的 ONNX Opset 版本，ONNX Opset 版本决定了模型中算子的版本，默认为18以确保兼容性。', 'The ONNX Opset version used when exporting the model; the ONNX Opset version determines the version of operators in the model, defaulting to 18 for compatibility.')">?</span>
          </label>
          <input v-model.number="form.export_onnx.opset_version" type="number" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('YAML 预览', 'YAML Preview') }}</h2>
      <div v-if="error" class="alert">{{ error }}</div>
      <div class="field">
        <textarea class="mono" readonly :value="yamlText"></textarea>
      </div>
      <div class="actions spaced-top">
        <button class="primary" @click="download" :disabled="!!error">{{ t('下载', 'Download') }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue';
import { buildTrainConfig } from '../utils/buildTrainConfig.js';
import { downloadText } from '../utils/download.js';
import { isIdtank } from '../utils/appEnv.js';
import { store } from '../store/configStore.js';

const form = store.train;

const error = ref('');
const lang = computed(() => store.ui.lang);
const t = (zh, en) => (lang.value === 'zh' ? zh : en);

const lossOptions = [
  'bce',
  'weighted_bce',
  'focal_loss',
  'cb_focal',
  'class_balanced_focal',
  'crossentropy',
  'ce',
  'mse',
  'mae',
  'bpr',
  'hinge',
  'triplet',
  'sampled_softmax',
  'softmax',
  'infonce',
  'listnet',
  'listmle',
  'approx_ndcg'
];

const metricsOptions = [
  'auc',
  'gauc',
  'ks',
  'logloss',
  'accuracy',
  'acc',
  'precision',
  'recall',
  'f1',
  'micro_f1',
  'macro_f1',
  'mse',
  'mae',
  'rmse',
  'r2',
  'mape',
  'msle',
  'precision@10',
  'hitrate@10',
  'map@10',
  'cosine',
  'recall@5',
  'recall@10',
  'recall@20',
  'ndcg@5',
  'ndcg@10',
  'ndcg@20',
  'mrr@5',
  'mrr@10',
  'mrr@20',
  'topk_recall@5',
  'topk_recall@10',
  'topk_recall@20',
  'topk_precision@5',
  'topk_precision@10',
  'topk_precision@20',
  'lift@5',
  'lift@10',
  'lift@20'
];

function addItem(list, defaultValue = '') {
  list.push(defaultValue);
}

function addTarget() {
  form.data.target_list.push('');
  addLoss();
}

function removeItem(list, index) {
  list.splice(index, 1);
}

function lossParamPlaceholder(index) {
  const lossName = form.train.loss_list[index];
  if (lossName === 'weighted_bce') {
    return `auto_balance: true/pos_weight: 1.0`;
  }
  if (lossName === 'focal_loss') {
    return '{"gamma": 2.0, "alpha": 0.25}';
  }
  return '';
}

function displayLossName(lossName) {
  if (lossName === 'weighted_bce') {
    return t('WEIGHTED_BCE', 'WEIGHTED_BCE');
  }
  if (lossName === 'bce' || lossName === 'binary_crossentropy') {
    return t('BCE', 'BCE');
  }
  const upper = String(lossName).toUpperCase();
  return t(upper, upper);
}

function addLoss() {
  form.train.loss_list.push(lossOptions[0]);
  form.train.loss_params_list.push('');
}

function removeLoss(index) {
  form.train.loss_list.splice(index, 1);
  if (Array.isArray(form.train.loss_params_list)) {
    form.train.loss_params_list.splice(index, 1);
  }
}

const yamlText = computed(() => {
  try {
    error.value = '';
    return buildTrainConfig(form);
  } catch (err) {
    error.value = `训练配置解析失败: ${err.message}`;
    return '';
  }
});

const lossMismatch = computed(() => form.train.loss_list.length > form.data.target_list.length);

function download() {
  if (error.value) {
    return;
  }
  downloadText('train_config.yaml', yamlText.value);
}
</script>
