<template>
  <section class="page">
    <div class="card">
      <h2>{{ t('训练配置', 'Training Configuration') }}</h2>
      <p style="color: var(--muted); margin-top: 0;">
        {{ t('用于生成训练相关的标准化 YAML 文件。', 'Generate the standardized YAML for training.') }}
      </p>
    </div>

    <div class="card">
      <h2>{{ t('Session 配置', 'Session Settings') }}</h2>
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
        <div class="field">
          <label>{{ t('训练数据路径', 'Training Data Path') }}</label>
          <input v-model="form.data.path" placeholder="/path/to/training/data" />
        </div>
        <div class="field">
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
          <label>{{ t('Streaming', 'Streaming') }}</label>
          <select v-model="form.data.streaming">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
      </div>
      <div style="margin-top: 16px;">
        <label style="font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;">
          {{ t('目标列 (targets)', 'Targets') }}
        </label>
        <div class="badge-row" style="margin-top: 8px;">
          <div v-for="(target, index) in form.data.target_list" :key="`target-${index}`" class="field" style="min-width: 160px;">
            <input v-model="form.data.target_list[index]" placeholder="label" />
            <button class="icon-button" style="margin-top: 6px;" @click="removeItem(form.data.target_list, index)" :disabled="form.data.target_list.length <= 1" aria-label="移除">
              ×
            </button>
          </div>
          <button class="secondary" @click="addTarget">{{ t('添加', 'Add') }}</button>
        </div>
      </div>
      <div class="grid-2" style="margin-top: 12px;">
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
          <label>{{ t('训练 Batch Size', 'Train Batch Size') }}</label>
          <input v-model.number="form.dataloader.train_batch_size" type="number" />
        </div>
        <div class="field">
          <label>{{ t('训练 Shuffle', 'Train Shuffle') }}</label>
          <select v-model="form.dataloader.train_shuffle">
            <option :value="true">true</option>
            <option :value="false">false</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('验证 Batch Size', 'Valid Batch Size') }}</label>
          <input v-model.number="form.dataloader.valid_batch_size" type="number" />
        </div>
        <div class="field">
          <label>{{ t('验证 Shuffle', 'Valid Shuffle') }}</label>
          <select v-model="form.dataloader.valid_shuffle">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('Num Workers', 'Num Workers') }}</label>
          <input v-model.number="form.dataloader.num_workers" type="number" />
        </div>
        <div class="field">
          <label>{{ t('Prefetch Factor', 'Prefetch Factor') }}</label>
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
          <label>{{ t('批大小', 'Batch Size') }}</label>
          <input v-model.number="form.train.batch_size" type="number" />
        </div>
        <div class="field">
          <label>{{ t('是否打乱', 'Shuffle') }}</label>
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
      </div>

      <div style="margin-top: 16px;">
        <label style="font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;">{{ t('Loss 列表', 'Loss List') }}</label>
        <div v-if="lossMismatch" class="alert alert-warn">
          {{ t('Loss 数量多于目标列，请检查。', 'Loss count exceeds targets. Please review.') }}
        </div>
        <div class="badge-row" style="margin-top: 8px;">
          <div v-for="(loss, index) in form.train.loss_list" :key="`loss-${index}`" class="field" style="min-width: 200px;">
            <select v-model="form.train.loss_list[index]">
              <option v-for="lossName in lossOptions" :key="lossName" :value="lossName">{{ lossName }}</option>
            </select>
            <button class="icon-button" style="margin-top: 6px;" @click="removeItem(form.train.loss_list, index)" :disabled="form.train.loss_list.length <= 1" aria-label="移除">
              ×
            </button>
          </div>
          <button class="secondary" @click="addItem(form.train.loss_list, lossOptions[0])">{{ t('添加', 'Add') }}</button>
        </div>
      </div>

      <div style="margin-top: 16px;">
        <label style="font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;">{{ t('Metrics 列表', 'Metrics List') }}</label>
        <div class="badge-row" style="margin-top: 8px;">
          <div v-for="(metric, index) in form.train.metrics_list" :key="`metric-${index}`" class="field" style="min-width: 200px;">
            <select v-model="form.train.metrics_list[index]">
              <option v-for="metricName in metricsOptions" :key="metricName" :value="metricName">{{ metricName }}</option>
            </select>
            <button class="icon-button" style="margin-top: 6px;" @click="removeItem(form.train.metrics_list, index)" :disabled="form.train.metrics_list.length <= 1" aria-label="移除">
              ×
            </button>
          </div>
          <button class="secondary" @click="addItem(form.train.metrics_list, metricsOptions[0])">{{ t('添加', 'Add') }}</button>
        </div>
      </div>

      <div class="grid-2" style="margin-top: 16px;">
        <div class="field">
          <label>{{ t('损失参数', 'Loss Params') }}</label>
          <textarea v-model="form.train.loss_params_text" placeholder="- pos_weight: 1.0\n  logits: false"></textarea>
        </div>
        <div class="field">
          <label>{{ t('损失权重', 'Loss Weights') }}</label>
          <textarea v-model="form.train.loss_weights_text" placeholder="method: grad_norm\nalpha: 1.5\nlr: 0.025"></textarea>
        </div>
      </div>

      <div class="grid-2" style="margin-top: 16px;">
        <div class="field">
          <label>{{ t('Use W&B', 'Use W&B') }}</label>
          <select v-model="form.train.use_wandb">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('Use SwanLab', 'Use SwanLab') }}</label>
          <select v-model="form.train.use_swanlab">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
      </div>

      <div v-if="form.train.use_wandb" class="grid-2" style="margin-top: 16px;">
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
      <h2>{{ t('YAML 预览', 'YAML Preview') }}</h2>
      <div v-if="error" class="alert">{{ error }}</div>
      <div class="field">
        <textarea class="mono" readonly :value="yamlText"></textarea>
      </div>
      <div class="actions spaced-top">
        <button class="primary" @click="download" :disabled="!!error">{{ t('下载 train_config.yaml', 'Download train_config.yaml') }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue';
import { buildTrainConfig } from '../utils/buildTrainConfig.js';
import { downloadText } from '../utils/download.js';
import { store } from '../store/configStore.js';

const form = store.train;

const error = ref('');
const lang = computed(() => store.ui.lang);
const t = (zh, en) => (lang.value === 'zh' ? zh : en);

const lossOptions = [
  'bce',
  'binary_crossentropy',
  'weighted_bce',
  'focal',
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
  form.train.loss_list.push(lossOptions[0]);
  form.train.metrics_list.push(metricsOptions[0]);
}

function removeItem(list, index) {
  list.splice(index, 1);
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
