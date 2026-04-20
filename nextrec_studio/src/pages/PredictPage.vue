<template>
  <section class="page">
    <div class="card">
      <h2>{{ t('推理配置', 'Prediction Configuration') }}</h2>
      <p style="color: var(--muted); margin-top: 0;">
        {{ t('配置推理所需的模型、数据与输出参数。', 'Configure checkpoint, input data, and output settings for prediction.') }}
      </p>
    </div>

    <div class="card">
      <h2>{{ t('模型与会话', 'Checkpoint & Session') }}</h2>
      <div class="grid-2">
        <div class="field">
          <label>
            {{ t('模型检查点路径', 'Checkpoint Path') }}
            <span class="help-icon" :data-tip="t('模型检查点文件的路径，用于加载训练完成模型。使用该路径时，推理结果将默认保存在该路径的pred目录下。', 'Path to the model checkpoint used for loading the trained model. When provided, predictions are saved under the pred directory of this path.')">?</span>
          </label>
          <input v-model="form.checkpoint_path" placeholder="/path/to/checkpoint" />
        </div>
        <div class="field">
          <label>{{ t('模型配置文件（可选）', 'Model Config (optional)') }}</label>
          <input v-model="form.model_config" placeholder="/path/to/model_config.yaml" />
        </div>
        <div class="field">
          <label>{{ t('会话 ID（可选）', 'Session ID (optional)') }}</label>
          <input v-model="form.session.id" placeholder="my_experiment_session" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('Predict 参数', 'Prediction Settings') }}</h2>
      <div class="grid-2">
        <div class="field">
          <label>{{ t('数据路径', 'Data Path') }}</label>
          <input v-model="form.predict.data_path" placeholder="/path/to/predict/data" />
        </div>
        <div class="field">
          <label>{{ t('数据格式', 'Source Data Format') }}</label>
          <select v-model="form.predict.source_data_format">
            <option value="parquet">parquet</option>
            <option value="csv">csv</option>
            <option value="auto">auto</option>
          </select>
        </div>
        <div class="field">
          <label>
            {{ t('ID 列', 'ID Column') }}
            <span class="help-icon" :data-tip="t('用于唯一标识每条数据的列名。未填写时，将会使用训练时的id列。', 'Column name that uniquely identifies each record. If empty, the training id column will be used.')">?</span>
          </label>
          <input v-model="form.predict.id_column" placeholder="user_id" />
        </div>
        <div class="field">
          <label>{{ t('输出文件名称', 'Output Name') }}</label>
          <input v-model="form.predict.name" placeholder="pred" />
        </div>
        <div class="field">
          <label>{{ t('保存格式', 'Save Data Format') }}</label>
          <select v-model="form.predict.save_data_format">
            <option value="csv">csv</option>
            <option value="parquet">parquet</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('预览行数', 'Preview Rows') }}</label>
          <input v-model.number="form.predict.preview_rows" type="number" />
        </div>
        <div class="field">
          <label>{{ t('推理集 Batch Size', 'Batch Size') }}</label>
          <input v-model.number="form.predict.batch_size" type="number" />
        </div>
        <div class="field">
          <label>
            {{ t('数据加载线程数', 'Num Workers') }}
            <span
              class="help-icon"
              :data-tip="t('用于数据加载的线程数量，影响数据预处理的并发度。流式推理且 num_processes>1 时会被强制为 0。', 'Number of worker threads for data loading; affects preprocessing parallelism. Forced to 0 when streaming and num_processes>1.')"
              >?</span
            >
          </label>
          <input
            v-model.number="form.predict.num_workers"
            type="number"
            :disabled="isMultiProcStreaming"
          />
        </div>
        <div class="field">
          <label>
            {{ t('并发推理进程数', 'Num Processes') }}
            <span
              class="help-icon"
              :data-tip="t('留空时自动根据系统负载选择 1~5。仅在 streaming=true 且未启用 ONNX 时生效。', 'Leave empty to auto-select 1~5 based on system load. Only effective when streaming=true and ONNX is disabled.')"
              >?</span
            >
          </label>
          <input
            v-model.number="form.predict.num_processes"
            type="number"
            :placeholder="t('自动(1~5)', 'Auto (1~5)')"
          />
        </div>
        <div class="field">
          <label>{{ t('设备', 'Device') }}</label>
          <input v-model="form.predict.device" placeholder="cpu" />
        </div>
        <div class="field">
          <label>{{ t('使用 ONNX 模型推理', 'Use ONNX') }}</label>
          <select v-model="form.predict.use_onnx">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('流式推理', 'Streaming') }}</label>
          <select v-model="form.predict.streaming">
            <option :value="true">true</option>
            <option :value="false">false</option>
          </select>
        </div>
        <div class="field" v-if="form.predict.streaming">
          <label>{{ t('分块大小', 'Chunk Size') }}</label>
          <input v-model.number="form.predict.chunk_size" type="number" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('YAML 预览', 'YAML Preview') }}</h2>
      <div class="field">
        <textarea class="mono" readonly :value="yamlText"></textarea>
      </div>
      <div class="actions spaced-top">
        <button class="primary" @click="download">{{ t('下载', 'Download') }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, watch } from 'vue';
import { dumpYaml } from '../utils/yaml.js';
import { downloadText } from '../utils/download.js';
import { store } from '../store/configStore.js';

const form = store.predict;
const lang = computed(() => store.ui.lang);
const t = (zh, en) => (lang.value === 'zh' ? zh : en);
const isMultiProcStreaming = computed(
  () => form.predict.streaming && Number(form.predict.num_processes || 0) > 1
);

const yamlText = computed(() => {
  const predict = { ...form.predict };
  if (
    predict.num_processes === null ||
    predict.num_processes === '' ||
    Number.isNaN(predict.num_processes)
  ) {
    delete predict.num_processes;
  }
  const output = {
    checkpoint_path: form.checkpoint_path,
    predict
  };

  if (form.model_config) {
    output.model_config = form.model_config;
  }

  if (form.session.id) {
    output.session = { id: form.session.id };
  }

  return dumpYaml(output);
});

watch(
  () => isMultiProcStreaming.value,
  (locked) => {
    if (locked) {
      form.predict.num_workers = 0;
    }
  },
  { immediate: true }
);

function download() {
  downloadText('predict_config.yaml', yamlText.value);
}
</script>
