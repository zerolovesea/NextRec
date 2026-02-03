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
          <label>{{ t('模型检查点路径', 'Checkpoint Path') }}</label>
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
          <label>{{ t('Data Path', 'Data Path') }}</label>
          <input v-model="form.predict.data_path" placeholder="/path/to/predict/data" />
        </div>
        <div class="field">
          <label>{{ t('Source Data Format', 'Source Data Format') }}</label>
          <select v-model="form.predict.source_data_format">
            <option value="parquet">parquet</option>
            <option value="csv">csv</option>
            <option value="auto">auto</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('Output Name', 'Output Name') }}</label>
          <input v-model="form.predict.name" placeholder="pred" />
        </div>
        <div class="field">
          <label>{{ t('Save Data Format', 'Save Data Format') }}</label>
          <select v-model="form.predict.save_data_format">
            <option value="csv">csv</option>
            <option value="parquet">parquet</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('Preview Rows', 'Preview Rows') }}</label>
          <input v-model.number="form.predict.preview_rows" type="number" />
        </div>
        <div class="field">
          <label>{{ t('Batch Size', 'Batch Size') }}</label>
          <input v-model.number="form.predict.batch_size" type="number" />
        </div>
        <div class="field">
          <label>{{ t('Num Workers', 'Num Workers') }}</label>
          <input v-model.number="form.predict.num_workers" type="number" />
        </div>
        <div class="field">
          <label>{{ t('Num Processes', 'Num Processes') }}</label>
          <input v-model.number="form.predict.num_processes" type="number" />
        </div>
        <div class="field">
          <label>{{ t('Device', 'Device') }}</label>
          <input v-model="form.predict.device" placeholder="cpu" />
        </div>
        <div class="field">
          <label>{{ t('Use ONNX', 'Use ONNX') }}</label>
          <select v-model="form.predict.use_onnx">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('Streaming', 'Streaming') }}</label>
          <select v-model="form.predict.streaming">
            <option :value="true">true</option>
            <option :value="false">false</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('Chunk Size', 'Chunk Size') }}</label>
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
        <button class="primary" @click="download">{{ t('下载 predict_config.yaml', 'Download predict_config.yaml') }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue';
import { dumpYaml } from '../utils/yaml.js';
import { downloadText } from '../utils/download.js';
import { store } from '../store/configStore.js';

const form = store.predict;
const lang = computed(() => store.ui.lang);
const t = (zh, en) => (lang.value === 'zh' ? zh : en);

const yamlText = computed(() => {
  const output = {
    checkpoint_path: form.checkpoint_path,
    predict: { ...form.predict }
  };

  if (form.model_config) {
    output.model_config = form.model_config;
  }

  if (form.session.id) {
    output.session = { id: form.session.id };
  }

  return dumpYaml(output);
});

function download() {
  downloadText('predict_config.yaml', yamlText.value);
}
</script>
