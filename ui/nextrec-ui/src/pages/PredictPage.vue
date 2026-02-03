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
          <label>{{ t('数据路径', 'Data Path') }}</label>
          <input v-model="form.predict.data_path" placeholder="/path/to/predict/data" />
        </div>
        <div class="field">
          <label>{{ t('源数据格式', 'Source Data Format') }}</label>
          <select v-model="form.predict.source_data_format">
            <option value="parquet">parquet</option>
            <option value="csv">csv</option>
            <option value="auto">auto</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('ID 列', 'ID Column') }}</label>
          <input v-model="form.predict.id_column" placeholder="user_id" />
        </div>
        <div class="field">
          <label>{{ t('输出名称', 'Output Name') }}</label>
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
          <label>{{ t('批大小', 'Batch Size') }}</label>
          <input v-model.number="form.predict.batch_size" type="number" />
        </div>
        <div class="field">
          <label>{{ t('工作进程数', 'Num Workers') }}</label>
          <input v-model.number="form.predict.num_workers" type="number" />
        </div>
        <div class="field">
          <label>{{ t('推理进程数', 'Num Processes') }}</label>
          <input v-model.number="form.predict.num_processes" type="number" />
        </div>
        <div class="field">
          <label>{{ t('设备', 'Device') }}</label>
          <input v-model="form.predict.device" placeholder="cpu" />
        </div>
        <div class="field">
          <label>{{ t('使用 ONNX', 'Use ONNX') }}</label>
          <select v-model="form.predict.use_onnx">
            <option :value="false">false</option>
            <option :value="true">true</option>
          </select>
        </div>
        <div class="field">
          <label>{{ t('流式处理', 'Streaming') }}</label>
          <select v-model="form.predict.streaming">
            <option :value="true">true</option>
            <option :value="false">false</option>
          </select>
        </div>
        <div class="field">
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
