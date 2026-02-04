<template>
  <section class="page">
    <div class="card">
      <h2>{{ t('模型配置', 'Model Configuration') }}</h2>
      <p style="color: var(--muted); margin-top: 0;">
        {{ t('选择模型并配置参数，生成对应的模型结构配置文件。', 'Select a model and tune parameters to build the model config.') }}
      </p>
    </div>

    <div class="card">
      <h2>{{ t('模型选择', 'Model Selection') }}</h2>
      <div class="grid-2">
        <div class="field">
          <label>{{ t('模型名称', 'Model Name') }}</label>
          <select v-model="form.model" @change="applyPreset">
            <option v-for="preset in presetList" :key="preset" :value="preset">{{ preset.toUpperCase() }}</option>
          </select>
        </div>
      </div>
      <div v-if="modelNote" class="alert model-note" style="margin-top: 12px;">
        {{ modelNote }}
      </div>
    </div>

    <div class="card">
      <h2>{{ t('参数配置', 'Parameter Configuration') }}</h2>
      <div class="inline-list">
        <div v-for="(entry, index) in visibleParamEntries" :key="entry.id" class="inline-item">
          <div class="grid-2">
            <div class="field">
              <label>{{ t('参数名', 'Parameter Key') }}</label>
              <input v-model="entry.key" placeholder="mlp_params" />
            </div>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label>
              {{ t('值', 'Value') }}
              <span
                v-if="paramHelp[entry.key]"
                class="help-icon"
                :title="paramHelp[entry.key]"
                :data-tip="paramHelp[entry.key]"
              >?</span>
            </label>
            <textarea v-model="entry.raw" class="mono" placeholder="relu 或 123 或 true 或 \nmlp_params:\n  hidden_dims: [256, 128]"></textarea>
          </div>
          <div class="actions">
            <button class="icon-button icon-button-small icon-button-danger" @click="removeEntryById(entry.id)" :disabled="paramEntries.length <= 1" aria-label="移除">×</button>
          </div>
        </div>
      </div>
      <div class="actions" style="margin-top: 12px;">
        <button class="secondary" @click="addEntry">{{ t('添加', 'Add') }}</button>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('正则配置', 'Regularization') }}</h2>
      <div class="grid-inline">
        <div class="field">
          <label>embedding_l1_reg</label>
          <input v-model.number="regEmbeddingL1" type="number" step="0.000001" />
        </div>
        <div class="field">
          <label>embedding_l2_reg</label>
          <input v-model.number="regEmbeddingL2" type="number" step="0.000001" />
        </div>
        <div class="field">
          <label>dense_l1_reg</label>
          <input v-model.number="regDenseL1" type="number" step="0.000001" />
        </div>
        <div class="field">
          <label>dense_l2_reg</label>
          <input v-model.number="regDenseL2" type="number" step="0.000001" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('YAML 预览', 'YAML Preview') }}</h2>
      <div v-if="error" class="alert">{{ error }}</div>
      <div class="field" style="margin-top: 12px;">
        <textarea class="mono" readonly :value="yamlText"></textarea>
      </div>
      <div class="actions spaced-top">
        <button class="primary" @click="download" :disabled="!!error">{{ t('下载', 'Download') }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, ref, reactive, watch, onMounted } from 'vue';
import { dumpYaml, parseYaml } from '../utils/yaml.js';
import { downloadText } from '../utils/download.js';
import { store } from '../store/configStore.js';

const presetModules = import.meta.glob('../presets/*.yaml', { as: 'raw', eager: true });

function filenameToModel(path) {
  const file = path.split('/').pop() || '';
  return file.replace(/\.yaml$/i, '');
}

function toParamsText(raw) {
  const data = parseYaml(raw) || {};
  if (data.params && typeof data.params === 'object') {
    return dumpYaml(data.params);
  }
  return '';
}

function parsePresetComments(raw) {
  const lines = raw.split(/\r?\n/);
  const topComments = [];
  let inTop = true;
  const paramComments = {};
  const pendingByIndent = new Map();
  let inParams = false;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();
    if (inTop) {
      if (trimmed.startsWith('#')) {
        topComments.push(trimmed.replace(/^#+\s?/, ''));
        continue;
      }
      if (trimmed === '') {
        continue;
      }
      inTop = false;
    }

    if (trimmed.startsWith('params:')) {
      inParams = true;
      continue;
    }
    if (!inParams) {
      continue;
    }

    const indent = line.match(/^\s*/)?.[0].length ?? 0;
    if (trimmed.startsWith('#') && indent >= 2) {
      pendingByIndent.set(indent, trimmed.replace(/^#+\s?/, ''));
      continue;
    }

    const match = line.match(/^\s{2,}([A-Za-z0-9_]+):/);
    if (!match) {
      continue;
    }
    const key = match[1];
    let comment = '';
    const inlineIdx = line.indexOf('#');
    if (inlineIdx !== -1) {
      comment = line.slice(inlineIdx + 1).trim();
    } else {
      comment = pendingByIndent.get(indent) || '';
    }
    if (comment) {
      paramComments[key] = comment;
    }
    pendingByIndent.delete(indent);
  }

  return { top: topComments.join('\n'), params: paramComments };
}

function getPresetRaw(name) {
  const entry = Object.entries(presetModules).find(([path]) => path.endsWith(`/${name}.yaml`));
  return entry ? entry[1] : '';
}

const presets = Object.entries(presetModules).reduce((acc, [path, raw]) => {
  const data = parseYaml(raw) || {};
  const modelName = data.model || filenameToModel(path);
  acc[modelName] = {
    model: modelName,
    paramsText: toParamsText(raw)
  };
  return acc;
}, {});

const presetList = Object.keys(presets).sort();

const form = store.model;
const lang = computed(() => store.ui.lang);
const t = (zh, en) => (lang.value === 'zh' ? zh : en);
const error = ref('');
const paramEntries = reactive([]);
const paramHelp = reactive({});
const modelNote = ref('');
const hiddenParamKeys = new Set(['embedding_l1_reg', 'embedding_l2_reg', 'dense_l1_reg', 'dense_l2_reg']);
const visibleParamEntries = computed(() => paramEntries.filter((entry) => !hiddenParamKeys.has(entry.key)));

const uid = () => (crypto?.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));

function entryFromValue(key, value) {
  if (value === null || value === undefined) {
    return { id: uid(), key, raw: 'null' };
  }
  if (Array.isArray(value) || typeof value === 'object') {
    return { id: uid(), key, raw: dumpYaml(value) };
  }
  if (typeof value === 'number') {
    return { id: uid(), key, raw: String(value) };
  }
  if (typeof value === 'boolean') {
    return { id: uid(), key, raw: value ? 'true' : 'false' };
  }
  return { id: uid(), key, raw: String(value) };
}

function setEntriesFromParams(params) {
  paramEntries.splice(0, paramEntries.length);
  const keys = Object.keys(params || {});
  if (keys.length === 0) {
    paramEntries.push(entryFromValue('', ''));
    return;
  }
  for (const key of keys) {
    paramEntries.push(entryFromValue(key, params[key]));
  }
}

function applyPreset() {
  const modelName = form.model;
  const preset = presets[modelName];
  if (!preset) {
    return;
  }
  form.model = preset.model;
  form.paramsText = preset.paramsText;
  const params = parseYaml(form.paramsText) || {};
  setEntriesFromParams(params);
  const comments = parsePresetComments(getPresetRaw(modelName));
  modelNote.value = comments.top;
  Object.keys(paramHelp).forEach((key) => delete paramHelp[key]);
  Object.entries(comments.params).forEach(([key, note]) => {
    paramHelp[key] = note;
  });
}

function addEntry() {
  paramEntries.push(entryFromValue('', ''));
}

function removeEntryById(id) {
  const idx = paramEntries.findIndex((entry) => entry.id === id);
  if (idx >= 0) {
    paramEntries.splice(idx, 1);
  }
}

function buildParams(entries) {
  const params = {};
  for (const entry of entries) {
    if (!entry.key) {
      continue;
    }
    const raw = entry.raw ?? '';
    if (!raw.trim()) {
      params[entry.key] = '';
      continue;
    }
    params[entry.key] = parseYaml(raw);
  }
  return params;
}

function getParamValue(key) {
  const entry = paramEntries.find((item) => item.key === key);
  if (!entry || entry.raw === undefined || entry.raw === null) {
    return '';
  }
  const value = Number(entry.raw);
  return Number.isNaN(value) ? '' : value;
}

function setParamValue(key, value) {
  const existing = paramEntries.find((item) => item.key === key);
  if (existing) {
    existing.raw = value === '' || value === null || value === undefined ? '' : String(value);
    return;
  }
  paramEntries.push(entryFromValue(key, value));
}

const regEmbeddingL1 = computed({
  get: () => getParamValue('embedding_l1_reg'),
  set: (val) => setParamValue('embedding_l1_reg', val)
});

const regEmbeddingL2 = computed({
  get: () => getParamValue('embedding_l2_reg'),
  set: (val) => setParamValue('embedding_l2_reg', val)
});

const regDenseL1 = computed({
  get: () => getParamValue('dense_l1_reg'),
  set: (val) => setParamValue('dense_l1_reg', val)
});

const regDenseL2 = computed({
  get: () => getParamValue('dense_l2_reg'),
  set: (val) => setParamValue('dense_l2_reg', val)
});

watch(
  () => paramEntries,
  () => {
    try {
      error.value = '';
      const params = buildParams(paramEntries);
      form.paramsText = dumpYaml(params);
    } catch (err) {
      error.value = `参数解析失败: ${err.message}`;
    }
  },
  { deep: true }
);

const yamlText = computed(() => {
  try {
    error.value = '';
    const params = buildParams(paramEntries);
    const obj = {
      model: form.model,
      params
    };
    return dumpYaml(obj);
  } catch (err) {
    error.value = `参数解析失败: ${err.message}`;
    return '';
  }
});

function download() {
  if (error.value) {
    return;
  }
  downloadText('model_config.yaml', yamlText.value);
}

onMounted(() => {
  if (!form.model && presetList.length > 0) {
    form.model = presetList[0];
  }
  const preset = presets[form.model];
  if (preset && (!form.paramsText || !form.paramsText.trim())) {
    form.paramsText = preset.paramsText;
  }
  const params = parseYaml(form.paramsText || '') || {};
  setEntriesFromParams(params);
  if (preset) {
    const comments = parsePresetComments(getPresetRaw(preset.model));
    modelNote.value = comments.top;
    Object.keys(paramHelp).forEach((key) => delete paramHelp[key]);
    Object.entries(comments.params).forEach(([key, note]) => {
      paramHelp[key] = note;
    });
  }
});
</script>
