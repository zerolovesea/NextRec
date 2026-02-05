<template>
  <div class="app-shell">
    <aside class="sidebar" ref="sidebarRef">
      <div class="brand">
        <img :src="logoUrl" alt="NextRec Logo" />
      </div>
      <nav class="nav">
        <RouterLink to="/train">{{ t('训练配置', 'Training') }}</RouterLink>
        <RouterLink to="/feature">{{ t('特征配置', 'Features') }}</RouterLink>
        <RouterLink to="/model">{{ t('模型配置', 'Models') }}</RouterLink>
        <RouterLink to="/predict">{{ t('推理配置', 'Prediction') }}</RouterLink>
      </nav>
      <div class="lang-switch">
        <button class="lang-pill" :class="{ active: store.ui.lang === 'zh' }" @click="store.ui.lang = 'zh'">中文</button>
        <button class="lang-pill" :class="{ active: store.ui.lang === 'en' }" @click="store.ui.lang = 'en'">EN</button>
      </div>
    </aside>
    <main class="main" ref="mainRef">
      <div class="hero">
        <h1>
          NextRec Studio
          <span v-if="isIdtank" class="idtank-tag">IDTANK VERSION</span>
        </h1>
        <p>{{ t('为训练、特征、模型、推理提供标准化的 YAML 文件配置。', 'Provide standardized YAML configuration files for training, features, models, and prediction.') }}</p>
        <div v-if="bundleError" class="alert">{{ bundleError }}</div>
      </div>
      <RouterView />
    </main>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { RouterLink, RouterView } from 'vue-router';
import { store } from './store/configStore.js';
import { dumpYaml, parseYaml } from './utils/yaml.js';
import { buildTrainConfig } from './utils/buildTrainConfig.js';
import logoUrl from './assets/logo.png';
import { isIdtank } from './utils/appEnv.js';

const mainRef = ref(null);
const sidebarRef = ref(null);
const lang = computed(() => store.ui.lang);
const t = (zh, en) => (lang.value === 'zh' ? zh : en);

watch(
  () => store.ui.lang,
  () => {
    if (!mainRef.value) return;
    mainRef.value.classList.remove('lang-flash');
    // force reflow to restart animation
    void mainRef.value.offsetWidth;
    mainRef.value.classList.add('lang-flash');
    if (sidebarRef.value) {
      sidebarRef.value.classList.remove('lang-flash');
      void sidebarRef.value.offsetWidth;
      sidebarRef.value.classList.add('lang-flash');
    }
  }
);

function buildFeatureSection(items) {
  const section = {};
  for (const item of items) {
    if (!item.name) {
      continue;
    }
    section[item.name] = {
      processor_config: { ...item.processor },
      embedding_config: { ...item.embedding }
    };
  }
  return section;
}

const trainYamlResult = computed(() => {
  try {
    return { text: buildTrainConfig(store.train), error: '' };
  } catch (err) {
    return { text: '', error: `训练配置解析失败: ${err.message}` };
  }
});

const featureYamlResult = computed(() => {
  try {
    const obj = {
      dense: buildFeatureSection(store.feature.dense),
      sparse: buildFeatureSection(store.feature.sparse),
      sequence: buildFeatureSection(store.feature.sequence)
    };
    return { text: dumpYaml(obj), error: '' };
  } catch (err) {
    return { text: '', error: `特征配置解析失败: ${err.message}` };
  }
});

const modelYamlResult = computed(() => {
  try {
    const params = parseYaml(store.model.paramsText) || {};
    const obj = {
      model: store.model.model,
      params
    };
    return { text: dumpYaml(obj), error: '' };
  } catch (err) {
    return { text: '', error: `模型配置解析失败: ${err.message}` };
  }
});

const predictYaml = computed(() => {
  const output = {
    checkpoint_path: store.predict.checkpoint_path,
    predict: { ...store.predict.predict }
  };

  if (store.predict.model_config) {
    output.model_config = store.predict.model_config;
  }

  if (store.predict.session.id) {
    output.session = { id: store.predict.session.id };
  }

  return dumpYaml(output);
});

const bundleError = computed(() => {
  if (trainYamlResult.value.error) {
    return trainYamlResult.value.error;
  }
  if (featureYamlResult.value.error) {
    return featureYamlResult.value.error;
  }
  if (modelYamlResult.value.error) {
    return modelYamlResult.value.error;
  }
  return '';
});

</script>
