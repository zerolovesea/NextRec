<template>
  <section class="page">
    <div class="card">
      <h2>{{ t('特征配置', 'Feature Configuration') }}</h2>
      <p style="color: var(--muted); margin-top: 0;">
        {{ t('为 稠密特征、稀疏特征、序列特征 分别配置特征处理方式和嵌入方式。', 'Configure processing and embeddings for dense, sparse, and sequence features.') }}
      </p>
    </div>

    <div class="card">
      <h2>{{ t('Dense 特征', 'Dense Features') }}</h2>
      <div class="inline-list">
        <div v-for="(item, index) in dense" :key="item.id" class="inline-item">
          <div class="grid-inline">
            <div class="field">
              <label>{{ t('特征名', 'Feature Name') }}</label>
              <input v-model="item.name" placeholder="user_active_days_7" />
            </div>
            <div class="field">
              <label>{{ t('归一化方式', 'scaler') }}</label>
              <select v-model="item.processor.scaler">
                <option value="standard">standard</option>
                <option value="minmax">minmax</option>
                <option value="robust">robust</option>
                <option value="none">none</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t('输入维度', 'input_dim') }}</label>
              <input v-model.number="item.embedding.input_dim" type="number" />
            </div>
            <div class="field">
              <label>{{ t('投影维度', 'proj_dim') }}</label>
              <input v-model.number="item.embedding.proj_dim" type="number" />
            </div>
            <div class="field">
              <label>{{ t('使用投影', 'use_projection') }}</label>
              <select v-model="item.embedding.use_projection">
                <option :value="false">false</option>
                <option :value="true">true</option>
              </select>
            </div>
          </div>
          <div class="actions">
            <button class="ghost" @click="remove(dense, index)">{{ t('移除', 'Remove') }}</button>
          </div>
        </div>
      </div>
      <div class="actions" style="margin-top: 12px;">
        <button class="secondary" @click="addDense">{{ t('添加', 'Add') }}</button>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('Sparse 特征', 'Sparse Features') }}</h2>
      <div class="inline-list">
        <div v-for="(item, index) in sparse" :key="item.id" class="inline-item">
          <div class="grid-inline">
            <div class="field">
              <label>{{ t('特征名', 'Feature Name') }}</label>
              <input v-model="item.name" placeholder="user_id" />
            </div>
            <div class="field">
              <label>{{ t('嵌入名称', 'Embedding Name') }}</label>
              <input v-model="item.embedding.embedding_name" placeholder="user_id" />
            </div>
            <div class="field">
              <label>{{ t('编码方式', 'encode_method') }}</label>
              <select v-model="item.processor.encode_method">
                <option value="hash">hash</option>
                <option value="vocab">vocab</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t('哈希桶大小', 'hash_size') }}</label>
              <input v-model.number="item.processor.hash_size" type="number" />
            </div>
            <div class="field">
              <label>{{ t('最小频次', 'min_freq') }}</label>
              <input v-model.number="item.processor.min_freq" type="number" />
            </div>
            <div class="field">
              <label>{{ t('嵌入维度', 'embedding_dim') }}</label>
              <input v-model.number="item.embedding.embedding_dim" type="number" />
            </div>
            <div class="field">
              <label>{{ t('填充索引', 'padding_idx') }}</label>
              <input v-model.number="item.embedding.padding_idx" type="number" />
            </div>
          </div>
          <div class="actions">
            <button class="ghost" @click="remove(sparse, index)">{{ t('移除', 'Remove') }}</button>
          </div>
        </div>
      </div>
      <div class="actions" style="margin-top: 12px;">
        <button class="secondary" @click="addSparse">{{ t('添加', 'Add') }}</button>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('Sequence 特征', 'Sequence Features') }}</h2>
      <div class="inline-list">
        <div v-for="(item, index) in sequence" :key="item.id" class="inline-item">
          <div class="grid-inline">
            <div class="field">
              <label>{{ t('特征名', 'Feature Name') }}</label>
              <input v-model="item.name" placeholder="hist_item_seq" />
            </div>
            <div class="field">
              <label>{{ t('嵌入名称', 'Embedding Name') }}</label>
              <input v-model="item.embedding.embedding_name" placeholder="hist_item_seq" />
            </div>
            <div class="field">
              <label>{{ t('编码方式', 'encode_method') }}</label>
              <select v-model="item.processor.encode_method">
                <option value="hash">hash</option>
                <option value="vocab">vocab</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t('哈希桶大小', 'hash_size') }}</label>
              <input v-model.number="item.processor.hash_size" type="number" />
            </div>
            <div class="field">
              <label>{{ t('最小频次', 'min_freq') }}</label>
              <input v-model.number="item.processor.min_freq" type="number" />
            </div>
            <div class="field">
              <label>{{ t('最大长度', 'max_len') }}</label>
              <input v-model.number="item.processor.max_len" type="number" />
            </div>
            <div class="field">
              <label>{{ t('填充值', 'pad_value') }}</label>
              <input v-model.number="item.processor.pad_value" type="number" />
            </div>
            <div class="field">
              <label>{{ t('截断方式', 'truncate') }}</label>
              <select v-model="item.processor.truncate">
                <option value="post">post</option>
                <option value="pre">pre</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t('分隔符', 'separator') }}</label>
              <input v-model="item.processor.separator" placeholder="," />
            </div>
            <div class="field">
              <label>{{ t('词表大小', 'vocab_size') }}</label>
              <input v-model.number="item.embedding.vocab_size" type="number" />
            </div>
            <div class="field">
              <label>{{ t('最大长度', 'max_len') }}</label>
              <input v-model.number="item.embedding.max_len" type="number" />
            </div>
            <div class="field">
              <label>{{ t('池化方式', 'combiner') }}</label>
              <select v-model="item.embedding.combiner">
                <option value="mean">mean</option>
                <option value="sum">sum</option>
                <option value="concat">concat</option>
                <option value="dot_attention">dot_attention</option>
                <option value="self_attention">self_attention</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t('嵌入维度', 'embedding_dim') }}</label>
              <input v-model.number="item.embedding.embedding_dim" type="number" />
            </div>
            <div class="field">
              <label>{{ t('填充索引', 'padding_idx') }}</label>
              <input v-model.number="item.embedding.padding_idx" type="number" />
            </div>
          </div>
          <div class="actions">
            <button class="ghost" @click="remove(sequence, index)">{{ t('移除', 'Remove') }}</button>
          </div>
        </div>
      </div>
      <div class="actions" style="margin-top: 12px;">
        <button class="secondary" @click="addSequence">{{ t('添加', 'Add') }}</button>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('YAML 预览', 'YAML Preview') }}</h2>
      <div class="field" style="margin-top: 12px;">
        <textarea class="mono" readonly :value="yamlText"></textarea>
      </div>
      <div class="actions spaced-top">
        <button class="primary" @click="download">{{ t('下载 feature_config.yaml', 'Download feature_config.yaml') }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue';
import { dumpYaml } from '../utils/yaml.js';
import { downloadText } from '../utils/download.js';
import { store, addFeatureItem, removeFeatureItem } from '../store/configStore.js';

const dense = store.feature.dense;
const sparse = store.feature.sparse;
const sequence = store.feature.sequence;

const lang = computed(() => store.ui.lang);
const t = (zh, en) => (lang.value === 'zh' ? zh : en);

function addDense() {
  addFeatureItem(dense, 'dense');
}

function addSparse() {
  addFeatureItem(sparse, 'sparse');
}

function addSequence() {
  addFeatureItem(sequence, 'sequence');
}

function remove(list, index) {
  removeFeatureItem(list, index);
}

function buildSection(items) {
  const section = {};
  for (const item of items) {
    if (!item.name) {
      continue;
    }
    const embedding = { ...item.embedding };
    embedding.name = item.name;
    section[item.name] = {
      processor_config: { ...item.processor },
      embedding_config: embedding
    };
  }
  return section;
}

const yamlText = computed(() => {
  const obj = {
    dense: buildSection(dense),
    sparse: buildSection(sparse),
    sequence: buildSection(sequence)
  };
  return dumpYaml(obj);
});

function download() {
  downloadText('feature_config.yaml', yamlText.value);
}
</script>
