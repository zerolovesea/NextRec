<template>
  <section class="page">
    <div class="card">
      <h2>{{ t('特征配置', 'Feature Configuration') }}</h2>
      <p style="color: var(--muted); margin-top: 0;">
        <span v-if="isIdtank">
          {{ t('基于列名规则批量生成特征配置代码。', 'Generate feature config code in batch based on column rules.') }}
        </span>
        <span v-else>
          {{ t('为 稠密特征、稀疏特征、序列特征 分别配置特征处理方式和嵌入方式。', 'Configure processing and embeddings for dense, sparse, and sequence features.') }}
        </span>
      </p>
    </div>

    <template v-if="isIdtank">
      <div class="card">
        <h2>{{ t('稠密特征批量参数', 'Dense Batch Settings') }}</h2>
        <div class="field" style="margin-top: 12px; margin-bottom: 12px;">
          <label>{{ t('稠密特征列表达式', 'Dense Columns Expression') }}</label>
          <input v-model="batchRules.dense" class="mono" placeholder="[col for col in columns if 'cnt' in col]" />
        </div>
        <div class="grid-2">
          <div class="field">
            <label>{{ t('缩放方式', 'Scaler') }}</label>
            <select v-model="batchDense.processor.scaler">
              <option value="standard">standard</option>
              <option value="minmax">minmax</option>
              <option value="robust">robust</option>
              <option value="maxabs">maxabs</option>
              <option value="log">log</option>
              <option value="none">none</option>
            </select>
          </div>
          <div class="field">
            <label>{{ t('输入维度', 'input_dim') }}</label>
            <input v-model.number="batchDense.embedding.input_dim" type="number" />
          </div>
          <div class="field" v-if="batchDense.embedding.use_projection">
            <label>{{ t('投影维度', 'proj_dim') }}</label>
            <input v-model.number="batchDense.embedding.proj_dim" type="number" />
          </div>
          <div class="field">
            <label>
              {{ t('使用投影', 'use_projection') }}
              <span class="help-icon" :data-tip="t('对于部分场景和模型，会要求对稠密特征做线性变换进行升维，例如 MASKNET。', 'Some scenarios/models require a linear projection to expand dense features, e.g. MASKNET.')">?</span>
            </label>
            <select v-model="batchDense.embedding.use_projection">
              <option :value="false">false</option>
              <option :value="true">true</option>
            </select>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>{{ t('稀疏特征批量参数', 'Sparse Batch Settings') }}</h2>
        <div class="field" style="margin-top: 12px;margin-bottom: 16px;">
          <label>{{ t('稀疏特征列表达式', 'Sparse Columns Expression') }}</label>
          <input v-model="batchRules.sparse" class="mono" placeholder="[col for col in columns if 'tag' in col]" />
        </div>
        <div class="grid-2">
          <div class="field">
            <label>
              {{ t('编码方式', 'encode_method') }}
              <span class="help-icon" :data-tip="t('编码方式决定如何将特征值转换为数值表示。支持哈希编码和标签编码。选择哈希时，需要配置哈希桶大小', 'Encoding method determines how feature values are converted to numeric representations. Supports hash and label encoding. When using hash, configure hash size.')">?</span>
            </label>
            <select v-model="batchSparse.processor.encode_method">
              <option value="hash">hash</option>
              <option value="label">label</option>
            </select>
          </div>
          <div class="field" v-if="batchSparse.processor.encode_method === 'hash'">
            <label>
              {{ t('哈希桶大小', 'hash_size') }}
              <span class="help-icon" :data-tip="t('哈希编码时使用的桶大小，决定哈希函数的取值范围。', 'Bucket size used for hash encoding; defines the hash value range.')">?</span>
            </label>
            <input v-model.number="batchSparse.processor.hash_size" type="number" />
          </div>
          <div class="field">
            <label>
              {{ t('最小频次', 'min_freq') }}
              <span class="help-icon" :data-tip="t('特征值出现的最小频次，低于该频次的值将被视为词表外。', 'Minimum frequency for a feature value; values below this are treated as OOV.')">?</span>
            </label>
            <input v-model.number="batchSparse.processor.min_freq" type="number" />
          </div>
          <div class="field">
            <label>{{ t('嵌入维度', 'embedding_dim') }}</label>
            <input v-model.number="batchSparse.embedding.embedding_dim" type="number" />
          </div>
          <div class="field">
            <label>{{ t('PADDING IDX', 'padding_idx') }}</label>
            <input v-model.number="batchSparse.embedding.padding_idx" type="number" />
          </div>
          <div class="field">
            <label>
              {{ t('初始化方式', 'init_type') }}
              <span class="help-icon" :data-tip="t('嵌入参数的初始化方式，支持 xavier/kaiming/orthogonal/normal/uniform。', 'Embedding initializer type: xavier/kaiming/orthogonal/normal/uniform.')">?</span>
            </label>
            <select v-model="batchSparse.embedding.init_type">
              <option v-for="opt in initTypeOptions" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </div>
          <div class="field">
            <label>{{ t('L1 正则', 'l1_reg') }}</label>
            <input v-model.number="batchSparse.embedding.l1_reg" type="number" step="0.0001" />
          </div>
          <div class="field">
            <label>{{ t('L2 正则', 'l2_reg') }}</label>
            <input v-model.number="batchSparse.embedding.l2_reg" type="number" step="0.0001" />
          </div>
        </div>
      </div>

      <div class="card">
        <h2>{{ t('序列特征批量参数', 'Sequence Batch Settings') }}</h2>
        <div class="field" style="margin-top: 12px;margin-bottom: 16px;">
          <label>{{ t('序列特征列表达式', 'Sequence Columns Expression') }}</label>
          <input v-model="batchRules.sequence" class="mono" placeholder="[col for col in columns if 'textlist' in col] + ['outerBizSorted']" />
        </div>
        <div class="grid-2">
          <div class="field">
            <label>
              {{ t('编码方式', 'encode_method') }}
              <span class="help-icon" :data-tip="t('编码方式决定如何将特征值转换为数值表示。支持哈希编码和标签编码。选择哈希时，需要配置哈希桶大小', 'Encoding method determines how feature values are converted to numeric representations. Supports hash and label encoding. When using hash, configure hash size.')">?</span>
            </label>
            <select v-model="batchSequence.processor.encode_method">
              <option value="hash">hash</option>
              <option value="label">label</option>
            </select>
          </div>
          <div class="field" v-if="batchSequence.processor.encode_method === 'hash'">
            <label>
              {{ t('哈希桶大小', 'hash_size') }}
              <span class="help-icon" :data-tip="t('哈希编码时使用的桶大小，决定哈希函数的取值范围。', 'Bucket size used for hash encoding; defines the hash value range.')">?</span>
            </label>
            <input v-model.number="batchSequence.processor.hash_size" type="number" />
          </div>
          <div class="field">
            <label>
              {{ t('截断最大长度', 'max_len') }}
              <span class="help-icon" :data-tip="t('序列特征的最大截断长度，超过该长度的序列将被截断。', 'Maximum truncation length for sequence features; sequences longer than this are truncated.')">?</span>
            </label>
            <input v-model.number="batchSequence.processor.max_len" type="number" />
          </div>
          <div class="field">
            <label>{{ t('填充值', 'pad_value') }}</label>
            <input v-model.number="batchSequence.processor.pad_value" type="number" />
          </div>
          <div class="field">
            <label>
              {{ t('截断方式', 'truncate') }}
              <span class="help-icon" :data-tip="t('截断方式决定序列特征在超过最大长度时如何截断。当选择post时，从序列末尾截断；选择pre时，从序列开头截断。', 'Truncation strategy when exceeding max length. post truncates from the end; pre truncates from the beginning.')">?</span>
            </label>
            <select v-model="batchSequence.processor.truncate">
              <option value="post">post</option>
              <option value="pre">pre</option>
            </select>
          </div>
          <div class="field">
            <label>
              {{ t('分隔符', 'separator') }}
              <span class="help-icon" :data-tip="t('序列特征中元素的分隔符，用于将字符串拆分为序列。', 'Separator for elements in a sequence feature, used to split a string into a sequence.')">?</span>
            </label>
            <input v-model="batchSequence.processor.separator" placeholder="," />
          </div>
          <div class="field">
            <label>{{ t('词表大小', 'vocab_size') }}</label>
            <input v-model.number="batchSequence.embedding.vocab_size" type="number" />
          </div>
          <div class="field">
            <label>{{ t('嵌入最大长度', 'max_len (embedding)') }}</label>
            <input v-model.number="batchSequence.embedding.max_len" type="number" />
          </div>
          <div class="field">
            <label>{{ t('池化方式', 'combiner') }}</label>
            <select v-model="batchSequence.embedding.combiner">
              <option value="mean">mean</option>
              <option value="sum">sum</option>
              <option value="concat">concat</option>
              <option value="dot_attention">dot_attention</option>
              <option value="self_attention">self_attention</option>
            </select>
          </div>
          <div class="field">
            <label>{{ t('嵌入维度', 'embedding_dim') }}</label>
            <input v-model.number="batchSequence.embedding.embedding_dim" type="number" />
          </div>
          <div class="field">
            <label>{{ t('PADDING IDX', 'padding_idx') }}</label>
            <input v-model.number="batchSequence.embedding.padding_idx" type="number" />
          </div>
          <div class="field">
            <label>
              {{ t('初始化方式', 'init_type') }}
              <span class="help-icon" :data-tip="t('嵌入参数的初始化方式，支持 xavier/kaiming/orthogonal/normal/uniform。', 'Embedding initializer type: xavier/kaiming/orthogonal/normal/uniform.')">?</span>
            </label>
            <select v-model="batchSequence.embedding.init_type">
              <option v-for="opt in initTypeOptions" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </div>
          <div class="field">
            <label>{{ t('L1 正则', 'l1_reg') }}</label>
            <input v-model.number="batchSequence.embedding.l1_reg" type="number" step="0.0001" />
          </div>
          <div class="field">
            <label>{{ t('L2 正则', 'l2_reg') }}</label>
            <input v-model.number="batchSequence.embedding.l2_reg" type="number" step="0.0001" />
          </div>
        </div>
      </div>

      <div class="card">
        <h2>{{ t('代码预览', 'Code Preview') }}</h2>
        <div class="field" style="margin-top: 12px;">
          <textarea class="mono code-preview" readonly :value="idtankScript" rows="24"></textarea>
        </div>
        <div class="actions spaced-top">
          <button class="primary" @click="download">{{ t('下载', 'Download') }}</button>
        </div>
      </div>
    </template>

    <template v-else>
    <div class="card">
      <h2>{{ t('稠密特征', 'Dense Features') }}</h2>
      <div class="inline-list">
        <div v-for="(item, index) in dense" :key="item.id" class="inline-item">
          <div class="grid-inline">
            <div class="field">
              <label>{{ t('特征名', 'Feature Name') }}</label>
              <input v-model="item.name" placeholder="user_active_days_7" />
            </div>
            <div class="field">
              <label>{{ t('缩放方式', 'Scaler') }}</label>
              <select v-model="item.processor.scaler">
                <option value="standard">standard</option>
                <option value="minmax">minmax</option>
                <option value="robust">robust</option>
                <option value="maxabs">maxabs</option>
                <option value="log">log</option>
                <option value="none">none</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t('输入维度', 'input_dim') }}</label>
              <input v-model.number="item.embedding.input_dim" type="number" />
            </div>
            <div class="field" v-if="item.embedding.use_projection">
              <label>{{ t('投影维度', 'proj_dim') }}</label>
              <input v-model.number="item.embedding.proj_dim" type="number" />
            </div>
            <div class="field">
              <label>
                {{ t('使用投影', 'use_projection') }}
                <span class="help-icon" :data-tip="t('对于部分场景和模型，会要求对稠密特征做线性变换进行升维，例如 MASKNET。', 'Some scenarios/models require a linear projection to expand dense features, e.g. MASKNET.')">?</span>
              </label>
              <select v-model="item.embedding.use_projection">
                <option :value="false">false</option>
                <option :value="true">true</option>
              </select>
            </div>
          </div>
          <div class="actions">
            <button class="icon-button icon-button-small icon-button-danger" @click="remove(dense, index)" aria-label="移除">×</button>
          </div>
        </div>
      </div>
      <div class="actions" style="margin-top: 12px;">
        <button class="secondary" @click="addDense">{{ t('添加', 'Add') }}</button>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('稀疏特征', 'Sparse Features') }}</h2>
      <div class="inline-list">
        <div v-for="(item, index) in sparse" :key="item.id" class="inline-item">
          <div class="grid-inline">
            <div class="field">
              <label>{{ t('特征名', 'Feature Name') }}</label>
              <input v-model="item.name" placeholder="user_id" />
            </div>
            <div class="field">
              <label>
                {{ t('嵌入名称', 'Embedding Name') }}
                <span class="help-icon" :data-tip="t('为特征嵌入指定名称，当多个特征使用统一嵌入名称时，它们共享同一嵌入向量。', 'Name for the embedding. Features sharing the same embedding name share one embedding table.')">?</span>
              </label>
              <input v-model="item.embedding.embedding_name" placeholder="user_id" />
            </div>
            <div class="field">
              <label>
                {{ t('编码方式', 'encode_method') }}
                <span class="help-icon" :data-tip="t('编码方式决定如何将特征值转换为数值表示。支持哈希编码和标签编码。选择哈希时，需要配置哈希桶大小', 'Encoding method determines how feature values are converted to numeric representations. Supports hash and label encoding. When using hash, configure hash size.')">?</span>
              </label>
              <select v-model="item.processor.encode_method">
                <option value="hash">hash</option>
                <option value="label">label</option>
              </select>
            </div>
            <div class="field" v-if="item.processor.encode_method === 'hash'">
              <label>
                {{ t('哈希桶大小', 'hash_size') }}
                <span class="help-icon" :data-tip="t('哈希编码时使用的桶大小，决定哈希函数的取值范围。', 'Bucket size used for hash encoding; defines the hash value range.')">?</span>
              </label>
              <input v-model.number="item.processor.hash_size" type="number" />
            </div>
            <div class="field">
              <label>
                {{ t('最小频次', 'min_freq') }}
                <span class="help-icon" :data-tip="t('特征值出现的最小频次，低于该频次的值将被视为词表外。', 'Minimum frequency for a feature value; values below this are treated as OOV.')">?</span>
              </label>
              <input v-model.number="item.processor.min_freq" type="number" />
            </div>
            <div class="field">
              <label>{{ t('嵌入维度', 'embedding_dim') }}</label>
              <input v-model.number="item.embedding.embedding_dim" type="number" />
            </div>
            <div class="field">
              <label>{{ t('PADDING IDX', 'padding_idx') }}</label>
              <input v-model.number="item.embedding.padding_idx" type="number" />
            </div>
            <div class="field">
              <label>
                {{ t('初始化方式', 'init_type') }}
                <span class="help-icon" :data-tip="t('嵌入参数的初始化方式，支持 xavier/kaiming/orthogonal/normal/uniform。', 'Embedding initializer type: xavier/kaiming/orthogonal/normal/uniform.')">?</span>
              </label>
              <select v-model="item.embedding.init_type">
                <option v-for="opt in initTypeOptions" :key="opt" :value="opt">{{ opt }}</option>
              </select>
            </div>
          </div>
          <div class="actions">
            <button class="icon-button icon-button-small icon-button-danger" @click="remove(sparse, index)" aria-label="移除">×</button>
          </div>
        </div>
      </div>
      <div class="actions" style="margin-top: 12px;">
        <button class="secondary" @click="addSparse">{{ t('添加', 'Add') }}</button>
      </div>
    </div>

    <div class="card">
      <h2>{{ t('序列特征', 'Sequence Features') }}</h2>
      <div class="inline-list">
        <div v-for="(item, index) in sequence" :key="item.id" class="inline-item">
          <div class="grid-inline">
            <div class="field">
              <label>{{ t('特征名', 'Feature Name') }}</label>
              <input v-model="item.name" placeholder="hist_item_seq" />
            </div>
            <div class="field">
              <label>
                {{ t('嵌入名称', 'Embedding Name') }}
                <span class="help-icon" :data-tip="t('为特征嵌入指定名称，当多个特征使用统一嵌入名称时，它们共享同一嵌入向量。', 'Name for the embedding. Features sharing the same embedding name share one embedding table.')">?</span>
              </label>
              <input v-model="item.embedding.embedding_name" placeholder="hist_item_seq" />
            </div>
            <div class="field">
              <label>
                {{ t('编码方式', 'encode_method') }}
                <span class="help-icon" :data-tip="t('编码方式决定如何将特征值转换为数值表示。支持哈希编码和标签编码。选择哈希时，需要配置哈希桶大小', 'Encoding method determines how feature values are converted to numeric representations. Supports hash and label encoding. When using hash, configure hash size.')">?</span>
              </label>
              <select v-model="item.processor.encode_method">
                <option value="hash">hash</option>
                <option value="label">label</option>
              </select>
            </div>
            <div class="field" v-if="item.processor.encode_method === 'hash'">
              <label>
                {{ t('哈希桶大小', 'hash_size') }}
                <span class="help-icon" :data-tip="t('哈希编码时使用的桶大小，决定哈希函数的取值范围。', 'Bucket size used for hash encoding; defines the hash value range.')">?</span>
              </label>
              <input v-model.number="item.processor.hash_size" type="number" />
            </div>
            <div class="field">
              <label>
                {{ t('最小频次', 'min_freq') }}
                <span class="help-icon" :data-tip="t('特征值出现的最小频次，低于该频次的值将被视为词表外。', 'Minimum frequency for a feature value; values below this are treated as OOV.')">?</span>
              </label>
              <input v-model.number="item.processor.min_freq" type="number" />
            </div>
            <div class="field">
              <label>
                {{ t('截断最大长度', 'max_len') }}
                <span class="help-icon" :data-tip="t('序列特征的最大截断长度，超过该长度的序列将被截断。', 'Maximum truncation length for sequence features; sequences longer than this are truncated.')">?</span>
              </label>
              <input v-model.number="item.processor.max_len" type="number" />
            </div>
            <div class="field">
              <label>{{ t('填充值', 'pad_value') }}</label>
              <input v-model.number="item.processor.pad_value" type="number" />
            </div>
            <div class="field">
              <label>
                {{ t('截断方式', 'truncate') }}
                <span class="help-icon" :data-tip="t('截断方式决定序列特征在超过最大长度时如何截断。当选择post时，从序列末尾截断；选择pre时，从序列开头截断。', 'Truncation strategy when exceeding max length. post truncates from the end; pre truncates from the beginning.')">?</span>
              </label>
              <select v-model="item.processor.truncate">
                <option value="post">post</option>
                <option value="pre">pre</option>
              </select>
            </div>
            <div class="field">
              <label>
                {{ t('分隔符', 'separator') }}
                <span class="help-icon" :data-tip="t('序列特征中元素的分隔符，用于将字符串拆分为序列。', 'Separator for elements in a sequence feature, used to split a string into a sequence.')">?</span>
              </label>
              <input v-model="item.processor.separator" placeholder="," />
            </div>
            <div class="field">
              <label>{{ t('词表大小', 'vocab_size') }}</label>
              <input v-model.number="item.embedding.vocab_size" type="number" />
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
              <label>{{ t('PADDING IDX', 'padding_idx') }}</label>
              <input v-model.number="item.embedding.padding_idx" type="number" />
            </div>
            <div class="field">
              <label>
                {{ t('初始化方式', 'init_type') }}
                <span class="help-icon" :data-tip="t('嵌入参数的初始化方式，支持 xavier/kaiming/orthogonal/normal/uniform。', 'Embedding initializer type: xavier/kaiming/orthogonal/normal/uniform.')">?</span>
              </label>
              <select v-model="item.embedding.init_type">
                <option v-for="opt in initTypeOptions" :key="opt" :value="opt">{{ opt }}</option>
              </select>
            </div>
          </div>
          <div class="actions">
            <button class="icon-button icon-button-small icon-button-danger" @click="remove(sequence, index)" aria-label="移除">×</button>
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
        <button class="primary" @click="download">{{ t('下载', 'Download') }}</button>
      </div>
    </div>
    </template>
  </section>
</template>

<script setup>
import { computed } from 'vue';
import { dumpYaml } from '../utils/yaml.js';
import { downloadText } from '../utils/download.js';
import { store, addFeatureItem, removeFeatureItem } from '../store/configStore.js';
import { isIdtank } from '../utils/appEnv.js';

const dense = store.feature.dense;
const sparse = store.feature.sparse;
const sequence = store.feature.sequence;
const batchRules = store.featureBatch.rules;
const batchDense = store.featureBatch.dense;
const batchSparse = store.featureBatch.sparse;
const batchSequence = store.featureBatch.sequence;
const initTypeOptions = [
  'xavier_uniform',
  'xavier_normal',
  'kaiming_uniform',
  'kaiming_normal',
  'orthogonal',
  'normal',
  'uniform'
];

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
    embedding.max_len = item.processor.max_len;
    embedding.name = item.name;
    section[item.name] = {
      processor_config: { ...item.processor },
      embedding_config: embedding
    };
  }
  return section;
}

function toPythonLiteral(value) {
  if (value === null || value === undefined) {
    return 'None';
  }
  if (typeof value === 'string') {
    const escaped = value.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    return `'${escaped}'`;
  }
  if (typeof value === 'boolean') {
    return value ? 'True' : 'False';
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || Number.isNaN(value)) {
      return 'None';
    }
    return String(value);
  }
  return 'None';
}

const idtankRules = computed(() => {
  const denseExpr = (batchRules.dense || '').trim() || "[col for col in columns if 'cnt' in col]";
  const sparseExpr = (batchRules.sparse || '').trim() || "[col for col in columns if 'tag' in col]";
  const sequenceExpr =
    (batchRules.sequence || '').trim() || "[col for col in columns if 'textlist' in col] + ['outerBizSorted']";
  return `dense_features_list = ${denseExpr}\n` +
    `sparse_features_list = ${sparseExpr}\n` +
    `sequence_features_list = ${sequenceExpr}`;
});

const idtankRulesIndented = computed(() =>
  idtankRules.value
    .split('\n')
    .map((line) => `    ${line}`)
    .join('\n')
);

const idtankScript = computed(() => {
  const dense = batchDense;
  const sparse = batchSparse;
  const sequence = batchSequence;

  return `
import sys
import yaml
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any


def get_parquet_columns(dir_path: str) -> List[str]:
    path = Path(dir_path)
    
    first_parquet = None
    for f in path.iterdir():
        if f.suffix == '.parquet':
            first_parquet = f
            break
    
    if first_parquet is None:
        raise FileNotFoundError(f"目录 {dir_path} 中没有找到 parquet 文件")
    
    parquet_file = pq.ParquetFile(first_parquet)
    
    schema = parquet_file.schema_arrow
    columns = [field.name for field in schema]
    
    return columns

def generate_feature_config(
    dense_features_list: List[str],
    sparse_features_list: List[str],
    sequence_features_list: List[str],
    target_path: str,
) -> None:
    feature_config: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
        'dense': {},
        'sparse': {},
        'sequence': {},
    }

    for feat in dense_features_list:
        feature_config['dense'][feat] = {
            'processor_config': {
                'type': 'numeric',
                'scaler': ${toPythonLiteral(dense.processor.scaler)},
            },
            'embedding_config': {
                'name': feat,
                'input_dim': ${toPythonLiteral(dense.embedding.input_dim)},
                'use_projection': ${toPythonLiteral(dense.embedding.use_projection)},
            },
        }

    for feat in sparse_features_list:
        feature_config['sparse'][feat] = {
            'processor_config': {
                'type': 'sparse',
                'encode_method': ${toPythonLiteral(sparse.processor.encode_method)},
                'hash_size': ${toPythonLiteral(sparse.processor.hash_size)},
                'min_freq': ${toPythonLiteral(sparse.processor.min_freq)},
            },
            'embedding_config': {
                'name': feat,
                'embedding_dim': ${toPythonLiteral(sparse.embedding.embedding_dim)},
                'embedding_name': feat,
                'padding_idx': ${toPythonLiteral(sparse.embedding.padding_idx)},
                'init_type': ${toPythonLiteral(sparse.embedding.init_type)},
                'init_params': {},
                'l1_reg': ${toPythonLiteral(sparse.embedding.l1_reg)},
                'l2_reg': ${toPythonLiteral(sparse.embedding.l2_reg)},
            },
        }

    for feat in sequence_features_list:
        print(f'Processing sequence feature: {feat}')
        if '_product_textlist' in feat:
            embedding_name = 'product_textlist'
        elif 'algmodel_' in feat:
            embedding_name = 'algmodel_textlist'
        else:
            embedding_name = feat
        feature_config['sequence'][feat] = {
            'processor_config': {
                'type': 'sequence',
                'encode_method': ${toPythonLiteral(sequence.processor.encode_method)},
                'hash_size': ${toPythonLiteral(sequence.processor.hash_size)},
                'max_len': ${toPythonLiteral(sequence.processor.max_len)},
                'pad_value': ${toPythonLiteral(sequence.processor.pad_value)},
                'truncate': ${toPythonLiteral(sequence.processor.truncate)},
                'separator': ${toPythonLiteral(sequence.processor.separator)},
            },
            'embedding_config': {
                'name': feat,
                'vocab_size': ${toPythonLiteral(sequence.embedding.vocab_size)},
                'max_len': ${toPythonLiteral(sequence.embedding.max_len)},
                'embedding_name': embedding_name,
                'combiner': ${toPythonLiteral(sequence.embedding.combiner)},
                'embedding_dim': ${toPythonLiteral(sequence.embedding.embedding_dim)},
                'padding_idx': ${toPythonLiteral(sequence.embedding.padding_idx)},
                'init_type': ${toPythonLiteral(sequence.embedding.init_type)},
                'init_params': {},
                'l1_reg': ${toPythonLiteral(sequence.embedding.l1_reg)},
                'l2_reg': ${toPythonLiteral(sequence.embedding.l2_reg)},
            },
        }

    feature_config_path = Path(target_path)

    with feature_config_path.open('w', encoding='utf-8') as fp:
        yaml.safe_dump(
            feature_config,
            fp,
            allow_unicode=True,
            sort_keys=False,
        )
    print(f'Feature config saved to {feature_config_path.resolve()}')


def main():
    if len(sys.argv) != 3:
        print("错误：参数数量不正确！")
        print("当前参数数量: {}，需要: 2".format(len(sys.argv) - 1))
        print("用法: python generate_feature_config.py <数据文件路径> <输出文件名>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_filename = sys.argv[2]
    
    target_config_path = Path.cwd() / output_filename
    
    print("正在处理数据文件: {}".format(file_path))
    print("输出配置文件路径: {}".format(target_config_path))
    
    columns = get_parquet_columns(file_path)
    print(f"共 {len(columns)} 列")
    
${idtankRulesIndented.value}
    
    print(f"Dense features: {len(dense_features_list)}")
    print(f"Sparse features: {len(sparse_features_list)}")
    print(f"Sequence features: {len(sequence_features_list)}")
    print(f"所有列名: {columns}")
    generate_feature_config(
        dense_features_list=dense_features_list,
        sparse_features_list=sparse_features_list,
        sequence_features_list=sequence_features_list,
        target_path=str(target_config_path),
    )


if __name__ == '__main__':
    main()
`;
});

const yamlText = computed(() => {
  const obj = {
    dense: buildSection(dense),
    sparse: buildSection(sparse),
    sequence: buildSection(sequence)
  };
  return dumpYaml(obj);
});

function download() {
  if (isIdtank) {
    downloadText('generate_feature_config_original.txt', idtankScript.value);
    return;
  }
  downloadText('feature_config.yaml', yamlText.value);
}
</script>
