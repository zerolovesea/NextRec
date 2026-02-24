<script setup lang="ts">
const features = [
  {
    title: '统一特征抽象',
    description: '支持 Dense、Sparse、Sequence 三类特征，统一编码、转换与输入接口',
    icon: '📊'
  },
  {
    title: '模块化模型组件设计',
    description: '将特征交互、骨干网络、任务塔等模块解耦，支持按需组合与快速迭代',
    icon: '🧩'
  },
  {
    title: '统一训推验流程',
    description: '统一训练、推理、评估接口与配置，减少环境切换和流程割裂',
    icon: '🔄'
  },
  {
    title: '支持流式预处理',
    description: '支持大规模样本的流式读取、在线转换与高效预处理，降低内存压力',
    icon: '🌊'
  },
  {
    title: '流式与分布式训推',
    description: '支持流式训练推理与分布式训练推理，兼顾吞吐、时效与扩展性',
    icon: '🚀'
  },
  {
    title: '丰富任务与模型支持',
    description: '覆盖排序、召回、多任务与生成式建模，满足多场景推荐需求',
    icon: '🧠'
  },
  {
    title: '全面日志管理',
    description: '支持 Weights & Biases、SwanLab、TensorBoard，统一记录指标、曲线与实验产物',
    icon: '📈'
  },
  {
    title: '命令行配置驱动',
    description: '通过配置文件一键完成训练与推理，降低工程接入与复现实验成本',
    icon: '⚡'
  }
]
</script>

<template>
  <div class="home-features">
    <div class="feature-grid">
      <div v-for="feature in features" :key="feature.title" class="feature-card">
        <div class="feature-icon">{{ feature.icon }}</div>
        <h3>{{ feature.title }}</h3>
        <p>{{ feature.description }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.home-features {
  margin: 40px 0;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
}

.feature-card {
  padding: 28px;
  border-radius: 16px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  transition: all 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand-2);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
}

.feature-icon {
  font-size: 2rem;
  margin-bottom: 16px;
}

.feature-card h3 {
  color: var(--vp-c-text-1);
  margin-bottom: 10px;
  font-size: 1.15rem;
  font-weight: 600;
}

.feature-card p {
  color: var(--vp-c-text-2);
  line-height: 1.6;
  font-size: 0.95rem;
  margin: 0;
}
</style>
