<script setup lang="ts">
const features = [
  {
    title: '统一特征抽象',
    description: '支持 Dense、Sparse、Sequence 三种特征类型，统一的数据处理与转换流程',
    icon: '📊'
  },
  {
    title: '多任务学习',
    description: '支持多任务学习，包括 ESMM、MMoE、PLE 等经典多任务架构',
    icon: '🎯'
  },
  {
    title: 'ONNX 导出',
    description: '支持模型导出为 ONNX 格式，跨框架推理部署更便捷',
    icon: '🚀'
  },
  {
    title: 'CLI & Studio',
    description: '提供命令行工具和可视化配置工具，降低使用门槛',
    icon: '⚡'
  },
  {
    title: '丰富模型库',
    description: '内置 DeepFM、DIN、DIEN、DCN、MVMoE 等主流推荐模型',
    icon: '🧠'
  },
  {
    title: '高性能数据加载',
    description: '支持大规模数据的流式加载与高效预处理',
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
