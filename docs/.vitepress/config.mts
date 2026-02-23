import { defineConfig } from 'vitepress'
import { resolve } from 'path'

export default defineConfig({
  base: '/nextrec/',
  title: 'NextRec',
  description: '基于 PyTorch 的现代推荐系统框架',
  lang: 'zh-CN',
  lastUpdated: true,

  // 主题配置
  themeConfig: {
    logo: '/logo.svg',

    // 导航栏仅显示 Logo（隐藏站点标题文字）
    siteTitle: false,

    // 导航栏
    nav: [
      { text: '首页', link: '/zh/' },
      {
        text: '指南',
        items: [
          { text: '快速开始', link: '/zh/getting-started' },
          { text: '安装', link: '/zh/installatiton' }
        ]
      },
      {
        text: 'API',
        items: [
          { text: '概览', link: '/zh/apis/' },
          { text: '定义特征', link: '/zh/apis/features' },
          { text: '数据预处理', link: '/zh/apis/data-processor' },
          { text: '数据加载', link: '/zh/apis/dataloader' },
          { text: '基类模型的生命周期', link: '/zh/apis/base-model' },
          { text: '样本加权', link: '/zh/apis/sample-weighting' },
          { text: '损失函数', link: '/zh/apis/loss' },
          { text: '评估指标', link: '/zh/apis/metrics' },
          { text: '日志管理', link: '/zh/apis/session-logging' },
          { text: '分布式训练', link: '/zh/apis/distributed-training' },
        ]
      },
      {
        text: '教程',
        items: [
          { text: '概览', link: '/zh/tutorial/index' },
          { text: '训练精排模型', link: '/zh/tutorial/ranking' },
          { text: '训练多任务模型', link: '/zh/tutorial/multitask' },
          { text: '训练召回模型', link: '/zh/tutorial/retrieval' },
        ]
      },
      {
        text: '命令行工具',
        items: [
          { text: 'NextRec CLI', link: '/zh/cli/nextrec-cli' },
          { text: 'NextRec Studio', link: '/zh/cli/nextrec-studio' }
        ]
      },
      { text: 'FAQ', link: '/zh/faq' }
    ],

    // 侧边栏
    sidebar: {
      '/zh/': [
        {
          text: '开始',
          collapsed: false,
          items: [
            { text: '首页', link: '/zh/' },
            { text: '安装', link: '/zh/installatiton' },
            { text: '快速开始', link: '/zh/getting-started' }
          ]
        },
        {
          text: 'API 文档',
          collapsed: false,
          items: [
            { text: '概览', link: '/zh/apis/' },
            { text: '定义特征', link: '/zh/apis/features' },
            { text: '数据预处理', link: '/zh/apis/data-processor' },
            { text: '数据加载', link: '/zh/apis/dataloader' },
            { text: '基类模型的生命周期', link: '/zh/apis/base-model' },
            { text: '样本加权', link: '/zh/apis/sample-weighting' },
            { text: '损失函数', link: '/zh/apis/loss' },
            { text: '评估指标', link: '/zh/apis/metrics' },
            { text: '日志管理', link: '/zh/apis/session-logging' },
            { text: '分布式训练', link: '/zh/apis/distributed-training' },

          ]
        },
        {
          text: '教程',
          collapsed: false,
          items: [
            { text: '概览', link: '/zh/tutorial/index' },
            { text: '训练精排模型', link: '/zh/tutorial/ranking' },
            { text: '训练多任务模型', link: '/zh/tutorial/multitask' },
            { text: '训练召回模型', link: '/zh/tutorial/retrieval' },
          ]
        },
        {
          text: '命令行工具',
          collapsed: false,
          items: [
            { text: 'NextRec CLI', link: '/zh/cli/nextrec-cli' },
            { text: 'NextRec Studio', link: '/zh/cli/nextrec-studio' }
          ]
        },
        { text: 'FAQ', link: '/zh/faq' }
      ]
    },

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/zerolovesea/NextRec' }
    ],

    // 搜索
    search: {
      provider: 'local'
    },

    // 页脚
    footer: {
      message: '基于 MIT 许可证开源',
      copyright: 'Copyright © 2026 NextRec'
    },

    // 编辑链接
    editLink: {
      pattern: 'https://github.com/zerolovesea/NextRec/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    // 轮廓 (右侧大纲)
    outline: {
      level: [2, 3],
      label: '目录'
    }
  },

  // Markdown 配置
  markdown: {
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  // Vite 配置
  vite: {
    resolve: {
      alias: {
        '@': resolve(__dirname, '../src')
      }
    }
  }
})
