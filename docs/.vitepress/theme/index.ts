import DefaultTheme from 'vitepress/theme'
import './custom.css'
import HomeFeatures from './components/HomeFeatures.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // 注册全局组件
    app.component('HomeFeatures', HomeFeatures)
  }
}
