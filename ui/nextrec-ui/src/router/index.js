import { createRouter, createWebHistory } from 'vue-router';
import TrainPage from '../pages/TrainPage.vue';
import FeaturePage from '../pages/FeaturePage.vue';
import ModelPage from '../pages/ModelPage.vue';
import PredictPage from '../pages/PredictPage.vue';

const routes = [
  { path: '/', redirect: '/train' },
  { path: '/train', component: TrainPage },
  { path: '/feature', component: FeaturePage },
  { path: '/model', component: ModelPage },
  { path: '/predict', component: PredictPage }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;
