import fs from 'fs';
import path from 'path';
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';



function readRootEnvValue(key) {
  const rootDir = path.resolve(__dirname, '..');
  const candidates = [path.join(rootDir, '.env'), path.join(rootDir, '.ENV')];
  for (const filePath of candidates) {
    if (!fs.existsSync(filePath)) {
      continue;
    }
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) {
        continue;
      }
      const match = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
      if (!match) {
        continue;
      }
      const name = match[1];
      let value = match[2] ?? '';
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }
      if (name === key) {
        return value.trim();
      }
    }
  }
  return '';
}

const appEnv = readRootEnvValue('ENV');

export default defineConfig({
  plugins: [vue()],
  define: {
    __APP_ENV__: JSON.stringify(appEnv)
  },
  server: {
    host: '0.0.0.0',
    port: 5173
  }
});

