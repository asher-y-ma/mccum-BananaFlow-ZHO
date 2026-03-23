import path from 'path';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        // 自定义 Gemini 兼容 API 根地址（中转），例如 https://your-proxy.com/v1beta
        'process.env.GEMINI_API_BASE_URL': JSON.stringify(
          (env.GEMINI_API_BASE_URL || '').replace(/\/+$/, '')
        ),
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
