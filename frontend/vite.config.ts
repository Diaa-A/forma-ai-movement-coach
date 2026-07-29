/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In production FastAPI serves the built files, so the app and the API sit on
// one origin and there's no CORS at all. In dev they're on different ports, so
// proxy the API routes across and keep the same relative fetch paths in both —
// that way api.ts never needs to know which mode it's running in.
export default defineConfig({
  plugins: [react()],
  server: {
    // 0.0.0.0 so a phone on the same wifi can reach the dev server. Note that
    // getUserMedia won't work over a LAN ip (not a secure context) — see the
    // plan, doc/WP02_PWA_PLAN.md 7.1. Everything except voice *recording* works.
    host: true,
    port: 5173,
    proxy: {
      '/analyze': 'http://127.0.0.1:8000',
      '/exercises': 'http://127.0.0.1:8000',
      '/results': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
    // the backend mounts this directory; keep the hashed filenames so the
    // service worker cache versioning has something to key off
    assetsDir: 'assets',
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/__tests__/setup.ts',
  },
})
