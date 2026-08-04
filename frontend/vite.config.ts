/// <reference types="vitest/config" />
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type { Plugin } from 'vite'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Give the service worker a version that actually changes between builds.
 *
 * public/sw.js ships a __BUILD_ID__ placeholder. Vite copies public/ verbatim,
 * so this rewrites the emitted copy with a hash of the built asset filenames.
 * Those filenames are content hashes already, so the id changes when and only
 * when the build output changes — which is exactly when a worker should
 * reinstall, and never on a rebuild that produced identical output.
 *
 * Without this the worker's bytes were identical every build, and a browser only
 * reinstalls a worker whose file has changed. Installed users kept whatever
 * shell they first got.
 */
function stampServiceWorker(): Plugin {
  return {
    name: 'stamp-service-worker',
    apply: 'build',
    closeBundle() {
      const dist = resolve(__dirname, 'dist')
      const swPath = resolve(dist, 'sw.js')
      let sw: string
      try {
        sw = readFileSync(swPath, 'utf8')
      } catch {
        return                       // no worker in this build, nothing to do
      }
      const manifest = readFileSync(resolve(dist, 'index.html'), 'utf8')
      const assets = [...manifest.matchAll(/assets\/[A-Za-z0-9_.-]+/g)]
        .map((m) => m[0])
        .sort()
        .join('|')
      const id = createHash('sha256').update(assets).digest('hex').slice(0, 12)
      writeFileSync(swPath, sw.replace('__BUILD_ID__', id))
    },
  }
}

// In production FastAPI serves the built files, so the app and the API sit on
// one origin and there's no CORS at all. In dev they're on different ports, so
// proxy the API routes across and keep the same relative fetch paths in both —
// that way api.ts never needs to know which mode it's running in.
export default defineConfig({
  plugins: [react(), stampServiceWorker()],
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
