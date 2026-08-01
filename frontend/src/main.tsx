import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

const root = document.getElementById('root')
if (!root) throw new Error('#root missing from index.html')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Register after load so the worker never competes with the first render for
// bandwidth. Service workers need a secure context, so this is a no-op over a
// plain LAN address during development — which is fine, the app doesn't depend
// on it for anything except installing and the offline shell.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch((err) => {
      // not fatal in any way — log it rather than bothering the user about a
      // feature they didn't ask for
      console.warn('service worker registration failed:', err)
    })
  })
}
