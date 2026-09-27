/* Service worker. Hand-written — Workbox would be a build plugin and a generated
   file, and the policy here is short enough to read in one sitting, which is
   worth more than the lines it would save.

   The rule that isn't negotiable: /results/* is NEVER cached. Those are videos
   and stills of the user's body. The app promises deletion and no retention, and
   writing them into a device cache outside that promise would contradict the
   consent people give before uploading. Storage is cheap; the guarantee
   isn't. This worker simply doesn't intercept those requests.

   Everything else:
     navigation      network first, cached shell as the fallback. New builds land
                     straight away when online, and the app still boots offline.
     hashed assets   cache first. The filename changes when the content does, so
                     a cached one is never stale.
     icons/manifest  cache first, stale is fine.
     fonts           cache first, and precached, because a display face that
                     arrives late is a layout shift and one that never arrives
                     offline is a shift on every cold start. The filename carries
                     a content hash for the same reason the hashed assets do —
                     public/ is copied verbatim, so Vite doesn't hash it and the
                     build id below wouldn't change for a font-only edit.
     /exercises      not cached. Filming guidance that's quietly out of date is
                     worse than a spinner, and the app already carries its own
                     fallback for when this call fails.
     anything POST   not intercepted. A silently replayed upload of somebody's
                     body video is not a feature.
*/

// Stamped at build time from the emitted asset filenames — see the
// stamp-service-worker plugin in vite.config.ts.
//
// This was the literal string 'v1' and never changed, which is worse than it
// looks. A browser only re-installs a worker when the FILE'S BYTES differ, so a
// worker whose contents are identical between builds is never reinstalled:
// install and activate never re-run, the precached shell from whenever the user
// installed the app stays forever, and the activate cleanup can never delete
// anything because KEEP is constant. Anyone who added the app to their home
// screen would keep that build until they deleted it.
//
// That is survivable while the app is only on the developer's phone. It is not
// survivable during a testing round, where the entire point is shipping fixes
// between Round 1 and Round 2 to people who already installed it.
const VERSION = '__BUILD_ID__'
// Still 'formcoach-' after the rename to Forma, on purpose. These names are
// invisible to users, and the activate handler deletes by this exact prefix — a
// rename would leave the caches on every phone that installed the app during a
// testing round orphaned, with nothing left that matches to clean them up.
const SHELL_CACHE = `formcoach-shell-${VERSION}`
const ASSET_CACHE = `formcoach-assets-${VERSION}`
const KEEP = [SHELL_CACHE, ASSET_CACHE]

// Stable paths only. The JS and CSS filenames are hashed at build time so this
// file can't know them — they get picked up at runtime on first request instead.
const SHELL = [
  '/',
  '/manifest.webmanifest',
  '/icons/icon-192.png',
  '/icons/apple-touch-icon.png',
  // precached rather than left to first request: the heading font is what the
  // no-layout-shift work in styles.css depends on, and offline it would never
  // arrive at all
  '/fonts/space-grotesk-var-336df680.woff2',
]

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      // addAll is all-or-nothing; one 404 would leave us with no shell at all,
      // so add them individually and let the stragglers get picked up at runtime
      .then((cache) => Promise.allSettled(SHELL.map((url) => cache.add(url))))
      .then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(
        names.filter((n) => n.startsWith('formcoach-') && !KEEP.includes(n))
             .map((n) => caches.delete(n)),
      ))
      .then(() => self.clients.claim()),
  )
})

self.addEventListener('fetch', (event) => {
  const { request } = event
  if (request.method !== 'GET') return

  const url = new URL(request.url)
  if (url.origin !== self.location.origin) return

  // the user's own footage — leave it entirely alone, see the note at the top
  if (url.pathname.startsWith('/results/')) return

  // live data, deliberately never served stale
  if (url.pathname === '/exercises' || url.pathname === '/health') return

  if (request.mode === 'navigate') {
    event.respondWith(networkFirstDocument(request))
    return
  }

  if (url.pathname.startsWith('/assets/') || url.pathname.startsWith('/icons/')
      || url.pathname.startsWith('/fonts/')
      || url.pathname === '/manifest.webmanifest') {
    event.respondWith(cacheFirst(request, ASSET_CACHE))
  }
})

async function networkFirstDocument(request) {
  try {
    const fresh = await fetch(request)
    const cache = await caches.open(SHELL_CACHE)
    cache.put('/', fresh.clone())
    return fresh
  } catch {
    // offline: hand back whatever shell we have. The app detects the connection
    // itself and explains the situation — it does not pretend to work.
    const cached = await caches.match('/', { cacheName: SHELL_CACHE })
    return cached || Response.error()
  }
}

async function cacheFirst(request, cacheName) {
  const cached = await caches.match(request)
  if (cached) return cached
  const response = await fetch(request)
  if (response.ok) {
    const cache = await caches.open(cacheName)
    cache.put(request, response.clone())
  }
  return response
}
