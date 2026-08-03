import { useEffect, useState } from 'react'

const DISMISSED_KEY = 'formcoach.installHintDismissed'

/** How long to wait for Chromium to fire beforeinstallprompt before assuming it
 *  is not coming. It usually arrives within a second of load; this is generous. */
const EVENT_GRACE_MS = 2500

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

declare global {
  interface Window {
    __installEvent: BeforeInstallPromptEvent | null
  }
}

type Mode = 'waiting' | 'native' | 'ios' | 'in-app' | 'manual'

/**
 * Getting the app onto the home screen.
 *
 * This is the point of building a PWA rather than a website, so it is deliberately
 * an explicit control at the top of the first screen rather than a hint at the
 * bottom. The previous version rendered below the exercise list, which on a phone
 * put it under the fold, and returned nothing at all when Chromium had not fired
 * its event yet — so the one feature the whole delivery choice rests on was
 * invisible on the device it was built for.
 *
 * Three routes, because the platforms genuinely differ:
 *
 *   Chromium fires `beforeinstallprompt` and we hold it behind a real button.
 *   That event is captured by an inline script in index.html, not here — it fires
 *   before a module bundle has mounted, so a listener added in an effect arrives
 *   too late and the button silently never appears.
 *
 *   iOS has no such event and never will. Installing is Share, then Add to Home
 *   Screen, by hand. No site can trigger it, so the honest thing is to say so
 *   clearly rather than wait for a prompt that is not coming.
 *
 *   Inside an in-app webview (a link opened from a messaging app) Add to Home
 *   Screen is not in the share sheet at all, so telling someone to look for it
 *   wastes their time. Those get told to reopen in a real browser.
 *
 * It never renders nothing when the app is uninstallable-but-uninstalled: every
 * path ends in something the user can act on.
 */
export default function InstallPrompt() {
  const [event, setEvent] = useState<BeforeInstallPromptEvent | null>(
    () => (typeof window === 'undefined' ? null : window.__installEvent),
  )
  const [waited, setWaited] = useState(false)
  const [dismissed, setDismissed] = useState(() => safeGet(DISMISSED_KEY) === '1')
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const pick = () => setEvent(window.__installEvent)
    window.addEventListener('installpromptready', pick)
    window.addEventListener('beforeinstallprompt', pick)
    const installed = () => setDismissed(true)
    window.addEventListener('appinstalled', installed)
    const timer = setTimeout(() => setWaited(true), EVENT_GRACE_MS)
    return () => {
      window.removeEventListener('installpromptready', pick)
      window.removeEventListener('beforeinstallprompt', pick)
      window.removeEventListener('appinstalled', installed)
      clearTimeout(timer)
    }
  }, [])

  // already running from the home screen, or explicitly dismissed
  if (isInstalled() || dismissed) return null

  const mode = resolveMode(event, waited)
  if (mode === 'waiting') return null

  function hide() {
    safeSet(DISMISSED_KEY, '1')
    setDismissed(true)
  }

  async function installNow() {
    if (!event) return
    await event.prompt()
    await event.userChoice
    window.__installEvent = null
    setEvent(null)
    hide()
  }

  return (
    <div className="install-bar">
      <div className="install-bar-row">
        <span className="install-bar-text">
          <strong>Add to your home screen</strong>
          <span className="muted small"> — opens like an app, handy at the gym</span>
        </span>
        {mode === 'native' ? (
          <button className="btn btn-primary install-bar-btn" onClick={installNow}>
            Install
          </button>
        ) : (
          <button className="btn install-bar-btn" onClick={() => setOpen(!open)}>
            {open ? 'Hide' : 'How'}
          </button>
        )}
      </div>

      {open && mode !== 'native' && (
        <div className="install-steps small">
          {mode === 'ios' && (
            <>
              <p style={{ marginBottom: 6 }}>
                On iPhone and iPad this is always manual — Safari has no install
                button, so no app can pop one up for you.
              </p>
              <ol style={{ margin: 0, paddingLeft: 18 }}>
                <li>Tap the <strong>Share</strong> icon (the square with an arrow
                  pointing up, in the toolbar).</li>
                <li>Scroll down and tap <strong>Add to Home Screen</strong>.</li>
                <li>Tap <strong>Add</strong>. The icon appears on your home screen.</li>
              </ol>
            </>
          )}

          {mode === 'in-app' && (
            <p style={{ margin: 0 }}>
              You have opened this inside another app's browser, which cannot add
              anything to your home screen. Tap the <strong>⋯</strong> or{' '}
              <strong>Share</strong> menu and choose{' '}
              <strong>Open in Safari</strong> (or <strong>Open in Chrome</strong>),
              then try again from there.
            </p>
          )}

          {mode === 'manual' && (
            <>
              <p style={{ marginBottom: 6 }}>
                Your browser has not offered an install button yet. You can still
                do it from the menu:
              </p>
              <ol style={{ margin: 0, paddingLeft: 18 }}>
                <li>Open the browser menu (<strong>⋮</strong>).</li>
                <li>Tap <strong>Install app</strong>, or{' '}
                  <strong>Add to Home screen</strong>.</li>
              </ol>
            </>
          )}

          <button className="btn btn-quiet" style={{ marginTop: 10 }} onClick={hide}>
            Don't show this again
          </button>
        </div>
      )}
    </div>
  )
}

function resolveMode(event: BeforeInstallPromptEvent | null, waited: boolean): Mode {
  if (event) return 'native'
  if (isIos()) return isIosSafari() ? 'ios' : 'in-app'
  if (!waited) return 'waiting'   // the event may still be coming; do not flicker
  return isInAppWebview() ? 'in-app' : 'manual'
}

/** Already launched from the home screen. display-mode covers Android and modern
 *  iOS; navigator.standalone is the older iOS-only flag and is still needed. */
function isInstalled(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia?.('(display-mode: standalone)').matches === true
    || (navigator as unknown as { standalone?: boolean }).standalone === true
}

function isIos(): boolean {
  if (typeof navigator === 'undefined') return false
  const ua = navigator.userAgent
  return /iPad|iPhone|iPod/.test(ua)
    // iPadOS 13+ reports itself as a Mac; the touch points give it away
    || (ua.includes('Macintosh') && navigator.maxTouchPoints > 1)
}

function isInAppWebview(): boolean {
  return /FBAN|FBAV|Instagram|Line\/|Twitter|WhatsApp|Snapchat|LinkedInApp|Pinterest|WebView/
    .test(navigator.userAgent)
}

/** Real Safari, not a webview and not another browser using Safari's engine.
 *  Only real Safari has Add to Home Screen in its share sheet. */
function isIosSafari(): boolean {
  const ua = navigator.userAgent
  const otherBrowser = /CriOS|FxiOS|EdgiOS|OPiOS|YaBrowser/.test(ua)
  return !otherBrowser && !isInAppWebview()
}

// localStorage throws in private mode on some browsers, and a dismissed hint is
// not worth a crash
function safeGet(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}

function safeSet(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* nothing worth doing */ }
}
