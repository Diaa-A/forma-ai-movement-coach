import { useEffect, useState } from 'react'

const DISMISSED_KEY = 'formcoach.installHintDismissed'

/** How long to wait for Chrome to fire beforeinstallprompt before assuming it
 *  isn't coming and falling back to written instructions. It normally fires
 *  within a second of load; this is generous. */
const EVENT_GRACE_MS = 3500

/** Chromium fires this so the page can offer installation at a sensible moment
 *  instead of the browser nagging. It isn't in TS's lib yet. */
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

declare global {
  interface Window {
    __installEvent: BeforeInstallPromptEvent | null
  }
}

type Mode = 'waiting' | 'native' | 'ios' | 'in-app' | 'manual' | 'none'

/**
 * Getting the app onto a home screen, which works differently everywhere.
 *
 * Chromium fires `beforeinstallprompt` and we hold it behind a real button.
 * Crucially the event is captured by an inline script in index.html, not here —
 * it fires before a module bundle has mounted, so a listener added in an effect
 * arrives too late and the button silently never appears. That was a real bug,
 * found on a phone.
 *
 * iOS has no such event and never will: installing is Share, then Add to Home
 * Screen. But that menu item only exists in Safari proper — inside the webview
 * that Instagram, WhatsApp and the rest use for links, it isn't there at all, so
 * telling someone to look for it is worse than saying nothing. Those get told to
 * open the page in Safari instead.
 *
 * Anything else falls back to pointing at the browser menu, since every desktop
 * and mobile Chromium can install from there whether or not the event fired.
 */
export default function InstallPrompt() {
  const [event, setEvent] = useState<BeforeInstallPromptEvent | null>(
    () => (typeof window === 'undefined' ? null : window.__installEvent),
  )
  const [waited, setWaited] = useState(false)
  const [dismissed, setDismissed] = useState(() => safeGet(DISMISSED_KEY) === '1')

  useEffect(() => {
    // it may already be sitting on window from the inline script; if not, it
    // may still arrive
    const pick = () => setEvent(window.__installEvent)
    window.addEventListener('installpromptready', pick)
    window.addEventListener('beforeinstallprompt', pick)
    const timer = setTimeout(() => setWaited(true), EVENT_GRACE_MS)
    return () => {
      window.removeEventListener('installpromptready', pick)
      window.removeEventListener('beforeinstallprompt', pick)
      clearTimeout(timer)
    }
  }, [])

  const mode = resolveMode(event, waited)
  if (dismissed || mode === 'none' || mode === 'waiting') return null

  function hide() {
    safeSet(DISMISSED_KEY, '1')
    setDismissed(true)
  }

  return (
    <div className="card card-tight install-hint">
      {mode === 'native' && (
        <>
          <p className="small" style={{ marginBottom: 10 }}>
            Add this to your home screen and it opens like an app — handy when
            you're filming at the gym.
          </p>
          <div className="btn-row">
            <button
              className="btn btn-primary"
              onClick={async () => {
                await event!.prompt()
                await event!.userChoice
                window.__installEvent = null
                setEvent(null)
                hide()
              }}
            >
              Install
            </button>
            <button className="btn btn-quiet" onClick={hide}>Not now</button>
          </div>
        </>
      )}

      {mode === 'ios' && (
        <>
          <p className="small" style={{ marginBottom: 8 }}>
            To keep this on your home screen: tap <strong>Share</strong>, then{' '}
            <strong>Add to Home Screen</strong>.
          </p>
          <button className="btn btn-quiet" onClick={hide}>Got it</button>
        </>
      )}

      {mode === 'in-app' && (
        <>
          <p className="small" style={{ marginBottom: 8 }}>
            You've opened this inside another app's browser, which can't add it to
            your home screen. Tap the <strong>⋯</strong> menu and choose{' '}
            <strong>Open in Safari</strong> first.
          </p>
          <button className="btn btn-quiet" onClick={hide}>Got it</button>
        </>
      )}

      {mode === 'manual' && (
        <>
          <p className="small" style={{ marginBottom: 8 }}>
            You can install this from your browser's menu — look for{' '}
            <strong>Install app</strong> or <strong>Add to Home screen</strong>.
          </p>
          <button className="btn btn-quiet" onClick={hide}>Got it</button>
        </>
      )}
    </div>
  )
}

function resolveMode(event: BeforeInstallPromptEvent | null, waited: boolean): Mode {
  if (isInstalled()) return 'none'
  if (event) return 'native'
  if (isIos()) return isIosSafari() ? 'ios' : 'in-app'
  // no event yet and it might still be coming — say nothing rather than flash a
  // fallback and then replace it with a button
  if (!waited) return 'waiting'
  return supportsInstall() ? 'manual' : 'none'
}

/** Already launched from the home screen — display-mode covers Android and
 *  modern iOS; navigator.standalone is the old iOS-only flag, still needed. */
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

/** Real Safari, not a webview and not another browser wearing Safari's engine.
 *  Only real Safari has Add to Home Screen in its share sheet. */
function isIosSafari(): boolean {
  const ua = navigator.userAgent
  const otherBrowser = /CriOS|FxiOS|EdgiOS|OPiOS|YaBrowser/.test(ua)
  // the usual suspects for link-opening webviews
  const inAppWebview = /FBAN|FBAV|Instagram|Line\/|Twitter|WhatsApp|Snapchat|LinkedInApp|Pinterest/.test(ua)
  return !otherBrowser && !inAppWebview
}

/** Chromium can install from its own menu even when the event didn't reach us. */
function supportsInstall(): boolean {
  if (typeof navigator === 'undefined') return false
  return /Chrome|Chromium|Edg\//.test(navigator.userAgent)
}

// localStorage throws in private mode on some browsers, and a dismissed hint is
// not worth a crash
function safeGet(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}

function safeSet(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* nothing worth doing */ }
}
