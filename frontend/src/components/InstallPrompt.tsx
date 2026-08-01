import { useEffect, useState } from 'react'

const DISMISSED_KEY = 'formcoach.installHintDismissed'

/** Chromium fires this so the page can offer installation at a sensible moment
 *  instead of the browser nagging. It isn't in TS's lib yet. */
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

/**
 * Two routes to a home-screen icon, because the platforms differ.
 *
 * Android/Chrome fires `beforeinstallprompt`, which we hold onto and fire from a
 * real button. iOS has no such event and never will — installing there is
 * Share, then Add to Home Screen — so Safari gets a short written hint instead.
 *
 * Dismissal is remembered in localStorage. That is a UI preference, not user
 * content, so it doesn't touch the no-persistence constraint.
 */
export default function InstallPrompt() {
  const [deferred, setDeferred] = useState<BeforeInstallPromptEvent | null>(null)
  const [dismissed, setDismissed] = useState(
    () => safeGet(DISMISSED_KEY) === '1',
  )

  useEffect(() => {
    const onPrompt = (e: Event) => {
      e.preventDefault()          // stop the browser's own mini-infobar
      setDeferred(e as BeforeInstallPromptEvent)
    }
    window.addEventListener('beforeinstallprompt', onPrompt)
    return () => window.removeEventListener('beforeinstallprompt', onPrompt)
  }, [])

  if (dismissed || isInstalled()) return null

  function hide() {
    safeSet(DISMISSED_KEY, '1')
    setDismissed(true)
  }

  if (deferred) {
    return (
      <div className="card card-tight install-hint">
        <p className="small" style={{ marginBottom: 10 }}>
          Add this to your home screen and it opens like an app — handy when
          you're filming at the gym.
        </p>
        <div className="btn-row">
          <button
            className="btn btn-primary"
            onClick={async () => {
              await deferred.prompt()
              await deferred.userChoice
              setDeferred(null)
              hide()
            }}
          >
            Install
          </button>
          <button className="btn btn-quiet" onClick={hide}>Not now</button>
        </div>
      </div>
    )
  }

  if (isIosSafari()) {
    return (
      <div className="card card-tight install-hint">
        <p className="small" style={{ marginBottom: 8 }}>
          To keep this on your home screen: tap <strong>Share</strong>, then
          <strong> Add to Home Screen</strong>.
        </p>
        <button className="btn btn-quiet" onClick={hide}>Got it</button>
      </div>
    )
  }

  return null
}

/** Already launched from the home screen — display-mode covers Android and
 *  modern iOS; navigator.standalone is the old iOS-only flag, still needed. */
function isInstalled(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia?.('(display-mode: standalone)').matches === true
    || (navigator as unknown as { standalone?: boolean }).standalone === true
}

function isIosSafari(): boolean {
  if (typeof navigator === 'undefined') return false
  const ua = navigator.userAgent
  const ios = /iPad|iPhone|iPod/.test(ua)
    // iPadOS 13+ reports itself as a Mac, and the touch points give it away
    || (ua.includes('Macintosh') && navigator.maxTouchPoints > 1)
  // Chrome and Firefox on iOS are Safari underneath but don't offer Add to Home
  // Screen from the share sheet in the same way, so don't tell them to look
  const realSafari = !/CriOS|FxiOS|EdgiOS|OPiOS/.test(ua)
  return ios && realSafari
}

// localStorage throws in private mode on some browsers, and a dismissed hint is
// not worth a crash
function safeGet(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}

function safeSet(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* nothing worth doing */ }
}
