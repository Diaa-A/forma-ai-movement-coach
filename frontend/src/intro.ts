// Whether this device has seen the one-time explainer.
//
// Same shape and failure mode as consent.ts, on a deliberately separate key.
// Consent records what someone agreed to and is re-asked when the retention
// period changes; this is only a note that a screen has been shown. One shared
// key would mean a retention change re-explained the app, or a copy change
// re-asked for consent.
//
// localStorage throws in a private window; swallowing that shows the explainer
// again, which is the harmless direction.

const KEY = 'intro-v1'

export function introSeen(): boolean {
  try {
    return localStorage.getItem(KEY) === '1'
  } catch {
    return false
  }
}

export function markIntroSeen(): void {
  try {
    localStorage.setItem(KEY, '1')
  } catch {
    /* it will show once more; not worth a fallback */
  }
}
