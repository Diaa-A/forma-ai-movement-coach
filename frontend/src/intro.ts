// Whether this device has seen the one-time explainer.
//
// Same shape and the same failure mode as consent.ts, and deliberately a
// separate key: consent is a record of what someone agreed to and is re-asked
// when the retention period changes, while this is only a note that a screen has
// been shown. Sharing one key would mean a change to the retention policy
// re-explained the app, or a copy change re-asked for consent.
//
// localStorage throws in a private window. Swallowing that means the explainer
// shows again, which is the harmless direction.

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
