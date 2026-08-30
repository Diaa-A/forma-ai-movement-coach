// Whether this person has agreed to the upload-and-processing terms, remembered
// per device so the dialogue is shown once rather than before every clip.
//
// The record is keyed to the retention period it was shown with. Every line of
// the consent copy is a promise the code keeps, and the retention period is the
// one most likely to change — if it does, agreement to the old number is not
// agreement to the new one, so the dialogue comes back.
//
// localStorage can throw (private browsing, storage cleared mid-session). Both
// functions swallow that: the cost is being asked again next time, which is the
// safe direction to fail in.

const KEY = 'consent-v1'

export function consentGiven(retentionHours: number): boolean {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return false
    return JSON.parse(raw).retention_hours === retentionHours
  } catch {
    return false
  }
}

export function recordConsent(retentionHours: number): void {
  try {
    localStorage.setItem(KEY, JSON.stringify({
      retention_hours: retentionHours,
      at: new Date().toISOString(),
    }))
  } catch {
    /* ask again next time */
  }
}
