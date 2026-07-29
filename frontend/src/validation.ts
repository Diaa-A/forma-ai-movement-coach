// Client-side checks, run before anything leaves the phone.
//
// The server enforces all of this too and is the actual authority — this is here
// so the user finds out in a second rather than after a three-minute upload on
// mobile data, and so the message can name the value that was wrong.
//
// Kept in step with backend/api/routes.py by hand. If the allowlists there change,
// change them here.

export const MAX_VIDEO_BYTES = 100 * 1024 * 1024
export const MAX_AUDIO_BYTES = 10 * 1024 * 1024

export const ALLOWED_VIDEO_SUFFIXES = ['.mp4', '.mov', '.webm', '.m4v']
export const ALLOWED_AUDIO_SUFFIXES = ['.wav', '.mp3', '.m4a', '.webm', '.ogg']

// The spec asks for 5-30 s. The server gate is looser (3-45 s) so a slightly
// long clip isn't thrown away, and we warn rather than block inside that margin.
export const IDEAL_MIN_SECONDS = 5
export const IDEAL_MAX_SECONDS = 30
export const HARD_MIN_SECONDS = 3
export const HARD_MAX_SECONDS = 45

/** Roughly where an upload stops being a few seconds and starts being a wait,
 *  on a connection the user is paying for by the megabyte. */
export const CHUNKY_UPLOAD_BYTES = 25 * 1024 * 1024

export interface Check {
  ok: boolean
  /** true when it's worth saying something but not worth refusing */
  warning?: boolean
  message?: string
}

const OK: Check = { ok: true }

export function suffixOf(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot).toLowerCase()
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function formatSeconds(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s - m * 60)}s`
}

export function checkVideoFile(file: File): Check {
  const suffix = suffixOf(file.name)
  if (!ALLOWED_VIDEO_SUFFIXES.includes(suffix)) {
    return {
      ok: false,
      message: suffix
        ? `${suffix} files aren't supported. Try one of ${ALLOWED_VIDEO_SUFFIXES.join(', ')}.`
        : `That file has no extension, so we can't tell what format it is.`,
    }
  }
  if (file.size > MAX_VIDEO_BYTES) {
    return {
      ok: false,
      message: `That clip is ${formatBytes(file.size)} and the limit is ${formatBytes(MAX_VIDEO_BYTES)}. A shorter clip, or a lower recording quality, will get you under it.`,
    }
  }
  if (file.size === 0) {
    return { ok: false, message: `That file is empty — the recording may not have saved properly.` }
  }
  return OK
}

export function checkAudioFile(file: File): Check {
  const suffix = suffixOf(file.name)
  if (!ALLOWED_AUDIO_SUFFIXES.includes(suffix)) {
    return { ok: false, message: `${suffix || 'That'} audio isn't a format we can read.` }
  }
  if (file.size > MAX_AUDIO_BYTES) {
    return { ok: false, message: `That voice note is ${formatBytes(file.size)}; keep it under ${formatBytes(MAX_AUDIO_BYTES)}.` }
  }
  return OK
}

export function checkDuration(seconds: number): Check {
  if (!isFinite(seconds) || seconds <= 0) {
    // Some containers don't report duration until they've buffered. Not the
    // user's problem, and the server checks anyway — let it through.
    return OK
  }
  if (seconds < HARD_MIN_SECONDS) {
    return { ok: false, message: `That clip is ${formatSeconds(seconds)} — too short to find a full rep in. Aim for ${IDEAL_MIN_SECONDS}-${IDEAL_MAX_SECONDS} seconds.` }
  }
  if (seconds > HARD_MAX_SECONDS) {
    return { ok: false, message: `That clip is ${formatSeconds(seconds)} and the limit is ${HARD_MAX_SECONDS} seconds. Trim it to a few good reps.` }
  }
  if (seconds < IDEAL_MIN_SECONDS || seconds > IDEAL_MAX_SECONDS) {
    return {
      ok: true,
      warning: true,
      message: `That's ${formatSeconds(seconds)}. It'll work, but ${IDEAL_MIN_SECONDS}-${IDEAL_MAX_SECONDS} seconds of a few clean reps gives the best read.`,
    }
  }
  return OK
}

/**
 * Read a clip's duration without uploading it.
 *
 * Loads metadata only into a detached <video>. Resolves to NaN rather than
 * rejecting if the browser won't tell us — an unknown duration should not block
 * an upload, the server will make the final call.
 */
export function readDuration(file: File): Promise<number> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const probe = document.createElement('video')
    probe.preload = 'metadata'

    const done = (value: number) => {
      URL.revokeObjectURL(url)
      resolve(value)
    }

    probe.onloadedmetadata = () => done(probe.duration)
    probe.onerror = () => done(NaN)
    // Safari has been known to sit on this forever for a codec it can decode but
    // not probe cheaply. Don't hold the UI hostage over a nice-to-have.
    setTimeout(() => done(NaN), 4000)

    probe.src = url
  })
}
