// Client-side checks, run before anything leaves the phone.
//
// The server enforces all of this too and is the actual authority — this is here
// so the user finds out in a second rather than after a three-minute upload on
// mobile data, and so the message can name the value that was wrong.
//
// The numbers come from GET /exercises. They used to be written out here by hand
// with a note saying to keep them in step, and they had already fallen out of
// step: the comment claimed the server enforced a 3-45 second gate, and the
// server had no duration check at all. FALLBACK below is what these checks use
// before the catalogue arrives, or if it never does — the app has to stay usable
// offline, and a stale-but-close limit is better than no check.

import type { Limits } from './types'

export const FALLBACK: Limits = {
  max_video_bytes: 100 * 1024 * 1024,
  max_audio_bytes: 10 * 1024 * 1024,
  video_suffixes: ['.mp4', '.mov', '.webm', '.m4v'],
  audio_suffixes: ['.m4a', '.mp3', '.mp4', '.ogg', '.wav', '.webm'],
  min_seconds: 3,
  max_seconds: 45,
  ideal_min_seconds: 5,
  ideal_max_seconds: 30,
}

/** Roughly where an upload stops being a few seconds and starts being a wait,
 *  on a connection the user is paying for by the megabyte. Ours, not the
 *  server's — it is about the user's patience, not about what /analyze accepts. */
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

export function checkVideoFile(file: File, limits: Limits = FALLBACK): Check {
  const suffix = suffixOf(file.name)
  if (!limits.video_suffixes.includes(suffix)) {
    return {
      ok: false,
      message: suffix
        ? `${suffix} files aren't supported. Try one of ${limits.video_suffixes.join(', ')}.`
        : `That file has no extension, so we can't tell what format it is.`,
    }
  }
  if (file.size > limits.max_video_bytes) {
    return {
      ok: false,
      message: `That clip is ${formatBytes(file.size)} and the limit is ${formatBytes(limits.max_video_bytes)}. A shorter clip, or a lower recording quality, will get you under it.`,
    }
  }
  if (file.size === 0) {
    return { ok: false, message: `That file is empty — the recording may not have saved properly.` }
  }
  return OK
}

export function checkAudioFile(file: File, limits: Limits = FALLBACK): Check {
  const suffix = suffixOf(file.name)
  if (!limits.audio_suffixes.includes(suffix)) {
    return { ok: false, message: `${suffix || 'That'} audio isn't a format we can read.` }
  }
  if (file.size > limits.max_audio_bytes) {
    return { ok: false, message: `That voice note is ${formatBytes(file.size)}; keep it under ${formatBytes(limits.max_audio_bytes)}.` }
  }
  return OK
}

export function checkDuration(seconds: number, limits: Limits = FALLBACK): Check {
  const { min_seconds, max_seconds, ideal_min_seconds, ideal_max_seconds } = limits

  if (!isFinite(seconds) || seconds <= 0) {
    // Some containers don't report duration until they've buffered. Not the
    // user's problem, and the server checks too — let it through.
    return OK
  }
  if (seconds < min_seconds) {
    return { ok: false, message: `That clip is ${formatSeconds(seconds)} — too short to find a full rep in. Aim for ${ideal_min_seconds}-${ideal_max_seconds} seconds.` }
  }
  if (seconds > max_seconds) {
    return { ok: false, message: `That clip is ${formatSeconds(seconds)} and the limit is ${max_seconds} seconds. Trim it to a few good reps.` }
  }
  if (seconds < ideal_min_seconds || seconds > ideal_max_seconds) {
    return {
      ok: true,
      warning: true,
      message: `That's ${formatSeconds(seconds)}. It'll work, but ${ideal_min_seconds}-${ideal_max_seconds} seconds of a few clean reps gives the best read.`,
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
