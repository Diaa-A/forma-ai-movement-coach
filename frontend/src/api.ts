// The only module that talks to the backend.
//
// Kept deliberately small and isolated: it is still open whether /analyze stays a
// single synchronous POST or becomes submit-and-poll once we know what request
// duration the host actually allows. If that changes, it changes here and
// nowhere else.
//
// Paths are relative on purpose. In dev Vite proxies them to :8000; in production
// FastAPI serves the built app from the same origin. Neither case needs a base url,
// and not having one means there's no way to accidentally ship a hardcoded
// localhost into a deployed build.

import type { AnalyzeResponse, ApiError, Catalog } from './types'

/** Anything past this and something is wrong — the reference clip takes ~12 s and
 *  a 50 MB upload on a bad connection is still bounded by the size cap. Generous
 *  rather than tight, because a false timeout on a slow train is worse than
 *  waiting. */
const REQUEST_TIMEOUT_MS = 6 * 60 * 1000

/**
 * Everything the app needs before the user picks anything: what can be analysed,
 * how to film each one, and what /analyze will accept.
 *
 * The limits come back on the same call rather than from a second endpoint —
 * this is already the "what can this thing do" request, it is already made once
 * on load, and the service worker already refuses to cache it.
 */
export async function fetchCatalog(): Promise<Catalog> {
  const res = await fetch('/exercises')
  if (!res.ok) throw new Error(`GET /exercises returned ${res.status}`)
  return await res.json()
}

/**
 * Remove one analysis from the server now, without waiting for the retention
 * sweep. The consent copy points at this — "you can also delete an analysis
 * immediately" has to be a button, not a sentence.
 */
export async function deleteJob(jobId: string): Promise<void> {
  const res = await fetch(`/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`DELETE /jobs returned ${res.status}`)
}

export interface AnalyzeArgs {
  video: File
  exerciseType: string
  voiceNote?: File | null
  voiceNoteText?: string
  dryRunCoach?: boolean
  /** 0..1, real bytes-sent. Only meaningful during the upload half. */
  onUploadProgress?: (fraction: number, bytesSent: number, bytesTotal: number) => void
  /** called once the last byte is gone, i.e. when the honest progress ends and
   *  the estimated part begins */
  onUploadComplete?: () => void
}

export interface AnalyzeHandle {
  result: Promise<AnalyzeResponse>
  abort: () => void
}

/**
 * POST the clip and wait for the finished analysis.
 *
 * Uses XMLHttpRequest rather than fetch() for one reason: fetch has no upload
 * progress event. The upload is the long part on mobile data — a 50 MB clip at a
 * realistic uplink is minutes — and a real percentage there is worth more than
 * anything we can show for the server-side stage, which reports nothing.
 */
export function analyze(args: AnalyzeArgs): AnalyzeHandle {
  const xhr = new XMLHttpRequest()

  const result = new Promise<AnalyzeResponse>((resolve, reject) => {
    const form = new FormData()
    form.append('video', args.video, args.video.name)
    form.append('exercise_type', args.exerciseType)
    if (args.voiceNote) form.append('voice_note', args.voiceNote, args.voiceNote.name)
    if (args.voiceNoteText) form.append('voice_note_text', args.voiceNoteText)
    if (args.dryRunCoach) form.append('dry_run_coach', 'true')

    xhr.upload.addEventListener('progress', (e) => {
      if (!e.lengthComputable) return
      args.onUploadProgress?.(e.loaded / e.total, e.loaded, e.total)
      if (e.loaded >= e.total) args.onUploadComplete?.()
    })
    // Safari has been known not to fire a final progress event at exactly 100%,
    // so treat 'load' on the upload object as the backstop for "bytes are gone".
    xhr.upload.addEventListener('load', () => args.onUploadComplete?.())

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch {
          reject(err('server', 'The server sent back something we could not read.'))
        }
        return
      }
      reject(httpError(xhr))
    })

    xhr.addEventListener('error', () =>
      reject(err('network', 'Could not reach the server. Check your connection and try again.')))
    xhr.addEventListener('timeout', () =>
      reject(err('timeout', 'That took longer than expected and timed out. A shorter clip usually helps.')))
    xhr.addEventListener('abort', () =>
      reject(err('aborted', 'Cancelled.')))

    xhr.open('POST', '/analyze')
    xhr.timeout = REQUEST_TIMEOUT_MS
    xhr.send(form)
  })

  return { result, abort: () => xhr.abort() }
}

function httpError(xhr: XMLHttpRequest): ApiError {
  // 4xx messages from this API are written to be read by a person — the route
  // handlers say things like "unsupported video format '.avi'". Pass them
  // through. 5xx is the opposite: whatever it says is an internal detail, so we
  // substitute our own wording and let the server log carry the specifics.
  if (xhr.status >= 400 && xhr.status < 500) {
    return { kind: 'rejected', status: xhr.status, message: detailOf(xhr) }
  }
  // The reference travels in a header rather than the body, which is what lets
  // the rule above stay absolute: we still read nothing a 5xx says about itself,
  // and the user still gets something they can quote.
  return {
    kind: 'server',
    status: xhr.status,
    message: 'Something went wrong on our side while analysing that clip. Trying again often works.',
    reference: xhr.getResponseHeader('X-Error-Reference') ?? undefined,
  }
}

function detailOf(xhr: XMLHttpRequest): string {
  try {
    const body = JSON.parse(xhr.responseText)
    if (typeof body?.detail === 'string') return body.detail
    // FastAPI validation errors come back as a list of objects, which is no use
    // to anybody reading it on a phone
    if (Array.isArray(body?.detail)) return 'That request was missing something the server needed.'
  } catch {
    /* fall through to the generic line below */
  }
  return 'The server would not accept that upload.'
}

function err(kind: ApiError['kind'], message: string): ApiError {
  return { kind, message }
}
