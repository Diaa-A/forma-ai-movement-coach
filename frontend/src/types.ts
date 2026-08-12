// Mirrors backend/api/schemas.py. If you change one, change the other — there's
// no codegen step here, and the field that will bite you is an optional one that
// the UI quietly renders as empty.

export type AnalysisStatus = 'ok' | 'no_reps' | 'low_detection'

/** Where the coaching prose came from. Surfaced in the UI, not just logged —
 *  the user should know when they're reading the system's own wording rather
 *  than the language model's. */
export type ReportSource = 'llm' | 'dry_run' | 'dry_run_fallback' | 'not_analyzed'

export interface Exercise {
  id: string
  name: string
  view_label: string
  filming_guide: string
  /** anatomical plane -> what filming from that view lets the system assess */
  assesses: Record<string, string[]>
}

/** What /analyze will accept, served by the API rather than kept here.
 *
 *  These numbers used to live in validation.ts as a hand-maintained second copy,
 *  with a comment saying so — and it had already drifted: it documented a 3-45
 *  second server gate that the server did not actually have. Same reasoning as
 *  filming_guide: the copy the user is checked against has to be the copy the
 *  server enforces. */
export interface Limits {
  max_video_bytes: number
  max_audio_bytes: number
  video_suffixes: string[]
  audio_suffixes: string[]
  /** the band the server refuses outside of */
  min_seconds: number
  max_seconds: number
  /** the band that gives a good read — worth a warning, not a refusal */
  ideal_min_seconds: number
  ideal_max_seconds: number
  /** how long a clip and its results are kept, and the sentence to show for it.
   *  Served rather than written here for the same reason as everything above,
   *  and more so: this one is a promise, and a copy of it in the frontend is a
   *  promise that can drift away from the sweep that keeps it. */
  retention_hours: number
  retention_note: string
}

export interface Catalog {
  exercises: Exercise[]
  limits: Limits
}

export interface KeyFrame {
  url: string
  label: string
  timestamp: number
  /** whether joints were actually marked as at fault on this frame — decides
   *  whether the caption promises the user something to look at */
  highlighted?: boolean
}

export interface RepStat {
  start: number
  bottom: number
  end: number
  knee_min?: number | null
  knee_max?: number | null
  score?: number | null
  breakdown?: Record<string, number> | null
}

export interface CoachingReport {
  what_went_well: string[]
  primary_issue: string
  secondary_issues: string[]
  corrective_cues: string[]
  next_session_focus: string
  source: ReportSource
  model?: string | null
  /** deterministic camera-view guidance — never LLM-written (Decision 23), so
   *  it gets its own treatment in the UI rather than being mixed into the prose */
  filming_tip?: string | null
}

export interface AnalyzeResponse {
  job_id: string
  exercise_type: string
  status: AnalysisStatus
  fps: number
  frame_count: number
  side: string
  annotated_video_url: string
  angles_url: string
  key_frames: KeyFrame[]
  reps: RepStat[]
  coaching_report?: CoachingReport | null
  voice_transcript?: string | null
  warnings: string[]
}

/** What went wrong, in a shape the UI can render without guessing.
 *  `kind` decides the copy; `message` is either the server's own wording (which
 *  is written for users) or ours. */
export interface ApiError {
  kind: 'network' | 'timeout' | 'rejected' | 'server' | 'aborted'
  message: string
  status?: number
  /** short code the server logged the traceback under, on a 5xx. Shown so a
   *  tester can quote it instead of describing what they were doing. */
  reference?: string
}
