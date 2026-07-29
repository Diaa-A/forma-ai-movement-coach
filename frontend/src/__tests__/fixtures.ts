import type { AnalyzeResponse, CoachingReport } from '../types'

// Shaped from real /analyze responses. The no_reps and low_detection fixtures
// mirror what not_analyzed_report() actually emits — in particular an EMPTY
// what_went_well, which is the property the UI must not undo.

const okReport: CoachingReport = {
  what_went_well: [
    'you hit full depth on every rep',
    'your descent was controlled throughout',
  ],
  primary_issue:
    'Your torso leaned forward more than your shins through the bottom of each rep.',
  secondary_issues: ['Depth varied noticeably between reps.'],
  corrective_cues: [
    'Drive your chest up as you descend so the torso stays roughly parallel to your shins.',
    'Pick one depth target and hit it every rep.',
  ],
  next_session_focus: 'Keep the chest up on every rep.',
  source: 'llm',
  model: 'llama-3.3-70b-versatile',
  filming_tip:
    'This clip is side-on, which covers squat depth, forward lean, descent tempo and rep consistency. To check left/right symmetry, film a set from the front.',
}

const notAnalysedReport: CoachingReport = {
  what_went_well: [],
  primary_issue:
    "I couldn't detect a complete squat rep in this clip, so there's nothing to score yet.",
  secondary_issues: [],
  corrective_cues: [
    'Film side-on at about hip height, with your whole body in the frame.',
    'Wear fitted clothing and use a plain, uncluttered background.',
  ],
  next_session_focus: 'Re-record with the framing above and upload again.',
  source: 'not_analyzed',
  model: null,
}

export const okResult: AnalyzeResponse = {
  job_id: 'squat_20260728_234148',
  exercise_type: 'squat',
  status: 'ok',
  fps: 30,
  frame_count: 608,
  side: 'left',
  annotated_video_url: '/results/squat_20260728_234148/annotated.mp4',
  angles_url: '/results/squat_20260728_234148/angles.json',
  key_frames: [
    { url: '/results/squat_20260728_234148/worst.jpg', label: 'worst', timestamp: 4.2 },
    { url: '/results/squat_20260728_234148/best.jpg', label: 'best', timestamp: 9.8 },
  ],
  reps: [
    { start: 10, bottom: 40, end: 70 },
    { start: 80, bottom: 110, end: 140 },
  ],
  coaching_report: okReport,
  voice_transcript: 'I want to know if I am going deep enough into the squat movement.',
  warnings: [],
}

export const noRepsResult: AnalyzeResponse = {
  ...okResult,
  job_id: 'squat_noreps',
  status: 'no_reps',
  key_frames: [],
  reps: [],
  coaching_report: notAnalysedReport,
  voice_transcript: null,
  warnings: [],
}

export const lowDetectionResult: AnalyzeResponse = {
  ...noRepsResult,
  job_id: 'squat_lowdet',
  status: 'low_detection',
  coaching_report: {
    ...notAnalysedReport,
    primary_issue:
      'Body tracking was too unreliable on this clip to give trustworthy feedback.',
  },
}

export const okWithWarning: AnalyzeResponse = {
  ...okResult,
  warnings: ['voice transcription failed: no transcription backend available'],
}

export const fallbackReportResult: AnalyzeResponse = {
  ...okResult,
  coaching_report: { ...okReport, source: 'dry_run_fallback', model: null },
}
