import { useEffect, useRef, useState } from 'react'
import { formatBytes } from '../validation'

interface Props {
  uploadFraction: number
  bytesSent: number
  bytesTotal: number
  uploadDone: boolean
  onCancel: () => void
}

/**
 * Stages of the server-side pipeline, in the order runner.py actually runs them,
 * weighted by measured cost on the reference clip (576x1024, 608 frames):
 * decode 1.3 s, pose inference 7.4 s, render ~2.2 s, transcription and the LLM
 * about a second each. Pose dominates, so it gets the dwell time to match.
 *
 * The weights only decide how long each label sits on screen. They are not
 * progress.
 */
const STAGES: { label: string; weight: number }[] = [
  { label: 'Reading your video', weight: 1.3 },
  { label: 'Tracking your body through the clip', weight: 7.4 },
  { label: 'Smoothing the tracking and measuring joint angles', weight: 0.6 },
  { label: 'Finding your reps', weight: 0.4 },
  { label: 'Checking your form against the cue rules', weight: 0.5 },
  { label: 'Drawing the skeleton overlay', weight: 2.2 },
  { label: 'Writing your coaching report', weight: 1.2 },
]

const TOTAL_WEIGHT = STAGES.reduce((sum, s) => sum + s.weight, 0)

/** Reference clip took ~12.4 s end to end. Longer clips take proportionally
 *  longer and a cold server takes longer still, so this is a floor, not a
 *  promise — which is exactly why the UI says so. */
const NOMINAL_SECONDS = 12.4

export default function Processing({
  uploadFraction, bytesSent, bytesTotal, uploadDone, onCancel,
}: Props) {
  const [stageIndex, setStageIndex] = useState(0)
  const [overrun, setOverrun] = useState(false)
  const startedAt = useRef<number | null>(null)

  // Keep the screen awake. On a long upload a phone that locks can suspend the
  // page and kill the request, which is a far more likely failure during user
  // testing than any timeout. Not supported on iOS Safari before 16.4, hence the
  // feature check and the fallback line in the markup.
  useEffect(() => {
    let sentinel: WakeLockSentinel | null = null
    let released = false

    navigator.wakeLock?.request('screen')
      .then((s) => {
        if (released) { s.release(); return }
        sentinel = s
      })
      .catch(() => { /* denied or unavailable — the hint below covers it */ })

    return () => {
      released = true
      sentinel?.release().catch(() => {})
    }
  }, [])

  // Walk the stage labels once the bytes are gone. This is a timer, not a
  // progress feed — see the note rendered under the list.
  useEffect(() => {
    if (!uploadDone) return
    startedAt.current = Date.now()

    const id = setInterval(() => {
      const elapsed = (Date.now() - (startedAt.current ?? Date.now())) / 1000
      const fraction = elapsed / NOMINAL_SECONDS

      if (fraction >= 1) {
        setStageIndex(STAGES.length - 1)
        setOverrun(true)
        return
      }

      let acc = 0
      for (let i = 0; i < STAGES.length; i++) {
        acc += STAGES[i].weight / TOTAL_WEIGHT
        if (fraction <= acc) { setStageIndex(i); return }
      }
    }, 250)

    return () => clearInterval(id)
  }, [uploadDone])

  if (!uploadDone) {
    const pct = Math.round(uploadFraction * 100)
    return (
      <div className="stack">
        <header>
          <h1>Uploading your clip</h1>
          <p className="muted">Keep this screen open until it finishes.</p>
        </header>

        <div className="card">
          <div className="progress" role="progressbar" aria-valuenow={pct}
               aria-valuemin={0} aria-valuemax={100}>
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <p className="small muted" style={{ marginTop: 10, marginBottom: 0 }}>
            {pct}%
            {bytesTotal > 0 && ` · ${formatBytes(bytesSent)} of ${formatBytes(bytesTotal)}`}
          </p>
        </div>

        {!navigator.wakeLock && (
          <p className="small muted">
            Your phone may lock the screen and interrupt this — tap it now and
            then if it's a big file.
          </p>
        )}

        <button className="btn btn-quiet" onClick={onCancel}>Cancel upload</button>
      </div>
    )
  }

  return (
    <div className="stack">
      <header>
        <h1>Analysing</h1>
        <p className="muted">Your clip is on the server. This usually takes around 15 seconds.</p>
      </header>

      <div className="card">
        <ol className="stage-list">
          {STAGES.map((stage, i) => (
            <li key={stage.label} className={i < stageIndex ? 'done' : i === stageIndex ? 'active' : ''}>
              <span className="stage-mark">{i < stageIndex ? '✓' : i === stageIndex ? '·' : ''}</span>
              <span>{stage.label}</span>
            </li>
          ))}
        </ol>
      </div>

      {overrun && (
        <div className="banner banner-info">
          <p className="small" style={{ margin: 0 }}>
            Still working. Longer clips, and a server that has been idle, both take
            more time than usual.
          </p>
        </div>
      )}

      {/* The honest bit. This distinction goes in the report, so it belongs in
          the interface too rather than only in the write-up. */}
      <p className="small muted">
        These steps are an estimate based on timing, not a live report. The server
        analyses the whole clip in one go and doesn't send progress back as it works.
      </p>

      <button className="btn btn-quiet" onClick={onCancel}>Cancel</button>
    </div>
  )
}
