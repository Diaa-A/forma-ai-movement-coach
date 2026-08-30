import { useEffect, useRef, useState } from 'react'
import type { AnalyzeResponse } from '../types'
import { deleteJob } from '../api'
import { staggerIn } from '../motion'
import StatusBanner from '../components/StatusBanner'
import ReportView from '../components/ReportView'
import KeyFrames from '../components/KeyFrames'
import VideoPlayer from '../components/VideoPlayer'

interface Props {
  result: AnalyzeResponse
  /** set only while a testing round is running — see the card below */
  feedbackFormUrl: string | null
  onRestart: () => void
}

/**
 * Order on this screen is deliberate: status first, then what we saw, then what
 * it means.
 *
 * On a non-ok run the rep count and key frames are suppressed entirely rather
 * than shown as zeroes. A "0 reps" line next to a coaching report is ambiguous;
 * an explicit banner and no numbers is not.
 */
export default function Results({ result, feedbackFormUrl, onRestart }: Props) {
  const analysed = result.status === 'ok'
  const worst = result.key_frames.find((f) => f.label === 'worst')

  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [deleted, setDeleted] = useState(false)
  const [deleteFailed, setDeleteFailed] = useState(false)

  // This is the longest screen in the app and it arrives all at once after a
  // fourteen-second wait. Fading the sections in gives the eye an order to read
  // them in. The status banner is excluded — see staggerIn and the data
  // attribute on it — because it is the one thing here that must not wait.
  const sections = useRef<HTMLDivElement>(null)
  useEffect(() => { staggerIn(sections.current) }, [])

  async function wipe() {
    setDeleting(true)
    setDeleteFailed(false)
    try {
      await deleteJob(result.job_id)
      setDeleted(true)
    } catch {
      setDeleteFailed(true)
    } finally {
      setDeleting(false)
    }
  }

  // Once the server has forgotten the job, the artefact URLs on this screen are
  // dead, so keeping the screen up would be showing a report we just promised
  // was gone. Replace it with the confirmation instead.
  if (deleted) {
    return (
      <div className="stack">
        <div className="banner banner-info">
          <h2>Deleted</h2>
          <p className="small" style={{ margin: 0 }}>
            Your clip, the annotated video and the report have been removed from
            the server. Nothing about this analysis is kept.
          </p>
        </div>
        <button className="btn btn-primary" onClick={onRestart}>
          Back to the start
        </button>
      </div>
    )
  }

  return (
    <div className="stack" ref={sections}>
      <StatusBanner status={result.status} warnings={result.warnings} />

      <header>
        <h1>{analysed ? 'Your form' : 'What we could see'}</h1>
        {analysed && (
          <p className="muted">
            {result.reps.length} {result.reps.length === 1 ? 'rep' : 'reps'} ·
            analysed from your {result.side} side
          </p>
        )}
      </header>

      <VideoPlayer src={result.annotated_video_url} poster={worst?.url} />

      {analysed && <KeyFrames frames={result.key_frames} />}

      {result.voice_transcript && (
        <section className="card card-tight">
          <h3>What you asked</h3>
          <p className="small" style={{ margin: 0, fontStyle: 'italic' }}>
            “{result.voice_transcript}”
          </p>
        </section>
      )}

      {result.coaching_report && <ReportView report={result.coaching_report} />}

      {feedbackFormUrl && (
        <section className="card card-tight">
          <h3>Helping test this app?</h3>
          <p className="small">
            Please fill in the short feedback form now, while the report is
            still fresh. The link carries this analysis's reference number and
            nothing about you.
          </p>
          <a
            className="btn"
            href={withJobId(feedbackFormUrl, result.job_id)}
            target="_blank"
            rel="noreferrer"
          >
            Open the feedback form
          </a>
        </section>
      )}

      <button className="btn btn-primary" onClick={onRestart}>
        Analyse another clip
      </button>

      {!confirmingDelete && (
        <button className="btn btn-quiet" onClick={() => setConfirmingDelete(true)}>
          Delete this analysis from the server
        </button>
      )}
      {confirmingDelete && (
        <div className="banner banner-warn">
          <p className="small">
            This removes your clip, the annotated video and the report from the
            server permanently. It can't be undone.
          </p>
          <div className="btn-row">
            <button className="btn btn-bad" disabled={deleting} onClick={wipe}>
              {deleting ? 'Deleting…' : 'Yes, delete it'}
            </button>
            <button
              className="btn"
              disabled={deleting}
              onClick={() => setConfirmingDelete(false)}
            >
              Keep it
            </button>
          </div>
          {deleteFailed && (
            <p className="small" style={{ marginTop: 10, marginBottom: 0 }}>
              That didn't go through — check your connection and try again. The
              retention period still applies either way.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

/** Fill the {job_id} placeholder in the served form URL. Google's pre-filled
 *  link generator percent-encodes whatever you type into the field, so the
 *  placeholder can arrive spelled either way. */
function withJobId(url: string, jobId: string): string {
  const id = encodeURIComponent(jobId)
  return url.replace('{job_id}', id).replace('%7Bjob_id%7D', id)
}
