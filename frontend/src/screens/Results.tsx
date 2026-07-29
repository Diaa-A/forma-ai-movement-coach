import type { AnalyzeResponse } from '../types'
import StatusBanner from '../components/StatusBanner'
import ReportView from '../components/ReportView'
import KeyFrames from '../components/KeyFrames'
import VideoPlayer from '../components/VideoPlayer'

interface Props {
  result: AnalyzeResponse
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
export default function Results({ result, onRestart }: Props) {
  const analysed = result.status === 'ok'
  const worst = result.key_frames.find((f) => f.label === 'worst')

  return (
    <div className="stack">
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

      <button className="btn btn-primary" onClick={onRestart}>
        Analyse another clip
      </button>
    </div>
  )
}
