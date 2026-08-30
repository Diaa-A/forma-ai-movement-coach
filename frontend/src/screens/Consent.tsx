import type { Limits } from '../types'

interface Props {
  limits: Limits
  onAccept: () => void
  onDecline: () => void
}

/**
 * Shown once, after the exercise is chosen and before the capture screen — the
 * last point at which nothing has been recorded or uploaded. Declining goes back
 * to the picker; accepting is remembered on this device (see consent.ts).
 *
 * Every line below is a promise the code keeps, which is why the retention
 * sentence is the served one rather than a copy: a period typed here could
 * drift away from the TTL the sweep actually enforces. It also means this
 * screen cannot honestly render before the catalogue has arrived — without the
 * served sentence there is no number to agree to, so the accept button waits.
 */
export default function Consent({ limits, onAccept, onDecline }: Props) {
  const haveRetention = Boolean(limits.retention_note)

  return (
    <div className="stack">
      <header>
        <h1>Before you record</h1>
        <p className="muted">What happens to your clip, in plain terms.</p>
      </header>

      <div className="card">
        <ul className="consent-list">
          <li>
            Your clip and voice note are uploaded to our server and analysed
            there — nothing is processed on your phone.
          </li>
          <li>
            Transcribing your voice note and wording the coaching use a
            third-party service (Groq). The video itself never leaves our server.
          </li>
          <li>
            {haveRetention
              ? limits.retention_note
              : 'Clips and results are deleted automatically after a fixed period.'}
          </li>
          <li>
            You can also delete an analysis immediately, from its results screen.
          </li>
          <li>
            Using this app is voluntary and you can stop at any point — nothing
            is uploaded until you press analyse.
          </li>
          <li>
            The feedback is exercise-form coaching, not medical advice.
          </li>
        </ul>
      </div>

      {!haveRetention && (
        <div className="banner banner-warn">
          <p className="small" style={{ margin: 0 }}>
            We couldn't load the data-handling details just now, and won't ask
            you to agree to a retention period we can't state. Check your
            connection and try again.
          </p>
        </div>
      )}

      <button
        className="btn btn-primary"
        disabled={!haveRetention}
        onClick={onAccept}
      >
        I agree — continue
      </button>
      <button className="btn btn-quiet" onClick={onDecline}>
        Not now
      </button>
    </div>
  )
}
