import type { ApiError } from '../types'

interface Props {
  error: ApiError
  onRetry: () => void
  onRestart: () => void
}

/**
 * Terminal failures. Each kind gets its own heading and its own advice, because
 * "something went wrong, try again" is useless when the real answer is "your
 * clip is 90 seconds long".
 *
 * For a 4xx the message is the server's own — those are written to be read by a
 * person. For a 5xx it is ours, because whatever the server said is an internal
 * detail and shouldn't reach the user.
 */
export default function ErrorPanel({ error, onRetry, onRestart }: Props) {
  const { heading, advice } = copyFor(error)

  return (
    <div className="stack">
      <div className="banner banner-bad">
        <h2>{heading}</h2>
        <p className="small" style={{ marginBottom: 0 }}>{error.message}</p>
      </div>

      {advice && <p className="muted small">{advice}</p>}

      <button className="btn btn-primary" onClick={onRetry}>Try again</button>
      <button className="btn btn-quiet" onClick={onRestart}>Start over</button>
    </div>
  )
}

function copyFor(error: ApiError): { heading: string; advice?: string } {
  switch (error.kind) {
    case 'network':
      return {
        heading: 'Lost the connection',
        advice: 'Nothing was analysed, so nothing was charged against your data twice — the whole upload will start again.',
      }
    case 'timeout':
      return {
        heading: 'That took too long',
        advice: 'A shorter clip is the reliable fix. Five to thirty seconds of a few clean reps is plenty.',
      }
    case 'rejected':
      return {
        heading: "The server wouldn't accept that",
        advice: 'Pick a different clip and try again.',
      }
    case 'server':
      return {
        heading: 'Something broke on our side',
        advice: 'This one is not your fault. Trying again often works — the failure has been logged either way.',
      }
    case 'aborted':
      return { heading: 'Cancelled' }
  }
}
