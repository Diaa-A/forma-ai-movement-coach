import type { Exercise } from '../types'

interface Props {
  exercises: Exercise[]
  loadFailed: boolean
  onPick: (exercise: Exercise) => void
}

/**
 * First screen. The list comes from GET /exercises rather than being written out
 * here, so when push-up is registered on the backend it turns up in this picker
 * with no frontend change at all.
 */
export default function ExerciseSelect({ exercises, loadFailed, onPick }: Props) {
  return (
    <div className="stack">
      <header>
        <h1>What did you film?</h1>
        <p className="muted">
          Pick the exercise, then we'll show you how to frame the shot before you
          upload anything.
        </p>
      </header>

      {loadFailed && (
        <div className="banner banner-warn">
          <p className="small" style={{ margin: 0 }}>
            Couldn't reach the server, so this list might be out of date. You can
            still carry on — the upload will tell you if something isn't supported.
          </p>
        </div>
      )}

      <div className="stack">
        {exercises.map((ex) => (
          <button
            key={ex.id}
            className="btn exercise-btn"
            onClick={() => onPick(ex)}
          >
            {ex.name}
            <span className="view">filmed {ex.view_label}</span>
          </button>
        ))}
      </div>

      {exercises.length === 0 && !loadFailed && (
        <p className="muted small">Loading…</p>
      )}
    </div>
  )
}
