import BrandMark from '../components/BrandMark'
import Disclaimer from '../components/Disclaimer'
import ExerciseGlyph from '../components/ExerciseGlyph'
import type { Exercise } from '../types'

interface Props {
  exercises: Exercise[]
  loadFailed: boolean
  /** true on the first arrival, false coming back. Replaying the opening on the
   *  way back from the guide looked like a stutter -- mark redrawing and sheet
   *  sliding up underneath a screen transition that was already running */
  opening: boolean
  onPick: (exercise: Exercise) => void
}

/**
 * First screen, laid out for a thumb.
 *
 * Brand and promise up top where they get read once, list in a sheet on the
 * bottom edge where a hand holding a phone at the gym actually reaches it.
 *
 * The list still comes from GET /exercises rather than being written out here,
 * so registering an exercise on the backend is enough on its own.
 *
 * Skeleton rows while the catalogue loads, same height as the real ones. The
 * loading text that used to sit here was shorter than the list that replaced
 * it, and that was half the layout shift this screen measured at
 */
export default function ExerciseSelect({ exercises, loadFailed, opening, onPick }: Props) {
  const loading = exercises.length === 0 && !loadFailed

  return (
    <div className={opening ? 'home home-opening' : 'home'}>
      <div className="home-hero">
        <BrandMark />
        <p className="home-headline">
          Film a set.
          <br />
          <span className="home-headline-accent">Get coaching on your form.</span>
        </p>
        <p className="muted home-subline">
          Analysis takes under a minute. You'll see what it measured, and what it
          couldn't.
        </p>
      </div>

      <section className="home-sheet">
        <h1 className="home-sheet-title">What did you film?</h1>

        {loadFailed && (
          <div className="banner banner-warn">
            <p className="small" style={{ margin: 0 }}>
              Couldn't reach the server, so this list might be out of date. You can
              still carry on — the upload will tell you if something isn't supported.
            </p>
          </div>
        )}

        <div className="home-list">
          {exercises.map((ex) => (
            <button
              key={ex.id}
              type="button"
              className="exercise-row"
              onClick={() => onPick(ex)}
            >
              <ExerciseGlyph id={ex.id} size={30} />
              <span className="exercise-text">
                <span className="exercise-name">{ex.name}</span>
                <span className="exercise-view">filmed {ex.view_label}</span>
              </span>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                aria-hidden="true"
                focusable="false"
                className="exercise-chevron"
              >
                <path
                  d="M9 6 L15 12 L9 18"
                  fill="none"
                  stroke="var(--text-dim)"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          ))}

          {loading && (
            <>
              <span className="sr-only">Loading…</span>
              <span className="exercise-skeleton" aria-hidden="true" />
              <span className="exercise-skeleton" aria-hidden="true" />
              <span className="exercise-skeleton" aria-hidden="true" />
            </>
          )}
        </div>

        <Disclaimer />
      </section>
    </div>
  )
}
