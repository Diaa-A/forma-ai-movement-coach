import BrandMark from '../components/BrandMark'
import Disclaimer from '../components/Disclaimer'
import ExerciseGlyph from '../components/ExerciseGlyph'
import type { Exercise } from '../types'

interface Props {
  exercises: Exercise[]
  loadFailed: boolean
  onPick: (exercise: Exercise) => void
}

/**
 * First screen, laid out for a thumb.
 *
 * The brand and the promise take the top half, where they are read once; the
 * list sits in a sheet on the bottom edge, where a hand holding a phone at the
 * gym can reach it without shifting grip. The list still comes from
 * GET /exercises rather than being written out here, so an exercise registered
 * on the backend turns up with no frontend change.
 *
 * Three skeleton rows stand in while the catalogue loads. They are the same
 * height as the real rows, so the screen does not jump when it arrives — the
 * loading text that used to sit here was shorter than the list that replaced it,
 * which is half of the layout shift this screen was measured at.
 */
export default function ExerciseSelect({ exercises, loadFailed, onPick }: Props) {
  const loading = exercises.length === 0 && !loadFailed

  return (
    <div className="home">
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
              <span className="exercise-tile" aria-hidden="true">
                <ExerciseGlyph id={ex.id} />
              </span>
              <span className="exercise-name">{ex.name}</span>
              <span className="exercise-view">filmed {ex.view_label}</span>
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
