import type { Exercise, Limits } from '../types'

interface Props {
  exercise: Exercise
  /** served, not written here: the band below is the one the server actually
   *  warns on, and a copy of it in this file is a copy that can drift */
  limits: Limits
  onContinue: () => void
  onBack: () => void
}

// The API hands back plane keys ('sagittal', 'frontal'). The user does not need
// the anatomy vocabulary, they need to know which way to point the phone.
const VIEW_FOR_PLANE: Record<string, string> = {
  sagittal: 'side-on',
  frontal: 'front-on',
}

/**
 * Shown before the user picks a file, which is the entire point of it.
 *
 * Camera placement is the biggest single cause of a clip the system can't assess
 * properly — a side-on view physically cannot show left/right symmetry, and
 * guessing at an occluded limb is how you get a confidently wrong answer.
 * Telling someone up front costs one screen. Detecting it
 * afterwards costs them a re-shoot.
 *
 * Every string here comes from the backend profile, so this screen and the
 * coaching output can't drift apart.
 */
export default function FilmingGuide({ exercise, limits, onContinue, onBack }: Props) {
  const thisView = VIEW_FOR_PLANE[planeFor(exercise)] ?? exercise.view_label

  return (
    <div className="stack">
      <header>
        <h1>Filming a {exercise.name.toLowerCase()}</h1>
        <p className="muted">Worth thirty seconds — it decides how much we can tell you.</p>
      </header>

      <div className="card">
        <h3>How to set up</h3>
        <p className="guide-quote">{exercise.filming_guide}</p>
      </div>

      <div className="card">
        <h3>What a {thisView} clip can show</h3>
        <ul className="plane-list">
          {Object.entries(exercise.assesses).map(([plane, items]) =>
            items.map((item) => {
              const covered = VIEW_FOR_PLANE[plane] === thisView
              return (
                <li key={`${plane}-${item}`}>
                  <span className={covered ? 'tick' : 'cross'}>{covered ? '✓' : '—'}</span>
                  <span className={covered ? undefined : 'muted'}>
                    {item}
                    {!covered && (
                      <span className="small muted">
                        {' '}· needs a {VIEW_FOR_PLANE[plane] ?? plane} view
                      </span>
                    )}
                  </span>
                </li>
              )
            }),
          )}
        </ul>
      </div>

      <div className="card card-tight">
        <p className="small muted" style={{ margin: 0 }}>
          Keep the clip between {limits.ideal_min_seconds} and{' '}
          {limits.ideal_max_seconds} seconds, with your whole body in frame
          the whole time. A few good reps beat a long set.
        </p>
      </div>

      <button className="btn btn-primary" onClick={onContinue}>
        Got it — choose a clip
      </button>
      <button className="btn btn-quiet" onClick={onBack}>
        Pick a different exercise
      </button>
    </div>
  )
}

/** Which plane this exercise's own recommended view corresponds to. Falls back to
 *  sagittal, which is the one every current exercise is filmed in. */
function planeFor(exercise: Exercise): string {
  const match = Object.keys(VIEW_FOR_PLANE).find(
    (plane) => VIEW_FOR_PLANE[plane] === exercise.view_label,
  )
  return match ?? 'sagittal'
}
