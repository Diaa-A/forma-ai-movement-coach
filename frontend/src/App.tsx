import { useEffect, useReducer } from 'react'
import { fetchExercises } from './api'
import type { Exercise } from './types'
import Disclaimer from './components/Disclaimer'
import ExerciseSelect from './screens/ExerciseSelect'
import FilmingGuide from './screens/FilmingGuide'

// Five screens, no deep links, no back/forward requirement — so `screen` is a
// field in the reducer rather than a router. A router here would be a dependency
// added to model a state machine we already have. The cost is that the phone's
// back button doesn't step through screens; every screen carries its own back
// control instead.
type Screen = 'select' | 'guide' | 'capture' | 'processing' | 'results'

interface State {
  screen: Screen
  exercises: Exercise[]
  exercisesFailed: boolean
  chosen: Exercise | null
}

type Action =
  | { type: 'exercises-loaded'; exercises: Exercise[] }
  | { type: 'exercises-failed' }
  | { type: 'pick'; exercise: Exercise }
  | { type: 'to'; screen: Screen }
  | { type: 'restart' }

const initial: State = {
  screen: 'select',
  exercises: [],
  exercisesFailed: false,
  chosen: null,
}

// A single exercise so the app is usable even if /exercises is unreachable on
// first load. Deliberately minimal — real guidance comes from the backend, and
// this is a fallback, not a second copy of the truth.
const OFFLINE_FALLBACK: Exercise[] = [
  {
    id: 'squat',
    name: 'Squat',
    view_label: 'side-on',
    filming_guide:
      'Film side-on at about hip height with your whole body in frame.',
    assesses: { sagittal: ['squat depth', 'forward lean'], frontal: ['left/right symmetry'] },
  },
]

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'exercises-loaded':
      return { ...state, exercises: action.exercises, exercisesFailed: false }
    case 'exercises-failed':
      return { ...state, exercises: OFFLINE_FALLBACK, exercisesFailed: true }
    case 'pick':
      return { ...state, chosen: action.exercise, screen: 'guide' }
    case 'to':
      return { ...state, screen: action.screen }
    case 'restart':
      // keep the loaded exercise list, drop everything about the last run
      return { ...initial, exercises: state.exercises, exercisesFailed: state.exercisesFailed }
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initial)

  useEffect(() => {
    let cancelled = false
    fetchExercises()
      .then((exercises) => {
        if (!cancelled) dispatch({ type: 'exercises-loaded', exercises })
      })
      .catch(() => {
        if (!cancelled) dispatch({ type: 'exercises-failed' })
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main>
      {state.screen === 'select' && (
        <ExerciseSelect
          exercises={state.exercises}
          loadFailed={state.exercisesFailed}
          onPick={(exercise) => dispatch({ type: 'pick', exercise })}
        />
      )}

      {state.screen === 'guide' && state.chosen && (
        <FilmingGuide
          exercise={state.chosen}
          onContinue={() => dispatch({ type: 'to', screen: 'capture' })}
          onBack={() => dispatch({ type: 'restart' })}
        />
      )}

      {state.screen === 'capture' && (
        <div className="stack">
          <h1>Choose a clip</h1>
          <p className="muted">Not built yet — next commit.</p>
          <button className="btn btn-quiet" onClick={() => dispatch({ type: 'restart' })}>
            Start again
          </button>
        </div>
      )}

      <Disclaimer />
    </main>
  )
}
