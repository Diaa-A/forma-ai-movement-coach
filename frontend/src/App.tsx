import { useEffect, useReducer, useRef } from 'react'
import { analyze, fetchCatalog } from './api'
import type { AnalyzeResponse, ApiError, Exercise, Limits } from './types'
import { FALLBACK } from './validation'
import { useOnline } from './useOnline'
import Disclaimer from './components/Disclaimer'
import ErrorPanel from './components/ErrorPanel'
import InstallPrompt from './components/InstallPrompt'
import OfflineNotice from './components/OfflineNotice'
import ExerciseSelect from './screens/ExerciseSelect'
import FilmingGuide from './screens/FilmingGuide'
import Capture from './screens/Capture'
import Processing from './screens/Processing'
import Results from './screens/Results'

// Five screens, no deep links, no back/forward requirement — so `screen` is a
// field in the reducer rather than a router. A router here would be a dependency
// added to model a state machine we already have. The cost is that the phone's
// back button doesn't step through screens; every screen carries its own back
// control instead.
type Screen = 'select' | 'guide' | 'capture' | 'processing' | 'results' | 'error'

interface Submission {
  video: File
  voiceNote: File | null
  voiceText: string
}

interface State {
  screen: Screen
  exercises: Exercise[]
  limits: Limits
  exercisesFailed: boolean
  chosen: Exercise | null
  submission: Submission | null
  uploadFraction: number
  bytesSent: number
  bytesTotal: number
  uploadDone: boolean
  result: AnalyzeResponse | null
  error: ApiError | null
}

type Action =
  | { type: 'exercises-loaded'; exercises: Exercise[]; limits: Limits }
  | { type: 'exercises-failed' }
  | { type: 'pick'; exercise: Exercise }
  | { type: 'to'; screen: Screen }
  | { type: 'submit'; submission: Submission }
  | { type: 'upload-progress'; fraction: number; sent: number; total: number }
  | { type: 'upload-done' }
  | { type: 'succeeded'; result: AnalyzeResponse }
  | { type: 'failed'; error: ApiError }
  | { type: 'restart' }

const initial: State = {
  screen: 'select',
  exercises: [],
  // the server's limits replace these as soon as /exercises answers; until then
  // the checks still run, so a file is never sent off unchecked
  limits: FALLBACK,
  exercisesFailed: false,
  chosen: null,
  submission: null,
  uploadFraction: 0,
  bytesSent: 0,
  bytesTotal: 0,
  uploadDone: false,
  result: null,
  error: null,
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
      return { ...state, exercises: action.exercises, limits: action.limits,
               exercisesFailed: false }
    case 'exercises-failed':
      // keep whatever limits we have; FALLBACK is already the initial value
      return { ...state, exercises: OFFLINE_FALLBACK, exercisesFailed: true }
    case 'pick':
      return { ...state, chosen: action.exercise, screen: 'guide' }
    case 'to':
      return { ...state, screen: action.screen }
    case 'submit':
      return {
        ...state,
        submission: action.submission,
        screen: 'processing',
        uploadFraction: 0,
        bytesSent: 0,
        bytesTotal: action.submission.video.size,
        uploadDone: false,
        error: null,
      }
    case 'upload-progress':
      return { ...state, uploadFraction: action.fraction,
               bytesSent: action.sent, bytesTotal: action.total }
    case 'upload-done':
      return { ...state, uploadDone: true, uploadFraction: 1 }
    case 'succeeded':
      return { ...state, result: action.result, screen: 'results' }
    case 'failed':
      // a cancel goes back to the capture screen with the file still chosen,
      // rather than to an error page that says "you did that on purpose"
      if (action.error.kind === 'aborted') {
        return { ...state, screen: 'capture', error: null }
      }
      return { ...state, error: action.error, screen: 'error' }
    case 'restart':
      // keep the loaded catalogue, drop everything about the last run
      return { ...initial, exercises: state.exercises, limits: state.limits,
               exercisesFailed: state.exercisesFailed }
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initial)
  const inFlight = useRef<{ abort: () => void } | null>(null)
  const online = useOnline()

  // refetch when the connection comes back, so an app opened offline heals
  // itself rather than needing a manual reload
  useEffect(() => {
    if (!online) return
    let cancelled = false
    fetchCatalog()
      .then(({ exercises, limits }) => {
        if (!cancelled) dispatch({ type: 'exercises-loaded', exercises, limits })
      })
      .catch(() => {
        if (!cancelled) dispatch({ type: 'exercises-failed' })
      })
    return () => {
      cancelled = true
    }
  }, [online])

  // Opened with no connection and nothing to show yet: that deserves the whole
  // screen. Once there's a result on screen we keep it and drop to a banner —
  // throwing away someone's analysis because the wifi dropped would be worse
  // than the disconnection.
  const deadStart = !online && state.screen === 'select' && state.exercisesFailed
  if (deadStart) {
    return (
      <main>
        <OfflineNotice standalone />
        <Disclaimer />
      </main>
    )
  }

  function send(submission: Submission) {
    dispatch({ type: 'submit', submission })

    const handle = analyze({
      video: submission.video,
      exerciseType: state.chosen?.id ?? 'squat',
      voiceNote: submission.voiceNote,
      voiceNoteText: submission.voiceText,
      onUploadProgress: (fraction, sent, total) =>
        dispatch({ type: 'upload-progress', fraction, sent, total }),
      onUploadComplete: () => dispatch({ type: 'upload-done' }),
    })
    inFlight.current = handle

    handle.result
      .then((result) => dispatch({ type: 'succeeded', result }))
      .catch((error: ApiError) => dispatch({ type: 'failed', error }))
      .finally(() => { inFlight.current = null })
  }

  function cancel() {
    inFlight.current?.abort()
  }

  return (
    <main>
      {!online && <OfflineNotice />}

      {/* Above the exercise list on purpose. Installing to the home screen is the
          whole reason this is a PWA rather than a website, and when it sat at the
          bottom of this screen it fell below the fold on a phone -- invisible on
          the one device it exists for. */}
      {state.screen === 'select' && <InstallPrompt />}

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

      {state.screen === 'capture' && state.chosen && (
        <Capture
          exercise={state.chosen}
          limits={state.limits}
          online={online}
          onSubmit={(video, voiceNote, voiceText) =>
            send({ video, voiceNote, voiceText })}
          onBack={() => dispatch({ type: 'to', screen: 'guide' })}
        />
      )}

      {state.screen === 'processing' && (
        <Processing
          uploadFraction={state.uploadFraction}
          bytesSent={state.bytesSent}
          bytesTotal={state.bytesTotal}
          uploadDone={state.uploadDone}
          onCancel={cancel}
        />
      )}

      {state.screen === 'results' && state.result && (
        <Results
          result={state.result}
          onRestart={() => dispatch({ type: 'restart' })}
        />
      )}

      {state.screen === 'error' && state.error && (
        <ErrorPanel
          error={state.error}
          onRetry={() => state.submission && send(state.submission)}
          onRestart={() => dispatch({ type: 'restart' })}
        />
      )}

      <Disclaimer />
    </main>
  )
}
