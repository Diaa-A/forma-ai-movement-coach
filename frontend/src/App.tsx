import { useEffect, useReducer, useRef } from 'react'
import { analyze, fetchCatalog } from './api'
import type { AnalyzeResponse, ApiError, Exercise, Limits } from './types'
import { FALLBACK } from './validation'
import { consentGiven, recordConsent } from './consent'
import { useOnline } from './useOnline'
import { buzzOnResult, withViewTransition } from './motion'
import Disclaimer from './components/Disclaimer'
import ErrorPanel from './components/ErrorPanel'
import InstallPrompt from './components/InstallPrompt'
import Masthead from './components/Masthead'
import Stepper from './components/Stepper'
import OfflineNotice from './components/OfflineNotice'
import ExerciseSelect from './screens/ExerciseSelect'
import FilmingGuide from './screens/FilmingGuide'
import Consent from './screens/Consent'
import Capture from './screens/Capture'
import Processing from './screens/Processing'
import Results from './screens/Results'

// Six screens, no deep links, no back/forward requirement — so `screen` is a
// field in the reducer rather than a router. A router here would be a dependency
// added to model a state machine we already have. The cost is that the phone's
// back button doesn't step through screens; every screen carries its own back
// control instead.
//
// 'consent' sits between the guide and capture: after the exercise is chosen,
// before anything can be recorded — the last point at which nothing has been
// uploaded. It is skipped once accepted (remembered per device, re-shown if the
// retention period changes), so the common path stays five screens long.
type Screen = 'select' | 'guide' | 'consent' | 'capture' | 'processing' | 'results' | 'error'

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
  feedbackFormUrl: string | null
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
  | { type: 'exercises-loaded'; exercises: Exercise[]; limits: Limits;
      feedbackFormUrl: string | null }
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
  feedbackFormUrl: null,
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
               feedbackFormUrl: action.feedbackFormUrl, exercisesFailed: false }
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
               feedbackFormUrl: state.feedbackFormUrl,
               exercisesFailed: state.exercisesFailed }
  }
}

// Which of Stepper's four steps each screen belongs to. Consent shares its step
// with the filming guide -- it is shown once per device, and a progress bar that
// is four steps long for most people and five on their first run is worse than
// one that is a little coarse. The error screen returns 0 and renders nothing:
// it is not a place in the flow, it is a place the flow stopped.
const STEP_OF: Record<Screen, number> = {
  select: 1, guide: 2, consent: 2, capture: 3, processing: 3, results: 4, error: 0,
}


export default function App() {
  const [state, dispatch] = useReducer(reducer, initial)
  const inFlight = useRef<{ abort: () => void } | null>(null)
  const online = useOnline()

  // Anything that swaps the screen goes through here rather than dispatch, so it
  // gets a View Transition where the browser has one. Everything else — upload
  // progress, the catalogue arriving — dispatches directly, because animating a
  // progress bar's own updates would fight the transition on .progress-fill.
  //
  // withViewTransition falls straight through to update() where the API is
  // missing or motion is reduced, so this is the same call either way.
  function go(action: Action) {
    withViewTransition(() => dispatch(action))
  }

  // refetch when the connection comes back, so an app opened offline heals
  // itself rather than needing a manual reload
  useEffect(() => {
    if (!online) return
    let cancelled = false
    fetchCatalog()
      .then(({ exercises, limits, feedback_form_url }) => {
        if (!cancelled) dispatch({ type: 'exercises-loaded', exercises, limits,
                                   feedbackFormUrl: feedback_form_url ?? null })
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
        <Masthead />
        <OfflineNotice standalone />
        <Disclaimer />
      </main>
    )
  }

  function send(submission: Submission) {
    go({ type: 'submit', submission })

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
      .then((result) => {
        // Before the screen swaps, so the buzz lands with the change rather than
        // after it. No-op on iOS, and often ignored elsewhere — see motion.ts.
        buzzOnResult()
        go({ type: 'succeeded', result })
      })
      .catch((error: ApiError) => go({ type: 'failed', error }))
      .finally(() => { inFlight.current = null })
  }

  function cancel() {
    inFlight.current?.abort()
  }

  return (
    <main>
      <Masthead />

      <Stepper current={STEP_OF[state.screen]} />

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
          onPick={(exercise) => go({ type: 'pick', exercise })}
        />
      )}

      {state.screen === 'guide' && state.chosen && (
        <FilmingGuide
          exercise={state.chosen}
          onContinue={() => go({
            type: 'to',
            screen: consentGiven(state.limits.retention_hours) ? 'capture' : 'consent',
          })}
          onBack={() => go({ type: 'restart' })}
        />
      )}

      {state.screen === 'consent' && (
        <Consent
          limits={state.limits}
          onAccept={() => {
            recordConsent(state.limits.retention_hours)
            go({ type: 'to', screen: 'capture' })
          }}
          onDecline={() => go({ type: 'restart' })}
        />
      )}

      {state.screen === 'capture' && state.chosen && (
        <Capture
          exercise={state.chosen}
          limits={state.limits}
          online={online}
          onSubmit={(video, voiceNote, voiceText) =>
            send({ video, voiceNote, voiceText })}
          onBack={() => go({ type: 'to', screen: 'guide' })}
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
          feedbackFormUrl={state.feedbackFormUrl}
          onRestart={() => go({ type: 'restart' })}
        />
      )}

      {state.screen === 'error' && state.error && (
        <ErrorPanel
          error={state.error}
          onRetry={() => state.submission && send(state.submission)}
          onRestart={() => go({ type: 'restart' })}
        />
      )}

      <Disclaimer />
    </main>
  )
}
