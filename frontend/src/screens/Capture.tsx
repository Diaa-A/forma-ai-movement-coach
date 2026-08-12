import { useRef, useState } from 'react'
import type { Exercise, Limits } from '../types'
import VoiceNote from '../components/VoiceNote'
import {
  CHUNKY_UPLOAD_BYTES, FALLBACK, checkDuration, checkVideoFile,
  formatBytes, formatSeconds, readDuration,
} from '../validation'

interface Props {
  exercise: Exercise
  /** what the server will accept, from GET /exercises. Defaulted so this screen
   *  still works if the catalogue never arrived. */
  limits?: Limits
  online: boolean
  onSubmit: (video: File, voiceNote: File | null, voiceText: string) => void
  onBack: () => void
}

interface Picked {
  file: File
  duration: number
  warning?: string
}

/**
 * Two ways to get a clip in, gallery first.
 *
 * Gallery is the primary path: most people filming a set already have the clip,
 * and it's the only route that works on a laptop. "Record now" is the same input
 * with capture="environment", which hands off to the phone's own camera app —
 * full viewfinder, proper focus and stabilisation, and no secure-context
 * requirement, unlike getUserMedia.
 *
 * Everything is validated here before a byte leaves the device, because finding
 * out a clip was too long after three minutes of uploading on 4G is miserable.
 */
export default function Capture({ exercise, limits = FALLBACK, online,
                                  onSubmit, onBack }: Props) {
  const [picked, setPicked] = useState<Picked | null>(null)
  const [rejection, setRejection] = useState<string | null>(null)
  const [voiceNote, setVoiceNote] = useState<File | null>(null)
  const [voiceText, setVoiceText] = useState('')
  const [checking, setChecking] = useState(false)

  const galleryInput = useRef<HTMLInputElement>(null)
  const cameraInput = useRef<HTMLInputElement>(null)

  async function take(file: File | undefined) {
    if (!file) return
    setRejection(null)

    const basic = checkVideoFile(file, limits)
    if (!basic.ok) {
      setPicked(null)
      setRejection(basic.message ?? 'That file cannot be used.')
      return
    }

    setChecking(true)
    const duration = await readDuration(file)
    setChecking(false)

    const timing = checkDuration(duration, limits)
    if (!timing.ok) {
      setPicked(null)
      setRejection(timing.message ?? 'That clip is the wrong length.')
      return
    }

    setPicked({ file, duration, warning: timing.warning ? timing.message : undefined })
  }

  const chunky = picked && picked.file.size > CHUNKY_UPLOAD_BYTES

  return (
    <div className="stack">
      <header>
        <h1>Choose your clip</h1>
        <p className="muted">
          {exercise.name}, filmed {exercise.view_label}.
        </p>
      </header>

      {/* Both are file inputs. The second adds capture=environment, which opens
          the native camera rather than the photo library. */}
      <input
        ref={galleryInput}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => take(e.target.files?.[0])}
      />
      <input
        ref={cameraInput}
        type="file"
        accept="video/*"
        capture="environment"
        hidden
        onChange={(e) => take(e.target.files?.[0])}
      />

      <div className="stack">
        <button className="btn" onClick={() => galleryInput.current?.click()}>
          Choose from gallery
        </button>
        <button className="btn" onClick={() => cameraInput.current?.click()}>
          Record now
        </button>
      </div>

      {checking && <p className="small muted">Checking that clip…</p>}

      {rejection && (
        <div className="banner banner-bad">
          <h2>Can't use that one</h2>
          <p className="small" style={{ margin: 0 }}>{rejection}</p>
        </div>
      )}

      {picked && (
        <div className="card">
          <h3>Ready to analyse</h3>
          <p className="small" style={{ marginBottom: 4 }}>
            {picked.file.name}
          </p>
          <p className="small muted" style={{ margin: 0 }}>
            {formatBytes(picked.file.size)}
            {isFinite(picked.duration) && picked.duration > 0
              ? ` · ${formatSeconds(picked.duration)}`
              : ''}
          </p>
          {picked.warning && (
            <p className="small" style={{ color: 'var(--warn)', marginTop: 10, marginBottom: 0 }}>
              {picked.warning}
            </p>
          )}
        </div>
      )}

      {chunky && (
        <div className="banner banner-warn">
          <p className="small" style={{ margin: 0 }}>
            That's a {formatBytes(picked!.file.size)} upload. On mobile data it may
            take a few minutes — keep the screen on while it goes.
          </p>
        </div>
      )}

      <VoiceNote
        audio={voiceNote}
        text={voiceText}
        limits={limits}
        onAudio={setVoiceNote}
        onText={setVoiceText}
      />

      {/* Said here, on the screen where the upload actually happens, rather than
          buried somewhere the user has already passed. The wording comes from
          the server so it always matches the TTL the sweep enforces — writing
          "24 hours" here would be a promise this file cannot keep. */}
      {limits?.retention_note && (
        <p className="small muted" style={{ marginBottom: 0 }}>
          {limits.retention_note}
        </p>
      )}

      <button
        className="btn btn-primary"
        disabled={!picked || checking || !online}
        onClick={() => picked && onSubmit(picked.file, voiceNote, voiceText.trim())}
      >
        {online ? 'Analyse my form' : 'Needs a connection'}
      </button>
      {!online && picked && (
        <p className="small muted" style={{ marginTop: 0 }}>
          Your clip is still selected. This button comes back as soon as you're
          reconnected.
        </p>
      )}
      <button className="btn btn-quiet" onClick={onBack}>
        Back to filming tips
      </button>
    </div>
  )
}
