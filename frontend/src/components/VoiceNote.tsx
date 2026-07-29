import { useEffect, useRef, useState } from 'react'
import { checkAudioFile } from '../validation'

interface Props {
  audio: File | null
  text: string
  onAudio: (file: File | null) => void
  onText: (text: string) => void
}

/**
 * The optional voice note — three ways to give it, and skipping is one of them.
 *
 * This is the only part of the app that needs getUserMedia, and therefore the
 * only part that needs a secure context. Over plain http on a LAN address
 * navigator.mediaDevices is undefined, so rather than showing a button that
 * throws when tapped, we detect it and offer the text box instead.
 *
 * Container note: Safari's MediaRecorder produces audio/mp4 and Chrome's produces
 * audio/webm. The backend allowlist has .webm and .m4a but not .mp4, so the blob
 * gets named from the recorder's own mimeType and audio/mp4 is written as .m4a —
 * same container, and it's on the list.
 */
export default function VoiceNote({ audio, text, onAudio, onText }: Props) {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const canRecord = supportsRecording()

  // tick the counter while recording
  useEffect(() => {
    if (!recording) return
    const started = Date.now()
    const id = setInterval(() => setElapsed((Date.now() - started) / 1000), 200)
    return () => clearInterval(id)
  }, [recording])

  // don't leave the mic light on if the user navigates away mid-recording
  useEffect(() => {
    return () => {
      const rec = recorderRef.current
      if (rec && rec.state !== 'inactive') {
        rec.stop()
        rec.stream.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  async function start() {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const rec = new MediaRecorder(stream)
      chunksRef.current = []

      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const type = rec.mimeType || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type })
        const file = new File([blob], `voice${extensionFor(type)}`, { type })

        const check = checkAudioFile(file)
        if (!check.ok) {
          setError(check.message ?? 'That recording could not be used.')
          return
        }
        onAudio(file)
      }

      rec.start()
      recorderRef.current = rec
      setElapsed(0)
      setRecording(true)
    } catch {
      // Almost always a declined mic permission. Occasionally no input device.
      setError("Couldn't access the microphone. You can type your note instead.")
    }
  }

  function stop() {
    recorderRef.current?.stop()
    recorderRef.current = null
    setRecording(false)
  }

  return (
    <div className="card">
      <h3>Anything you want feedback on? (optional)</h3>
      <p className="small muted">
        Tell us what you're working on or what you're unsure about, and the
        coaching will answer it directly. Skip it and you still get the full
        analysis.
      </p>

      {audio ? (
        <div className="banner banner-info">
          <p className="small" style={{ margin: 0 }}>
            Voice note attached ({audio.name}).{' '}
            <button className="link-btn" onClick={() => onAudio(null)}>
              Remove
            </button>
          </p>
        </div>
      ) : canRecord ? (
        <div className="stack">
          {recording ? (
            <button className="btn btn-bad" onClick={stop}>
              ■ Stop recording — {elapsed.toFixed(0)}s
            </button>
          ) : (
            <button className="btn" onClick={start}>
              ● Record a voice note
            </button>
          )}
        </div>
      ) : (
        <p className="small muted">
          Recording needs a secure (https) connection, which this one isn't — type
          your note below instead.
        </p>
      )}

      {error && (
        <p className="small" style={{ color: 'var(--bad)' }}>
          {error}
        </p>
      )}

      {!audio && (
        <textarea
          className="textarea"
          rows={3}
          placeholder="e.g. I want to know if I'm going deep enough"
          value={text}
          onChange={(e) => onText(e.target.value)}
        />
      )}
    </div>
  )
}

/** navigator.mediaDevices is undefined outside a secure context, so this doubles
 *  as the https check — no need to sniff location.protocol separately. */
function supportsRecording(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== 'undefined'
  )
}

/** mimeType comes back like 'audio/webm;codecs=opus' — take the container part.
 *  audio/mp4 is deliberately written as .m4a: it's the same container and it's on
 *  the server's allowlist, .mp4 isn't. */
function extensionFor(mimeType: string): string {
  const base = mimeType.split(';')[0].trim()
  switch (base) {
    case 'audio/mp4':
    case 'audio/x-m4a':
      return '.m4a'
    case 'audio/mpeg':
      return '.mp3'
    case 'audio/ogg':
      return '.ogg'
    case 'audio/wav':
    case 'audio/x-wav':
      return '.wav'
    default:
      return '.webm'
  }
}
