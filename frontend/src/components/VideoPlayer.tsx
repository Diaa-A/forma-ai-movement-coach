import { useState } from 'react'

interface Props {
  src: string
  poster?: string
}

/**
 * The annotated clip.
 *
 * It is written as H.264 in an MP4 with faststart, which every
 * target browser plays — but "should play" and "does play on this particular
 * phone" are different claims, and this is the one screen where being wrong
 * leaves the user staring at a black box. So the error path is real: if the
 * element can't decode it, fall back to a download link and let the rest of the
 * results stand. Losing the video must not lose the coaching.
 *
 * playsInline matters on iOS — without it Safari takes the video fullscreen the
 * moment it starts, which is jarring inside an installed app.
 */
export default function VideoPlayer({ src, poster }: Props) {
  const [failed, setFailed] = useState(false)

  if (failed) {
    return (
      <div className="banner banner-warn">
        <p className="small" style={{ margin: 0 }}>
          This browser wouldn't play the annotated clip.{' '}
          <a href={src} download>
            Download it
          </a>{' '}
          to watch it in another app — the key frames and your report below are
          unaffected.
        </p>
      </div>
    )
  }

  return (
    <video
      className="annotated-video"
      src={src}
      poster={poster}
      controls
      playsInline
      /* muted + autoPlay + loop: the clip is a few seconds of the user's own
         set with the overlay on it, and it is the thing they came to see.
         Waiting for a play tap wastes the moment the screen arrives. Muted is
         not a preference here -- iOS and Chrome both refuse to autoplay with
         sound, and an unmuted autoplay attempt simply does not start. There is
         no audio track on the render anyway. Controls stay, so it can be
         paused, scrubbed or replayed. */
      muted
      autoPlay
      loop
      preload="metadata"
      onError={() => setFailed(true)}
    />
  )
}
