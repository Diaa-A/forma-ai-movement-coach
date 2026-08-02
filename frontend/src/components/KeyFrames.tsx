import type { KeyFrame } from '../types'

interface Props {
  frames: KeyFrame[]
}

function captionFor(frame: KeyFrame): { title: string; note?: string } {
  if (frame.label === 'best') return { title: 'Your cleanest rep' }
  if (frame.label !== 'worst') return { title: frame.label }
  // On a set where nothing was flagged there is no red limb in this image, and
  // promising "where form drifted most" sends the user hunting for one. Say what
  // it actually is: the rep that came closest to a limit without crossing it.
  return frame.highlighted
    ? { title: 'Where form drifted most', note: 'the marked joints are the fault' }
    : { title: 'Your tightest rep', note: 'nothing was flagged — this one came closest' }
}

/**
 * Worst and best frames side by side. The contrast is the point — seeing the two
 * next to each other explains a fault faster than any amount of prose about
 * trunk angles, and both images already carry the measured overlay.
 */
export default function KeyFrames({ frames }: Props) {
  if (frames.length === 0) return null

  return (
    <section>
      <h3>Key moments</h3>
      <div className="keyframes">
        {frames.map((frame) => {
          const caption = captionFor(frame)
          return (
            <figure key={frame.label} className="keyframe">
              {/* deliberately not loading="lazy" — there are two images, they are
                  the reason the user is on this screen, and they're already paid
                  for by the upload that produced them. Lazy loading saves nothing
                  here and leaves a blank box on any browser that handles the
                  intersection check oddly. */}
              <img src={frame.url} alt={`${frame.label} form frame`} />
              <figcaption className="small">
                <strong>{caption.title}</strong>
                <span className="muted"> · {frame.timestamp.toFixed(1)}s in</span>
                {caption.note && (
                  <span className="muted"> — {caption.note}</span>
                )}
              </figcaption>
            </figure>
          )
        })}
      </div>
    </section>
  )
}
