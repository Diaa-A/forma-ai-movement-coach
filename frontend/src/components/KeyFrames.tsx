import type { KeyFrame } from '../types'

interface Props {
  frames: KeyFrame[]
}

const CAPTIONS: Record<string, string> = {
  worst: 'Where form drifted most',
  best: 'Your cleanest rep',
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
        {frames.map((frame) => (
          <figure key={frame.label} className="keyframe">
            {/* deliberately not loading="lazy" — there are two images, they are
                the reason the user is on this screen, and they're already paid
                for by the upload that produced them. Lazy loading saves nothing
                here and leaves a blank box on any browser that handles the
                intersection check oddly. */}
            <img src={frame.url} alt={`${frame.label} form frame`} />
            <figcaption className="small">
              <strong>{CAPTIONS[frame.label] ?? frame.label}</strong>
              <span className="muted"> · {frame.timestamp.toFixed(1)}s in</span>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}
