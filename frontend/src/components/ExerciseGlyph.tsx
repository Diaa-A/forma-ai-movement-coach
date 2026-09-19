/**
 * A small figure for each exercise, in the pose the app expects to be filmed.
 *
 * Drawn rather than photographed, and drawn per exercise rather than one generic
 * dumbbell, because the shape carries the camera angle: the pull-up hangs from a
 * bar seen front-on, the push-up is side-on against the floor. The registry ids
 * are the keys, so an exercise the backend adds falls back to the squat figure
 * rather than rendering nothing.
 *
 * Figure strokes take the accent; the bar and floor take the dim text colour, so
 * the equipment reads as context rather than as part of the body.
 */
interface Props {
  id: string
  size?: number
}

const ACCENT = 'var(--accent)'
const DIM = 'var(--text-dim)'

export default function ExerciseGlyph({ id, size = 26 }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      style={{ display: 'block' }}
    >
      {id === 'pullup' && (
        <>
          <path d="M3 3.2 L21 3.2" {...line(DIM, 1.6)} />
          <circle cx="12" cy="7.8" r="2.1" fill={ACCENT} />
          <path
            d="M7 3.2 L8.8 11.2 L15.2 11.2 L17 3.2 M12 11.2 L12 16.8 M10.2 21.8 L12 16.8 L13.8 21.8"
            {...line(ACCENT, 2)}
          />
        </>
      )}

      {id === 'pushup' && (
        <>
          <path d="M2.6 21.4 L21.4 21.4" {...line(DIM, 1.6)} />
          <circle cx="5" cy="12.4" r="2.1" fill={ACCENT} />
          <path d="M21 19.4 L8.4 14 L8.6 19.6" {...line(ACCENT, 2)} />
        </>
      )}

      {id !== 'pullup' && id !== 'pushup' && (
        <>
          <circle cx="16" cy="3.6" r="2.3" fill={ACCENT} />
          <path d="M14 7.5 L7.5 14.5 L16.5 17 L13 22" {...line(ACCENT, 2.2)} />
        </>
      )}
    </svg>
  )
}

function line(stroke: string, width: number) {
  return {
    fill: 'none',
    stroke,
    strokeWidth: width,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  }
}
