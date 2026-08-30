interface Props {
  /** 1-based, or 0 to render nothing (error screens have no place in the flow) */
  current: number
}

const STEPS = ['Exercise', 'Framing', 'Clip', 'Results']

/**
 * Where you are in a four-step flow.
 *
 * Added because the app never said how long it was. Someone uploading a video of
 * themselves to a tool they have not used before is deciding, at every screen,
 * whether this is worth continuing — and "one of four" answers that in a way no
 * amount of reassuring copy does.
 *
 * Consent shares a step with framing rather than taking its own. It appears once
 * per device, so giving it a dot would make the flow four steps long for most
 * runs and five for the first, and a progress indicator that changes length is
 * worse than one that is slightly coarse.
 *
 * The current step is named; the others are dots. Naming all four costs a line
 * of text on a 375px screen and buys nothing — the labels only matter for the
 * step you are on.
 */
export default function Stepper({ current }: Props) {
  if (current < 1) return null

  return (
    <nav className="stepper" aria-label="progress">
      <ol>
        {STEPS.map((label, i) => {
          const n = i + 1
          const state = n < current ? 'done' : n === current ? 'now' : 'todo'
          return (
            <li key={label} className={`step step-${state}`}>
              {/* the dot is decorative; the accessible name is the label text,
                  which is present for every step but visually hidden except on
                  the current one */}
              <span className="step-dot" aria-hidden="true" />
              <span className={n === current ? 'step-label' : 'sr-only'}>
                {label}
              </span>
            </li>
          )
        })}
      </ol>
      <span className="sr-only">{`step ${current} of ${STEPS.length}`}</span>
    </nav>
  )
}
