interface Props {
  /** 1-based, or 0 to render nothing (error screens have no place in the flow) */
  current: number
}

const STEPS = ['Exercise', 'Framing', 'Clip', 'Results']

/**
 * Where you are in a four-step flow.
 *
 * Added because the app never said how long it was, and someone uploading a
 * video of themselves is deciding at every screen whether to carry on.
 *
 * Consent shares a step with framing rather than taking its own: it appears once
 * per device, so a dot for it would make the flow five steps on the first run
 * and four after, and an indicator that changes length is worse than a coarse
 * one.
 *
 * Only the current step is named. Naming all four costs a line of text at 375px,
 * and the labels only matter for the step you are on.
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
