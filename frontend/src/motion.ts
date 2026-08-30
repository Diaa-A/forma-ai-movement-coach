import { flushSync } from 'react-dom'

/**
 * Motion, in one file, with no animation library.
 *
 * Framer Motion is around 30 kB gzipped against a 57 kB bundle — a 50% increase
 * to do what two CSS transitions and one element.animate() call do here. The
 * same argument the WP-02 plan made against Tailwind applies: the library's
 * motion language would quietly become the app's.
 *
 * Everything below is optional. Each function checks for the capability it needs
 * and does nothing where it is absent, so the app behaves identically on a
 * browser that has none of them — which is most of them, today.
 */

/** True when the OS asks for less movement. matchMedia is absent in jsdom, hence
 *  the optional call rather than a bare matchMedia(...). */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true
}

/**
 * Run a state change inside a View Transition where the browser has one.
 *
 * flushSync is not optional: React batches updates, so without it the callback
 * returns before the DOM has changed and the transition captures the old state
 * twice, producing a cross-fade from a screen to itself.
 *
 * Chrome and Edge have this; Safari from 18.2; Firefox does not ship it. So most
 * users will not see it, which is the right weight for a garnish on a state
 * change that already worked. Where it is missing — or where motion is reduced —
 * the update runs directly and the resulting DOM is identical.
 */
export function withViewTransition(update: () => void): void {
  // lib.dom types startViewTransition as always present; it is not, in Firefox
  // or in jsdom, so the runtime check stays regardless of what TypeScript thinks
  //
  // document.hidden is in this list because of a real failure, found driving the
  // results screen in a background tab: a transition started while the document
  // is hidden is skipped, its update callback never runs, and the state change
  // inside it is lost for good -- bringing the tab back does not replay it. The
  // screen sat on "Uploading your clip" after the server had already returned
  // 200. On a phone that is the ordinary case, not an edge one: the analysis
  // takes fifteen to thirty seconds and people switch apps while they wait.
  // Losing somebody's result to a cosmetic animation is not a trade worth making.
  if (typeof document.startViewTransition !== 'function'
      || prefersReducedMotion()
      || document.hidden) {
    update()
    return
  }
  const transition = document.startViewTransition(() => flushSync(update))
  // Belt and braces for the same failure arriving another way -- a transition
  // superseded by the next one, or the document hidden between the check above
  // and the callback. updateCallbackDone rejects when the callback did not run,
  // so apply the change directly rather than dropping it. Re-applying is safe:
  // every action this wraps sets state to a fixed value rather than toggling it.
  transition.updateCallbackDone.catch(() => update())
  // A transition that gets superseded or runs while the document is hidden
  // rejects `ready`, and with nothing attached that surfaces as an uncaught
  // promise rejection in the console — seen here as "Transition was aborted
  // because of invalid state". Nothing needs to react to it: the DOM update has
  // already happened inside the callback and only the animation was lost. It is
  // swallowed rather than logged because it is not a fault, and a console error
  // in a shipped build is something a marker opening devtools would find.
  transition.ready.catch(() => {})
}

/**
 * Fade the results sections in, staggered.
 *
 * The reason this is Web Animations and not CSS: the section count varies
 * between four and ten depending on the analysis status, whether there are
 * warnings, whether a voice note was transcribed and whether a feedback-form URL
 * was served. nth-child delays would need a hardcoded ceiling and would silently
 * stop staggering past it.
 *
 * Elements marked data-no-stagger are left alone and do not consume a step. That
 * is the status banner: it is the guard against "we could not analyse this"
 * reading as "your form was fine", and putting even 40 ms in front of the most
 * important sentence on the screen is the wrong trade.
 */
export function staggerIn(root: HTMLElement | null): void {
  if (!root || prefersReducedMotion()) return

  const children = Array.from(root.children) as HTMLElement[]
  // jsdom has no Web Animations, and neither does Safari before 13.1
  if (children.length === 0 || typeof children[0].animate !== 'function') return

  let step = 0
  for (const el of children) {
    if (el.dataset.noStagger !== undefined) continue
    el.animate(
      [
        { opacity: 0, transform: 'translateY(6px)' },
        { opacity: 1, transform: 'none' },
      ],
      {
        duration: 200,
        // capped so a long results screen doesn't end up with half a second of
        // dead time before the last card appears
        delay: Math.min(step, 8) * 40,
        easing: 'cubic-bezier(0.2, 0, 0, 1)',
        // without this the element sits at full opacity during its delay and
        // then jumps to the start of the keyframes
        fill: 'backwards',
      },
    )
    step += 1
  }
}

/**
 * One short pulse when a result lands.
 *
 * Absent on iOS Safari entirely, so it is called optionally rather than checked
 * — it must not throw there. It is also commonly ignored by Chrome without prior
 * engagement with the origin, and the analysis takes about fourteen seconds, by
 * which time the activation from the tap that started it may have lapsed. So
 * this fires sometimes, on some devices, and is not worth claiming as a feature.
 *
 * Gated on reduced motion, which is a judgement call rather than a rule: that
 * media query is about movement on screen, and haptics are a different channel.
 * The argument the other way is that someone who turned off animation may be
 * exactly the person who wants to know it finished without watching the screen.
 * Recorded in docs/UI_DESIGN_PLAN.md 6.2 as open.
 */
export function buzzOnResult(): void {
  if (prefersReducedMotion()) return
  navigator.vibrate?.(18)
}
