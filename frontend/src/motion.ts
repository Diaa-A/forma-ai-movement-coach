import { flushSync } from 'react-dom'

/**
 * Motion, in one file, with no animation library.
 *
 * Framer Motion is around 30 kB gzipped against a 57 kB bundle — a 50% increase
 * to do what two CSS transitions and one element.animate() call do here.
 *
 * Everything here is optional: each function checks for the capability it needs
 * and does nothing without it, so the app behaves the same on a browser that has
 * none of them.
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
 * returns before the DOM has changed and the transition cross-fades a screen
 * with itself.
 *
 * Chrome, Edge and Safari 18.2+ have it; Firefox does not. Where it is missing,
 * or motion is reduced, the update runs directly and the DOM ends up identical.
 */
export function withViewTransition(update: () => void): void {
  // lib.dom types startViewTransition as always present; it is not, in Firefox
  // or jsdom, so the runtime check stays whatever TypeScript thinks.
  //
  // document.hidden is here because of a real failure, found driving the results
  // screen in a background tab: a transition started while the document is
  // hidden is skipped, its update callback never runs, and the state change
  // inside it is lost for good. The screen sat on "Uploading your clip" after
  // the server had returned 200. On a phone that is the ordinary case — the
  // analysis takes fifteen to thirty seconds and people switch apps while they
  // wait.
  if (typeof document.startViewTransition !== 'function'
      || prefersReducedMotion()
      || document.hidden) {
    update()
    return
  }
  const transition = document.startViewTransition(() => flushSync(update))
  // Same failure arriving another way — superseded by the next transition, or
  // hidden between the check above and the callback. updateCallbackDone rejects
  // when the callback did not run, so apply the change directly rather than
  // dropping it. Re-applying is safe: everything this wraps sets state to a
  // fixed value rather than toggling it.
  transition.updateCallbackDone.catch(() => update())
  // An aborted transition also rejects `ready` — "Transition was aborted because
  // of invalid state" — which surfaces as an uncaught rejection with nothing
  // attached. Swallowed rather than logged because it is not a fault: the DOM
  // update already happened inside the callback and only the animation was lost.
  transition.ready.catch(() => {})
}

/**
 * Fade the results sections in, staggered.
 *
 * Web Animations and not CSS because the section count varies between four and
 * ten, depending on analysis status, warnings, whether a voice note was
 * transcribed and whether a feedback-form URL was served. nth-child delays would
 * need a hardcoded ceiling and would stop staggering past it.
 *
 * data-no-stagger elements are skipped and do not consume a step. That is the
 * status banner, which guards against "we could not analyse this" reading as
 * "your form was fine" — delaying the most important sentence on the screen is
 * the wrong trade.
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
 * Called optionally because iOS Safari has no vibrate at all and it must not
 * throw there. Chrome also ignores it without prior engagement with the origin,
 * and by the time the analysis finishes the activation from the starting tap may
 * have lapsed — so this fires sometimes, on some devices, and is not worth
 * claiming as a feature.
 *
 * Whether reduced motion should gate haptics at all is still an open question:
 * that media query is about movement on screen, and haptics are a different
 * channel.
 */
export function buzzOnResult(): void {
  if (prefersReducedMotion()) return
  navigator.vibrate?.(18)
}
