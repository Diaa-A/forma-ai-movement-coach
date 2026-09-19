import type { ReactNode } from 'react'

/**
 * The app's name, on screen.
 *
 * Before this it existed only in the <title> and the manifest, so the one place
 * it never appeared was the app. It sits above every screen and is deliberately
 * smaller than the screen's own h1 — a persistent strip, not a headline.
 *
 * The mark is the figure scripts/make_icons.py draws for the PWA icons, side on
 * at the bottom of a squat. Single-colour, unlike the icon, which picks the trunk
 * out in --bad: red means "this joint is at fault" on the overlay and in the key
 * frames, and spending it on decoration sitting next to a red failure banner
 * would wear the meaning off it.
 *
 * Inline SVG because one mark does not justify an icon font or a library. The
 * screens carry the accessible structure, so the mark is hidden from assistive
 * tech and the wordmark is a plain div — every screen already has its own h1,
 * and the results screen's tests assert on the exact set of h3s present.
 *
 * `aside` rides at the right end of the row. The row's height is fixed, so the
 * install pill arriving a couple of seconds after load — Chromium fires its event
 * late — moves nothing that is already on screen. Anything the aside renders at
 * full width wraps underneath the row instead of stretching it.
 */
interface Props {
  aside?: ReactNode
}

export default function Masthead({ aside }: Props) {
  return (
    <div className="masthead">
      {/* Redrawn rather than scaled down from the icon. The icon has seven
          segments and reads fine at 192 px; the same figure at 24 px came out a
          squiggle, each segment landing at about five pixels. Three segments and
          a head — trunk, thigh, shin. Direction matches the icon: shoulder ahead
          of the hip, knee forward, ankle back under it. */}
      <svg
        className="masthead-mark"
        width="13"
        height="24"
        viewBox="6 1 12.5 22.5"
        aria-hidden="true"
        focusable="false"
      >
        <circle cx="16" cy="3.6" r="2.4" fill="var(--accent)" />
        <path
          d="M14 7.5 L7.5 14.5 L16.5 17 L13 22"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="wordmark">Forma</span>
      {aside}
    </div>
  )
}
