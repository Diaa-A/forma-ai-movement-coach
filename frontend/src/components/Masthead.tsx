import type { ReactNode } from 'react'

/**
 * The app's name on screen. A persistent strip, not a headline.
 *
 * Mark is hidden from assistive tech and the wordmark is a plain div -- every
 * screen has its own h1 already, and the results tests assert on the exact set
 * of headings.
 *
 * Single colour, unlike the PWA icon. --bad means "this joint is at fault" on
 * the overlay and in the key frames; spending it on decoration next to a
 * failure banner wears the meaning off it.
 *
 * Row height is fixed so the install pill lands without shoving anything --
 * Chromium offers it a second or two after load
 */
interface Props {
  aside?: ReactNode
  /** where back goes from the screen below, or nothing on the screens that
   *  have nowhere to go back to */
  onBack?: () => void
}

export default function Masthead({ aside, onBack }: Props) {
  return (
    <div className="masthead">
      {onBack && (
        <button type="button" className="masthead-back" onClick={onBack}
                aria-label="Back">
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true"
               focusable="false">
            <path d="M15 5 L8 12 L15 19" fill="none" stroke="currentColor"
                  strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      )}
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
