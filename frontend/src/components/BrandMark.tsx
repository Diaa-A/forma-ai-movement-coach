/**
 * The Forma mark at hero size.
 *
 * The same figure the masthead carries, drawn at 44 x 80 for the home screen.
 * Hidden from assistive tech — the wordmark beside it in the masthead already
 * says the name.
 *
 * pathLength normalises the outline to 1 so the opening animation can draw it
 * with a dash offset of 1 to 0, without anyone having to measure the real path.
 * Remove the attribute and the mark stops drawing, silently.
 */
export default function BrandMark() {
  return (
    <svg
      className="brand-mark"
      width="44"
      height="80"
      viewBox="6 1 12.5 22.5"
      aria-hidden="true"
      focusable="false"
    >
      <circle cx="16" cy="3.6" r="2.4" fill="var(--accent)" />
      <path
        d="M14 7.5 L7.5 14.5 L16.5 17 L13 22"
        pathLength="1"
        fill="none"
        stroke="var(--accent)"
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
