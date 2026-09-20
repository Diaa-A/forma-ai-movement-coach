/**
 * The Forma mark at hero size -- same figure the masthead carries.
 *
 * pathLength normalises the outline to 1 so the opening can draw it from a dash
 * offset of 1 down to 0. Take the attribute off and the mark just stops
 * drawing, no error anywhere
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
