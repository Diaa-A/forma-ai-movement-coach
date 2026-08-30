/**
 * The app's name, on screen.
 *
 * Before this it existed only in the <title> and the manifest — so the one place
 * it never appeared was the app. It sits above every screen, deliberately
 * smaller than the screen's own h1: it is a persistent strip, not a headline,
 * and a wordmark larger than the page heading would invert the hierarchy on all
 * six screens to serve the brand on none of them.
 *
 * The mark is the figure scripts/make_icons.py draws for the PWA icons — side
 * on, at the bottom of a squat, trunk tipped forward over the hip. That is the
 * honest version of an app mark, because it is roughly what the overlay draws on
 * the user's own video.
 *
 * One difference from the icon, on purpose: the icon picks the trunk out in
 * --bad, where the forward-lean cue usually fires. This one is single-colour.
 * Red means "this joint is at fault" on the skeleton overlay and in the key
 * frames, and spending it on decoration that is on screen at the same time as a
 * red failure banner would wear the meaning off it. The icon is seen once, out
 * of context, on a home screen; this is seen next to results.
 *
 * Inline SVG rather than an icon font or an icon library — one mark does not
 * justify either, and a font would be a second network request for a glyph set
 * we would use one character of.
 *
 * The heading elements on each screen carry the accessible structure, so the
 * mark is hidden from assistive tech and the wordmark is plain text in a div
 * rather than a heading — six screens already have their own h1, and the results
 * screen's tests assert on the exact set of h3s present.
 */
export default function Masthead() {
  return (
    <div className="masthead">
      {/* Redrawn rather than scaled down from the icon's coordinates. The icon
          has seven segments and reads fine at 192 px; the same figure at 24 px
          came out as a squiggle, because each segment lands at about five
          pixels. This is three segments and a head — trunk, thigh, shin — which
          is the least that still says "side-on, at the bottom of a squat". The
          direction matters and matches the icon: shoulder ahead of the hip,
          knee forward, ankle back under it. A trunk tilted the other way is not
          a squat, and this is the mark of an app that judges exactly that. */}
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
    </div>
  )
}
