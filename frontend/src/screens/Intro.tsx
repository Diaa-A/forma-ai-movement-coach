interface Props {
  onStart: () => void
}

/**
 * Shown once per device, before anything else.
 *
 * It exists for the wait. The analysis takes tens of seconds, and until now the
 * first thing a new user learned about that was a progress screen appearing
 * after they had already committed their video. Saying it up front costs one
 * screen on one visit and changes what the wait means.
 *
 * The third point is the one that is actually unusual about this app, so it goes
 * in the introduction rather than being discovered on the results screen: it
 * tells you what the clip could not show. Everything here is deliberately
 * modest — no accuracy claim, no adjective about the coaching. The system's own
 * output is careful about what it does not know, and an introduction that
 * oversold it would be the first thing to contradict.
 */
export default function Intro({ onStart }: Props) {
  return (
    <div className="stack">
      <header>
        <h1>How this works</h1>
        <p className="muted">Three things worth knowing before you start.</p>
      </header>

      <ol className="intro-list">
        <li>
          <strong>Film one set from the side.</strong>
          <span>
            Five to thirty seconds, whole body in frame. We show you how to
            frame it before you record anything.
          </span>
        </li>
        <li>
          <strong>It is analysed on our server, not on your phone.</strong>
          <span>
            That takes under a minute, and the screen tells you roughly where it
            has got to. Your clip is deleted afterwards — you will see exactly
            when, before you upload.
          </span>
        </li>
        <li>
          <strong>You get what it measured, and what it could not.</strong>
          <span>
            An annotated video, the moments that mattered, and written coaching
            — plus the things a side-on clip cannot show, said out loud rather
            than left out.
          </span>
        </li>
      </ol>

      <button className="btn btn-primary" onClick={onStart}>
        Start
      </button>
    </div>
  )
}
