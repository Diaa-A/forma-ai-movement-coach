interface Props {
  /** true when the app was opened with no connection and has nothing loaded —
   *  worth a whole screen rather than a strip along the top */
  standalone?: boolean
}

/**
 * What the app says when there's no connection.
 *
 * The requirement is that it degrades honestly rather than looking broken. All
 * the analysis runs server-side, so offline genuinely means no feedback — and
 * saying that plainly is better than an upload form that accepts a file and then
 * fails, or a spinner that never resolves. There is deliberately no offline
 * queue: silently uploading somebody's body video later, when they've moved on,
 * is not a trade worth making for convenience.
 */
export default function OfflineNotice({ standalone }: Props) {
  if (standalone) {
    return (
      <div className="stack">
        <header>
          <h1>You're offline</h1>
        </header>
        <div className="banner banner-warn">
          <p style={{ marginBottom: 8 }}>
            The app has loaded, but analysing a clip happens on the server, so it
            needs a connection.
          </p>
          <p className="small" style={{ margin: 0 }}>
            Nothing is lost — reconnect and this screen will move on by itself.
          </p>
        </div>
        <p className="small muted">
          Your clips are never analysed on the phone, which is why this can't work
          offline. It also means nothing is left behind on the device.
        </p>
      </div>
    )
  }

  return (
    <div className="banner banner-warn">
      <p className="small" style={{ margin: 0 }}>
        <strong>You're offline.</strong> You can look at what's already here, but
        analysing a new clip needs a connection.
      </p>
    </div>
  )
}
