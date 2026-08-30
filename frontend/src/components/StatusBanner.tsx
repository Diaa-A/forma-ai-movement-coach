import type { AnalysisStatus } from '../types'

interface Props {
  status: AnalysisStatus
  warnings: string[]
}

/**
 * The guard against the worst thing this app could do: let "we couldn't analyse
 * this" read as "your form was fine".
 *
 * The backend already does the structural half — when status isn't ok it emits
 * not_analyzed_report() *instead of* coaching, with an empty what_went_well, so
 * there's no praise to misread. This is the visual half. It renders above the
 * video, not below the report, so it's read first, and it uses a different card
 * treatment rather than only different words.
 *
 * data-no-stagger keeps it out of the entrance animation the rest of the results
 * screen gets (staggerIn in motion.ts). Everything else on that screen fades in
 * over a couple of hundred milliseconds; this appears immediately. Putting even
 * 40 ms in front of the sentence that stops "we could not analyse this" reading
 * as "your form was fine" is not a trade worth making for a nicer entrance.
 */
export default function StatusBanner({ status, warnings }: Props) {
  const hasWarnings = warnings.length > 0

  if (status === 'ok') {
    // Nothing to announce about the analysis itself, but a warning (a voice note
    // that failed to transcribe, say) still has to be shown rather than dropped.
    if (!hasWarnings) return null
    return (
      <div className="banner banner-warn" data-no-stagger>
        <h2>One thing to flag</h2>
        <WarningList warnings={warnings} />
      </div>
    )
  }

  const copy = status === 'no_reps'
    ? {
        heading: 'No complete rep found',
        body: 'We could not pick out a full repetition in this clip, so nothing has been scored. This is not a judgement on your form — there was simply nothing to measure.',
      }
    : status === 'rotated'
    ? {
        // the file's fault, not the person's — analysing it anyway is how the
        // reference clip's best frame got reported as a forward-lean fault
        heading: 'This clip decodes sideways',
        body: 'The file is stored rotated without the flag that says which way up it goes — clips sent through messaging apps often lose it. Measuring it would judge a tipped-over image, so we have not scored it. Upload the original from your camera roll and it should analyse normally.',
      }
    : {
        heading: 'Tracking was too unreliable',
        body: 'Body tracking on this clip was not steady enough to give feedback we would stand behind. Rather than guess, we have not scored it.',
      }

  return (
    <div className="banner banner-bad" data-no-stagger>
      <h2>{copy.heading}</h2>
      <p className="small" style={{ marginBottom: hasWarnings ? 10 : 0 }}>{copy.body}</p>
      {hasWarnings && <WarningList warnings={warnings} />}
    </div>
  )
}

function WarningList({ warnings }: { warnings: string[] }) {
  return (
    <ul className="small" style={{ marginBottom: 0 }}>
      {warnings.map((w) => <li key={w}>{w}</li>)}
    </ul>
  )
}
