import type { CoachingReport } from '../types'

interface Props {
  report: CoachingReport
}

/**
 * The five-part coaching structure, rendered in its locked order:
 * what went well, primary issue, secondary issues, corrective cues, next focus.
 * Plus the filming tip, which gets its own treatment because it is generated
 * deterministically and never by the language model (Decision 23) — mixing it
 * into the prose would blur a distinction the project makes on purpose.
 *
 * Every section is conditional on having content. That is not tidiness: on a
 * no_reps run what_went_well is empty by design, and rendering an empty
 * "What you did well" heading would reintroduce exactly the false-pass reading
 * the backend went to trouble to prevent.
 */
export default function ReportView({ report }: Props) {
  return (
    <div className="stack">
      {report.what_went_well.length > 0 && (
        <section className="card">
          <h3>What you did well</h3>
          <ul style={{ marginBottom: 0 }}>
            {report.what_went_well.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </section>
      )}

      {report.primary_issue && (
        <section className="card">
          <h3>Main thing to work on</h3>
          <p style={{ marginBottom: 0 }}>{report.primary_issue}</p>
        </section>
      )}

      {report.secondary_issues.length > 0 && (
        <section className="card">
          <h3>Also worth noting</h3>
          <ul style={{ marginBottom: 0 }}>
            {report.secondary_issues.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </section>
      )}

      {report.corrective_cues.length > 0 && (
        <section className="card">
          <h3>Try this</h3>
          <ul style={{ marginBottom: 0 }}>
            {report.corrective_cues.map((cue) => <li key={cue}>{cue}</li>)}
          </ul>
        </section>
      )}

      {report.next_session_focus && (
        <section className="card">
          <h3>Next session</h3>
          <p style={{ marginBottom: 0 }}>{report.next_session_focus}</p>
        </section>
      )}

      {report.filming_tip && (
        <section className="card filming-tip">
          <h3>Filming tip</h3>
          <p style={{ marginBottom: 0 }}>{report.filming_tip}</p>
        </section>
      )}

      <Provenance report={report} />
    </div>
  )
}

/**
 * Where the wording came from. Small, but it matters: when the language model
 * fails we fall back to the cue database's own text, and the user is entitled to
 * know they're reading the system's phrasing rather than something written for
 * them. Hiding the fallback would make the failure invisible, which is the
 * opposite of what the backend does with it.
 */
function Provenance({ report }: { report: CoachingReport }) {
  let line: string
  switch (report.source) {
    case 'llm':
      line = `Written by a language model from the measurements above${report.model ? ` (${report.model})` : ''}. It rephrases the findings; it does not decide them.`
      break
    case 'dry_run':
      line = 'Written directly from the system\'s cue database, without the language model.'
      break
    case 'dry_run_fallback':
      line = 'The language model was unavailable, so this is the system\'s own wording from its cue database. The findings are unaffected — only the phrasing is plainer.'
      break
    case 'not_analyzed':
      line = 'No coaching was generated, because there was nothing measurable to coach.'
      break
  }
  return <p className="provenance small muted">{line}</p>
}
