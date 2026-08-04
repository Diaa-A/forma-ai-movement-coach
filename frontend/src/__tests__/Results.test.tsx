/**
 * The safety-critical rendering: a clip we could not analyse must never read as
 * a clip that came back clean.
 *
 * The backend guarantees half of this structurally — when status isn't ok it
 * emits not_analyzed_report() in place of coaching, with an empty
 * what_went_well, so there is no praise available to misread. These tests exist
 * because the frontend is perfectly capable of undoing that: an empty-array map
 * that still renders its heading, a rep count of zero shown as a stat, a green
 * card because the default card is green.
 *
 * Driven off fixtures rather than reasoning about the components.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import Results from '../screens/Results'
import {
  fallbackReportResult, lowDetectionResult, noRepsResult, okResult, okWithWarning,
} from './fixtures'

const noop = () => {}

describe('a clip that analysed cleanly', () => {
  it('shows the rep count and both key frames', () => {
    render(<Results result={okResult} onRestart={noop} />)
    expect(screen.getByText(/2 reps/)).toBeInTheDocument()
    expect(screen.getByAltText('worst form frame')).toBeInTheDocument()
    expect(screen.getByAltText('best form frame')).toBeInTheDocument()
  })

  it('renders the five-part report in order', () => {
    render(<Results result={okResult} onRestart={noop} />)
    // the page has other h3s too (key frames, the transcript), so check the
    // report's own headings appear in the right order relative to each other
    // rather than demanding they are the only ones present
    const all = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    const wanted = [
      'What you did well',
      'Main thing to work on',
      'Also worth noting',
      'Try this',
      'Next session',
      'Filming tip',
    ]
    expect(all.filter((h) => wanted.includes(h ?? ''))).toEqual(wanted)
  })

  it('keeps the filming tip separate from the coaching prose', () => {
    // it is deterministic, never LLM-written (Decision 23) — if it were folded
    // into the report body that distinction would be invisible to the user
    const { container } = render(<Results result={okResult} onRestart={noop} />)
    const tip = container.querySelector('.filming-tip')
    expect(tip).toBeInTheDocument()
    expect(tip).toHaveTextContent(/film a set from the front/i)
  })
})

describe.each([
  ['no_reps', noRepsResult, /No complete rep found/i],
  ['low_detection', lowDetectionResult, /Tracking was too unreliable/i],
])('a clip that could not be analysed (%s)', (_label, result, heading) => {
  it('leads with an explicit banner saying so', () => {
    render(<Results result={result} onRestart={noop} />)
    // by role, because the banner wording and the report's primary_issue say
    // similar things — it's the *heading* that has to be there
    expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument()
  })

  it('renders NO "what you did well" section at all', () => {
    // not an empty list, not a placeholder — the heading must be absent
    render(<Results result={result} onRestart={noop} />)
    expect(screen.queryByText('What you did well')).not.toBeInTheDocument()
  })

  it('does not show a rep count', () => {
    render(<Results result={result} onRestart={noop} />)
    expect(screen.queryByText(/\d+ reps?/)).not.toBeInTheDocument()
  })

  it('does not show key frames', () => {
    render(<Results result={result} onRestart={noop} />)
    expect(screen.queryByText('Key moments')).not.toBeInTheDocument()
  })

  it('uses the failure card treatment, not the neutral one', () => {
    const { container } = render(<Results result={result} onRestart={noop} />)
    expect(container.querySelector('.banner-bad')).toBeInTheDocument()
  })

  it('still explains what to do next', () => {
    render(<Results result={result} onRestart={noop} />)
    expect(screen.getByText(/Re-record with the framing above/i)).toBeInTheDocument()
  })
})

describe('warnings', () => {
  it('are shown even when the analysis itself succeeded', () => {
    // a voice note that failed to transcribe on an otherwise fine run is exactly
    // where silently dropping the warning would be dishonest
    render(<Results result={okWithWarning} onRestart={noop} />)
    expect(screen.getByText(/voice note couldn't be transcribed/i)).toBeInTheDocument()
  })

  it('say what happened without quoting the provider at the user', () => {
    // this carried the raw exception, so an invalid key put "Groq transcription
    // error 401: {...}" in the banner
    render(<Results result={okWithWarning} onRestart={noop} />)
    const banner = document.querySelector('.banner-warn')
    expect(banner?.textContent).not.toMatch(/401|Groq|Traceback/)
  })

  it('produce no banner at all when there are none', () => {
    const { container } = render(<Results result={okResult} onRestart={noop} />)
    expect(container.querySelector('.banner-warn')).not.toBeInTheDocument()
  })
})

describe('provenance', () => {
  it('says when the wording came from the language model', () => {
    render(<Results result={okResult} onRestart={noop} />)
    expect(screen.getByText(/it does not decide them/i)).toBeInTheDocument()
  })

  it('tells the user when the model was unavailable and this is the fallback', () => {
    render(<Results result={fallbackReportResult} onRestart={noop} />)
    expect(screen.getByText(/language model was unavailable/i)).toBeInTheDocument()
  })
})
