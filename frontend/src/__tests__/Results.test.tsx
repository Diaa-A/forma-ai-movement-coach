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
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import Results from '../screens/Results'
import type { AnalyzeResponse } from '../types'
import {
  fallbackReportResult, lowDetectionResult, noRepsResult, okResult,
  okWithWarning, rotatedResult,
} from './fixtures'

const noop = () => {}

function show(result: AnalyzeResponse, feedbackFormUrl: string | null = null) {
  return render(<Results result={result} feedbackFormUrl={feedbackFormUrl}
                         onRestart={noop} />)
}

describe('a clip that analysed cleanly', () => {
  it('shows the rep count and both key frames', () => {
    // the count moved into the summary strip on 30 Aug, where the number and its
    // unit are separate elements, so a contiguous /2 reps/ matcher no longer sees
    // it. Asserting on the strip rather than loosening the matcher: the point of
    // the test is that the count is on screen, and it should fail if it is not.
    show(okResult)
    const stats = document.querySelector('.stats')
    expect(stats).toHaveTextContent('2')
    expect(stats).toHaveTextContent('reps')
    expect(screen.getByAltText('worst form frame')).toBeInTheDocument()
    expect(screen.getByAltText('best form frame')).toBeInTheDocument()
  })

  it('states the clip length and the side analysed', () => {
    show(okResult)
    const stats = document.querySelector('.stats')
    // 608 frames at 30 fps
    expect(stats).toHaveTextContent('20.3s')
    expect(stats).toHaveTextContent('left')
  })

  it('names both arms for an exercise scored on both', () => {
    // the pull-up is filmed front-on and scores the two arms together, so a chip
    // naming one side would contradict the coaching under it
    show({ ...okResult, exercise_type: 'pullup', side: 'both' })
    const stats = document.querySelector('.stats')
    expect(stats).toHaveTextContent('both')
    expect(stats).toHaveTextContent('arms analysed')
    expect(stats).not.toHaveTextContent('side analysed')
  })

  it('reports flagged or clean from the same flag the key frame uses', () => {
    // the strip must not disagree with the caption beside it: both read
    // key_frames.worst.highlighted, and okResult has no flagged worst frame
    show(okResult)
    expect(document.querySelector('.stat-clean')).toBeInTheDocument()
    expect(document.querySelector('.stat-flagged')).not.toBeInTheDocument()
  })

  it('renders the five-part report in order', () => {
    show(okResult)
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
    // it is deterministic, never LLM-written — if it were folded
    // into the report body that distinction would be invisible to the user
    const { container } = show(okResult)
    const tip = container.querySelector('.filming-tip')
    expect(tip).toBeInTheDocument()
    expect(tip).toHaveTextContent(/film a set from the front/i)
  })
})

describe.each([
  ['no_reps', noRepsResult, /No complete rep found/i,
   /Re-record with the framing above/i],
  ['low_detection', lowDetectionResult, /Tracking was too unreliable/i,
   /Re-record with the framing above/i],
  ['rotated', rotatedResult, /decodes sideways/i,
   /Re-upload the original file/i],
])('a clip that could not be analysed (%s)', (_label, result, heading, nextStep) => {
  it('leads with an explicit banner saying so', () => {
    show(result)
    // by role, because the banner wording and the report's primary_issue say
    // similar things — it's the *heading* that has to be there
    expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument()
  })

  it('renders NO "what you did well" section at all', () => {
    // not an empty list, not a placeholder — the heading must be absent
    show(result)
    expect(screen.queryByText('What you did well')).not.toBeInTheDocument()
  })

  it('does not show a rep count', () => {
    show(result)
    expect(screen.queryByText(/\d+ reps?/)).not.toBeInTheDocument()
  })

  it('does not show key frames', () => {
    show(result)
    expect(screen.queryByText('Key moments')).not.toBeInTheDocument()
  })

  it('uses the failure card treatment, not the neutral one', () => {
    const { container } = show(result)
    expect(container.querySelector('.banner-bad')).toBeInTheDocument()
  })

  it('still explains what to do next', () => {
    show(result)
    expect(screen.getAllByText(nextStep).length).toBeGreaterThan(0)
  })
})

describe('warnings', () => {
  it('are shown even when the analysis itself succeeded', () => {
    // a voice note that failed to transcribe on an otherwise fine run is exactly
    // where silently dropping the warning would be dishonest
    show(okWithWarning)
    expect(screen.getByText(/voice note couldn't be transcribed/i)).toBeInTheDocument()
  })

  it('say what happened without quoting the provider at the user', () => {
    // this carried the raw exception, so an invalid key put "Groq transcription
    // error 401: {...}" in the banner
    show(okWithWarning)
    const banner = document.querySelector('.banner-warn')
    expect(banner?.textContent).not.toMatch(/401|Groq|Traceback/)
  })

  it('produce no banner at all when there are none', () => {
    const { container } = show(okResult)
    expect(container.querySelector('.banner-warn')).not.toBeInTheDocument()
  })
})

describe('provenance', () => {
  it('says when the wording came from the language model', () => {
    show(okResult)
    expect(screen.getByText(/it does not decide them/i)).toBeInTheDocument()
  })

  it('tells the user when the model was unavailable and this is the fallback', () => {
    show(fallbackReportResult)
    expect(screen.getByText(/language model was unavailable/i)).toBeInTheDocument()
  })
})

describe('the feedback-form link', () => {
  // Round-scoped: the server sends a URL only while a testing round is running,
  // and the link joins the response to this exact analysis via the job id.
  it('appears when a form URL was served, with the job id filled in', () => {
    show(okResult, 'https://example.test/form?entry.7={job_id}')
    const link = screen.getByRole('link', { name: /open the feedback form/i })
    expect(link).toHaveAttribute('href',
      'https://example.test/form?entry.7=squat_20260728_234148')
  })

  it('fills the percent-encoded placeholder Google\'s prefill generator makes', () => {
    show(okResult, 'https://example.test/form?entry.7=%7Bjob_id%7D')
    const link = screen.getByRole('link', { name: /open the feedback form/i })
    expect(link.getAttribute('href')).toContain('squat_20260728_234148')
  })

  it('does not exist outside a testing round', () => {
    show(okResult, null)
    expect(screen.queryByText(/feedback form/i)).not.toBeInTheDocument()
  })
})

describe('deleting an analysis now', () => {
  // The consent copy promises this, so it has to be a working button. The
  // endpoint is WP-07's DELETE /jobs/{id}; this screen is the only caller.
  afterEach(() => vi.unstubAllGlobals())

  function stubDelete(response: Partial<Response>) {
    const spy = vi.fn().mockResolvedValue({ ok: true, ...response })
    vi.stubGlobal('fetch', spy)
    return spy
  }

  it('asks before doing anything irreversible', () => {
    const spy = stubDelete({})
    show(okResult)
    fireEvent.click(screen.getByText(/delete this analysis/i))
    expect(screen.getByText(/can't be undone/i)).toBeInTheDocument()
    expect(spy).not.toHaveBeenCalled()
  })

  it('sends the DELETE and then says everything is gone', async () => {
    const spy = stubDelete({})
    show(okResult)
    fireEvent.click(screen.getByText(/delete this analysis/i))
    fireEvent.click(screen.getByText(/yes, delete it/i))

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Deleted' })).toBeInTheDocument())
    expect(spy).toHaveBeenCalledWith('/jobs/squat_20260728_234148',
                                     { method: 'DELETE' })
    // the report is no longer on screen — we just told them it is gone
    expect(screen.queryByText('What you did well')).not.toBeInTheDocument()
  })

  it('keeps the report and says so when the delete fails', async () => {
    stubDelete({ ok: false, status: 500 })
    show(okResult)
    fireEvent.click(screen.getByText(/delete this analysis/i))
    fireEvent.click(screen.getByText(/yes, delete it/i))

    await waitFor(() =>
      expect(screen.getByText(/didn't go through/i)).toBeInTheDocument())
    expect(screen.getByText('What you did well')).toBeInTheDocument()
  })

  it('can be backed out of', () => {
    stubDelete({})
    show(okResult)
    fireEvent.click(screen.getByText(/delete this analysis/i))
    fireEvent.click(screen.getByText('Keep it'))
    expect(screen.queryByText(/can't be undone/i)).not.toBeInTheDocument()
  })
})

describe('a question asked out loud', () => {
  const asked = 'Did I go all the way down?'
  const answered: AnalyzeResponse = {
    ...okResult,
    voice_transcript: asked,
    coaching_report: {
      ...okResult.coaching_report!,
      answer_to_question: 'Not on the last two reps.',
    },
  }

  it('puts the question and its answer in one card, not two', () => {
    show(answered)

    const headings = screen.getAllByRole('heading', { level: 3 })
      .filter((h) => h.textContent === 'What you asked')
    expect(headings).toHaveLength(1)

    // both halves live inside that one card
    const card = headings[0].closest('section')!
    expect(card).toHaveTextContent(asked)
    expect(card).toHaveTextContent('Not on the last two reps.')

    // the answer has its own cue -- nothing distinguished it from the question
    // before this, and a reader scrolling past several look-alike cards had no
    // way to tell the second paragraph was the reply
    expect(card).toHaveTextContent(/answer/i)
  })

  it('keeps the quote on its own when nothing answered it', () => {
    // okResult carries a transcript and no answer_to_question, which is what a
    // run that transcribed but could not be coached looks like
    show(okResult)

    const headings = screen.getAllByRole('heading', { level: 3 })
      .filter((h) => h.textContent === 'What you asked')
    expect(headings).toHaveLength(1)
    expect(headings[0].closest('section')).toHaveTextContent(/going deep enough/)
  })
})
