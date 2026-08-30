/**
 * The consent screen and the record it leaves.
 *
 * Every line of the consent copy is a promise the code keeps, so the two things
 * worth pinning are: the retention sentence is the served one, not a copy — and
 * without a served sentence the screen refuses to collect agreement at all; and
 * the stored record is keyed to the retention period, so changing the period
 * brings the dialogue back rather than treating old agreement as new agreement.
 */
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import Consent from '../screens/Consent'
import { consentGiven, recordConsent } from '../consent'
import { FALLBACK } from '../validation'
import type { Limits } from '../types'

const noop = () => {}

const served: Limits = {
  ...FALLBACK,
  retention_hours: 24,
  retention_note:
    'Your clip and results are deleted automatically after 24 hours.',
}

beforeEach(() => localStorage.clear())

describe('the stored record', () => {
  it('is absent until someone agrees', () => {
    expect(consentGiven(24)).toBe(false)
  })

  it('remembers agreement for the period it was shown with', () => {
    recordConsent(24)
    expect(consentGiven(24)).toBe(true)
  })

  it('does not carry over to a changed retention period', () => {
    // agreement to "deleted after 24 hours" is not agreement to any other
    // number, so the dialogue has to come back
    recordConsent(24)
    expect(consentGiven(48)).toBe(false)
  })

  it('fails safe when storage is unavailable', () => {
    // private browsing — the cost of throwing here is being asked again, which
    // is the right direction to fail in
    const broken = vi.spyOn(Storage.prototype, 'getItem')
      .mockImplementation(() => { throw new Error('denied') })
    expect(consentGiven(24)).toBe(false)
    broken.mockRestore()
  })
})

describe('the consent screen', () => {
  it('quotes the served retention sentence verbatim', () => {
    render(<Consent limits={served} onAccept={noop} onDecline={noop} />)
    expect(screen.getByText(served.retention_note)).toBeInTheDocument()
  })

  it('names the third party and the delete-now option', () => {
    render(<Consent limits={served} onAccept={noop} onDecline={noop} />)
    expect(screen.getByText(/Groq/)).toBeInTheDocument()
    expect(screen.getByText(/delete an analysis immediately/i)).toBeInTheDocument()
    expect(screen.getByText(/not medical advice/i)).toBeInTheDocument()
  })

  it('will not collect agreement without a served retention period', () => {
    // FALLBACK deliberately carries an empty retention_note: if the catalogue
    // never arrived we have not been told the period, and asking someone to
    // agree to an unstated one would be the consent form lying
    render(<Consent limits={FALLBACK} onAccept={noop} onDecline={noop} />)
    expect(screen.getByRole('button', { name: /i agree/i })).toBeDisabled()
    expect(screen.getByText(/couldn't load the data-handling details/i))
      .toBeInTheDocument()
  })

  it('continues on accept and goes back on decline', () => {
    const accepted = vi.fn()
    const declined = vi.fn()
    render(<Consent limits={served} onAccept={accepted} onDecline={declined} />)
    fireEvent.click(screen.getByRole('button', { name: /i agree/i }))
    fireEvent.click(screen.getByRole('button', { name: /not now/i }))
    expect(accepted).toHaveBeenCalledTimes(1)
    expect(declined).toHaveBeenCalledTimes(1)
  })
})
