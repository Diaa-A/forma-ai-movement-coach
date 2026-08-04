/**
 * Two things the client stopped deciding for itself.
 *
 * The upload limits used to be written out in validation.ts as a second copy of
 * the backend's constants, with a comment asking whoever changed one to change
 * the other. They had already diverged — the comment claimed the server enforced
 * a 3-45 second gate, and the server had no duration check at all, so the client
 * was describing a guarantee nobody was keeping. They come off GET /exercises
 * now, and these tests are about the checks honouring what arrives rather than
 * falling back to what they were built with.
 *
 * The error reference is the other half: a 5xx body is still never shown, but
 * the reference rides in a header, and it is the difference between a tester
 * saying "it broke" and a tester saying which run broke.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ErrorPanel from '../components/ErrorPanel'
import type { ApiError, Limits } from '../types'
import { FALLBACK, checkAudioFile, checkDuration, checkVideoFile } from '../validation'

const noop = () => {}

/** A server that is deliberately stricter than the built-in fallback, so a check
 *  reading the wrong one is visible rather than coincidentally right. */
const strict: Limits = {
  ...FALLBACK,
  max_video_bytes: 5 * 1024 * 1024,
  video_suffixes: ['.mp4'],
  audio_suffixes: ['.m4a'],
  min_seconds: 8,
  max_seconds: 20,
  ideal_min_seconds: 10,
  ideal_max_seconds: 15,
}

function fileOf(name: string, bytes: number): File {
  return new File([new Uint8Array(bytes)], name)
}

describe('limits that arrived from the server', () => {
  it('refuse a format the server does not take, even though the fallback does', () => {
    const check = checkVideoFile(fileOf('clip.mov', 10), strict)
    expect(check.ok).toBe(false)
    expect(check.message).toMatch(/\.mov/)
  })

  it('quote the server size cap rather than the built-in one', () => {
    const check = checkVideoFile(fileOf('clip.mp4', 6 * 1024 * 1024), strict)
    expect(check.ok).toBe(false)
    expect(check.message).toMatch(/5\.0 MB/)
    expect(check.message).not.toMatch(/100/)
  })

  it('use the server duration band', () => {
    // 5 s passes the fallback's 3 s floor and fails this server's 8 s floor
    expect(checkDuration(5, strict).ok).toBe(false)
    expect(checkDuration(5, FALLBACK).ok).toBe(true)
  })

  it('warn inside the ideal band without refusing', () => {
    const check = checkDuration(18, strict)
    expect(check.ok).toBe(true)
    expect(check.warning).toBe(true)
    expect(check.message).toMatch(/10-15/)
  })

  it('apply to the voice note too', () => {
    expect(checkAudioFile(fileOf('note.webm', 10), strict).ok).toBe(false)
    expect(checkAudioFile(fileOf('note.m4a', 10), strict).ok).toBe(true)
  })
})

describe('the fallback limits', () => {
  it('still check a file when the catalogue never arrived', () => {
    // offline on first load is a real state — the app must not respond by
    // waving everything through
    expect(checkVideoFile(fileOf('clip.avi', 10)).ok).toBe(false)
    expect(checkDuration(120).ok).toBe(false)
  })

  it('accept the iOS voice-note container', () => {
    // .mp4 was missing from the backend allowlist, which is why the recorder
    // renames audio/mp4 blobs. Both are fine now.
    expect(FALLBACK.audio_suffixes).toContain('.mp4')
    expect(FALLBACK.audio_suffixes).toContain('.m4a')
  })
})

describe('a server error', () => {
  const serverError: ApiError = {
    kind: 'server',
    status: 500,
    message: 'Something went wrong on our side while analysing that clip. Trying again often works.',
    reference: 'a1b2c3d4',
  }

  it('shows the reference so it can be quoted', () => {
    render(<ErrorPanel error={serverError} onRetry={noop} onRestart={noop} />)
    expect(screen.getByText('a1b2c3d4')).toBeInTheDocument()
  })

  it('says nothing about a reference when the server did not send one', () => {
    const { reference, ...withoutRef } = serverError
    void reference
    render(<ErrorPanel error={withoutRef} onRetry={noop} onRestart={noop} />)
    expect(screen.queryByText(/quote/i)).not.toBeInTheDocument()
  })

  it('gets no reference on a 4xx, where the server message is the whole story', () => {
    // 4xx wording is written for a person and passed through; there is nothing
    // to look up in a log, because nothing went wrong on our side
    const rejected: ApiError = {
      kind: 'rejected',
      status: 400,
      message: 'That clip is 300s and the limit is 45s. Trim it to a few good reps.',
    }
    render(<ErrorPanel error={rejected} onRetry={noop} onRestart={noop} />)
    expect(screen.getByText(/Trim it to a few good reps/)).toBeInTheDocument()
    expect(screen.queryByText(/quote/i)).not.toBeInTheDocument()
  })
})
