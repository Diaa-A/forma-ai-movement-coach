/**
 * Offline behaviour, and the one rule the service worker exists to respect.
 *
 * The requirement is that no connection degrades honestly rather than looking
 * broken — the app must say analysis needs a server, not accept a file and fail
 * later, and it must never present stale output as new.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import OfflineNotice from '../components/OfflineNotice'
import Capture from '../screens/Capture'
import type { Exercise } from '../types'

const squat: Exercise = {
  id: 'squat',
  name: 'Squat',
  view_label: 'side-on',
  filming_guide: 'Film side-on at about hip height.',
  assesses: { sagittal: ['squat depth'], frontal: ['left/right symmetry'] },
}

const noop = () => {}

describe('offline notice', () => {
  it('explains WHY it cannot work, not just that it cannot', () => {
    render(<OfflineNotice standalone />)
    expect(screen.getByText(/happens on the server/i)).toBeInTheDocument()
  })

  it('reassures rather than looking like a crash', () => {
    render(<OfflineNotice standalone />)
    expect(screen.getByText(/Nothing is lost/i)).toBeInTheDocument()
  })

  it('has a compact form for when there is already something on screen', () => {
    const { container } = render(<OfflineNotice />)
    expect(container.querySelector('h1')).not.toBeInTheDocument()
    expect(screen.getByText(/needs a connection/i)).toBeInTheDocument()
  })
})

describe('capture while offline', () => {
  it('disables submit and says why', () => {
    render(<Capture exercise={squat} online={false} onSubmit={noop} onBack={noop} />)
    const submit = screen.getByRole('button', { name: /needs a connection/i })
    expect(submit).toBeDisabled()
  })

  it('is enabled again when online', () => {
    render(<Capture exercise={squat} online onSubmit={noop} onBack={noop} />)
    // still disabled because no clip is chosen yet, but the label is the normal one
    expect(screen.getByRole('button', { name: /analyse my form/i })).toBeInTheDocument()
  })
})

describe('service worker caching policy', () => {
  // Read as source rather than executed — a worker needs a real SW environment,
  // and the property worth protecting is a policy decision, which is visible in
  // the source and cheap to assert here.
  const sw = readFileSync(resolve(__dirname, '../../public/sw.js'), 'utf8')

  it('never caches /results/* — those are videos of the user', () => {
    // Uploads are deleted after the retention period. Caching body footage on
    // the device would keep it past that and contradict the consent form.
    expect(sw).toMatch(/\/results\//)
    const line = sw.split('\n').find((l) => l.includes("startsWith('/results/')"))
    expect(line, 'no guard for /results/ at all').toBeDefined()
    expect(line).toMatch(/return/)
  })

  it('does not intercept non-GET, so an upload can never be replayed', () => {
    expect(sw).toMatch(/request\.method !== 'GET'/)
  })

  it('does not serve /exercises from cache', () => {
    expect(sw).toMatch(/pathname === '\/exercises'/)
  })

  it('cleans up old caches on activate so a new build cannot serve stale assets', () => {
    expect(sw).toMatch(/caches\.delete/)
  })

  it('leaves a build-id placeholder for the stamping plugin to replace', () => {
    // The version was the literal 'v1'. A browser only reinstalls a worker whose
    // file bytes changed, so an unchanging worker is never reinstalled: the
    // precached shell survives forever and the activate cleanup can never fire,
    // because the cache names it keeps are constant. Anyone who installed the
    // app to their home screen would hold that build until they deleted it —
    // which breaks the one thing user testing needs, shipping fixes between
    // Round 1 and Round 2 to people who already have it installed.
    expect(sw).toContain('__BUILD_ID__')
    expect(sw).not.toMatch(/const VERSION = 'v\d+'/)
  })

  it('derives both cache names from that version', () => {
    expect(sw).toMatch(/SHELL_CACHE = `formcoach-shell-\$\{VERSION\}`/)
    expect(sw).toMatch(/ASSET_CACHE = `formcoach-assets-\$\{VERSION\}`/)
  })
})

describe('the built service worker', () => {
  // Skipped on a clean checkout: dist/ is gitignored, so this only runs where a
  // build has actually happened. It is the half that proves the plugin ran.
  const built = resolve(__dirname, '../../dist/sw.js')
  const exists = (() => {
    try { readFileSync(built); return true } catch { return false }
  })()

  it.skipIf(!exists)('has a real build id, not the placeholder', () => {
    const out = readFileSync(built, 'utf8')
    expect(out).not.toContain('__BUILD_ID__')
    expect(out).toMatch(/const VERSION = '[a-f0-9]{12}'/)
  })
})
