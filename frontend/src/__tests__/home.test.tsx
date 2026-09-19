/**
 * The home screen's two moving parts: the list the catalogue fills, and the
 * mark the screen's opening animation draws.
 *
 * The list arrives from the server a moment after the screen does, which is the
 * place this screen can jump, so the stand-in rows are measured rather than
 * looked at. The mark is here because its animation depends on an attribute in
 * the markup that nothing else would miss if it went.
 */
import { readFileSync } from 'node:fs'

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import BrandMark from '../components/BrandMark'
import ExerciseSelect from '../screens/ExerciseSelect'
import type { Exercise } from '../types'

const noop = () => {}

const served: Exercise[] = [
  {
    id: 'squat', name: 'Squat', view_label: 'side-on',
    filming_guide: 'Film side-on at about hip height.',
    assesses: { sagittal: ['squat depth'], frontal: ['left/right symmetry'] },
  },
  {
    id: 'pushup', name: 'Push-up', view_label: 'side-on',
    filming_guide: 'Film side-on at about chest height.',
    assesses: { sagittal: ['depth'], frontal: [] },
  },
  {
    id: 'pullup', name: 'Pull-up', view_label: 'front-on',
    filming_guide: 'Film from in front of the bar.',
    assesses: { sagittal: [], frontal: ['rep height'] },
  },
]

describe('the exercise list', () => {
  it('renders one row per exercise the server served', () => {
    render(<ExerciseSelect exercises={served} loadFailed={false} onPick={noop} />)

    expect(screen.getAllByRole('button')).toHaveLength(3)
    expect(screen.getByRole('button', { name: /Pull-up/ })).toBeInTheDocument()
    // the camera angle belongs to the exercise, not to the app
    expect(screen.getByText('filmed front-on')).toBeInTheDocument()
  })

  it('holds the list height with three skeleton rows while it loads', () => {
    const { container } = render(
      <ExerciseSelect exercises={[]} loadFailed={false} onPick={noop} />,
    )

    expect(container.querySelectorAll('.exercise-skeleton')).toHaveLength(3)
    // still announced, just not drawn twice
    expect(screen.getByText('Loading…')).toBeInTheDocument()
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it('drops the skeletons when the catalogue failed, and says so', () => {
    const { container } = render(
      <ExerciseSelect exercises={[]} loadFailed onPick={noop} />,
    )

    expect(container.querySelectorAll('.exercise-skeleton')).toHaveLength(0)
    expect(screen.getByText(/might be out of date/i)).toBeInTheDocument()
  })
})

describe('the mark the screen draws', () => {
  // styles.css draws the outline with a dash offset of 1 to 0, which only means
  // anything because the path declares its own length as 1. Drop the attribute
  // and the CSS stays valid, the mark simply never appears — a failure with no
  // error attached to it, which is why it is asserted here.
  it('normalises its path length so the opening animation can draw it', () => {
    const { container } = render(<BrandMark />)
    const path = container.querySelector('.brand-mark path')

    expect(path).not.toBeNull()
    expect(path).toHaveAttribute('pathLength', '1')
  })
})

describe('the screen keeps its opening', () => {
  // This exists because the animation was deleted once, by a patch that replaced
  // a range of the stylesheet by index while the opening happened to sit inside
  // it. Nothing failed: the CSS stayed valid, the tests stayed green, and the
  // screen simply stopped moving. A missing animation has no error attached to
  // it, so it needs an assertion.
  // from the package root, which is where vitest runs. import.meta.url is
  // not a file: URL once the module has been transformed, so it cannot be
  // resolved against.
  const css = readFileSync('src/styles.css', 'utf8')

  it.each(['mark-draw', 'mark-head', 'rise-in', 'sheet-in'])(
    'still defines @keyframes %s',
    (name) => {
      expect(css).toContain(`@keyframes ${name}`)
    },
  )

  it('still drives the elements the screen actually renders', () => {
    const { container } = render(
      <ExerciseSelect exercises={served} loadFailed={false} onPick={noop} />,
    )

    // each of these is named in an animation rule; renaming one in the markup
    // without the stylesheet, or the reverse, leaves the rule pointing at
    // nothing and is silent in the browser
    for (const selector of ['.home-hero', '.brand-mark', '.home-headline',
                            '.home-subline', '.home-sheet']) {
      expect(container.querySelector(selector), selector).not.toBeNull()
      expect(css, selector).toContain(selector)
    }
  })
})
