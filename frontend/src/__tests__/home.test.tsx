/**
 * The home screen's two moving parts: the list the catalogue fills, and the
 * mark the screen's opening animation draws.
 *
 * The list arrives from the server a moment after the screen does, which is the
 * place this screen can jump, so the stand-in rows are measured rather than
 * looked at. The mark is here because its animation depends on an attribute in
 * the markup that nothing else would miss if it went.
 */
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
