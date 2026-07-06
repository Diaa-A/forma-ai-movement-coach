"""One Euro filter behaviour on synthetic series."""
import math

import numpy as np

from backend.pipeline.filter import OneEuroFilter, smooth_series

FPS = 30.0


def _timestamps(n):
    return np.arange(n) / FPS


def test_constant_signal_passes_through():
    v = np.full(60, 0.42)
    out = smooth_series(v, _timestamps(60))
    assert np.allclose(out, 0.42)


def test_first_sample_is_returned_unfiltered():
    out = smooth_series([7.0, 7.5], _timestamps(2))
    assert out[0] == 7.0


def test_nan_passes_through_without_breaking_the_filter():
    v = [1.0, 1.0, float("nan"), 1.0, 1.0]
    out = smooth_series(v, _timestamps(5))
    assert math.isnan(out[2])
    assert np.isfinite(out).sum() == 4


def test_jitter_is_reduced():
    # noisy slow sine: frame-to-frame jitter should shrink after filtering
    rng = np.random.default_rng(0)
    t = _timestamps(90)
    clean = 0.5 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
    noisy = clean + rng.normal(0, 0.02, len(t))
    out = smooth_series(noisy, t)
    assert np.diff(out).std() < np.diff(noisy).std()


def test_step_input_converges():
    v = np.concatenate([np.zeros(10), np.ones(80)])
    out = smooth_series(v, _timestamps(90))
    assert out[-1] > 0.9


def test_duplicate_timestamp_returns_previous_value():
    f = OneEuroFilter()
    assert f(1.0, 0.0) == 1.0
    assert f(5.0, 0.0) == 1.0  # dt == 0 must not divide by zero
