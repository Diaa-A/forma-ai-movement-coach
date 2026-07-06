"""One Euro filter — Casiez, Roussel, Vogel (SIGCHI 2012).

Adaptive low-pass smoothing for noisy time series. Cuts more aggressively when the
signal is slow (kills jitter) and lets fast transitions through. Two tunables:
`min_cutoff` (lower = smoother) and `beta` (higher = more responsive to speed).
"""
import math
import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x, t):
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x

        dt = t - self._t_prev
        if dt <= 0:
            # duplicate timestamp - just return the previous smoothed value
            return self._x_prev

        # derivative, smoothed with its own low-pass
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # adapt cutoff to signal speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


def smooth_series(values, timestamps, min_cutoff=1.0, beta=0.007):
    """Apply One Euro to a 1-D series. NaN inputs are passed through unchanged
    (we don't want the filter to drift on missing detections)."""
    f = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    out = np.empty(len(values), dtype=np.float64)
    for i, (v, t) in enumerate(zip(values, timestamps)):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out[i] = np.nan
            continue
        out[i] = f(float(v), float(t))
    return out
