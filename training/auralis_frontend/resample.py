from math import gcd
import numpy as np


def _kaiser_sinc(cutoff, half_len, beta=8.6):
    n = np.arange(-half_len, half_len + 1)
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(2 * half_len + 1, beta)
    return h / h.sum()


def resample(x, sr_in, sr_out, half_len=32):
    """Rational resampling: zero-stuff by `up`, low-pass with a Kaiser-windowed sinc, keep every `down`-th sample."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float64)
    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    cutoff = 0.5 / max(up, down)              # cycles per sample at the upsampled rate
    h = _kaiser_sinc(cutoff, half_len * max(up, down)) * up
    stuffed = np.zeros(len(x) * up)
    stuffed[::up] = x
    y = np.convolve(stuffed, h, mode="full")
    delay = (len(h) - 1) // 2
    y = y[delay : delay + len(stuffed)]
    return y[::down]
