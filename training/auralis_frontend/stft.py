import numpy as np
from .dft import fft
from .windows import get_window

def frame_count(n_samples, hop, center=True):
    return 1 + n_samples // hop if center else 1 + (n_samples - 1) // hop

def stft(x, n_fft=512, win_length=400, hop=160, window="hann", center=True):
    x = np.asarray(x, dtype=np.float64)
    w = np.zeros(n_fft)
    off = (n_fft - win_length) // 2
    w[off : off + win_length] = get_window(window, win_length, periodic=True)
    if center:
        x = np.pad(x, n_fft // 2, mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for t in range(n_frames):
        out[:, t] = fft(x[t * hop : t * hop + n_fft] * w)[: n_fft // 2 + 1]
    return out
