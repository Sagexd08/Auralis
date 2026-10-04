import numpy as np

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

def mel_filterbank(sr=16000, n_fft=512, n_mels=80, fmin=0.0, fmax=None):
    fmax = sr / 2 if fmax is None else fmax
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb
