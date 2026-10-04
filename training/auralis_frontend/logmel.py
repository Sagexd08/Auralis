import numpy as np
from .stft import stft
from .mel import mel_filterbank


def power_spectrogram(x, **kw):
    return np.abs(stft(x, **kw)) ** 2


def log_mel(x, sr=16000, n_fft=512, win_length=400, hop=160, n_mels=80, eps=1e-10):
    """waveform -> STFT -> power -> mel projection (E = M S) -> log(E + eps). Shape (n_mels, frames)."""
    s = power_spectrogram(x, n_fft=n_fft, win_length=win_length, hop=hop)
    return np.log(mel_filterbank(sr, n_fft, n_mels) @ s + eps)
