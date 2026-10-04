"""Auralis audio frontend (M0): WAV -> log-mel, written from first principles in numpy.

Numeric tolerances against the reference implementations are stated in tests/test_frontend.py.
"""
from .wav import read_wav
from .resample import resample
from .windows import get_window
from .dft import dft, fft
from .stft import stft, frame_count
from .mel import hz_to_mel, mel_to_hz, mel_filterbank
from .logmel import log_mel, power_spectrogram

__all__ = ["read_wav", "resample", "get_window", "dft", "fft", "stft", "frame_count",
           "hz_to_mel", "mel_to_hz", "mel_filterbank", "log_mel", "power_spectrogram"]
