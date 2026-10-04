"""M0 numeric-equivalence tests. Every stage is compared with a trusted implementation.

Documented tolerances (float64, max absolute error unless noted):
  fft vs numpy.fft ................ 1e-9
  dft vs numpy.fft ................ 1e-9
  windows vs scipy.signal ......... 1e-12
  stft vs torch.stft .............. 1e-8
  mel filterbank vs librosa ....... 1e-6   (htk=True, norm=None; librosa returns float32, observed 3e-8)
  log-mel vs torch pipeline ....... 1e-7
  resample vs scipy resample_poly . 5e-3 (different filter design; passband sine compared)
"""
import numpy as np
import pytest
import scipy.signal as ss
from auralis_frontend import (dft, fft, get_window, stft, hz_to_mel, mel_to_hz, mel_filterbank,
                              log_mel, resample, read_wav, frame_count)

rng = np.random.default_rng(0)
SR = 16000


def speechlike(n=SR):
    t = np.arange(n) / SR
    return 0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1800 * t) + 0.05 * rng.standard_normal(n)


def test_dft_hand_example():
    X = dft([1, 0, -1, 0])                      # worked by hand: [0, 2, 0, 2]
    assert np.allclose(X, [0, 2, 0, 2], atol=1e-12)


@pytest.mark.parametrize("n", [8, 64, 512])
def test_fft_and_dft_match_numpy(n):
    x = rng.standard_normal(n)
    assert np.max(np.abs(fft(x) - np.fft.fft(x))) < 1e-9
    assert np.max(np.abs(dft(x) - np.fft.fft(x))) < 1e-9


def test_fft_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fft(np.ones(12))


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "boxcar"])
@pytest.mark.parametrize("periodic", [True, False])
def test_windows_match_scipy(name, periodic):
    ref = ss.get_window(name, 400, fftbins=periodic)
    assert np.max(np.abs(get_window(name, 400, periodic) - ref)) < 1e-12


def test_stft_matches_torch():
    torch = pytest.importorskip("torch")
    x = speechlike()
    ref = torch.stft(torch.tensor(x), 512, hop_length=160, win_length=400,
                     window=torch.hann_window(400, dtype=torch.float64), center=True,
                     pad_mode="reflect", return_complex=True).numpy()
    mine = stft(x)
    assert mine.shape == ref.shape == (257, frame_count(len(x), 160))
    assert np.max(np.abs(mine - ref)) < 1e-8


def test_mel_roundtrip_and_known_values():
    f = np.array([0.0, 440.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f, atol=1e-9)
    assert abs(hz_to_mel(1000.0) - 1000.0) < 1.0   # ~1000 mel at 1 kHz by construction


def test_filterbank_properties():
    fb = mel_filterbank()
    assert fb.shape == (80, 257) and fb.min() >= 0 and fb.max() <= 1 + 1e-12
    assert (fb.sum(axis=1) > 0).all()


def test_filterbank_matches_librosa():
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(sr=SR, n_fft=512, n_mels=80, htk=True, norm=None)
    assert np.max(np.abs(mel_filterbank() - ref)) < 1e-6


def test_log_mel_matches_torch_pipeline():
    torch = pytest.importorskip("torch")
    x = speechlike()
    S = torch.stft(torch.tensor(x), 512, hop_length=160, win_length=400,
                   window=torch.hann_window(400, dtype=torch.float64), return_complex=True).abs() ** 2
    ref = torch.log(torch.tensor(mel_filterbank()) @ S + 1e-10).numpy()
    mine = log_mel(x)
    assert mine.shape == (80, 101) and np.max(np.abs(mine - ref)) < 1e-7


def test_resample_preserves_tone():
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 440 * t)
    y = resample(x, 48000, 16000)
    ref = ss.resample_poly(x, 1, 3)
    assert abs(len(y) - 16000) <= 1
    core = slice(500, 15500)
    assert np.max(np.abs(y[core] - ref[core])) < 5e-3


def test_silence_has_floor_log_mel():
    m = log_mel(np.zeros(SR))
    assert np.allclose(m, np.log(1e-10))


def test_read_wav_roundtrip(tmp_path):
    import wave
    x = (speechlike(8000) * 20000).astype("<i2")
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(x.tobytes())
    y, sr = read_wav(p)
    assert sr == SR and np.allclose(y, x / 32768.0)
