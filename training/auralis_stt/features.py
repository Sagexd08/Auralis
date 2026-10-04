from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from auralis_frontend import get_window, mel_filterbank


class LogMel(nn.Module):
    def __init__(self, sr=16000, n_fft=512, win_length=400, hop=160, n_mels=80, eps=1e-10):
        super().__init__()
        self.n_fft, self.hop, self.eps = n_fft, hop, eps
        win = np.zeros(n_fft)
        off = (n_fft - win_length) // 2
        win[off:off + win_length] = get_window("hann", win_length, periodic=True)
        self.register_buffer("window", torch.tensor(win, dtype=torch.float32))
        self.register_buffer("fb", torch.tensor(mel_filterbank(sr, n_fft, n_mels), dtype=torch.float32))

    def num_frames(self, n_samples: int) -> int:
        return 1 + n_samples // self.hop

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wave, self.n_fft, self.hop, self.n_fft, self.window, center=True, pad_mode="reflect", return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2
        return torch.log(torch.matmul(self.fb, power) + self.eps)


def normalize_features(feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(feats)
    for i, n in enumerate(lengths.tolist()):
        x = feats[i, :, :n]
        out[i, :, :n] = (x - x.mean()) / (x.std() + 1e-5)
    return out
