from __future__ import annotations
import math
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_mels: int = 80
    d_model: int = 144
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


def subsampled_length(lengths: torch.Tensor) -> torch.Tensor:
    return ((lengths - 1) // 2) // 2 + 1


class Subsampling(nn.Module):
    def __init__(self, n_mels: int, d_model: int, channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 2, 1), nn.ReLU(),
        )
        reduced = ((n_mels - 1) // 2) // 2 + 1
        self.proj = nn.Linear(channels * reduced, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2).unsqueeze(1))
        b, c, t, f = x.shape
        return self.proj(x.permute(0, 2, 1, 3).reshape(b, t, c * f))


class Feedforward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d * mult, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.gn = nn.GroupNorm(1, d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        y = self.norm(x).masked_fill(pad_mask.unsqueeze(-1), 0.0).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = self.pw2(F.silu(self.gn(self.dw(y))))
        return self.drop(y.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ff1 = Feedforward(d, cfg.ff_mult, cfg.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.conv = ConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = Feedforward(d, cfg.ff_mult, cfg.dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x, pad_mask):
        x = x + 0.5 * self.ff1(x)
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.attn_drop(y)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


def sinusoidal(length: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(length, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class AuralisSTT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = Subsampling(cfg.n_mels, cfg.d_model)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        x = self.subsample(feats)
        out_lengths = subsampled_length(lengths).clamp(max=x.shape[1])
        x = self.dropout(x * math.sqrt(self.cfg.d_model) + sinusoidal(x.shape[1], self.cfg.d_model, x.device))
        pad_mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) >= out_lengths.unsqueeze(1)
        for block in self.blocks:
            x = block(x, pad_mask)
        return F.log_softmax(self.head(x), dim=-1), out_lengths

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
