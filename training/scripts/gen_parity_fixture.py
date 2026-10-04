import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auralis_frontend import log_mel

OUT = Path(__file__).resolve().parents[2] / "runtime" / "core" / "tests" / "fixtures" / "logmel_parity.f64"

rng = np.random.default_rng(1004)
t = np.arange(16000) / 16000
x = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 1760 * t) + 0.05 * rng.standard_normal(16000)
x *= np.hanning(16000) ** 0.25
ref = log_mel(x)
assert ref.shape == (80, 101), ref.shape
OUT.parent.mkdir(parents=True, exist_ok=True)
np.concatenate([x, ref.ravel()]).astype("<f8").tofile(OUT)
print("wrote", OUT, ref.shape)
