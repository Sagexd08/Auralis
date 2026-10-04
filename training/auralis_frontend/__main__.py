import sys
import numpy as np
from . import read_wav, resample, log_mel

path, out = sys.argv[1], sys.argv[2]
x, sr = read_wav(path)
x = resample(x, sr, 16000)
m = log_mel(x)
np.save(out, m)
print(f"{path}: {len(x) / 16000:.2f} s -> log-mel {m.shape}")
if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imshow(m, origin="lower", aspect="auto"); plt.colorbar(); plt.savefig(sys.argv[sys.argv.index("--plot") + 1])
