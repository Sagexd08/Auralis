import numpy as np

def get_window(name, n, periodic=True):
    d = n if periodic else n - 1
    k = np.arange(n)
    if name in ("rect", "rectangular", "boxcar"):
        return np.ones(n)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / d)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * k / d)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(2 * np.pi * k / d) + 0.08 * np.cos(4 * np.pi * k / d)
    raise ValueError(f"unknown window: {name}")
