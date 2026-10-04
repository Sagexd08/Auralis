import numpy as np

def dft(x):
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    k = np.arange(n)[:, None]
    return np.exp(-2j * np.pi * k * np.arange(n)[None, :] / n) @ x

def fft(x):
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    if n & (n - 1):
        raise ValueError("fft length must be a power of two")
    if n == 1:
        return x
    even, odd = fft(x[0::2]), fft(x[1::2])
    tw = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + tw, even - tw])
