import numpy as np


def broken_powerlaw_1break(x, A, m1, m2, logE_break):
    out = np.zeros_like(x)
    mask1 = (x < logE_break)
    mask2 = (x >= logE_break)

    out[mask1] = A * 10 ** (m1 * (x[mask1] - logE_break))
    A2 = A
    out[mask2] = A2 * 10 ** (m2 * (x[mask2] - logE_break))

    return out


def broken_powerlaw_2break(x, A1, m1, m2, m3, logE1, logE2):
    out = np.zeros_like(x)
    mask1 = (x < logE1)
    mask2 = (x >= logE1) & (x < logE2)
    mask3 = (x >= logE2)

    out[mask1] = A1 * 10 ** (m1 * (x[mask1] - logE1))
    A2 = A1
    out[mask2] = A2 * 10 ** (m2 * (x[mask2] - logE1))
    A3 = A2 * 10 ** (m2 * (logE2 - logE1))
    out[mask3] = A3 * 10 ** (m3 * (x[mask3] - logE2))

    return out
