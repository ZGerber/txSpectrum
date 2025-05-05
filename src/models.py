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


def synthetic_energy_distribution(logE_min=17.0,
                                  logE_max=21.0,
                                  gamma=2.0,
                                  total_events=int(1e7),
                                  n_bins=1000,
                                  seed=None):

    rng = np.random.default_rng(seed)

    bin_edges = np.linspace(logE_min, logE_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    E_low = 10 ** bin_edges[:-1]
    E_high = 10 ** bin_edges[1:]

    # Compute integral of E^-gamma over each bin
    if gamma == 1.0:
        bin_integrals = np.log(E_high / E_low)
    else:
        bin_integrals = (E_high ** (1 - gamma) - E_low ** (1 - gamma)) / (1 - gamma)

    # Normalize to total_events
    probabilities = bin_integrals / np.sum(bin_integrals)
    n_per_bin = rng.multinomial(total_events, probabilities)

    # Now randomly sample uniformly within each bin
    logE_samples = []
    for i in range(n_bins):
        if n_per_bin[i] > 0:
            logEs = rng.uniform(bin_edges[i], bin_edges[i + 1], n_per_bin[i])
            logE_samples.append(logEs)

    logE_samples = np.concatenate(logE_samples)
    return logE_samples, bin_edges, bin_centers
