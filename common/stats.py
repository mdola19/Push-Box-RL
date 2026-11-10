import numpy as np

def bootstrap_ci(values, n_resamples=1000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    n_seeds = values.shape[0]
    idx = rng.integers(0, n_seeds, size=(n_resamples, n_seeds))
    means = values[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    low = np.percentile(means, 100 * alpha, axis=0)
    high = np.percentile(means, 100 * (1 - alpha), axis=0)
    return low, high
