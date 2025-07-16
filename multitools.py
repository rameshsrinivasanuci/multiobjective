import numpy as np
from scipy import stats

def generalized_dirichlet(n_samples, k, alpha, beta):
    rng = np.random.default_rng(seed = 23)
    samples = np.zeros((n_samples, k))
    betas = rng.beta(alpha, beta, size=(n_samples, k))
    for i in range(n_samples):
        q = 0
        for j in range(k):
            if j == k - 1:  # Last component is 1 - sum of previous
                samples[i, j] = 1 - q
            else:
                samples[i, j] = betas[i, j] * (1 - q)
                q += samples[i, j]
    return samples

def gamma_GC(r, n, shape,scale):
    mean = np.zeros(np.shape(r)[0])
    z = np.random.multivariate_normal(mean, r, size=n)
    u = stats.norm.cdf(z)
    data = stats.gamma.ppf(u, a=shape, scale=scale)
    return data