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
def make_pos_def(corr):
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals[eigvals < 1e-8] = 1e-8  
    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # normalize diagonal to 1
    d = np.sqrt(np.diag(corr_pd))
    corr_pd = corr_pd / d[:, None] / d[None, :]
    return corr_pd
def gamma_GC(R, n, shape,scale,rng=None):
    R = make_pos_def(R)
    mean = np.zeros(np.shape(R)[0])
    if rng is None:
        z = np.random.multivariate_normal(mean, R, size=n)
    else:
        z = rng.multivariate_normal(mean, R, size=n)
    u = stats.norm.cdf(z)
    data = np.zeros_like(u)
    for i in range(u.shape[1]):
        data[:, i] = stats.gamma.ppf(u[:, i], a=shape[i], scale=scale[i])
    return data




