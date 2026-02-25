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

# ---- old data generation functions ---
def cleanupsamples(samples, nobj, precision=1):
    """Clean up samples by rounding and removing duplicates."""
    samples = np.round(samples, precision)
    c, i = np.unique(samples[:, :nobj], axis=0, return_index=True)
    newsamples = samples[i, :]  # note - these have been sorted into increasing magnitude
    if precision == 0:
        newsamples = np.array(newsamples, dtype=np.int32)
    return newsamples

# def generate_example_data(r, shape, scale, n_items=100, seed=1124):
#     r = make_pos_def(r)
#     item_rng = random.default_rng(seed=seed)
#     items = gamma_GC(r, n_items, shape, scale, rng=item_rng)
#     items = cleanupsamples(items, nobj=3, precision=0)
    
#     return items

# biased sampling?
# def generate_example_data(r, shape, scale, n_items=100, seed=1124):
#     r = make_pos_def(r)
#     item_rng = random.default_rng(seed=seed)
#     items = gamma_GC(r, n_items*2, shape, scale, rng=item_rng)
#     items = cleanupsamples(items, nobj=3, precision=0)
#     selected_idx = item_rng.choice(items.shape[0], size=n_items, replace=False)
#     items = np.unique(items[selected_idx], axis=0) # make sure to obtain n_items unique items
#     print(f"Number of items: {items.shape[0]}")
#     return items

def generate_example_data(r, shape, scale, n_items=100, seed=1124, precision = 0):
    #r = make_pos_def(r)
    item_rng = random.default_rng(seed=seed)
    
    batch = max(5, n_items // 10)
    uniq = set()
    items = []
    while len(items) < n_items:
        new = gamma_GC(r, batch, shape, scale, rng=item_rng)
        new = cleanupsamples(new, nobj=3, precision=precision)
        for item in new:
            key = tuple(item)
            if key not in uniq:
                uniq.add(key)
                items.append(item)
                if len(items) == n_items:
                    break
    return np.unique(np.array(items), axis=0), r # here np.unique is used for sorting


# --- organize eda results ---
def organize_results(results):
    js_div_list = results['js_div_list']
    distribution_table = results['distribution_table']
    pareto_indices_table = []
    pareto_front_table = []
    for j in range(len(results['pareto_front_table'])):
        pareto_indices_table.append(np.unique(np.sort(results['pareto_indices_table'][j],axis = 1),axis =0))
        pareto_front_table.append(np.unique(results['pareto_front_table'][j], axis=0))
    return js_div_list, distribution_table, pareto_indices_table, pareto_front_table




