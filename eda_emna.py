import time
import numpy as np
from numpy import random
from scipy.stats import beta, norm, multivariate_normal
from scipy.spatial.distance import jensenshannon
from numba import jit, prange
from multitools import make_pos_def


def get_objectives(problem, population):
    """Evaluate given population."""
    return -problem.evaluate(population)


# new non dominated sort algorithm compatible with numba
@jit(nopython=True)
def non_dominated_sort(objectives):
    """
    Numba-compatible version of non_dominated_sort
    """
    n_solutions = objectives.shape[0]
    max_dominated = n_solutions - 1
    dominated_matrix = np.full((n_solutions, max_dominated), -1, dtype=np.int32) # solutions dominated by p
    dominated_counts = np.zeros(n_solutions, dtype=np.int32) # number of solutions p dominates
    domination_counts = np.zeros(n_solutions, dtype=np.int32) # number of solutions that dominate p
    ranks = np.full(n_solutions, -1, dtype=np.int32)
    
    for p in range(n_solutions):
        for q in range(n_solutions):
            if p == q:
                continue
            if np.all(objectives[p, :] >= objectives[q, :]) and \
                np.any(objectives[p, :] > objectives[q, :]):
                idx = int(dominated_counts[p])
                dominated_matrix[p, idx] = q
                dominated_counts[p] += 1
            elif np.all(objectives[q, :] >= objectives[p, :]) and \
                np.any(objectives[q, :] > objectives[p, :]):
                domination_counts[p] += 1
        
        if domination_counts[p] == 0:
            ranks[p] = 0
    
    fronts_list = list()
    first_front = []
    for p in range(n_solutions):
        if ranks[p] == 0:
            first_front.append(p)
    if len(first_front) > 0:
        first_front_arr = np.array(first_front, dtype=np.int32)
        fronts_list.append(first_front_arr)

    i = 0
    while i < len(fronts_list) and len(fronts_list[i]) > 0:
        next_front = []
        for p in fronts_list[i]:
            for j in range(dominated_counts[p]):
                q = dominated_matrix[p, j]     
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        if len(next_front) > 0:
            next_front_arr = np.array(next_front, dtype=np.int32)
            fronts_list.append(next_front_arr)
        i += 1
    
    return ranks, fronts_list

@jit(nopython=True)
def non_dominated(objectives):
    n_solutions = objectives.shape[0]
    non_dominated = np.ones(n_solutions)
    for i in prange(n_solutions):
        for j in prange(n_solutions):
            if i == j:
                continue

            if np.all(objectives[j, :] >= objectives[i, :]) and \
                np.any(objectives[j, :] > objectives[i, :]):
                non_dominated[i] = 0
                break

    return non_dominated


def assign_crowding_distance(objectives):
    """Assign crowding distance to solutions."""
    distances = np.zeros(objectives.shape[0], dtype=float)
    for m in range(np.shape(objectives)[1]):
        objective = objectives[:, m]
        sort_indices = np.argsort(objective)[::-1]
        sorted_objective = objective[sort_indices]
        min_val = sorted_objective[0]
        max_val = sorted_objective[-1]
        distances[sort_indices[0]] = np.inf
        distances[sort_indices[-1]] = np.inf
        for i in range(1, np.shape(objectives)[0] - 1):
            distances[sort_indices[i]] += (sorted_objective[i + 1] - sorted_objective[i - 1]) \
                / (max_val - min_val)
    return distances


def binary_tournament_selection(population, ranks, distances, rng):
    """Perform binary tournament selection based on rank and crowding distance."""
    indices = np.arange(len(population))
    i, j = rng.choice(indices, size=2, replace=False)
    if ranks[i] < ranks[j]:
        return i
    if ranks[j] < ranks[i]:
        return j
    else:
        if distances[i] > distances[j]:
            return i
        else:
            return j


def row_diff(A, B):
    A_set = set(map(tuple, A))
    B_set = set(map(tuple, B))
    return len(A_set.symmetric_difference(B_set))


def initial_sample_population(n_var, xl, xu, pop_size, rng):
    """
    Sample initial population from uniform distribution.
    """
    population = np.zeros((pop_size, n_var), dtype=np.float64)
    for i in range(pop_size):
        population[i, :] = rng.uniform(xl, xu, n_var)
    return population


# ------- MVN as probabilistic model -------
# def fit_multivariate_normal(population):
#     """Build multivariate normal distribution to population."""
#     mu = population.mean(axis=0)
#     cov = np.cov(
#         population,
#         rowvar=False,
#         bias=True
#     )
#     return mu, cov

# def sample_population(mu, cov, pop_size, xl, xu, rng):
#     population = []

#     while len(population) < pop_size:
#         n_needed = pop_size - len(population)
#         samples = rng.multivariate_normal(
#             mu, cov,
#             size=n_needed * 2
#         )
#         mask = np.all(
#             (samples >= xl) & (samples <= xu),
#             axis=1
#         )
#         population.extend(samples[mask])

#     return np.asarray(population[:pop_size])

# def js_divergence_mvn(mu, cov, mu_updated, cov_updated, n=10_000, rng=None):
#     """Calculate Jensen-Shannon divergence between two multivariate normal distributions."""
#     p = multivariate_normal(mu, cov, allow_singular=True)
#     q = multivariate_normal(mu_updated, cov_updated, allow_singular=True)

#     x = rng.multivariate_normal(mu, cov, size=n)
#     y = rng.multivariate_normal(mu_updated, cov_updated, size=n)

#     # log_m_x = np.log(p.pdf(x) + q.pdf(x)) - np.log(2) # literal version that may cause underflow to 0
#     # log_m_y = np.log(p.pdf(y) + q.pdf(y)) - np.log(2)
#     log_m_x = np.logaddexp(p.logpdf(x), q.logpdf(x)) - np.log(2)
#     log_m_y = np.logaddexp(p.logpdf(y), q.logpdf(y)) - np.log(2)
#     js = 0.5 * np.mean(p.logpdf(x) - log_m_x) + 0.5 * np.mean(q.logpdf(y) - log_m_y)
#     return js
# -------------------------------------------


# ------- Beta-GC as probabilistic model -------
def fit_beta_GC(population, eps=1e-10):
    """Fit Gaussian copula with Beta marginals."""

    X = np.asarray(population, dtype=float)
    n, d = X.shape

    beta_params = np.zeros((d, 2))
    U = np.zeros((n, d))
    X = np.clip(X, eps, 1 - eps) # avoid exact 0 or 1 when fitting Beta distributions

    for j in range(d):
        a, b, _, _ = beta.fit(
            X[:, j],
            floc=0,
            fscale=1,
            method="MM"
        )
        beta_params[j] = [a, b]

        U[:, j] = beta.cdf(
            X[:, j],
            a,
            b,
            loc=0,
            scale=1
        )

    U = np.clip(U, eps, 1 - eps)
    Z = norm.ppf(U)
    R = np.corrcoef(Z, rowvar=False)
    
    parameters = {
        'beta_params': beta_params,
        'R': R,
    }
    return parameters

def sample_population(parameters, pop_size, rng, eps=1e-10):
    """Sample population from Gaussian corpula model with beta marginals."""
    R = make_pos_def(parameters['R'])
    d = R.shape[0]
    
    Z = rng.multivariate_normal(
        mean=np.zeros(d), 
        cov=R, 
        size=pop_size)

    U = norm.cdf(Z)
    U = np.clip(U, eps, 1 - eps)

    X = np.zeros((pop_size, d))
    for j, (a, b) in enumerate(parameters['beta_params']):
        X[:, j] = beta.ppf(
            U[:, j], 
            a, 
            b,
            loc=0,
            scale=1
        )
    
    return X

def logpdf_beta_GC(X, parameters, eps=1e-10):
    """Calculate log probability density function of Beta-GC distribution."""

    X = np.asarray(X)
    beta_params = parameters["beta_params"]
    R = make_pos_def(parameters["R"])
    n, d = X.shape
    X = np.clip(X, eps, 1 - eps)

    U = np.zeros_like(X)
    log_marginals = np.zeros(n)
    for j, (a, b) in enumerate(beta_params):
        U[:, j] = beta.cdf(
            X[:, j],
            a,
            b,
            loc=0,
            scale=1
        )

        log_marginals += beta.logpdf(
            X[:, j],
            a,
            b,
            loc=0,
            scale=1
        )
    
    U = np.clip(U, eps, 1 - eps)
    Z = norm.ppf(U)
    R_inv = np.linalg.inv(R)
    sign, log_det_R = np.linalg.slogdet(R) # np.linalg.slogdet computes sign of determinant and natural log of absolute determinant
    if sign <= 0:
        raise ValueError("R must be positive definite.")
    A = R_inv - np.eye(d)
    quadratic = np.einsum(
        "ni,ij,nj->n",
        Z,
        A,
        Z
    )
    log_copula = - 0.5 * log_det_R - 0.5 * quadratic

    return log_copula + log_marginals

def JSD(parameters, updated_parameters, rng, n=10_000, eps=1e-10):
    """Calculate Jensen-Shannon divergence between two Beta-GC distributions."""
    x = sample_population(parameters, n, rng)
    y = sample_population(updated_parameters, n, rng)

    # evaluate densities at samples from p
    log_p_x = logpdf_beta_GC(x, parameters, eps=eps)
    log_q_x = logpdf_beta_GC(x, updated_parameters, eps=eps)

    # evaluate densities at samples from q
    log_p_y = logpdf_beta_GC(y, parameters, eps=eps)
    log_q_y = logpdf_beta_GC(y, updated_parameters, eps=eps)

    log_m_x = np.logaddexp(log_p_x, log_q_x) - np.log(2)
    log_m_y = np.logaddexp(log_p_y, log_q_y) - np.log(2)
    js = 0.5 * np.mean(log_p_x - log_m_x) + 0.5 * np.mean(log_q_y - log_m_y)
    return js
# -------------------------------------------

class ContEDA:
    """
    Estimation of Distribution Algorithm for Multi-Objective Continuous Problem.
    
    Encapsulates the EDA algorithm with state management for distribution,
    population, and objectives across generations.
    """
    
    def __init__(
        self, 
        problem,
        pop_size=1000, 
        generations=100, 
        max_no_improve_gen=20, 
        max_row_diff=0.1, 
        seed=1123
    ):
        """
        Initialize EDA algorithm.
        
        Parameters:
        -----------
        problem : pymoo.problems
            Problem to solve
        n_var : int
            Number of variables
        n_obj : int
            Number of objectives
        xl : np.ndarray
            Lower bounds
        xu : np.ndarray
        pop_size : int
            Population size
        generations : int
            Maximum number of generations to run
        max_no_improve_gen : int
            Maximum number of generations without improvement
        max_row_diff : int
            Maximum number of row differences (fraction) between consecutive Pareto fronts
        seed : int
            Random seed
        """
        self.problem = problem
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        self.xl = problem.xl
        self.xu = problem.xu
        self.pop_size = pop_size
        self.generations = generations
        self.max_no_improve_gen = max_no_improve_gen
        self.max_row_diff = max_row_diff
        self.rng = random.default_rng(seed=seed)
        
        self.params = None
        self.selected_population = None
        self.selected_objectives = None
        
        self.params_table = []
        self.pareto_set_table = []
        self.pareto_front_table = []
        self.js_div_list = []
        self.converged_pf_table = []
        self.converged_ps_table = []
    
    def _generate_initial_population(self):
        """Generate initial population based on tournament selection."""
        population = initial_sample_population(self.n_var, self.xl, self.xu, self.pop_size, self.rng)
        objectives = get_objectives(self.problem, population)
        
        ranks, fronts = non_dominated_sort(objectives)
        distances_all_solutions = np.zeros(population.shape[0], dtype=float)
        for f in fronts:
            distances = assign_crowding_distance(objectives[f, :])
            distances_all_solutions[f] = distances
        
        select_indices = np.array([], dtype=int)
        while len(select_indices) < self.pop_size:
            indice = binary_tournament_selection(
                population, ranks, distances_all_solutions, self.rng
            )
            select_indices = np.concatenate([select_indices, np.array([indice])])
        
        selected_population = population[select_indices]
        selected_objectives = objectives[select_indices]
        
        params = fit_beta_GC(selected_population)
        
        return params, selected_population, selected_objectives
    
    def _update_distribution(self):
        """Update distribution and select new population."""
        population = sample_population(
            self.params, self.pop_size, self.rng, eps=1e-10
        )
        objectives = get_objectives(self.problem, population)
        
        _, fronts_current = non_dominated_sort(objectives)
        pareto_set = population[fronts_current[0]]
        
        objectives = np.vstack((self.selected_objectives, objectives))
        population = np.vstack((self.selected_population, population))
        
        ranks, fronts = non_dominated_sort(objectives)
        select_indices = np.array([], dtype=np.int32)
        for f in fronts:
            if len(select_indices) + len(f) <= self.pop_size:
                select_indices = np.concatenate([select_indices, f])
            else:
                remaining_size = self.pop_size - len(select_indices)
                f_distance = assign_crowding_distance(objectives[f, :])
                sort_indices = np.argsort(f_distance)[::-1]
                remaining = f[sort_indices[:remaining_size]]
                select_indices = np.concatenate([select_indices, remaining])
                break
        
        selected_population = population[select_indices]
        selected_objectives = objectives[select_indices]
        
        params_updated = fit_beta_GC(selected_population)
        js_div = JSD(self.params, params_updated, self.rng, n=5_000, eps=1e-10)
        
        return params_updated, selected_population, selected_objectives, pareto_set, js_div

    def _converged_pf(self):
        """Find the converged Pareto Front using non-dominated, still updating distribution."""
        population = sample_population(
            self.params, self.pop_size, self.rng, eps=1e-10
        )
        objectives = get_objectives(self.problem, population)

        pareto_set = population[non_dominated(objectives).astype(bool)]

        population = np.unique(np.vstack((self.selected_population, population)), axis=0)
        objectives = get_objectives(self.problem, population)

        nd_idx = non_dominated(objectives).astype(bool)
        selected_population = population[nd_idx]
        selected_objectives = objectives[nd_idx]

        params_updated = fit_beta_GC(selected_population)
        js_div = JSD(self.params, params_updated, self.rng, n=5_000, eps=1e-10)
        
        return params_updated, selected_population, selected_objectives, pareto_set, js_div
    
    def run(self):
        """
        Run the EDA algorithm for specified number of generations.
        
        Returns:
        --------
        dict : Dictionary containing results
            - distribution_table : List of distributions per generation
            - pareto_indices_table : List of pareto indices per generation
            - pareto_front_table : List of pareto fronts per generation
            - js_div_list : Jensen-Shannon divergence per generation
        """
        t0 = time.perf_counter()
        # Initialize
        self.params, self.selected_population, self.selected_objectives = \
            self._generate_initial_population()
        
        # Mode 1: run until distribution converges
        no_improve_gen = 0
        prev_js_div = None
        generation = 0
        while (generation < self.generations
               and no_improve_gen < self.max_no_improve_gen):
            generation += 1
            print(f"Mode 1 generation {generation} (no improve count: {no_improve_gen})")
            self.params, self.selected_population, self.selected_objectives, \
                pareto_set, js_div = self._update_distribution()

            pareto_front = get_objectives(self.problem, pareto_set)
                
            self.params_table.append(self.params.copy())
            self.pareto_set_table.append(pareto_set.copy())
            self.pareto_front_table.append(-pareto_front.copy()) # only negates the copy but not actual pareto_front
            self.js_div_list.append(js_div)
                
            if prev_js_div is not None:
                diff = prev_js_div - js_div
                if np.abs(diff) > 0.0001:
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1
            else:
                no_improve_gen = 0
            prev_js_div = js_div

        # Mode 2: run until Pareto Front converges
        no_improve_gen = 0
        counter = 0 
        prev_front_0 = None
        while (counter < self.generations
               and no_improve_gen < self.max_no_improve_gen):
            counter += 1
            print(f"Mode 2 generation {counter} (no improve count: {no_improve_gen})")
            self.params, self.selected_population, self.selected_objectives, \
                pareto_set, js_div = self._converged_pf()

            pareto_front = get_objectives(self.problem, pareto_set)
            
            self.params_table.append(self.params.copy())
            self.pareto_set_table.append(pareto_set.copy())
            self.pareto_front_table.append(-pareto_front.copy())
            self.js_div_list.append(js_div)

            front_0, unique_idx = np.unique(self.selected_objectives, axis=0, return_index=True)
            set_0 = self.selected_population[unique_idx]
            if prev_front_0 is not None:
                if row_diff(prev_front_0, front_0) <= self.max_row_diff * len(prev_front_0):
                    no_improve_gen += 1
                else:
                    no_improve_gen = 0
            else:
                no_improve_gen = 0
            
            self.converged_pf_table.append(-front_0.copy())
            self.converged_ps_table.append(set_0.copy())
            prev_front_0 = front_0
        
        elapsed = time.perf_counter() - t0
        print(f"EDA finished in {elapsed:.2f}s "
              f"(mode 1: {generation} gens, mode 2: {counter} gens)")

        return {
            'params_table': self.params_table,
            'pareto_set_table': self.pareto_set_table,
            'pareto_front_table': self.pareto_front_table,
            'js_div_list': self.js_div_list,
            'converged_pf_table': self.converged_pf_table,
            'converged_ps_table': self.converged_ps_table,
            'mode 1 generations': generation,
            'mode 2 generations': counter,
            'elapsed_sec': elapsed,
        }




