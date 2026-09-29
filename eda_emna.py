import time
import numpy as np
from numpy import random
from scipy.stats import multivariate_normal
from scipy.spatial.distance import jensenshannon
from numba import jit, prange


def get_objectives(problem, population):
    """Evaluate given population."""
    return problem.evaluate(population)


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


def sample_population(mu, cov, pop_size, xl, xu, rng):
    population = []

    while len(population) < pop_size:
        n_needed = pop_size - len(population)
        samples = rng.multivariate_normal(
            mu, cov,
            size=n_needed * 2
        )
        mask = np.all(
            (samples >= xl) & (samples <= xu),
            axis=1
        )
        population.extend(samples[mask])

    return np.asarray(population[:pop_size])


def fit_multivariate_normal(population):
    """Build multivariate normal distribution to population."""
    mu = population.mean(axis=0)
    cov = np.cov(
        population,
        rowvar=False,
        bias=True
    )
    return mu, cov


def js_divergence_mvn(mu, cov, mu_updated, cov_updated, n=10_000, rng=None):
    """Calculate Jensen-Shannon divergence between two multivariate normal distributions."""
    p = multivariate_normal(mu, cov, allow_singular=True)
    q = multivariate_normal(mu_updated, cov_updated, allow_singular=True)

    x = rng.multivariate_normal(mu, cov, size=n)
    y = rng.multivariate_normal(mu_updated, cov_updated, size=n)

    # log_m_x = np.log(p.pdf(x) + q.pdf(x)) - np.log(2) # literal version that may cause underflow to 0
    # log_m_y = np.log(p.pdf(y) + q.pdf(y)) - np.log(2)
    log_m_x = np.logaddexp(p.logpdf(x), q.logpdf(x)) - np.log(2)
    log_m_y = np.logaddexp(p.logpdf(y), q.logpdf(y)) - np.log(2)
    js = 0.5 * np.mean(p.logpdf(x) - log_m_x) + 0.5 * np.mean(q.logpdf(y) - log_m_y)
    return js
    


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
        
        self.mu = None
        self.cov = None
        self.selected_population = None
        self.selected_objectives = None
        
        self.mu_table = []
        self.cov_table = []
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
        
        mu, cov = fit_multivariate_normal(selected_population)
        
        return mu, cov, selected_population, selected_objectives
    
    def _update_distribution(self):
        """Update distribution and select new population."""
        population = sample_population(
            self.mu, self.cov, self.pop_size, self.xl, self.xu, self.rng
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
        
        mu_updated, cov_updated = fit_multivariate_normal(selected_population)
        js_div = js_divergence_mvn(self.mu, self.cov, mu_updated, cov_updated, self.rng)
        
        return mu_updated, cov_updated, selected_population, selected_objectives, pareto_set, js_div

    def _converged_pf(self):
        """Find the converged Pareto Front using non-dominated, still updating distribution."""
        population = sample_population(
            self.mu, self.cov, self.pop_size, self.xl, self.xu, self.rng
        )
        objectives = get_objectives(self.problem, population)

        pareto_set = population[non_dominated(objectives).astype(bool)]

        population = np.unique(np.vstack((self.selected_population, population)), axis=0)
        objectives = get_objectives(self.problem, population)

        nd_idx = non_dominated(objectives).astype(bool)
        selected_population = population[nd_idx]
        selected_objectives = objectives[nd_idx]

        mu_updated, cov_updated = fit_multivariate_normal(selected_population)
        js_div = js_divergence_mvn(self.mu, self.cov, mu_updated, cov_updated)
        
        return mu_updated, cov_updated, selected_population, selected_objectives, pareto_set, js_div # js_div
    
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
        self.mu, self.cov, self.selected_population, self.selected_objectives = \
            self._generate_initial_population()
        
        # Mode 1: run until distribution converges
        no_improve_gen = 0
        prev_js_div = None
        generation = 0
        while (generation < self.generations
               and no_improve_gen < self.max_no_improve_gen):
            generation += 1
            print(f"Mode 1 generation {generation} (no improve count: {no_improve_gen})")
            self.mu, self.cov, self.selected_population, self.selected_objectives, \
                pareto_set, js_div = self._update_distribution()

            pareto_front = get_objectives(self.problem, pareto_set)
                
            self.mu_table.append(self.mu.copy())
            self.cov_table.append(self.cov.copy())
            self.pareto_set_table.append(pareto_set.copy())
            self.pareto_front_table.append(pareto_front.copy())
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
            self.mu, self.cov, self.selected_population, self.selected_objectives, \
                pareto_set, js_div = self._converged_pf()

            pareto_front = get_objectives(self.problem, pareto_set)
            
            self.mu_table.append(self.mu.copy())
            self.cov_table.append(self.cov.copy())
            self.pareto_set_table.append(pareto_set.copy())
            self.pareto_front_table.append(pareto_front.copy())
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
            
            self.converged_pf_table.append(front_0.copy())
            self.converged_ps_table.append(set_0.copy())
            prev_front_0 = front_0
        
        elapsed = time.perf_counter() - t0
        print(f"EDA finished in {elapsed:.2f}s "
              f"(mode 1: {generation} gens, mode 2: {counter} gens)")

        return {
            'mu_table': self.mu_table,
            'cov_table': self.cov_table,
            'pareto_set_table': self.pareto_set_table,
            'pareto_front_table': self.pareto_front_table,
            'js_div_list': self.js_div_list,
            'converged_pf_table': self.converged_pf_table,
            'converged_ps_table': self.converged_ps_table,
            'mode 1 generations': generation,
            'mode 2 generations': counter,
            'elapsed_sec': elapsed,
        }




