import numpy as np
import os
from multitools import gamma_GC, make_pos_def
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import islice, combinations
from numpy import random
from scipy.spatial.distance import jensenshannon
from numba import jit, prange, njit
from numba.typed import List
from hdf5storage import loadmat
import pickle

def get_objectives(samples, indices, nobj):
    """Calculate objectives for given solution indices."""
    objectives = np.zeros((indices.shape[0], nobj), dtype='int32')
    for j in range(indices.shape[0]):
        objectives[j, :] = np.sum(samples[indices[j], :nobj], axis=0, dtype=np.int32)
    return objectives


def get_constraints(samples, indices, nobj, ncon):
    """Calculate constraints for given solution indices."""
    constraints = np.zeros((indices.shape[0], ncon), dtype='int32')
    for j in range(indices.shape[0]):
        constraints[j, :] = np.sum(samples[indices[j], nobj:], axis=0, dtype=np.int32)
    constraints = np.squeeze(constraints)
    return constraints

# old non dominated sort algorithm
# def non_dominated_sort(objectives):
#     """Perform non-dominated sorting on objectives."""
#     n_solutions = objectives.shape[0]
#     dominated_sets = [[] for _ in range(n_solutions)]  # the set of solutions that p dominates
#     domination_counts = [0] * n_solutions  # the number of solutions that dominate p
#     ranks = np.full(n_solutions, -1, dtype=int)
#     fronts = [[]]

#     for p in range(n_solutions):
#         for q in range(n_solutions):
#             if p == q:
#                 continue

#             if np.all(objectives[p, :] >= objectives[q, :]) and \
#                     np.any(objectives[p, :] > objectives[q, :]):
#                 dominated_sets[p].append(q)
#             elif np.all(objectives[q, :] >= objectives[p, :]) and \
#                     np.any(objectives[q, :] > objectives[p, :]):
#                 domination_counts[p] += 1

#         if domination_counts[p] == 0:
#             ranks[p] = 0
#             fronts[0].append(p)

#     i = 0
#     while i < len(fronts) and len(fronts[i]) > 0:
#         next_front = []
#         for p in fronts[i]:
#             for q in dominated_sets[p]:
#                 domination_counts[q] -= 1
#                 if domination_counts[q] == 0:
#                     ranks[q] = i + 1
#                     next_front.append(q)

#         if next_front:
#             fronts.append(next_front)
#         i += 1

#     dominated_sets = np.array([np.array(s, dtype=int) for s in dominated_sets], dtype=object) # issue here
#     domination_counts = np.array(domination_counts, dtype=int)
#     ranks = np.array(ranks, dtype=int)
#     fronts = np.array([np.array(f, dtype=int) for f in fronts if len(f) > 0], dtype=object) # issue here

#     return dominated_sets, domination_counts, ranks, fronts


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


def sample_population(samples, distribution, pop_size, n_selected, capacity, rng, p_rank):
    """
        Sample population from distribution respecting capacity constraint.
        Human input(weights) is used to bias the sampling.
    """
    pop_count = 0
    population = np.zeros((pop_size, n_selected), dtype=np.int32)
    n_items = distribution.size

    if p_rank is not None:
        distribution = distribution * p_rank
        distribution /= distribution.sum()

    while pop_count < pop_size:
        knapsack = rng.choice(n_items, n_selected, p=distribution, replace=False)
        constraint = np.sum(samples[knapsack, -1])
        if constraint <= capacity:
            population[pop_count, :] = knapsack
            pop_count += 1

    return population



class KnapsackEDA:
    """
    Estimation of Distribution Algorithm for Multi-Objective Knapsack Problem.
    
    Encapsulates the EDA algorithm with state management for distribution,
    population, and objectives across generations.
    """
    
    def __init__(self, items, capacity, 
                 n_selected, n_obj, 
                 pop_size=1000, generations=100, max_no_improve_gen=20, max_row_diff=5, seed=1123,
                 p_rank=None):
        """
        Initialize EDA algorithm.
        
        Parameters:
        -----------
        items : np.ndarray
            Array of items with shape (n_items, n_obj + n_con)
        capacity : float
            Capacity constraint
        n_selected : int
            Number of items to select per solution
        n_obj : int
            Number of objectives
        pop_size : int
            Population size
        generations : int
            Number of generations to run
        max_no_improve_gen : int
            Maximum number of generations without improvement
        max_row_diff : int
            Maximum number of row differences between consecutive Pareto fronts
        seed : int
            Random seed
        p_rank : np.ndarray
            Human input weights used to bias the sampling
        """
        self.items = items
        self.capacity = capacity
        self.n_selected = n_selected
        self.n_obj = n_obj
        self.pop_size = pop_size
        self.generations = generations
        self.max_no_improve_gen = max_no_improve_gen
        self.max_row_diff = max_row_diff
        self.rng = random.default_rng(seed=seed)
        self.p_rank = p_rank
        
        # State variables (will be initialized during run)
        self.distribution = None
        self.selected_population = None
        self.selected_objectives = None
        
        # Results storage
        self.distribution_table = []
        self.pareto_indices_table = []
        self.pareto_front_table = []
        self.js_div_list = []
        self.converged_pf_table = []
        self.converged_pf_population_table = []
    
    def _generate_initial_population(self):
        """Generate initial population based on tournament selection."""
        n_items = self.items.shape[0]
        distribution = np.ones(n_items) / n_items
        population = sample_population(
            self.items, distribution, self.pop_size, self.n_selected, 
            self.capacity, self.rng, self.p_rank
        )
        objectives = get_objectives(self.items, population, self.n_obj)
        
        # _, _, ranks, fronts = non_dominated_sort(objectives)
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
        
        distribution = np.ones(n_items)
        distribution += np.bincount(selected_population.flatten(), minlength=n_items)
        distribution /= np.sum(distribution)
        
        return distribution, selected_population, selected_objectives
    
    def _update_distribution(self):
        """Update distribution and select new population."""
        population = sample_population(
            self.items, self.distribution, self.pop_size, self.n_selected,
            self.capacity, self.rng, self.p_rank
        )
        objectives = get_objectives(self.items, population, self.n_obj)
        
        # Find current pareto front
        # _, _, _, fronts_current = non_dominated_sort(objectives)
        _, fronts_current = non_dominated_sort(objectives)
        pareto_indices = population[fronts_current[0]]
        
        objectives = np.vstack((self.selected_objectives, objectives))
        population = np.vstack((self.selected_population, population))
        
        # _, _, ranks, fronts = non_dominated_sort(objectives)
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
        
        n_items = self.items.shape[0]
        updated_distribution = np.ones(n_items)
        updated_distribution += np.bincount(selected_population.flatten(), minlength=n_items)
        updated_distribution /= np.sum(updated_distribution)
        
        self.distribution[self.distribution < 1E-08] = 1E-08
        updated_distribution[updated_distribution < 1E-08] = 1E-08
        js_div = jensenshannon(self.distribution, updated_distribution)**2
        
        return updated_distribution, selected_population, selected_objectives, pareto_indices, js_div

    def _converged_pf(self):
        """Find the converged Pareto Front using non-dominated, still updating distribution."""
        population = sample_population(
            self.items, self.distribution, self.pop_size, self.n_selected,
            self.capacity, self.rng, self.p_rank
        )
        objectives = get_objectives(self.items, population, self.n_obj)

        # find current pareto front
        pareto_indices = population[non_dominated(objectives).astype(bool)]

        population = np.unique(np.sort(np.vstack((self.selected_population, population)), axis=1), axis=0)
        objectives = get_objectives(self.items, population, self.n_obj)

        nd_idx = non_dominated(objectives).astype(bool)
        selected_population = population[nd_idx]
        selected_objectives = objectives[nd_idx]

        n_items = self.items.shape[0]
        updated_distribution = np.ones(n_items)
        updated_distribution += np.bincount(selected_population.flatten(), minlength=n_items)
        updated_distribution /= np.sum(updated_distribution)
        
        self.distribution[self.distribution < 1E-08] = 1E-08
        updated_distribution[updated_distribution < 1E-08] = 1E-08
        js_div = jensenshannon(self.distribution, updated_distribution)**2

        return updated_distribution, selected_population, selected_objectives, pareto_indices, js_div
    
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
        # Initialize
        self.distribution, self.selected_population, self.selected_objectives = \
            self._generate_initial_population()
        
        # Mode 1: run until distribution converges
        no_improve_gen = 0
        prev_js_div = None
        generation = 0
        while no_improve_gen < self.max_no_improve_gen:
            generation += 1
            # print(f"Mode 1 generation {generation} (no improve count: {no_improve_gen})")
            self.distribution, self.selected_population, self.selected_objectives, \
                pareto_indices, js_div = self._update_distribution()
            # print(f"number of front 0: {pareto_indices.shape[0]}")

            pareto_front = np.zeros((pareto_indices.shape[0], self.items.shape[1]))
            for k in range(pareto_indices.shape[0]):
                pareto_front[k, :] = np.sum(self.items[pareto_indices[k, :], :], axis=0)
                
            self.distribution_table.append(self.distribution.copy())
            self.pareto_indices_table.append(pareto_indices.copy())
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
        while no_improve_gen < 1:
            counter += 1
            # print(f"Mode 2 generation {counter} (no improve count: {no_improve_gen})")
            self.distribution, self.selected_population, self.selected_objectives, \
                pareto_indices, js_div = self._converged_pf()
            # print(f"number of front 0: {pareto_indices.shape[0]}")

            pareto_front = np.zeros((pareto_indices.shape[0], self.items.shape[1]))
            for k in range(pareto_indices.shape[0]):
                pareto_front[k, :] = np.sum(self.items[pareto_indices[k, :], :], axis=0)
            
            self.distribution_table.append(self.distribution.copy())
            self.pareto_indices_table.append(pareto_indices.copy())
            self.pareto_front_table.append(pareto_front.copy())
            self.js_div_list.append(js_div)

            front_0, unique_idx = np.unique(self.selected_objectives, axis=0, return_index=True)
            # front_0 = front_0[np.lexsort(front_0.T[::-1])]
            front_0_population = self.selected_population[unique_idx]
            if prev_front_0 is not None:
                if row_diff(prev_front_0, front_0) <= self.max_row_diff:
                    no_improve_gen += 1
                else:
                    no_improve_gen = 0
            else:
                no_improve_gen = 0
            
            self.converged_pf_table.append(front_0.copy())
            self.converged_pf_population_table.append(front_0_population.copy())
            prev_front_0 = front_0

        return {
            'pareto_indices_table': self.pareto_indices_table,
            'pareto_front_table': self.pareto_front_table,
            'js_div_list': self.js_div_list,
            'converged_pf_table': self.converged_pf_table,
            'converged_pf_population_table': self.converged_pf_population_table,
            'mode 1 generations': generation,
            'mode 2 generations': counter
        }


# obtain converged Pareto Front independently of EDA
def converged_pf_from_dist(
    distribution, items, pareto_solutions, capacity, n_selected, n_obj,f_seed = 1234, 
    sample_size=1000, max_iters=100, max_no_change=2):

    rng = np.random.default_rng(f_seed)       
 
    pareto_solutions = np.unique(np.sort(pareto_solutions, axis=1), axis=0)
    pareto_objectives = get_objectives(items, pareto_solutions, n_obj)
    no_change = 0
    counter = 0
    while no_change < max_no_change and counter < max_iters:
        new_sample = sample_population(items, distribution, sample_size, n_selected, capacity, rng)
        all_solutions = np.unique(np.sort(np.vstack((pareto_solutions, new_sample)), axis=1), axis=0)
        all_objectives = get_objectives(items, all_solutions, n_obj)
        nd_idx = non_dominated(all_objectives)
        nd_idx = nd_idx.astype(bool)
        new_pareto_solutions = all_solutions[nd_idx]
        new_pareto_objectives = all_objectives[nd_idx]
        
        if np.array_equal(np.unique(new_pareto_objectives, axis=0), np.unique(pareto_objectives, axis=0)):
            no_change += 1
        else:
            no_change = 0

        pareto_solutions, pareto_objectives = new_pareto_solutions, new_pareto_objectives
        counter += 1
        #print(f"iter {counter}: {len(pareto_solutions)}")
    
    return pareto_solutions, pareto_objectives, counter

def main():
    # Set parameters
    n_items = 60
    n_selected = 6
    n_obj = 3
    n_con = 1
    # capacity = 60
    pop_size = 1_000
    generations = 100 
    max_no_improve_gen = 5
    max_row_diff = 5  # can set to be 1.5% of the number of Pareto front (depends on number of objectives)
    
    # Generate data
    # items = generate_example_data(r, shape, scale, n_items=n_items)
    kn = loadmat('/data/knapsack/runB/kn_2_3_allneg_60_6_3.mat')
    items = kn['items'][0]
    capacity = kn['capacity']

    # human input
    # aspi_item = np.array([15, 2, 4, 9, 8])
    # item_scores = items[:, :n_obj] @ aspi_item
    # r = item_scores.argsort().argsort().astype(float)
    # s = r / (r.max() + 1e-12)
    # logits = s / 0.1
    # logits -= logits.max() 
    # p_rank = np.exp(logits)
    # p_rank /= p_rank.sum()
    p_rank = None  # set p_rank to None if no human input

    # Run EDA
    eda = KnapsackEDA(
        items=items,
        capacity=capacity,
        n_selected=n_selected,
        n_obj=n_obj,
        pop_size=pop_size,
        generations=generations,
        max_no_improve_gen=max_no_improve_gen,
        max_row_diff=max_row_diff,
        seed=1123,
        p_rank=p_rank
    )
    results = eda.run()

    # Save results
    if p_rank is not None:
        result_type = "eda_human"
    else:
        result_type = "eda"

    output_dir = "/home/tailai/multiobjective/eda_results"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{result_type}_{n_items}_{n_selected}_{n_obj}.pkl")
    if os.path.exists(file_path):
        print(f"File {file_path} already exists")
    else:
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    results = main()
