import numpy as np
import os
from multitools import gamma_GC, make_pos_def
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import islice, combinations
from numpy import random
from scipy.spatial.distance import jensenshannon
from numba import jit
from numba.typed import List

def get_objectives(samples, indices, nobj):
    """Calculate objectives for given solution indices."""
    objectives = np.zeros((indices.shape[0], nobj), dtype='int16')
    for j in range(indices.shape[0]):
        objectives[j, :] = np.sum(samples[indices[j], :nobj], axis=0, dtype='int16')
    return objectives


def get_constraints(samples, indices, nobj, ncon):
    """Calculate constraints for given solution indices."""
    constraints = np.zeros((indices.shape[0], ncon), dtype='int16')
    for j in range(indices.shape[0]):
        constraints[j, :] = np.sum(samples[indices[j], nobj:], axis=0, dtype='int16')
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


def non_dominated(objectives):
    n_solutions = objectives.shape[0]
    non_dominated = np.ones(n_solutions, dtype=bool)
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i == j:
                continue

            if np.all(objectives[j, :] >= objectives[i, :]) and \
                np.any(objectives[j, :] > objectives[i, :]):
                non_dominated[i] = False
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


def sample_population(samples, distribution, pop_size, n_selected, capacity, rng):
    """Sample population from distribution respecting capacity constraint."""
    pop_count = 0
    population = np.zeros((pop_size, n_selected), dtype='int32')
    n_items = distribution.size

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
    
    def __init__(self, items, capacity, n_selected, n_obj, pop_size=1000, 
                 generations=100, max_no_improve_gen=20, seed=1123):
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
        seed : int
            Random seed
        """
        self.items = items
        self.capacity = capacity
        self.n_selected = n_selected
        self.n_obj = n_obj
        self.pop_size = pop_size
        self.generations = generations
        self.max_no_improve_gen = max_no_improve_gen
        self.rng = random.default_rng(seed=seed)
        
        # State variables (will be initialized during run)
        self.distribution = None
        self.selected_population = None
        self.selected_objectives = None
        
        # Results storage
        self.distribution_table = []
        self.pareto_indices_table = []
        self.pareto_front_table = []
        self.js_div_list = []
    
    def _generate_initial_population(self):
        """Generate initial population based on tournament selection."""
        n_items = self.items.shape[0]
        distribution = np.ones(n_items) / n_items
        population = sample_population(
            self.items, distribution, self.pop_size, self.n_selected, 
            self.capacity, self.rng
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
            self.capacity, self.rng
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
        select_indices = np.array([], dtype=int)
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
        
        # Run generations (fixed number of generations)
        # for g in range(self.generations):
        #     print(f"Generation {g+1}/{self.generations}")
        #     self.distribution, self.selected_population, self.selected_objectives, \
        #         pareto_indices, js_div = self._update_distribution()
            
        #     pareto_front = np.zeros((pareto_indices.shape[0], self.items.shape[1]))
        #     for k in range(pareto_indices.shape[0]):
        #         pareto_front[k, :] = np.sum(self.items[pareto_indices[k, :], :], axis=0)
            
        #     self.distribution_table.append(self.distribution.copy())
        #     self.pareto_indices_table.append(pareto_indices.copy())
        #     self.pareto_front_table.append(pareto_front.copy())
        #     self.js_div_list.append(js_div)
        # print()

        # Run generations (until convergence)
        no_improve_gen = 0
        prev_js_div = None
        generation = 0
        while no_improve_gen < self.max_no_improve_gen:
            generation += 1
            print(f"Generation {generation} (no improve count: {no_improve_gen})")
            self.distribution, self.selected_population, self.selected_objectives, \
                pareto_indices, js_div = self._update_distribution()

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

        return {
            'distribution_table': self.distribution_table,
            'pareto_indices_table': self.pareto_indices_table,
            'pareto_front_table': self.pareto_front_table,
            'js_div_list': self.js_div_list
        }


def cleanupsamples(samples, nobj, precision=1):
    """Clean up samples by rounding and removing duplicates."""
    samples = np.round(samples, precision)
    c, i = np.unique(samples[:, :nobj], axis=0, return_index=True)
    newsamples = samples[i, :]  # note - these have been sorted into increasing magnitude
    if precision == 0:
        newsamples = np.array(newsamples, dtype='int32')
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

def generate_example_data(r, shape, scale, n_items=100, seed=1124):
    r = make_pos_def(r)
    item_rng = random.default_rng(seed=seed)
    
    batch = max(5, n_items // 10)
    uniq = set()
    items = []
    while len(items) < n_items:
        new = gamma_GC(r, batch, shape, scale, rng=item_rng)
        new = cleanupsamples(new, nobj=3, precision=0)
        for item in new:
            key = tuple(item)
            if key not in uniq:
                uniq.add(key)
                items.append(item)
                if len(items) == n_items:
                    break
    return np.unique(np.array(items), axis=0) # here np.unique is used for sorting

def organize_results(results):
    js_div_list = results['js_div_list']
    distribution_table = results['distribution_table']
    pareto_indices_table = []
    pareto_front_table = []
    for j in range(len(results['pareto_front_table'])):
        pareto_indices_table.append(np.unique(np.sort(results['pareto_indices_table'][j],axis = 1),axis =0))
        pareto_front_table.append(np.unique(results['pareto_front_table'][j], axis=0))
    return js_div_list, distribution_table, pareto_indices_table, pareto_front_table

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
        new_pareto_solutions = all_solutions[nd_idx]
        new_pareto_objectives = all_objectives[nd_idx]
        
        if np.array_equal(np.unique(new_pareto_objectives, axis=0), np.unique(pareto_objectives, axis=0)):
            no_change += 1
        else:
            no_change = 0

        pareto_solutions, pareto_objectives = new_pareto_solutions, new_pareto_objectives
        counter += 1
        print(f"iter {counter}: {len(pareto_solutions)}")
    
    return pareto_solutions, pareto_objectives, counter

def main():
    # Set parameters
    n_items = 20
    n_selected = 5
    n_obj = 3
    n_con = 1
    shape = [3.0, 4.0, 2.0, 8.0]
    scale = [2.0, 3, 2, 1.0]
    r = np.array([      
        [1.0, 0.4, -0.5, 0.3],
        [0.4, 1.0, 0.5, 0.4],
        [-0.5, 0.5, 1.0, 0.2],
        [0.3, 0.4, 0.2, 1.0],
    ])
    capacity = int(shape[-1]*scale[-1]*n_selected)
    pop_size = 1000
    generations = 100 # do not matter if check convergence
    max_no_improve_gen = 20
    
    # Generate data
    items = generate_example_data(r, shape, scale, n_items=n_items)
    
    # Run EDA
    eda = KnapsackEDA(
        items=items,
        capacity=capacity,
        n_selected=n_selected,
        n_obj=n_obj,
        pop_size=pop_size,
        generations=generations,
        max_no_improve_gen=max_no_improve_gen,
        seed=1123
    )
    
    results = eda.run()

    # Print results
    print(f"Number of unique items: {items.shape[0]}")
    print(f"Final Pareto front size: {np.unique(results['pareto_front_table'][-1], axis=0).shape[0]}")
    print(f"Final Pareto front: {np.unique(results['pareto_front_table'][-1], axis=0)}")
    print(f"Final JS divergence: {results['js_div_list'][-1]:.6f}")
    
    # Save results
    output_dir = "/home/tailai/research/multiobjective/eda_results"
    os.makedirs(output_dir, exist_ok=True)

    distribution_table = results['distribution_table']
    if len(distribution_table) > 0:
        dist_df = pd.DataFrame(np.vstack(distribution_table))
        dist_df.index.name = 'generation'
        dist_df.columns = [f"item_{i}" for i in range(dist_df.shape[1])]
        dist_df.to_csv(os.path.join(output_dir, f"distribution_table_{n_items}_{n_selected}.csv"))

    js_div_list = results['js_div_list']
    if len(js_div_list) > 0:
        js_df = pd.DataFrame({
            'generation': np.arange(1, len(js_div_list) + 1, dtype=int),
            'js_divergence': js_div_list
        })
        js_df.to_csv(os.path.join(output_dir, f"js_div_list_{n_items}_{n_selected}.csv"), index=False)

    pareto_indices_table = results['pareto_indices_table']
    pareto_front_table = results['pareto_front_table']
    if len(pareto_indices_table) > 0:
        np.savez_compressed(
            os.path.join(output_dir, f"pareto_indices_table_{n_items}_{n_selected}.npz"),
            **{f"gen_{i+1}": arr for i, arr in enumerate(pareto_indices_table)}
        )
    if len(pareto_front_table) > 0:
        np.savez_compressed(
            os.path.join(output_dir, f"pareto_front_table_{n_items}_{n_selected}.npz"),
            **{f"gen_{i+1}": arr for i, arr in enumerate(pareto_front_table)}
        )

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(js_div_list) + 1, 1), js_div_list)
    plt.xlabel('Generations')
    plt.ylabel('Jensen-Shannon Divergence')
    plt.yscale('log')
    plt.title('Jensen-Shannon Divergence between successive generations')
    plt.grid(True)
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()
