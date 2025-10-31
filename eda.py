import numpy as np
import os
from multitools import gamma_GC, make_pos_def
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import islice, combinations
from numpy import random
from scipy.spatial.distance import jensenshannon


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


def non_dominated_sort(objectives):
    """Perform non-dominated sorting on objectives."""
    n_solutions = objectives.shape[0]
    dominated_sets = [[] for _ in range(n_solutions)]  # the set of solutions that p dominates
    domination_counts = [0] * n_solutions  # the number of solutions that dominate p
    ranks = np.full(n_solutions, -1, dtype=int)
    fronts = [[]]

    for p in range(n_solutions):
        for q in range(n_solutions):
            if p == q:
                continue

            if np.all(objectives[p, :] >= objectives[q, :]) and \
                    np.any(objectives[p, :] > objectives[q, :]):
                dominated_sets[p].append(q)
            elif np.all(objectives[q, :] >= objectives[p, :]) and \
                    np.any(objectives[q, :] > objectives[p, :]):
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)
        i += 1

    dominated_sets = np.array([np.array(s, dtype=int) for s in dominated_sets], dtype=object)
    domination_counts = np.array(domination_counts, dtype=int)
    ranks = np.array(ranks, dtype=int)
    fronts = np.array([np.array(f, dtype=int) for f in fronts if len(f) > 0], dtype=object)

    return dominated_sets, domination_counts, ranks, fronts


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
    population = np.zeros((pop_size, n_selected), dtype='int8')
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
                 generations=100, seed=1123):
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
        
        _, _, ranks, fronts = non_dominated_sort(objectives)
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
        _, _, _, fronts_current = non_dominated_sort(objectives)
        pareto_indices = population[fronts_current[0]]
        
        objectives = np.vstack((self.selected_objectives, objectives))
        population = np.vstack((self.selected_population, population))
        
        _, _, ranks, fronts = non_dominated_sort(objectives)
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
        
        # Run generations
        for g in range(self.generations):
            print(f"Generation {g+1}/{self.generations}", end='\r', flush=True)
            self.distribution, self.selected_population, self.selected_objectives, \
                pareto_indices, js_div = self._update_distribution()
            
            pareto_front = np.zeros((pareto_indices.shape[0], self.items.shape[1]))
            for k in range(pareto_indices.shape[0]):
                pareto_front[k, :] = np.sum(self.items[pareto_indices[k, :], :], axis=0)
            
            self.distribution_table.append(self.distribution.copy())
            self.pareto_indices_table.append(pareto_indices.copy())
            self.pareto_front_table.append(pareto_front.copy())
            self.js_div_list.append(js_div)
        print()
        
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
        newsamples = np.array(newsamples, dtype='int8')
    return newsamples

def generate_example_data(r, shape, scale, n_items=100, seed=1124):
    r = make_pos_def(r)
    item_rng = random.default_rng(seed=seed)
    items = gamma_GC(r, n_items, shape, scale, rng=item_rng)
    items = cleanupsamples(items, nobj=3, precision=0)
    
    return items


def main():
    # Set parameters
    n_items = 100
    n_selected = 10
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
    generations = 100
    
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
        dist_df.to_csv(os.path.join(output_dir, f"distribution_table_{n_items}_{n_selected}_{generations}.csv"))

    js_div_list = results['js_div_list']
    if len(js_div_list) > 0:
        js_df = pd.DataFrame({
            'generation': np.arange(1, len(js_div_list) + 1, dtype=int),
            'js_divergence': js_div_list
        })
        js_df.to_csv(os.path.join(output_dir, f"js_div_list_{n_items}_{n_selected}_{generations}.csv"), index=False)

    pareto_indices_table = results['pareto_indices_table']
    pareto_front_table = results['pareto_front_table']
    if len(pareto_indices_table) > 0:
        np.savez_compressed(
            os.path.join(output_dir, f"pareto_indices_table_{n_items}_{n_selected}_{generations}.npz"),
            **{f"gen_{i+1}": arr for i, arr in enumerate(pareto_indices_table)}
        )
    if len(pareto_front_table) > 0:
        np.savez_compressed(
            os.path.join(output_dir, f"pareto_front_table_{n_items}_{n_selected}_{generations}.npz"),
            **{f"gen_{i+1}": arr for i, arr in enumerate(pareto_front_table)}
        )

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, generations + 1, 1), results['js_div_list'])
    plt.xlabel('Generations')
    plt.ylabel('Jensen-Shannon Divergence')
    plt.yscale('log')
    plt.title('Jensen-Shannon Divergence between successive generations')
    plt.grid(True)
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()
