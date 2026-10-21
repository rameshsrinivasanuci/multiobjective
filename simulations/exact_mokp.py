"""Exact Pareto enumeration for a multiobjective 0-1 knapsack problem.

All objectives are maximized, total cost cannot exceed budget, and the
number of selected items is either exactly k or at most k.

Requires NumPy. Suitable for small problems because enumeration is exponential.
"""

from itertools import combinations
import numpy as np


def dominates(a, b):
    """True if objective vector a Pareto-dominates b (maximization)."""
    return np.all(a >= b) and np.any(a > b)


def exact_pareto_knapsack(values, costs, budget, n_selected, exactly=True):
    """Return all Pareto-optimal feasible solutions as (X, F, total_costs).

    Parameters
    ----------
    values : (n_items, n_objectives) array
        Each item's contribution to each maximized objective.
    costs : (n_items,) array
        Nonnegative cost of each item.
    budget : float
        Maximum allowable total cost (inclusive).
    n_selected : int
        Required number of items (or upper bound if exactly=False).
    exactly : bool
        If True, select exactly n_selected items; otherwise select at most.

    Returns
    -------
    X : (n_pareto, n_items) int array
        Binary selection vectors.
    F : (n_pareto, n_objectives) float array
        Objective vectors (larger is better).
    total_costs : (n_pareto,) float array
        Total cost of each solution.

    Distinct selections with identical objective vectors are both retained.
    """
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)

    if values.ndim != 2:
        raise ValueError("values must have shape (n_items, n_objectives)")
    n_items, n_objectives = values.shape
    if costs.shape != (n_items,):
        raise ValueError("costs must have shape (n_items,)")
    if not np.isfinite(values).all() or not np.isfinite(costs).all():
        raise ValueError("values and costs must contain only finite numbers")
    if (costs < 0).any() or not np.isfinite(budget) or budget < 0:
        raise ValueError("Costs and budget must be nonnegative and finite")
    if not isinstance(n_selected, (int, np.integer)) or not 0 <= n_selected <= n_items:
        raise ValueError("n_selected must be an integer between 0 and n_items")

    # Entries: (selected_indices, cost, objective_vector)
    pareto = []
    counts = [n_selected] if exactly else range(n_selected + 1)

    for k in counts:
        for indices in combinations(range(n_items), k):
            selected = list(indices)
            total_cost = float(costs[selected].sum())
            if total_cost > budget:
                continue

            f = values[selected].sum(axis=0)

            # Skip candidates dominated by an existing Pareto solution.
            if any(dominates(old_f, f) for _, _, old_f in pareto):
                continue

            # The new candidate may dominate some existing solutions.
            pareto = [p for p in pareto if not dominates(f, p[2])]
            pareto.append((indices, total_cost, f))

    # Stable, human-readable order: first objective descending.
    pareto.sort(key=lambda p: -p[2][0])

    X = np.zeros((len(pareto), n_items), dtype=int)
    F = np.empty((len(pareto), n_objectives), dtype=float)
    total_costs = np.empty(len(pareto), dtype=float)
    for i, (indices, cost, f) in enumerate(pareto):
        X[i, list(indices)] = 1
        F[i] = f
        total_costs[i] = cost

    return X, F, total_costs


if __name__ == "__main__":
    
    n_obj = 3
    n_items = 20
    n_selected = 5
    rho = 0.3
    exactly = False

    data = np.load(f'problems/items_{n_obj}obj_{n_items}items_{rho}rho_allneg.npz')
    values = data['items'][:, :n_obj]
    costs = data['items'][:, n_obj]
    budget = data['capacity']
    print(f"Capacity: {budget}")

    X, F, C = exact_pareto_knapsack(values, costs, budget, n_selected, exactly=exactly)
    print(f"Found {len(F)} Pareto-optimal selections")
    for i, (x, f, cost) in enumerate(zip(X, F, C), start=1):
        selected_items = np.flatnonzero(x).tolist()  # 0-based item IDs
        print(f"{i:2d}. Items: {selected_items}; cost: {cost:g}; objectives: {f.tolist()}")

    if exactly:
        folder = "exact"
    else:
        folder = "upper_bound"
    np.savez(f"results/{folder}/exact_front_{n_obj}obj_{n_items}items_{rho}rho_allneg.npz", X=X, F=F, C=C)
