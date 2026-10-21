#!/usr/bin/env python3
"""Approximate Pareto fronts for a constrained multiobjective 0-1 knapsack.

All objectives are maximized, subject to:
    costs @ x <= budget
    sum(x) == n_selected  if exactly=True
    sum(x) <= n_selected  if exactly=False
    x_i in {0, 1}

Algorithms: NSGA-II, NSGA-III, SPEA2. Uses pymoo's native constraint
handling (no repair operators). Install: pip install 'pymoo>=0.6.2' numpy
"""

from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


class MultiobjectiveKnapsack(ElementwiseProblem):
    """Binary knapsack with explicit cost and item-count constraints."""

    def __init__(self, values, costs, budget, n_selected, exactly=True):
        self.values = np.asarray(values, dtype=float)
        self.costs = np.asarray(costs, dtype=float)
        self.budget = float(budget)
        self.n_selected = int(n_selected)
        self.exactly = bool(exactly)

        if self.values.ndim != 2 or self.values.shape[1] < 2:
            raise ValueError("values must have shape (n_items, n_objectives >= 2)")
        n_items = self.values.shape[0]
        if self.costs.shape != (n_items,):
            raise ValueError("costs must have shape (n_items,)")
        if not np.isfinite(self.values).all() or not np.isfinite(self.costs).all():
            raise ValueError("values and costs must be finite")
        if np.any(self.costs < 0) or not np.isfinite(self.budget) or self.budget < 0:
            raise ValueError("costs and budget must be nonnegative and finite")
        if not 0 <= self.n_selected <= n_items:
            raise ValueError("n_selected must be between 0 and n_items")
        if (self.exactly and
                np.sort(self.costs)[:self.n_selected].sum() > self.budget + 1e-9):
            raise ValueError("Infeasible: cheapest n_selected items exceed budget")

        # pymoo requires G <= 0 for inequalities and H == 0 for equalities.
        super().__init__(
            n_var=n_items,
            n_obj=self.values.shape[1],
            n_ieq_constr=1 if self.exactly else 2,
            n_eq_constr=1 if self.exactly else 0,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=bool).astype(float)
        cost = x @ self.costs
        count = x.sum()

        # pymoo minimizes; negate our maximization objectives.
        out["F"] = -(x @ self.values)

        if self.exactly:
            out["G"] = np.array([cost - self.budget])
            out["H"] = np.array([count - self.n_selected])
        else:
            out["G"] = np.array([
                cost - self.budget,
                count - self.n_selected,
            ])


def make_algorithm(name, n_obj, pop_size, seed):
    """Use the same binary variation operators for all three methods."""
    common = dict(
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )

    if name == "nsga2":
        return NSGA2(pop_size=pop_size, **common)
    if name == "spea2":
        return SPEA2(pop_size=pop_size, **common)
    if name == "nsga3":
        if n_obj == 2:
            # Exact count of evenly spaced directions, cheap for 2 objectives.
            ref_dirs = get_reference_directions(
                "das-dennis", n_obj, n_partitions=pop_size - 1
            )
        else:
            # Works for arbitrary pop_size; may be slower for large pop_size.
            ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=seed)
        return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, **common)

    raise ValueError("algorithm must be 'nsga2', 'nsga3', or 'spea2'")


def solve_mokp(values, costs, budget, n_selected, *, algorithm="nsga2",
               exactly=True, pop_size=120, n_gen=200, seed=1, verbose=False):
    """Return X, F, C, n_eval for the feasible approximate Pareto front.

    X has shape (n_found, n_items), F has (n_found, n_obj), and C has
    (n_found,). F uses MAXIMIZATION values. n_found may be less than pop_size.
    """
    if pop_size < 4 or n_gen < 1:
        raise ValueError("pop_size must be >= 4 and n_gen must be >= 1")

    problem = MultiobjectiveKnapsack(values, costs, budget, n_selected, exactly)
    optimizer = make_algorithm(algorithm, problem.n_obj, pop_size, seed)
    res = minimize(problem, optimizer, ("n_gen", n_gen), seed=seed, verbose=verbose)
    n_eval = int(res.algorithm.evaluator.n_eval)

    # pymoo already returns the feasible nondominated set in res.X / res.F.
    # No manual population filtering or nondominated sorting is necessary.
    if res.X is None:
        X = np.empty((0, problem.n_var), dtype=np.int8)
        F = np.empty((0, problem.n_obj), dtype=float)
        C = np.empty(0, dtype=float)
    else:
        X = np.atleast_2d(np.asarray(res.X, dtype=np.int8))
        F = -np.atleast_2d(np.asarray(res.F, dtype=float))
        C = X @ problem.costs

    return X, F, C, n_eval


if __name__ == "__main__":
    # Problem configuration
    n_obj = 3
    n_items = 20
    n_selected = 5
    rho = 0.3
    exactly = False

    # Load the instance (columns: objective values, followed by item cost).
    input_file = f"problems/items_{n_obj}obj_{n_items}items_{rho}rho_allneg.npz"
    with np.load(input_file) as data:
        values = data["items"][:, :n_obj]
        costs = data["items"][:, n_obj]
        budget = float(np.asarray(data["capacity"]).item())
    print(f"Capacity: {budget:g}")

    # Algorithm configuration
    algorithms = ["nsga2", "nsga3", "spea2"]
    pop_size = 1000
    n_gen = 200
    seed = 1123

    # Separate output directories for equality and upper-bound formulations.
    folder = "exact" if exactly else "upper_bound"
    output_dir = Path("results") / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in algorithms:
        X, F, C, n_eval = solve_mokp(
            values, costs, budget, n_selected,
            algorithm=name, exactly=exactly,
            pop_size=pop_size, n_gen=n_gen, seed=seed,
        )

        print(f"\n{name.upper()}: {len(X)} feasible nondominated selections; "
              f"evaluations: {n_eval}")
        for i, (x, f, cost) in enumerate(zip(X, F, C), start=1):
            print(f"{i:2d}. Items: {np.flatnonzero(x).tolist()}; "
                  f"cost: {cost:g}; objectives: {f.tolist()}")

        filename = output_dir / f"{name}_{n_obj}obj_{n_items}items_{rho}rho_allneg.npz"
        np.savez(filename, X=X, F=F, C=C, n_eval=n_eval)
        print(f"Saved to: {filename}")
        if len(X) == 0:
            print("Warning: no feasible solution found; try more generations "
                  "or a different initialization.")
