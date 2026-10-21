#!/usr/bin/env python3
"""Multiobjective 0-1 knapsack with pymoo's NATIVE constraint handling.

Three multiobjective algorithms are supported: NSGA-II, NSGA-III, SPEA2.
No repair operator, custom feasibility-preserving representation, or penalty
objective is used. pymoo handles infeasibility with its built-in constraint
violation / feasibility-first machinery.

Maximize each column of values.T @ x, subject to
    costs @ x <= budget
    sum(x) == n_selected  (exactly=True), OR
    sum(x) <= n_selected  (exactly=False)
    x_i in {0, 1}.

Install: pip install 'pymoo>=0.6.2' numpy

Quick start (from a Python session):
    from pymoo_mokp_native_constraints import solve_mokp
    X, F, C = solve_mokp(values, costs, budget=50, n_selected=10,
                         algorithm='nsga2', exactly=True)

Command-line example:
    python pymoo_mokp_native_constraints.py

For your own problem, create an .npz with keys 'values' (n_items, n_obj)
 and 'costs' (n_items,), then call:
    python pymoo_mokp_native_constraints.py --input instance.npz \
      --budget 50 --n-selected 10 --algorithms nsga2 nsga3 spea2
"""

from __future__ import annotations

import argparse
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
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


class MultiobjectiveKnapsack(ElementwiseProblem):
    """Explicit cost and cardinality constraints registered with pymoo."""

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
            raise ValueError("costs and budget must be nonnegative")
        if not 0 <= self.n_selected <= n_items:
            raise ValueError("n_selected must be between 0 and n_items")
        if self.exactly and np.sort(self.costs)[:self.n_selected].sum() > self.budget + 1e-9:
            raise ValueError("Infeasible instance: even the cheapest n_selected items exceed budget")

        # In pymoo, inequality constraints G must be <= 0, and
        # equality constraints H must be == 0 (up to pymoo's numerical tolerance).
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
        total_cost = float(x @ self.costs)
        count = int(x.sum())

        # pymoo minimizes, so negate the objectives we want to MAXIMIZE.
        out["F"] = -(x @ self.values)

        # Cost (inequality): sum_i cost_i * x_i - budget <= 0.
        if self.exactly:
            out["G"] = np.array([total_cost - self.budget])
            # Cardinality (equality): sum_i x_i - k = 0.
            out["H"] = np.array([count - self.n_selected])
        else:
            # Both constraints are inequalities in the 'at most' case.
            out["G"] = np.array([
                total_cost - self.budget,
                count - self.n_selected,
            ])


def make_algorithm(name: str, n_obj: int, pop_size: int, seed: int):
    """Create a genetic multiobjective optimizer with BINARY operators."""
    common = dict(
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )
    name = name.lower().replace("-", "").replace("_", "")

    if name == "nsga2":
        return NSGA2(pop_size=pop_size, **common)
    if name == "nsga3":
        # Exactly pop_size reference directions, for any number of objectives.
        ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=seed)
        return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, **common)
    if name == "spea2":
        return SPEA2(pop_size=pop_size, **common)
    raise ValueError("algorithm must be one of: nsga2, nsga3, spea2")


def _solve_with_stats(
    values,
    costs,
    budget,
    n_selected,
    *,
    algorithm="nsga2",
    exactly=True,
    pop_size=120,
    n_gen=200,
    seed=1,
    verbose=False,
):
    if pop_size < 4 or n_gen < 1:
        raise ValueError("pop_size must be >= 4 and n_gen must be >= 1")
    problem = MultiobjectiveKnapsack(values, costs, budget, n_selected, exactly)
    optimizer = make_algorithm(algorithm, problem.n_obj, pop_size, seed)
    res = minimize(
        problem,
        optimizer,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
    )

    # res.pop is the final population; filter it explicitly to return ONLY
    # feasible non-dominated selections (rather than 'best infeasible' ones).
    pop = res.pop
    X_all = np.asarray(pop.get("X"), dtype=bool)
    total_costs = X_all.astype(float) @ problem.costs
    counts = X_all.sum(axis=1)
    feasible = total_costs <= problem.budget + 1e-9
    if exactly:
        feasible &= counts == problem.n_selected
    else:
        feasible &= counts <= problem.n_selected

    X = X_all[feasible]
    if not len(X):
        X_empty = np.empty((0, problem.n_var), dtype=np.int8)
        F_empty = np.empty((0, problem.n_obj), dtype=float)
        return X_empty, F_empty, np.empty(0), int(res.algorithm.evaluator.n_eval)

    # Multiple genotype rows can coincide after selection; remove duplicates.
    X = np.unique(X.astype(np.int8), axis=0)
    F = X.astype(float) @ problem.values     # Return positive values (MAXIMIZATION).
    nd_indices = NonDominatedSorting().do(-F, only_non_dominated_front=True)
    X, F = X[nd_indices], F[nd_indices]
    C = X.astype(float) @ problem.costs

    # Safety checks on output feasibility.
    assert np.all(C <= problem.budget + 1e-9)
    if exactly:
        assert np.all(X.sum(axis=1) == problem.n_selected)
    else:
        assert np.all(X.sum(axis=1) <= problem.n_selected)
    return X, F, C, int(res.algorithm.evaluator.n_eval)


def solve_mokp(
    values,
    costs,
    budget,
    n_selected,
    *,
    algorithm="nsga2",
    exactly=True,
    pop_size=120,
    n_gen=200,
    seed=1,
    verbose=False,
):
    """Return (X, F, C), like an exact MOKP solver, but APPROXIMATE.

    X: (number_found, n_items) binary matrix
    F: (number_found, n_obj) maximized objective vectors (not negated)
    C: (number_found,) total item costs

    If no feasible individual is found, all arrays are empty. Increase
    population/generations or relax constraints rather than treating an
    infeasible solution as valid.
    """
    X, F, C, _n_eval = _solve_with_stats(
        values, costs, budget, n_selected, algorithm=algorithm,
        exactly=exactly, pop_size=pop_size, n_gen=n_gen,
        seed=seed, verbose=verbose,
    )
    return X, F, C


if __name__ == "__main__":

    # Problem configuration
    n_obj = 3
    n_items = 20
    n_selected = 5
    rho = 0.3

    # Load problem instance
    data = np.load(
        f"problems/items_{n_obj}obj_{n_items}items_{rho}rho_allneg.npz"
    )
    values = data["items"][:, :n_obj]
    costs = data["items"][:, n_obj]
    budget = float(data["capacity"])

    print(f"Capacity: {budget}")

    # Algorithm configuration
    algorithms = ["nsga2", "nsga3", "spea2"]
    pop_size = 1000
    n_gen = 200
    seed = 1123

    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run each algorithm
    for name in algorithms:

        X, F, C, n_eval = _solve_with_stats(
            values,
            costs,
            budget,
            n_selected,
            algorithm=name,
            exactly=True,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            verbose=False,
        )

        print(f"\n{name.upper()}")
        print(f"Found {len(F)} feasible nondominated selections")
        print(f"Number of evaluations: {n_eval}")

        for i, (x, f, cost) in enumerate(zip(X, F, C), start=1):
            selected_items = np.flatnonzero(x).tolist()
            print(
                f"{i:2d}. Items: {selected_items}; "
                f"cost: {cost:g}; objectives: {f.tolist()}"
            )

        # Save results
        filename = (
            output_dir /
            f"{name}_{n_obj}obj_{n_items}items_{rho}rho_allneg.npz"
        )

        np.savez(
            filename,
            X=X,
            F=F,
            C=C,
            n_eval=n_eval,
        )

        print(f"Saved to: {filename}")
