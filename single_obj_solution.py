"""Solve each single-objective 0-1 knapsack problem with SciPy MILP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


DEFAULT_DATA = Path(__file__).parent / "data/items/block_1_trial_1_2016.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "block_1_trial_1_2016_single_solution.csv"
SALARY_CAPS = {
    2000: 34.000, 2001: 35.500, 2002: 42.500, 2003: 40.271,
    2004: 43.870, 2005: 43.870, 2006: 49.500, 2007: 53.135,
    2008: 55.630, 2009: 58.680, 2010: 57.700, 2011: 58.044,
    2012: 58.044, 2013: 58.679, 2014: 58.679, 2015: 63.065,
    2016: 70.000, 2017: 94.143, 2018: 99.093, 2019: 101.869,
    2020: 109.140, 2021: 109.140, 2022: 112.414, 2023: 123.655,
    2024: 136.021, 2025: 140.588, 2026: 154.647,
}


def infer_salary_cap(csv_path: Path) -> float:
    """Infer the NBA season and its salary cap from the CSV filename."""
    try:
        year = int(csv_path.stem.rsplit("_", 1)[-1])
        return SALARY_CAPS[year]
    except (ValueError, KeyError) as exc:
        raise ValueError(
            "Could not infer a supported season from the filename; "
            "provide the cap with --salary-cap."
        ) from exc


def solve_knapsack(
    data: pd.DataFrame, objective: str, salary_cap: float, maximize: bool
) -> tuple[pd.DataFrame, float, float]:
    """Optimize one objective while selecting exactly 10 players under the cap."""
    objective_values = data[objective].to_numpy(dtype=float)
    salaries = data["SALARY"].to_numpy(dtype=float)
    n_players = len(data)
    n_selected = 10

    # scipy.optimize.milp minimizes c @ x, so negate values when maximizing.
    result = milp(
        c=-objective_values if maximize else objective_values,
        integrality=np.ones(n_players, dtype=int),
        bounds=Bounds(np.zeros(n_players), np.ones(n_players)),
        constraints=LinearConstraint(
            np.vstack([salaries, np.ones(n_players)]),
            lb=[-np.inf, n_selected],
            ub=[salary_cap, n_selected],
        ),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        direction = "maximizing" if maximize else "minimizing"
        raise RuntimeError(
            f"MILP failed while {direction} {objective}: {result.message}"
        )

    selected = data.loc[result.x > 0.5, ["PLAYER", objective, "SALARY"]].copy()
    selected = selected.sort_values(objective, ascending=not maximize)
    return selected, float(selected[objective].sum()), float(selected["SALARY"].sum())


def solve_all_objectives(
    csv_path: Path, salary_cap: float
) -> dict[str, dict[str, tuple[pd.DataFrame, float, float]]]:
    """Find the best and worst feasible solution for every objective column."""
    data = pd.read_csv(csv_path)
    required = {"PLAYER", "SALARY"}
    if missing := required.difference(data.columns):
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("The input CSV contains no players.")

    objective_columns = [
        column for column in data.columns if column not in {"PLAYER", "SALARY"}
    ]
    if not objective_columns:
        raise ValueError("The input CSV contains no objective columns.")

    numeric_columns = objective_columns + ["SALARY"]
    if data[numeric_columns].isna().any().any():
        raise ValueError("Objective and SALARY columns must not contain missing values.")
    if (data["SALARY"] < 0).any():
        raise ValueError("SALARY values must be nonnegative.")

    return {
        objective: {
            "best": solve_knapsack(data, objective, salary_cap, maximize=True),
            "worst": solve_knapsack(data, objective, salary_cap, maximize=False),
        }
        for objective in objective_columns
    }


def save_results(
    solutions: dict[str, dict[str, tuple[pd.DataFrame, float, float]]],
    output_path: Path,
) -> None:
    """Save totals and selected source-row indices in the requested four rows."""
    objectives = list(solutions)
    best_salaries = [round(solutions[obj]["best"][2], 3) for obj in objectives]
    worst_salaries = [round(solutions[obj]["worst"][2], 3) for obj in objectives]

    rows = [
        {
            **{obj: round(solutions[obj]["best"][1], 6) for obj in objectives},
            "SALARY": json.dumps(best_salaries),
        },
        {
            **{
                obj: json.dumps(solutions[obj]["best"][0].index.tolist())
                for obj in objectives
            },
            "SALARY": "",
        },
        {
            **{obj: round(solutions[obj]["worst"][1], 6) for obj in objectives},
            "SALARY": json.dumps(worst_salaries),
        },
        {
            **{
                obj: json.dumps(solutions[obj]["worst"][0].index.tolist())
                for obj in objectives
            },
            "SALARY": "",
        },
    ]
    pd.DataFrame(rows, columns=objectives + ["SALARY"]).to_csv(
        output_path, index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve one salary-constrained 0-1 knapsack per objective."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA,
        help=f"input CSV (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--salary-cap",
        type=float,
        help="salary cap; by default it is inferred from the filename's year",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    salary_cap = (
        args.salary_cap
        if args.salary_cap is not None
        else infer_salary_cap(args.csv_path)
    )
    if salary_cap < 0:
        raise ValueError("The salary cap must be nonnegative.")

    solutions = solve_all_objectives(args.csv_path, salary_cap)
    save_results(solutions, args.output)
    print(f"Input: {args.csv_path}")
    print(f"Salary cap: {salary_cap:.3f}")

    for objective, extremes in solutions.items():
        for label, (selected, objective_total, salary_total) in extremes.items():
            print(f"\n=== {label.title()} {objective} ===")
            print(selected.to_string(index=False))
            print(f"Players selected: {len(selected)}")
            print(f"Total {objective}: {objective_total:.6f}")
            print(f"Total SALARY: {salary_total:.3f} / {salary_cap:.3f}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
