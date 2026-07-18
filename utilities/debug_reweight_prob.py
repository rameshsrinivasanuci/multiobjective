"""Minimal script to debug reweight_prob with a full Python debug session.

Usage:
  1. Set a breakpoint in eda_sol_reweight_noinit.py (e.g. line 343).
  2. Open Run and Debug (Ctrl+Shift+D).
  3. Select "Debug reweight_prob (init only)" and press F5.

This is more reliable than notebook Debug Cell for breakpoints in imported .py files.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import eda_sol_reweight_noinit as eda_mod


def load_items(stimuli_dir: Path, trial_id: int) -> np.ndarray:
    csv_path = stimuli_dir / f"civ_items_trial_{trial_id}.csv"
    df = pd.read_csv(csv_path)
    numeric = df.drop(columns=["Name"], errors="ignore")
    return numeric.to_numpy(dtype=np.int64)


def select_test_ref(trial_id: int) -> tuple[np.ndarray, np.ndarray]:
    with open(f"card_game/eda_results/eda_trial{trial_id}.pkl", "rb") as f:
        results = pickle.load(f)
    pf_actual = results["converged_pf_table"][-1]
    q5 = np.percentile(pf_actual, 5, axis=0)
    q95 = np.percentile(pf_actual, 95, axis=0)
    ref_sol = np.array([q95[0], q95[1], q95[2], q5[3], q5[4]])
    return ref_sol, pf_actual


def gen_aspi(ref_sol: np.ndarray, items: np.ndarray) -> np.ndarray:
    n_obj = items.shape[1] - 1
    n_selected = 10 if n_obj == 5 else 6
    items_q5 = np.percentile(items[:, :n_obj], 5, axis=0)
    items_q95 = np.percentile(items[:, :n_obj], 95, axis=0)
    return (ref_sol - items_q5 * n_selected) / (
        items_q95 * n_selected - items_q5 * n_selected + 1e-12
    )


def main() -> None:
    trial_id = 8
    seed = 1127
    if_rank = True
    temp = 1

    items = load_items(Path("card_game/stimuli"), trial_id=trial_id)
    n_obj = items.shape[1] - 1
    n_selected = 10 if n_obj == 5 else 6

    ref_sol, _ = select_test_ref(trial_id)
    unit_aspi = gen_aspi(ref_sol, items)
    unit_aspi = unit_aspi / (np.linalg.norm(unit_aspi) + 1e-12)

    eda = eda_mod.KnapsackEDA(
        items=items,
        capacity=n_selected * 10,
        n_selected=n_selected,
        n_obj=n_obj,
        pop_size=1_000,
        max_no_improve_gen=5,
        max_row_diff=500,
        seed=seed,
        aspi=unit_aspi,
        if_rank=if_rank,
        temp=temp,
    )

    ### Only runs init ###
    # distribution, selected_population, selected_objectives = eda._generate_initial_population()
    # print("distribution sum:", distribution.sum())
    # print("selected_population shape:", selected_population.shape)
    # print("selected_objectives shape:", selected_objectives.shape)

    ### runs the whole eda ###
    results = eda.run()
    # print("results:", results)


if __name__ == "__main__":
    main()
