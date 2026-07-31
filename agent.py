import pickle
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import eda
import eda_distance
from mrs import (
    OBJ_COLORS,
    compute_mrs_at_query,
    density_filter,
    get_mrs_params,
    get_pf_info,
    get_query_point,
    pf_to_csv,
    plot_bar_chart,
    save_mrs_results,
)


def get_items(trial_id):
    """Load item table for the trial sent by the client."""
    return pd.read_csv(
        f"data/items/items_{trial_id}.csv", header=None
    ).values


def get_aspi(slider_values):
    """Convert client slider values into an aspiration vector."""
    return np.asarray(slider_values, dtype=float)


def get_params(n_obj, items):
    if n_obj == 3:
        n_selected, max_row_diff = 6, 5
    elif n_obj == 5:
        n_selected, max_row_diff = 10, 500
    else:
        raise ValueError(f"Number of objectives {n_obj} not supported")

    return {
        "items": items,
        "capacity": n_selected * 10,
        "n_selected": n_selected,
        "n_obj": n_obj,
        "pop_size": 1_000,
        "generations": 100,
        "max_no_improve_gen": 5,
        "max_row_diff": max_row_diff,
    }


def run_eda(params, seed, aspi=None, temp=None):
    """Run plain EDA, or human-guided EDA when aspi/temp are provided."""
    common = dict(
        items=params["items"],
        capacity=params["capacity"],
        n_selected=params["n_selected"],
        n_obj=params["n_obj"],
        pop_size=params["pop_size"],
        generations=params["generations"],
        max_no_improve_gen=params["max_no_improve_gen"],
        max_row_diff=params["max_row_diff"],
        seed=seed,
    )
    if aspi is None:
        return eda.KnapsackEDA(**common, p_rank=None).run()
    return eda_distance.KnapsackEDA(
        **common, aspi=aspi, if_rank=True, temp=temp
    ).run()


def save_pickle(run_type, sub_id, run_id, results, output_dir):
    path = Path(output_dir) / f"{run_type}_{sub_id}_{run_id}.pkl"
    if path.exists():
        raise ValueError(f"File {path} already exists")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    return path


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def compute_game_score(query_pt, pf):
    """
    Percentage of PF solutions that query_pt beats in more than half the objectives.

    Maximization: query_pt beats a solution on an objective when its value is
    strictly greater. A solution counts if that holds for > n_obj / 2 objectives.
    """
    query_pt = np.asarray(query_pt, dtype=float)
    pf = np.asarray(pf, dtype=float)
    n_obj = pf.shape[1]
    n_wins = np.sum(query_pt > pf, axis=1)
    beaten = n_wins > (n_obj / 2)
    return 100.0 * beaten.mean()


def main(trial_id, slider_values):
    """
    Run one trial using values from the client request.

    Expected client payload fields:
        trial["trial_id"] -> items CSV lookup
        slider_values     -> aspiration levels (aspi)
    """
    sub_id = 0
    run_id = str(uuid.uuid4())
    run_type = "hg_eda"  # "eda" | "hg_eda"
    density_threshold = 0.8
    temp = 0.1  # only used for hg_eda

    items = get_items(trial_id)
    n_obj = items.shape[1] - 1
    params = get_params(n_obj, items)
    aspi = get_aspi(slider_values)
    seed = sub_id + int(trial_id)

    # --- EDA ---
    if run_type == "eda":
        results = run_eda(params, seed)
    elif run_type == "hg_eda":
        results = run_eda(params, seed, aspi=aspi, temp=temp)
    else:
        raise ValueError(f"Unknown run_type: {run_type}")

    out_eda = Path("data/results/eda")
    out_pf = Path("data/results/pf")
    out_mrs = Path("data/results/mrs")
    ensure_dirs(out_eda, out_pf, out_mrs)

    save_pickle(run_type, sub_id, run_id, results, out_eda)

    # --- PF post-process ---
    pf = density_filter(results["converged_pf_table"][-1], density_threshold)
    df, _ = pf_to_csv(pf, out_pf, run_type, sub_id, run_id)

    # --- MRS at aspiration-nearest point ---
    mrs_params = get_mrs_params()
    info = get_pf_info(df)
    query_idx, query_pt = get_query_point(info["data"], aspi, info["iqr"])
    betas_raw, betas = compute_mrs_at_query(
        info["data"], mrs_params["k"], info["d"], info["iqr"], query_idx
    )
    save_mrs_results(run_type, sub_id, run_id, betas_raw, betas, out_mrs, info["obj_names"])
    plot_bar_chart(
        betas_raw, info["obj_names"], query_pt, info["d"],
        out_mrs, run_type, sub_id, run_id, obj_colors=OBJ_COLORS,
    )
    
    # --- compute game score ---
    game_score = compute_game_score(query_pt, pf)

    return {
        "rec": query_pt,
        "mrs": betas_raw,
        "score": game_score,
    }


if __name__ == "__main__":
    # Example local call; production path passes these from the TCP client.
    raise SystemExit(
        "Call main(trial_id, slider_values) with values from the client "
        "(trial['trial_id'] and slider_values)."
    )
