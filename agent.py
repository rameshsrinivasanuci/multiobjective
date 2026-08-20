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
    compute_validity_radii,
    density_filter,
    get_mrs_params,
    get_pf_info,
    get_query_point,
    pf_to_csv,
    plot_bar_chart,
    plot_validity_bar_chart,
    save_mrs_results,
)


SALARY_CAP = {
        2000: 34.000, 2001: 35.500, 2002: 42.500, 2003: 40.271,
        2004: 43.870, 2005: 43.870, 2006: 49.500, 2007: 53.135,
        2008: 55.630, 2009: 58.680, 2010: 57.700, 2011: 58.044,
        2012: 58.044, 2013: 58.679, 2014: 58.679, 2015: 63.065,
        2016: 70.000, 2017: 94.143, 2018: 99.093, 2019: 101.869,
        2020: 109.140, 2021: 109.140, 2022: 112.414, 2023: 123.655,
        2024: 136.021, 2025: 140.588, 2026: 154.647,
    }


# def get_items(trial_id):
#     """Load item table for the trial sent by the client."""
#     df = pd.read_csv(f"data/items/trial_{trial_id}.csv")
#     return df.drop(columns=["PLAYER"], errors="ignore").to_numpy(float)

def get_items(block_id, trial_id):
    matches = list(
        Path("data/game_data").glob(
            f"block_{block_id}_trial_{trial_id}_[0-9][0-9][0-9][0-9].csv"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one file for trial {trial_id}, found {len(matches)}")
    csv_path = matches[0]
    year = int(csv_path.stem.rsplit("_", 1)[-1]) 
    df = pd.read_csv(csv_path)
    obj_names = df.columns[1:-1].tolist()
    items = df.drop(columns=["PLAYER"], errors="ignore").to_numpy(float)
    return items, year, obj_names


def get_aspi(slider_values):
    """Convert client slider values into an aspiration vector."""
    return np.asarray(slider_values, dtype=float)


def get_params(n_obj, items, year):

    return {
        "items": items,
        "capacity": SALARY_CAP[year],
        "n_selected": 10,
        "n_obj": n_obj,
        "pop_size": 1_000,
        "generations": 150,
        "max_no_improve_gen": 3,
        "max_row_diff": 0.1,
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


def get_submission_dir(sub_id, block_id, trial_id, submission_id):
    output_dir = (
        Path("data/results")
        / f"sub_{sub_id}"
        / f"block_{block_id}"
        / f"trial_{trial_id}"
        / f"submission_{submission_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_pickle(results, output_dir, run_type):
    path = Path(output_dir) / f"{run_type}.pkl"
    if path.exists():
        raise ValueError(f"File {path} already exists")
    with path.open("wb") as f:
        pickle.dump(results, f)
    return path


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


def main(
    sub_id: int,
    block_id: int,
    trial_id: int,
    submission_id: int,
    slider_values: list[float],
):
    """
    Run one trial using values from the client request.

    Expected client payload fields:
        trial["trial_id"] -> items CSV lookup
        slider_values     -> aspiration levels (aspi)
    """

    run_type = "hg_eda"  # "eda" or "hg_eda"
    density_threshold = 0.8
    temp = 0.1  # only used for hg_eda

    output_dir = get_submission_dir(
        sub_id, block_id, trial_id, submission_id
    )

    items, year, obj_names = get_items(block_id, trial_id)
    n_obj = items.shape[1] - 1
    params = get_params(n_obj, items, year)
    aspi = get_aspi(slider_values)
    seed = sub_id + int(trial_id)


    # --- EDA ---
    if run_type == "eda":
        results = run_eda(params, seed)
    elif run_type == "hg_eda":
        results = run_eda(params, seed, aspi=aspi, temp=temp)
    else:
        raise ValueError(f"Unknown run_type: {run_type}")

    save_pickle(results, output_dir, run_type)


    # --- PF post-process ---
    pf, kept_idx = density_filter(results["converged_pf_table"][-1], density_threshold)
    df, _ = pf_to_csv(pf, output_dir, obj_names)


    # --- Query point and indices ---
    mrs_params = get_mrs_params()
    info = get_pf_info(df)
    query_idx, query_pt = get_query_point(info["data"], aspi, info["iqr"])

    full_idx = int(kept_idx[query_idx])  # find the index of the query point in the full pf
    query_items_indices = results["converged_pf_population_table"][-1][full_idx]


    # --- MRS at query point ---
    betas_raw, betas = compute_mrs_at_query(
        info["data"], mrs_params["k"], info["d"], info["iqr"], query_idx
    )
    save_mrs_results(betas_raw, betas, output_dir, info["obj_names"])
    plot_bar_chart(
        betas, info["obj_names"], query_pt, info["d"],
        output_dir, obj_colors=OBJ_COLORS,
    )


    # --- Validity radii at query point ---
    radii_pct, radii_raw = compute_validity_radii(
        info["data"], mrs_params["k"], info["d"], info["iqr"],
        query_idx, mrs_params["epsilon"],
    )
    plot_validity_bar_chart(
        radii_pct, radii_raw, info["obj_names"], info["d"], mrs_params["epsilon"],
        output_dir, obj_colors=OBJ_COLORS,
    )


    # --- compute game score ---
    game_score = compute_game_score(query_pt, pf)


    return {
        "rec": np.round(query_pt, 1),
        "mrs": betas,
        "score": game_score,
        "rec_player_indices": query_items_indices,
    }


if __name__ == "__main__":
    # Example local call; production path passes these from the TCP client.
    raise SystemExit(
        "Call main(trial_id, slider_values) with values from the client "
        "(trial['trial_id'] and slider_values)."
    )
