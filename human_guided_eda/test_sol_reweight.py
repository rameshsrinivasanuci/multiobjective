from pathlib import Path
import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import eda_sol_reweight_noinit # no bias to initial distribution
import eda_standard

def load_items(stimuli_dir: Path, trial_id: int) -> np.ndarray:
    csv_path = stimuli_dir / f"civ_items_trial_{trial_id}.csv"
    df = pd.read_csv(csv_path)
    numeric = df.drop(columns=["Name"], errors="ignore")
    return numeric.to_numpy(dtype=np.int64)

def get_eda_params(items: np.ndarray) -> dict:
    n_obj = items.shape[1] - 1

    if n_obj == 3:
        n_selected = 6
        max_row_diff = 5
    elif n_obj == 5:
        n_selected = 10
        max_row_diff = 500 ## should use different criteria for human guided eda (default is 500)
    else:
        raise ValueError(f"Number of objectives {n_obj} not supported")

    return {
        "n_items": items.shape[0],
        "n_obj": n_obj,
        "n_con": 1,
        "n_selected": n_selected,
        "capacity": n_selected * 10,
        "pop_size": 1_000,
        "generations": 100,
        "max_no_improve_gen": 5,
        "max_row_diff": max_row_diff,
    }

def run_eda_pass(items: np.ndarray, params: dict, 
                 seed: int, aspi: np.ndarray, if_rank: bool, temp: float):
    eda_process = eda_standard.KnapsackEDA(
        items=items,
        capacity=params["capacity"],
        n_selected=params["n_selected"],
        n_obj=params["n_obj"],
        pop_size=params["pop_size"],
        generations=params["generations"],
        max_no_improve_gen=params["max_no_improve_gen"],
        max_row_diff=params["max_row_diff"],
        seed=seed,
        aspi=aspi,
        if_rank=if_rank,
        temp=temp
    )
    return eda_process.run()

def save_pass_results(run_name: str, results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"eda_human_reweight_{run_name}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    return file_path

def select_test_ref(pf_actual, percentile):
    percentile_unique = np.unique(percentile)
    percent_ref = {}
    for p in percentile_unique:
        qp = np.percentile(pf_actual, p, axis=0)
        percent_ref[p] = qp
    ref_sol = np.array([percent_ref[p][i] for i, p in enumerate(percentile)])
    
    return ref_sol

# def gen_aspi(ref_sol, pf_actual):
#     # normalize by 95 percentile
#     # q95 = np.percentile(pf_actual, 95, axis=0)
#     # aspi = ref_sol/q95

#     # normalize by z-score
#     # aspi = (ref_sol - np.mean(pf_actual, axis=0)) / np.std(pf_actual, axis=0)

#     # min-max quantile
#     q5 = np.percentile(pf_actual, 5, axis=0)
#     q95 = np.percentile(pf_actual, 95, axis=0)
#     aspi = (ref_sol - q5) / (q95 - q5 + 1e-12)

#     # robust scalar (ref_sol - q50) / (q75 - q25)
#     # scaler = RobustScaler()
#     # scaler.fit(pf_actual)
#     # aspi = scaler.transform(ref_sol.reshape(1, -1))[0]

#     return aspi

def gen_aspi(ref_sol, items):
    n_obj = items.shape[1] - 1
    if n_obj == 3:
        n_selected = 6
    elif n_obj == 5:
        n_selected = 10
    else:
        raise ValueError(f"Number of objectives {n_obj} not supported")

    items_q5 = np.percentile(items[:, :n_obj], 5, axis=0)
    items_q95 = np.percentile(items[:, :n_obj], 95, axis=0)
    aspi = (ref_sol - items_q5*n_selected) / (items_q95*n_selected - items_q5*n_selected + 1e-12)
    return aspi 

# def gen_aspi(percentile, items, n_selected):
#     percentile_unique = np.unique(percentile)
#     percent_ref = {}
#     for p in percentile_unique:
#         qp = np.percentile(items, p, axis=0)
#         percent_ref[p] = qp
#     aspi = np.array([percent_ref[p][i]*n_selected for i, p in enumerate(percentile)])
#     return aspi

def main():
    # run ref x temp pair
    # temps = np.linspace(0.1, 1, 10)
    temps = [0.1]
    percentiles = {
        # "qmax": np.array([100, 100, 100, 0, 0]),
        # "qmin": np.array([0, 0, 0, 100, 100]),
        "qmedian": np.array([50, 50, 50, 50, 50]),
        # "q95": np.array([95, 95, 95, 5, 5]),
        # "q75": np.array([75, 75, 75, 25, 25]),
        # "q60": np.array([60, 60, 60, 40, 40]),
        # "q40": np.array([40, 40, 40, 60, 60]),
        # "q25": np.array([25, 25, 25, 75, 75]),
        # "q5": np.array([5, 5, 5, 95, 95]),
        # "q100": np.array([100, 100, 100, 100, 100]),
    }

    trial_id = 8
    stimuli_dir = Path("card_game/stimuli")
    output_dir = Path("data/eda_results/test/")
    items = load_items(stimuli_dir=stimuli_dir, trial_id=trial_id)
    with open(f'card_game/eda_results/eda_trial{trial_id}.pkl', 'rb') as f:
        results = pickle.load(f)
    pf_actual = results['converged_pf_table'][-1]

    params = get_eda_params(items)
    n_obj = params["n_obj"]
    if_rank = True

    for name, percentile in percentiles.items():
        ref_sol = select_test_ref(pf_actual, percentile)
        aspi = gen_aspi(ref_sol, items)
        unit_aspi = aspi / (np.linalg.norm(aspi) + 1e-12)

        for temp in temps:
            # run_name = f"ref{name}_temp{temp:.1f}_trial{trial_id}"
            run_name = "test3"
            eda_results = run_eda_pass(
                items=items,
                params=params,
                seed=1127,
                aspi=unit_aspi,
                if_rank=if_rank,
                temp=temp
            )
            save_path = save_pass_results(run_name, eda_results, output_dir)
            file_path = output_dir / f"history_{run_name}.pkl"
            history = {
                "trial_id": trial_id,
                "ref_name": name,
                "percentile": percentile,
                "original_aspi": ref_sol,
                "normalized_aspi": aspi,
                "unit_aspi": unit_aspi,
                "if_rank": if_rank,
                "temp": temp
            }
            with open(file_path, "wb") as f:
                pickle.dump(history, f)

if __name__ == "__main__":
    main()