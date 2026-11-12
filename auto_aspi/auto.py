import sys
from pathlib import Path
import numpy as np
import pickle

from scipy.stats import percentileofscore

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
import mrs
import agent
import eda_distance_copy
import eda_distance


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EDA_PARAMS = {
    "pop_size": 1000,
    "generations": 150, ## changed from 200 to match agent.py
    "max_no_improve_gen": 3,
    "max_row_diff": 0.1,
    "seed": 2+17,  ## change from 1123 to match agent.py
    "if_rank": True,
    "temp": 0.1,
}

DENSITY_THRESHOLD = 0.8

SALARY_CAP = {
        2000: 34.000, 2001: 35.500, 2002: 42.500, 2003: 40.271,
        2004: 43.870, 2005: 43.870, 2006: 49.500, 2007: 53.135,
        2008: 55.630, 2009: 58.680, 2010: 57.700, 2011: 58.044,
        2012: 58.044, 2013: 58.679, 2014: 58.679, 2015: 63.065,
        2016: 70.000, 2017: 94.143, 2018: 99.093, 2019: 101.869,
        2020: 109.140, 2021: 109.140, 2022: 112.414, 2023: 123.655,
        2024: 136.021, 2025: 140.588, 2026: 154.647,
    }

def save_records(records, output_path):
    """Save a list of step records in a common format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(records, f)

# ---------------------------------------------------------------------------
# Inner-loop pieces
# ---------------------------------------------------------------------------

def analyze_point(info, mrs_params, query_idx):
    """Compute mrs and validity radius at query point."""
    betas_raw, betas = mrs.compute_mrs_at_query(
        info["data"], 
        mrs_params["k"], 
        info["d"], 
        info["iqr"], 
        query_idx
    )
    radii_pct, radii_raw = mrs.compute_validity_radii(
        info["data"], 
        mrs_params["k"], 
        info["d"], 
        info["iqr"],
        query_idx, 
        mrs_params["epsilon"],
    )
    return betas_raw, betas, radii_raw, radii_pct

def choose_step(betas, radii_raw, obj_weight):
    """Obtain weighted mrs, which objective to change, and the smallest valid change size."""
    betas_weighted = betas @ obj_weight.T
    obj_change_i= int(np.argmax(betas_weighted))
    finite_radii = radii_raw[obj_change_i][np.isfinite(radii_raw[obj_change_i])]
    change = float(np.min(finite_radii)) if finite_radii.size else np.nan
    return betas_weighted, obj_change_i, change


def move_query(info, query_pt, betas_raw, obj_change_i, change):
    """Find target point given betas and change. Then find the nearest point on the pareto front
    as the new query point. Return (new_query_idx, new_query_pt, target) for the new point."""
    if not np.isfinite(change):
        return None
    target = query_pt.copy()
    for j in range(len(query_pt)):
        if j != obj_change_i:
            target[j] = query_pt[j] + betas_raw[obj_change_i, j] * change
    target[obj_change_i] = query_pt[obj_change_i] + change

    candidates = np.flatnonzero(info["data"][:, obj_change_i] > query_pt[obj_change_i])
    if candidates.size == 0:
        return None
    dists = np.abs((info["data"][candidates] - target) / info["iqr"]).sum(axis=1)
    new_idx = int(candidates[np.argmin(dists)])
    return new_idx, info["data"][new_idx], target


# ---------------------------------------------------------------------------
# Run inner loop
# ---------------------------------------------------------------------------

def run_inner_loop(
    info, mrs_params, aspiration, obj_weight, 
    query_idx, query_pt, n_inner, run_dir, outer_i
):
    records = []
    visited = {query_idx}
    stop_reason = "max_steps"

    for inner_i in range(n_inner):
        betas_raw, betas, radii_raw, radii_pct = analyze_point(info, mrs_params, query_idx)
        betas_weighted, obj_change_i, change = choose_step(betas, radii_raw, obj_weight)

        record = {
            "outer_step": outer_i,
            "inner_step": inner_i,
            "aspiration": aspiration.copy(),
            "obj_weight": obj_weight.copy(),
            "query_idx": query_idx,
            "query_pt": query_pt.copy(),
            "betas_weighted": betas_weighted,
            "obj_to_change_idx": obj_change_i,
            "obj_to_change": info["obj_names"][obj_change_i],
            "change": change,
            "target": None,
            "new_query_idx": None,
        }
        records.append(record)
        print(f"[outer step {outer_i} inner step {inner_i}] query={np.round(query_pt, 1)}"
              f"improve {info['obj_names'][obj_change_i]} by {change:.2f}")

        moved = move_query(info, query_pt, betas_raw, obj_change_i, change)
        if moved is None:
            stop_reason = "no_candidate"
            break
        new_idx, new_pt, target = moved
        record["target"] = target
        record["new_query_idx"] = new_idx
        record["new_query_pt"] = new_pt
        if new_idx in visited:
            stop_reason = "cycle"
            break
        visited.add(new_idx)
        query_idx, query_pt = new_idx, new_pt

    save_records(records, run_dir / "inner_records.pkl")
    print(f"inner loop stopped: {stop_reason}")
    return query_idx, query_pt, records, stop_reason


# ---------------------------------------------------------------------------
# Outer-loop pieces
# ---------------------------------------------------------------------------
def obtain_front(aspiration, items, capacity, setup, cache_path):
    """Return (pf_raw, ps_raw) for `aspiration`.
    Tries `cache_path` first, and otherwise runs the EDA.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            print(f"loading cached front: {cache_path}")
            with open(cache_path, "rb") as f:
                results = pickle.load(f)
            return (
                results["converged_pf_table"][-1],
                results["converged_pf_population_table"][-1],
            )
        except Exception as exc:
            print(f"could not load cached front ({exc})")

    print(f"running EDA with aspiration {np.round(aspiration, 1)}")
    eda_run = eda_distance.KnapsackEDA(  ##changed from eda_distance_copy to eda_distance to match agent.py
        items=items,
        capacity=capacity,
        n_selected=setup["n_selected"],
        n_obj=setup["n_obj"],
        aspi=np.asarray(aspiration, dtype=float),
        **EDA_PARAMS,
    )
    results = eda_run.run()
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    return results["converged_pf_table"][-1], results["converged_pf_population_table"][-1]


def prepare_front(pf_raw, output_dir, setup):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pf, kept_index = mrs.density_filter(pf_raw[:, :setup["n_obj"]], DENSITY_THRESHOLD)
    df, _ = mrs.pf_to_csv(pf, output_dir, setup["obj_names"])
    info = mrs.get_pf_info(df)
    return pf, kept_index, info


def compute_obj_weight(pf, reference_point):
    obj_weight = np.array([
        percentileofscore(pf[:, i], reference_point[i])
        for i in range(pf.shape[1])
    ])
    return obj_weight / np.sum(obj_weight)


# ---------------------------------------------------------------------------
# Run outer loop (run inner loop once inside)
# ---------------------------------------------------------------------------
def run_outer_loop(n_outer, run_dir, setup):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    items, year, _ = agent.get_items(setup["block_id"], setup["trial_id"])
    capacity = SALARY_CAP[year]
    mrs_params = mrs.get_mrs_params()
    aspiration = np.asarray(setup["aspiration"], dtype=float)

    all_records = []
    outer_summary = []
    for outer_i in range(n_outer):
        print(f"=== outer step {outer_i}: aspiration {np.round(aspiration, 1)} ===")
        outer_dir = run_dir / f"outer_{outer_i:02d}"
        outer_dir.mkdir(parents=True, exist_ok=True) 

        pf_raw, _ = obtain_front(
            aspiration, items, capacity, setup,
            cache_path=outer_dir / "eda_results.pkl",
        )
        pf, _, info = prepare_front(pf_raw, outer_dir, setup)
        obj_weight = compute_obj_weight(pf, aspiration)
        query_idx, query_pt = mrs.get_query_point(info["data"], aspiration, info["iqr"])

        # run inner loop once during each outer loop
        final_idx, final_pt, records, stop_reason = run_inner_loop(
            info, mrs_params, aspiration, obj_weight, 
            query_idx, query_pt, 1, outer_dir, outer_i
        )
        all_records.extend(records)

        # extra information for outer loop
        outer_summary.append({
            "pf_size": len(pf),
            "n_inner_steps": len(records),
            "inner_stop_reason": stop_reason,
        })

        aspiration = final_pt.astype(float).copy()

    save_records(all_records, run_dir / "outer_records.pkl")
    return all_records, outer_summary