"""
Iterative MRS-guided aspiration search.

Outer loop: fix an aspiration, obtain the Pareto front (pf_raw, ps_raw) for it,
            and fix obj_weight from that aspiration.
Inner loop: starting from the PF point closest to the aspiration, repeatedly
            compute MRS + validity radii, pick the objective to improve
            (argmax of weighted MRS), and move the query point. Stops after
            n_inner steps, on cycling, or when no PF point improves the chosen
            objective.
The final inner query point becomes the next outer aspiration.

Usage:
    python mrs_loop.py --n-outer 3 --n-inner 5
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
import mrs
import eda_distance_copy

# initial aspiration predefinedfor each problem (seed: aspiration)
ASPI = {
    1300: np.array([ 8856.,  9058., 10015., 11802., 14529.]),
    1400: np.array([ 8137., 10991., 14076., 12892.,  7928.]),
    1500: np.array([13703., 10232., 11229.,  8183.,  9798.]),
    1600: np.array([12329.,  6994.,  9638., 10307., 13385.]),
}

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = REPO_DIR / "simulations"
RESULTS_ROOT = SCRIPT_DIR / "results"

SETUP = {
    "n_obj": 5,
    "n_con": 1,
    "n_items": 100,
    "n_selected": 25,
    "rho": 0.2,
    "exactly": True,
    "aspiration": ASPI[1500],
    "obj_names": ["obj_1", "obj_2", "obj_3", "obj_4", "obj_5"],
    "seed": 1500,
}

EDA_PARAMS = {
    "pop_size": 1000,
    "generations": 200,
    "max_no_improve_gen": 3,
    "max_row_diff": 0.1,
    "seed": 1123,
    "if_rank": True,
    "temp": 0.1,
}

OBJ_COLORS = {
    "obj_1": "#4C78A8",   # blue
    "obj_2": "#F58518",   # orange
    "obj_3": "#54A24B",   # green
    "obj_4": "#E45756",   # red
    "obj_5": "#B279A2",   # purple
    "obj_6": "#FF9DA6",   # pink
    "obj_7": "#9D755D",   # brown
    "obj_8": "#2CB1BC",   # teal
}

DENSITY_THRESHOLD = 0.8


def problem_tag(setup):
    return f"{setup['n_obj']}obj_{setup['n_items']}items_{setup['rho']}rho_allneg_{setup['seed']}seed"


def load_problem(setup):
    data = np.load(DATA_DIR / "problems" / f"items_{problem_tag(setup)}.npz")
    return data["items"], data["capacity"]


def initial_results_path(setup):
    results_dir = "exact" if setup["exactly"] else "upper_bound"
    return (
        DATA_DIR / "results" / results_dir / "prefer"
        / f"eda_mode2_{problem_tag(setup)}_allneg_withtime.pkl"
    )


# ---------------------------------------------------------------------------
# Outer-loop pieces
# ---------------------------------------------------------------------------
def obtain_front(aspiration, items, capacity, setup, cache_path, initial_path=None):
    """Return (pf_raw, ps_raw) for `aspiration`.

    Loads `cache_path` if it exists, else `initial_path` if given, else runs the EDA
    with `aspiration` and caches the full results dict to `cache_path`.
    """
    if cache_path.exists():
        print(f"  loading cached front: {cache_path}")
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
    elif initial_path is not None:
        print(f"  loading initial front: {initial_path}")
        with open(initial_path, "rb") as f:
            results = pickle.load(f)
    else:
        print(f"  running EDA with aspiration {np.round(aspiration, 1)}")
        eda_run = eda_distance_copy.KnapsackEDA(
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
# Inner-loop pieces
# ---------------------------------------------------------------------------
def analyze_point(info, mrs_params, query_idx):
    betas_raw, betas = mrs.compute_mrs_at_query(
        info["data"], mrs_params["k"], info["d"], info["iqr"], query_idx
    )
    radii_pct, radii_raw = mrs.compute_validity_radii(
        info["data"], mrs_params["k"], info["d"], info["iqr"],
        query_idx, mrs_params["epsilon"],
    )
    return betas_raw, betas, radii_raw, radii_pct


def choose_step(betas, radii_raw, obj_weight):
    betas_weighted = betas @ obj_weight.T
    change_i = int(np.argmax(betas_weighted))
    # inf radii (no curvature) would make the step infinite, so only finite radii count
    finite = radii_raw[change_i][np.isfinite(radii_raw[change_i])]
    change = float(np.min(finite)) if finite.size else np.nan
    return betas_weighted, change_i, change


def move_query(info, query_pt, betas_raw, change_i, change):
    """Return (query_idx, query_pt, target) for the new point, or None if no PF
    point improves objective `change_i` or the step size is undefined."""
    if not np.isfinite(change):
        return None
    target = query_pt.copy()
    for j in range(len(query_pt)):
        if j != change_i:
            target[j] = query_pt[j] + betas_raw[change_i, j] * change
    target[change_i] = query_pt[change_i] + change

    candidates = np.flatnonzero(info["data"][:, change_i] > query_pt[change_i])
    if candidates.size == 0:
        return None
    dists = np.abs((info["data"][candidates] - target) / info["iqr"]).sum(axis=1)
    new_idx = int(candidates[np.argmin(dists)])
    return new_idx, info["data"][new_idx], target


def save_step(step_dir, info, mrs_params, query_pt, betas_raw, betas, radii_raw, radii_pct):
    step_dir.mkdir(parents=True, exist_ok=True)
    names = info["obj_names"]
    pd.DataFrame(betas_raw, columns=names, index=names).to_csv(step_dir / "betas_raw.csv")
    pd.DataFrame(betas, columns=names, index=names).to_csv(step_dir / "betas.csv")
    pd.DataFrame(radii_raw, columns=names, index=names).to_csv(step_dir / "radii_raw.csv")
    mrs.plot_bar_chart(betas, names, query_pt, info["d"], step_dir, obj_colors=OBJ_COLORS)
    mrs.plot_validity_bar_chart(
        radii_pct, radii_raw, names, info["d"], mrs_params["epsilon"],
        step_dir, obj_colors=OBJ_COLORS,
    )


def run_inner_loop(info, mrs_params, obj_weight, query_idx, query_pt, n_inner, outer_dir, outer):
    records = []
    visited = {query_idx}
    stop_reason = "max_steps"

    for step in range(n_inner):
        betas_raw, betas, radii_raw, radii_pct = analyze_point(info, mrs_params, query_idx)
        save_step(outer_dir / f"step_{step:02d}", info, mrs_params, query_pt,
                  betas_raw, betas, radii_raw, radii_pct)
        betas_weighted, change_i, change = choose_step(betas, radii_raw, obj_weight)

        record = {
            "outer": outer,
            "step": step,
            "query_idx": query_idx,
            "query_pt": query_pt.copy(),
            "betas_weighted": betas_weighted,
            "change_i": change_i,
            "change_obj": info["obj_names"][change_i],
            "change": change,
            "target": None,
            "next_query_idx": None,
        }
        records.append(record)
        print(f"  [outer {outer} step {step}] query={np.round(query_pt, 1)} "
              f"improve {info['obj_names'][change_i]} by {change:.2f}")

        moved = move_query(info, query_pt, betas_raw, change_i, change)
        if moved is None:
            stop_reason = "no_candidate"
            break
        new_idx, new_pt, target = moved
        record["target"] = target
        record["next_query_idx"] = new_idx
        if new_idx in visited:
            stop_reason = "cycle"
            break
        visited.add(new_idx)
        query_idx, query_pt = new_idx, new_pt

    print(f"  inner loop stopped: {stop_reason}")
    return query_idx, query_pt, records, stop_reason


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------
def run_outer_loop(n_outer, n_inner, run_dir, setup):
    items, capacity = load_problem(setup)
    mrs_params = mrs.get_mrs_params()
    aspiration = np.asarray(setup["aspiration"], dtype=float)

    all_records = []
    outer_summary = []
    for outer in range(n_outer):
        print(f"=== outer {outer}: aspiration {np.round(aspiration, 1)} ===")
        outer_dir = run_dir / f"outer_{outer:02d}"
        outer_dir.mkdir(parents=True, exist_ok=True)

        pf_raw, ps_raw = obtain_front(
            aspiration, items, capacity, setup,
            cache_path=outer_dir / "eda_results.pkl",
            initial_path=initial_results_path(setup) if outer == 0 else None,
        )
        pf, kept_index, info = prepare_front(pf_raw, outer_dir, setup)
        obj_weight = compute_obj_weight(pf, aspiration)
        query_idx, query_pt = mrs.get_query_point(info["data"], aspiration, info["iqr"])

        final_idx, final_pt, records, stop_reason = run_inner_loop(
            info, mrs_params, obj_weight, query_idx, query_pt, n_inner, outer_dir, outer
        )
        for r in records:
            r["aspiration"] = aspiration.copy()
            r["obj_weight"] = obj_weight
        all_records.extend(records)

        outer_summary.append({
            "outer": outer,
            "aspiration": aspiration.copy(),
            "obj_weight": obj_weight,
            "pf_size": len(pf),
            "start_query_pt": query_pt.copy(),
            "final_query_idx": final_idx,
            "final_query_pt": final_pt.copy(),
            "final_solution": ps_raw[kept_index[final_idx]],
            "n_steps": len(records),
            "stop_reason": stop_reason,
        })

        aspiration = final_pt.astype(float).copy()

    save_history(run_dir, all_records, outer_summary, setup["obj_names"])
    return all_records, outer_summary


def save_history(run_dir, all_records, outer_summary, obj_names):
    with open(run_dir / "history.pkl", "wb") as f:
        pickle.dump({"steps": all_records, "outer": outer_summary}, f)

    rows = []
    for r in all_records:
        row = {k: r[k] for k in ("outer", "step", "query_idx", "change_obj",
                                 "change", "next_query_idx")}
        row.update({f"query_{n}": v for n, v in zip(obj_names, r["query_pt"])})
        row.update({f"aspi_{n}": v for n, v in zip(obj_names, r["aspiration"])})
        row.update({f"weight_{n}": v for n, v in zip(obj_names, r["obj_weight"])})
        row.update({f"betas_w_{n}": v for n, v in zip(obj_names, r["betas_weighted"])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "history.csv", index=False)

    summary_rows = []
    for s in outer_summary:
        row = {k: s[k] for k in ("outer", "pf_size", "final_query_idx", "n_steps", "stop_reason")}
        row.update({f"aspi_{n}": v for n, v in zip(obj_names, s["aspiration"])})
        row.update({f"final_{n}": v for n, v in zip(obj_names, s["final_query_pt"])})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(run_dir / "outer_summary.csv", index=False)
    print(f"Saved history → {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-outer", type=int, default=3, help="number of aspiration updates")
    parser.add_argument("--n-inner", type=int, default=5, help="max query-point moves per aspiration")
    parser.add_argument("--run-name", type=str, default=None,
                        help="subfolder name; reuse an existing one to reuse cached EDA fronts")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / problem_tag(SETUP) / "loop" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_outer_loop(args.n_outer, args.n_inner, run_dir, SETUP)


if __name__ == "__main__":
    main()
