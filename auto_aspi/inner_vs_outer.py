"""
Compare one inner-loop step against one outer-loop step for the same aspiration.

Case k (k = 1..n_cases) uses as aspiration:
    case 1: the original aspiration
    case k: the query point reached after k-1 inner-loop steps (on the original front)

Inner step: stays on the original front with the original obj_weight (as in
            run_inner_loop) and takes one step from the current inner query point.
Outer step: obtains the front for the aspiration, recomputes obj_weight from it,
            snaps to the closest PF point, and takes one step from there
            (i.e. the first inner step of a new outer iteration).

For each case both steps report betas, validity radii, the chosen objective,
the step size, the target, and the new query point.

Usage:
    python inner_vs_outer.py --n-cases 5
"""

import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from mrs_loop import (
    RESULTS_ROOT, SETUP, analyze_point, choose_step, compute_obj_weight,
    initial_results_path, load_problem, move_query, obtain_front, prepare_front,
    problem_tag, save_step,
)
import mrs  # importable once mrs_loop has put the repo root on sys.path


def one_step(info, mrs_params, obj_weight, query_idx, query_pt, step_dir):
    betas_raw, betas, radii_raw, radii_pct = analyze_point(info, mrs_params, query_idx)
    save_step(step_dir, info, mrs_params, query_pt, betas_raw, betas, radii_raw, radii_pct)
    betas_weighted, change_i, change = choose_step(betas, radii_raw, obj_weight)
    moved = move_query(info, query_pt, betas_raw, change_i, change)
    new_idx, new_pt, target = moved if moved is not None else (None, None, None)
    return {
        "obj_weight": obj_weight,
        "query_idx": query_idx,
        "query_pt": query_pt.copy(),
        "betas_raw": betas_raw,
        "betas": betas,
        "radii_raw": radii_raw,
        "radii_pct": radii_pct,
        "betas_weighted": betas_weighted,
        "change_i": change_i,
        "change_obj": info["obj_names"][change_i],
        "change": change,
        "target": target,
        "new_query_idx": new_idx,
        "new_query_pt": None if new_pt is None else new_pt.copy(),
    }


def print_case(case, aspiration, inner, outer):
    print(f"\n=== case {case}: aspiration {np.round(aspiration, 1)} ===")
    for name, r in (("inner", inner), ("outer", outer)):
        new_pt = "none" if r["new_query_pt"] is None else np.round(r["new_query_pt"], 1)
        print(f"  {name}: query={np.round(r['query_pt'], 1)} "
              f"weight={np.round(r['obj_weight'], 3)}")
        print(f"         improve {r['change_obj']} by {r['change']:.2f} -> new query={new_pt}")
    diff_b = np.linalg.norm(inner["betas"] - outer["betas"])
    both_finite = np.isfinite(inner["radii_raw"]) & np.isfinite(outer["radii_raw"])
    diff_r = np.abs(inner["radii_raw"] - outer["radii_raw"])[both_finite]
    print(f"  ||betas_inner - betas_outer||_F = {diff_b:.4f}; "
          f"mean |radii diff| (finite) = {diff_r.mean() if diff_r.size else np.nan:.4f}")


def save_comparison(run_dir, cases, obj_names):
    with open(run_dir / "comparison.pkl", "wb") as f:
        pickle.dump(cases, f)

    rows = []
    for c in cases:
        for mode in ("inner", "outer"):
            r = c[mode]
            row = {"case": c["case"], "mode": mode, "pf_size": c[f"{mode}_pf_size"],
                   "query_idx": r["query_idx"], "change_obj": r["change_obj"],
                   "change": r["change"], "new_query_idx": r["new_query_idx"]}
            row.update({f"aspi_{n}": v for n, v in zip(obj_names, c["aspiration"])})
            row.update({f"weight_{n}": v for n, v in zip(obj_names, r["obj_weight"])})
            row.update({f"query_{n}": v for n, v in zip(obj_names, r["query_pt"])})
            row.update({f"betas_w_{n}": v for n, v in zip(obj_names, r["betas_weighted"])})
            for key in ("target", "new_query_pt"):
                vals = r[key] if r[key] is not None else [np.nan] * len(obj_names)
                row.update({f"{key}_{n}": v for n, v in zip(obj_names, vals)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "comparison.csv", index=False)
    print(f"\nSaved comparison → {run_dir}")


def run_comparison(n_cases, run_dir, setup, fresh_eda):
    items, capacity = load_problem(setup)
    mrs_params = mrs.get_mrs_params()
    aspiration0 = np.asarray(setup["aspiration"], dtype=float)

    orig_dir = run_dir / "original_front"
    orig_dir.mkdir(parents=True, exist_ok=True)
    pf_raw0, _ = obtain_front(aspiration0, items, capacity, setup,
                              cache_path=orig_dir / "eda_results.pkl",
                              initial_path=None if fresh_eda else initial_results_path(setup))
    pf0, _, info0 = prepare_front(pf_raw0, orig_dir, setup)
    obj_weight0 = compute_obj_weight(pf0, aspiration0)
    inner_idx, inner_pt = mrs.get_query_point(info0["data"], aspiration0, info0["iqr"])
    visited = {inner_idx}

    cases = []
    for case in range(1, n_cases + 1):
        aspiration = aspiration0 if case == 1 else inner_pt.astype(float).copy()
        case_dir = run_dir / f"case_{case:02d}"

        inner = one_step(info0, mrs_params, obj_weight0, inner_idx, inner_pt,
                         case_dir / "inner")

        outer_dir = case_dir / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        if case == 1:
            pf_raw, pf, info = pf_raw0, pf0, info0
        else:
            pf_raw, _ = obtain_front(aspiration, items, capacity, setup,
                                     cache_path=outer_dir / "eda_results.pkl")
            pf, _, info = prepare_front(pf_raw, outer_dir, setup)
        obj_weight = compute_obj_weight(pf, aspiration)
        outer_idx, outer_pt = mrs.get_query_point(info["data"], aspiration, info["iqr"])
        outer = one_step(info, mrs_params, obj_weight, outer_idx, outer_pt, outer_dir)

        cases.append({"case": case, "aspiration": aspiration, "inner": inner, "outer": outer,
                      "inner_pf_size": len(pf0), "outer_pf_size": len(pf)})
        print_case(case, aspiration, inner, outer)

        if inner["new_query_idx"] is None:
            print(f"  inner loop has no next point after case {case} (no candidate); stopping")
            break
        if inner["new_query_idx"] in visited:
            print(f"  inner loop cycles after case {case}; stopping")
            break
        visited.add(inner["new_query_idx"])
        inner_idx, inner_pt = inner["new_query_idx"], inner["new_query_pt"]

    save_comparison(run_dir, cases, setup["obj_names"])
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cases", type=int, default=5,
                        help="number of cases; case k starts after k-1 inner steps")
    parser.add_argument("--run-name", type=str, default=None,
                        help="subfolder name; reuse an existing one to reuse cached EDA fronts")
    parser.add_argument("--fresh-eda", action="store_true",
                        help="rerun the EDA for the original aspiration instead of loading "
                             "the precomputed initial front")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / problem_tag(SETUP) / "inner_vs_outer" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_comparison(args.n_cases, run_dir, SETUP, args.fresh_eda)


if __name__ == "__main__":
    main()
