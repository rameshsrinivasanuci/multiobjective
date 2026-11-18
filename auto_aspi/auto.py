import sys
from pathlib import Path
import numpy as np
import pickle

from scipy.stats import percentileofscore
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures

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

# Round objective vectors when checking for outer-loop cycles.
OUTER_CYCLE_ROUND = 1
CYCLE_DAMP = 0.5

# How choose_step turns the validity radii of the chosen objective into a step size:
# "median", "weighted" (obj_weight-weighted average), "min" (most conservative), or
# "absolute" (min over j of the step where the curvature error in o_j reaches ABS_TOL * IQR_j).
RADIUS_AGG = "absolute"

# Allowed curvature error for "absolute", as a fraction of each objective's IQR.
ABS_TOL = 0.1

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

# round the point to the nearest OUTER_CYCLE_ROUND decimal place
def point_key(pt, decimals=OUTER_CYCLE_ROUND):
    """Hashable rounded vector for outer-loop cycle detection."""
    return tuple(np.round(np.asarray(pt, dtype=float), decimals))


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

def compute_curvature(info, mrs_params, query_idx):
    """Raw quadratic coefficient c[i, j] of o_i^2 when o_j is fit on the other objectives.

    Same local fit as mrs.compute_validity_radii (k nearest neighbours, degree-2
    polynomial). Units are o_j per (o_i)^2; the diagonal is NaN.
    """
    data, k, d, iqr = info["data"], mrs_params["k"], info["d"], info["iqr"]
    poly = PolynomialFeatures(degree=2, include_bias=True)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(data / iqr)
    _, nn_idx = nn.kneighbors((data[query_idx] / iqr).reshape(1, -1))
    nbrs = data[nn_idx[0, 1:]] - data[query_idx]

    curv = np.full((d, d), np.nan)
    for j in range(d):
        input_cols = [c for c in range(d) if c != j]
        feat_names = [f"x{p}" for p in range(len(input_cols))]
        X_poly = poly.fit_transform(nbrs[:, input_cols])
        all_names = list(poly.get_feature_names_out(feat_names))
        coef, _, _, _ = np.linalg.lstsq(X_poly, nbrs[:, j], rcond=None)
        for pos, i in enumerate(input_cols):
            curv[i, j] = coef[all_names.index(f"x{pos}^2")]
    return curv

def absolute_radii(curv_row, iqr, abs_tol):
    """Step in o_i at which the curvature error |c_ij| * step^2 reaches abs_tol * IQR_j.

    Returns sqrt(abs_tol * IQR_j / |c_ij|) per j; inf where there is no curvature,
    NaN on the diagonal.
    """
    curv_row = np.asarray(curv_row, dtype=float)
    radii = np.full(curv_row.shape, np.nan)
    known = np.isfinite(curv_row)
    flat = known & (np.abs(curv_row) < 1e-10)
    curved = known & ~flat
    radii[flat] = np.inf
    radii[curved] = np.sqrt(abs_tol * iqr[curved] / np.abs(curv_row[curved]))
    return radii

def choose_step(betas, radii_raw, radii_pct, obj_weight, radius_agg=None,
                curv=None, iqr=None, abs_tol=None):
    """Obtain weighted mrs, which objective to change, and the change size.

    The change size aggregates the validity radii of the chosen objective i over the
    other objectives j (all radii in row i are in raw units of objective i):
      "median":   median of the finite radii
      "weighted": average of the finite radii, weighted by obj_weight[j]
      "min":      smallest finite radius (most conservative; the original rule)
      "absolute": smallest step at which the curvature error in any o_j reaches
                  abs_tol * IQR_j (needs `curv` from compute_curvature and `iqr`)
    """
    radius_agg = radius_agg or RADIUS_AGG
    betas_weighted = betas @ obj_weight.T
    obj_change_i= int(np.argmax(betas_weighted))

    if radius_agg == "absolute":
        if curv is None or iqr is None:
            raise ValueError('radius_agg="absolute" needs curv and iqr')
        abs_tol = ABS_TOL if abs_tol is None else abs_tol
        abs_row = absolute_radii(curv[obj_change_i], np.asarray(iqr, dtype=float), abs_tol)
        finite = abs_row[np.isfinite(abs_row)]
        change = float(np.min(finite)) if finite.size else np.nan
        return betas_weighted, obj_change_i, change

    pct_row = radii_pct[obj_change_i]
    raw_row = radii_raw[obj_change_i]
    # skip the diagonal (NaN) and inf radii (no curvature)
    valid = np.isfinite(pct_row) & np.isfinite(raw_row)
    if not valid.any():
        return betas_weighted, obj_change_i, np.nan
    valid_idx = np.flatnonzero(valid)

    if radius_agg == "median":
        change = float(np.median(raw_row[valid_idx]))
    elif radius_agg == "weighted":
        w = np.asarray(obj_weight, dtype=float)[valid_idx]
        if w.sum() > 0:
            change = float(np.sum(w * raw_row[valid_idx]) / w.sum())
        else:
            change = float(np.mean(raw_row[valid_idx]))
    elif radius_agg == "min":
        change = float(raw_row[valid_idx[np.argmin(pct_row[valid_idx])]])
    else:
        raise ValueError(f"unknown radius_agg: {radius_agg}")
    return betas_weighted, obj_change_i, change

def move_query(info, query_pt, betas_raw, obj_change_i, change):
    """Find target point given betas and change. Then find the nearest point on the pareto front
    as the new query point. Return (new_query_idx, new_query_pt, target) for the new point."""
    if not np.isfinite(change):
        return None
    target = query_pt.copy()
    for j in range(len(query_pt)):
        if j != obj_change_i:
            target[j] = query_pt[j] + betas_raw[obj_change_i, j] * change ##CHECK
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
    query_idx, query_pt, n_inner, run_dir, outer_i,
    change_scale=1.0,
):
    records = []
    visited = {query_idx}
    stop_reason = "max_steps"

    for inner_i in range(n_inner):
        betas_raw, betas, radii_raw, radii_pct = analyze_point(info, mrs_params, query_idx)
        curv = compute_curvature(info, mrs_params, query_idx) if RADIUS_AGG == "absolute" else None
        betas_weighted, obj_change_i, change = choose_step(
            betas, radii_raw, radii_pct, obj_weight, curv=curv, iqr=info["iqr"],
        )
        change_raw = change
        if np.isfinite(change) and change_scale != 1.0:
            change = float(change) * change_scale

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
            "change_raw": change_raw,
            "change": change,
            "change_scale": change_scale,
            "target": None,
            "new_query_idx": None,
        }
        records.append(record)
        scale_note = f" (scale={change_scale})" if change_scale != 1.0 else ""
        print(f"[outer step {outer_i} inner step {inner_i}] query={np.round(query_pt, 1)} "
              f"improve {info['obj_names'][obj_change_i]} by {change:.2f}{scale_note}")

        moved = move_query(info, query_pt, betas_raw, obj_change_i, change)
        if moved is None:
            stop_reason = "no_candidate"
            break
        new_idx, new_pt, target = moved
        record["target"] = target
        record["new_query_idx"] = new_idx
        record["new_query_pt"] = new_pt
        # if new_idx in visited:  # remove loop check to obtain a fixed number of rounds
        #     stop_reason = "cycle"
        #     break
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
    # Track visited aspirations (rounded). If a move lands on a seen point,
    # damp the next outer step's change by CYCLE_DAMP instead of stopping.
    visited = {point_key(aspiration)}
    damp_next = False

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

        change_scale = CYCLE_DAMP if damp_next else 1.0
        if damp_next:
            print(f"outer cycle previously detected; damping change by {CYCLE_DAMP}")
        damp_next = False

        # run inner loop once during each outer loop
        final_idx, final_pt, records, stop_reason = run_inner_loop(
            info, mrs_params, aspiration, obj_weight,
            query_idx, query_pt, 1, outer_dir, outer_i,
            change_scale=change_scale,
        )
        all_records.extend(records)

        # Next aspiration is the (unsnapped) target of the last inner step, so the
        # aspiration moves continuously and step size / damping take effect.
        # Fall back to the snapped point if the last step produced no target.
        target = records[-1]["target"] if records else None
        used_target = target is not None
        next_aspiration = target if used_target else final_pt

        new_key = point_key(next_aspiration)
        # Same idea as the inner-loop visited check: revisit => cycle.
        cycle_detected = new_key in visited
        if cycle_detected:
            damp_next = True
            print(
                f"outer cycle detected at {np.round(next_aspiration, OUTER_CYCLE_ROUND)}; "
                f"next change will be damped by {CYCLE_DAMP}"
            )
        visited.add(new_key)

        # extra information for outer loop
        outer_summary.append({
            "pf_size": len(pf),
            "n_inner_steps": len(records),
            "inner_stop_reason": stop_reason,
            "change_scale": change_scale,
            "cycle_detected": cycle_detected,
            "used_target": used_target,
            "next_aspiration": np.asarray(next_aspiration, dtype=float).copy(),
        })

        aspiration = np.asarray(next_aspiration, dtype=float).copy()

    save_records(all_records, run_dir / "outer_records.pkl")
    return all_records, outer_summary