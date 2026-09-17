import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import glob
import os


def _resolve_path(path):
    """Return a concrete file path. Glob patterns must match exactly one file."""
    path = os.fspath(path)
    if glob.has_magic(path):
        matches = sorted(glob.glob(path))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one file matching {path!r}, found {len(matches)}"
            )
        return matches[0]
    return path


def load_front_and_solution(pareto_path, objectives=None, new_solution=None, demo_percentile=70):
    """Load the reference Pareto front and the new solution to score.

    `pareto_path` may be a concrete file or a glob that matches exactly one
    file (e.g. ``data/game_data/pf_block_2_trial_1_*.csv``).

    Returns
    -------
    X : ndarray, shape (N, D)
    new : ndarray, shape (D,)
    objectives : list[str]
    """
    pareto_path = _resolve_path(pareto_path)
    objectives = list(objectives) if objectives else []
    ext = os.path.splitext(pareto_path)[1].lower()
    if ext == '.csv':
        front = pd.read_csv(pareto_path)
        if objectives:
            missing = [c for c in objectives if c not in front.columns]
            assert not missing, f"columns not found in CSV header: {missing}"
            X = front[objectives].to_numpy(dtype=float)
        else:
            objectives = list(front.columns)
            X = front.to_numpy(dtype=float)
    else:
        X = np.load(pareto_path).astype(float)
        assert objectives, "for a .npy front you must list `objectives`"
    N, D = X.shape
    assert D == len(objectives), f"{D} columns but {len(objectives)} objective names"

    def _load_vec(spec):
        if isinstance(spec, dict):
            return np.array([float(spec[o]) for o in objectives])
        if isinstance(spec, str):
            if os.path.splitext(spec)[1].lower() == '.csv':
                df = pd.read_csv(spec)
                cols = [o for o in objectives if o in df.columns]
                return (df[cols] if len(cols) == D else df).to_numpy(float)[0]
            a = np.load(spec).astype(float)
            return a.reshape(-1)[:D] if a.ndim == 1 else a[0]
        return np.asarray(spec, float).reshape(-1)

    if new_solution is None:
        new = np.percentile(X, demo_percentile, axis=0)
        # print(f"new_solution is None -> using a DEMO point "
        #       f"({demo_percentile}th percentile of each objective).")
        # print("Pass new_solution to score your own solution.\n")
    else:
        new = _load_vec(new_solution)
    assert new.shape == (D,), f"new solution has {new.shape}, expected ({D},)"

    # print(f"reference front: {N:,} points x {D} objectives  (from {os.path.basename(pareto_path)})")
    # print("objective order:", objectives)
    return X, new, objectives

def compute_matchup_outcomes(X, new):
    """Win / loss / draw of `new` vs every point on the front `X`.

    A win is a strictly higher value on a majority of objectives
    (m = floor(D/2)+1). Draws occur only when neither side has a majority,
    which requires value ties.

    Returns
    -------
    new_wins, front_win, draws : bool ndarray, shape (N,)
    gt : bool ndarray, shape (N, D)
        True where new beats the front point on that objective.
    wins_j : int ndarray, shape (N,)
        Number of objectives the new solution takes per matchup.
    m : int
        Majority threshold.
    win_pct : float
    """
    N, D = X.shape
    m = D // 2 + 1
    gt = new[None, :] > X
    lt = new[None, :] < X
    wins_j = gt.sum(1)
    losses_j = lt.sum(1)

    new_wins = wins_j >= m
    front_win = losses_j >= m
    draws = ~(new_wins | front_win)
    win_pct = new_wins.mean()

    # print(f"NEW SOLUTION WIN RATE vs the front : {win_pct:.3f}")
    # print(f"   wins {int(new_wins.sum()):,}   losses {int(front_win.sum()):,}   "
    #       f"draws {int(draws.sum()):,}   (of {N:,})")
    # print(f"   a 'win' = higher value on a majority (>= {m} of {D}) of objectives")
    return new_wins, front_win, draws, gt, wins_j, m, win_pct

def compute_coalition_coverage(gt, new_wins, objectives, m):
    """Coverage of each size-m coalition over matchups the new solution wins.

    For every won matchup, credit every size-m subset of the objectives the
    new solution beat that point on. Coalitions overlap, so coverages can
    sum to more than total_wins.

    Returns
    -------
    coalitions : list[dict]
        Each has 'indices', 'label', 'bitmask'.
    coverage : Counter
        bitmask -> number of won matchups containing that coalition.
    total_wins : int
    """
    D = gt.shape[1]

    def label_of(c):
        return '+'.join(objectives[d] for d in c)

    def bitmask_of(c):
        return sum(1 << d for d in c)

    coalitions = [{'indices': c, 'label': label_of(c), 'bitmask': bitmask_of(c)}
                  for c in combinations(range(D), m)]

    weights = (1 << np.arange(D)).astype(np.int64)
    bm = (gt * weights).sum(1)
    bit_counts = np.bincount(bm[new_wins], minlength=1 << D)
    total_wins = int(new_wins.sum())

    coverage = Counter()
    for W in np.flatnonzero(bit_counts):
        c = int(bit_counts[W])
        members = [d for d in range(D) if (W >> d) & 1]
        for sub in combinations(members, m):
            coverage[bitmask_of(sub)] += c

    # print(f"m = {m}   |   minimal coalitions = C({D},{m}) = {len(coalitions)}")
    # print(f"matchups won (credited to coalitions): {total_wins:,}")
    if total_wins == 0:
        print("The new solution wins no matchups -> no coalitions to report.")
    return coalitions, coverage, total_wins



# X, new, objectives = load_front_and_solution(PARETO_PATH, objectives, NEW_SOLUTION)
# N, D = X.shape
# pd.DataFrame({'objective': objectives, 'new solution': new}).set_index('objective').T

# new_wins, front_win, draws, gt, wins_j, m, win_pct = compute_matchup_outcomes(X, new)

# coalitions, coverage, total_wins = compute_coalition_coverage(gt, new_wins, objectives, m)