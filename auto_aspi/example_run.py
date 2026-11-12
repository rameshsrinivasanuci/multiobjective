"""Example standalone runs for the inner and outer loops.

Run from the repository root:
    python auto_aspi/example_run.py inner
    python auto_aspi/example_run.py outer
"""

import argparse
from pathlib import Path
import numpy as np


from auto import (
    SALARY_CAP,
    compute_obj_weight,
    obtain_front,
    prepare_front,
    run_inner_loop,
    run_outer_loop,
)
import mrs
import agent


def make_setup():
    block_id = 2
    trial_id = 17
    _, _, obj_names = agent.get_items(block_id, trial_id)

    return {
        "block_id": block_id,
        "trial_id": trial_id,
        "n_selected": 10,
        "n_obj": len(obj_names),
        "obj_names": obj_names,
        "aspiration": np.array([120.8, 38.1, 21.1, 13.5, 211.4]),
    }


def run_inner_example(run_dir, setup, n_inner=5):
    """Run multiple inner steps on one fixed Pareto front."""
    run_dir.mkdir(parents=True, exist_ok=True)
    items, year, _ = agent.get_items(setup["block_id"], setup["trial_id"])
    aspiration = np.asarray(setup["aspiration"], dtype=float)

    pf_raw, _ = obtain_front(
        aspiration=aspiration,
        items=items,
        capacity=SALARY_CAP[year],
        setup=setup,
        cache_path=run_dir / "eda_results.pkl",
    )
    pf, _, info = prepare_front(pf_raw, run_dir, setup)
    mrs_params = mrs.get_mrs_params()
    obj_weight = compute_obj_weight(pf, aspiration)
    query_idx, query_pt = mrs.get_query_point(
        info["data"], aspiration, info["iqr"]
    )

    return run_inner_loop(
        info=info,
        mrs_params=mrs_params,
        aspiration=aspiration,
        obj_weight=obj_weight,
        query_idx=query_idx,
        query_pt=query_pt,
        n_inner=n_inner,
        run_dir=run_dir,
        outer_i=0,
    )


def run_outer_example(run_dir, setup, n_outer=5):
    """Run one inner step per outer iteration and update the aspiration."""
    return run_outer_loop(
        n_outer=n_outer,
        run_dir=run_dir,
        setup=setup,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inner", "outer"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("auto_aspi/example_results"))
    args = parser.parse_args()

    setup = make_setup()
    if args.mode == "inner":
        run_inner_example(args.output / "inner", setup, args.steps)
    else:
        run_outer_example(args.output / "outer", setup, args.steps)


if __name__ == "__main__":
    main()
