"""
Parameter sweep simulations for plain EDA and human-guided EDA (hg_eda).

Varies: aspi, temp, pop_size, max_no_improve_gen, max_row_diff.
Results and an index CSV are written under data/simulation/.

Sweep modes
-----------
- "oat"  : one-at-a-time around BASELINE (default; modest run count)
- "full" : full factorial product of all grids (can be large)
"""

from __future__ import annotations

import csv
import itertools
import pickle
from pathlib import Path

import numpy as np

from agent import (
    ensure_dirs,
    get_aspi,
    get_items,
    get_params,
    run_eda,
)

# ---------------------------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------------------------
N_OBJ = 5
ITEMS_SEED = 1125
BASE_SEED = 99  # matches agent.py sub_id convention
OUTPUT_ROOT = Path("data/simulation")
SWEEP_MODE = "oat"  # "oat" | "full"

# ---------------------------------------------------------------------------
# Baseline + grids (edit these to expand / shrink the sweep)
# ---------------------------------------------------------------------------
BASELINE = {
    "aspi_name": "default",
    "temp": 0.1,
    "pop_size": 1_000,
    "max_no_improve_gen": 5,
    "max_row_diff": 500,
}

# Named aspiration vectors. Keys appear in filenames / the index CSV.
# "default" uses agent.ASPI; other entries are absolute objective targets.
ASPI_GRID = {
    "default": None,  # resolved via get_aspi(N_OBJ)
    "high_o0": np.array([120, 55, 110, 100, 100]),
    "high_o2": np.array([80, 55, 150, 100, 100]),
    "balanced": np.array([90, 90, 90, 90, 90]),
    "low_o1": np.array([80, 30, 110, 100, 100]),
}

TEMP_GRID = [0.1, 0.3, 0.5, 1.0]
POP_SIZE_GRID = [500, 1_000, 2_000]
MAX_NO_IMPROVE_GEN_GRID = [3, 5, 10]
MAX_ROW_DIFF_GRID = [100, 500, 1000]

# Run types to simulate. Plain EDA ignores aspi/temp.
RUN_TYPES = ("eda", "hg_eda")


def resolve_aspi(aspi_name: str) -> np.ndarray:
    value = ASPI_GRID[aspi_name]
    if value is None:
        return get_aspi(N_OBJ)
    return np.asarray(value, dtype=float).copy()


def run_tag(
    run_type: str,
    aspi_name: str | None,
    temp: float | None,
    pop_size: int,
    max_no_improve_gen: int,
    max_row_diff: int,
    seed: int,
) -> str:
    """Build a filesystem-safe tag that encodes the full config."""
    parts = [
        run_type,
        f"seed{seed}",
        f"pop{pop_size}",
        f"noi{max_no_improve_gen}",
        f"mrd{max_row_diff}",
    ]
    if run_type == "hg_eda":
        parts.extend([f"aspi-{aspi_name}", f"temp{temp}"])
    return "_".join(parts)


def _dedupe(configs):
    seen = set()
    unique = []
    for cfg in configs:
        if cfg in seen:
            continue
        seen.add(cfg)
        unique.append(cfg)
    return unique


def iter_configs(mode: str = SWEEP_MODE):
    """
    Yield (run_type, aspi_name, temp, pop_size, max_no_improve_gen, max_row_diff).

    For plain EDA, aspi_name and temp are always None.
    """
    configs = []

    if mode == "full":
        for run_type in RUN_TYPES:
            for pop_size, max_no_improve_gen, max_row_diff in itertools.product(
                POP_SIZE_GRID, MAX_NO_IMPROVE_GEN_GRID, MAX_ROW_DIFF_GRID
            ):
                if run_type == "eda":
                    configs.append(
                        (run_type, None, None, pop_size, max_no_improve_gen, max_row_diff)
                    )
                else:
                    for aspi_name, temp in itertools.product(ASPI_GRID, TEMP_GRID):
                        configs.append(
                            (
                                run_type,
                                aspi_name,
                                temp,
                                pop_size,
                                max_no_improve_gen,
                                max_row_diff,
                            )
                        )
    elif mode == "oat":
        b = BASELINE
        for run_type in RUN_TYPES:
            # baseline
            if run_type == "eda":
                configs.append(
                    (
                        run_type,
                        None,
                        None,
                        b["pop_size"],
                        b["max_no_improve_gen"],
                        b["max_row_diff"],
                    )
                )
            else:
                configs.append(
                    (
                        run_type,
                        b["aspi_name"],
                        b["temp"],
                        b["pop_size"],
                        b["max_no_improve_gen"],
                        b["max_row_diff"],
                    )
                )

            # one-at-a-time deviations
            for pop_size in POP_SIZE_GRID:
                configs.append(
                    (
                        run_type,
                        None if run_type == "eda" else b["aspi_name"],
                        None if run_type == "eda" else b["temp"],
                        pop_size,
                        b["max_no_improve_gen"],
                        b["max_row_diff"],
                    )
                )
            for max_no_improve_gen in MAX_NO_IMPROVE_GEN_GRID:
                configs.append(
                    (
                        run_type,
                        None if run_type == "eda" else b["aspi_name"],
                        None if run_type == "eda" else b["temp"],
                        b["pop_size"],
                        max_no_improve_gen,
                        b["max_row_diff"],
                    )
                )
            for max_row_diff in MAX_ROW_DIFF_GRID:
                configs.append(
                    (
                        run_type,
                        None if run_type == "eda" else b["aspi_name"],
                        None if run_type == "eda" else b["temp"],
                        b["pop_size"],
                        b["max_no_improve_gen"],
                        max_row_diff,
                    )
                )
            if run_type == "hg_eda":
                for aspi_name in ASPI_GRID:
                    configs.append(
                        (
                            run_type,
                            aspi_name,
                            b["temp"],
                            b["pop_size"],
                            b["max_no_improve_gen"],
                            b["max_row_diff"],
                        )
                    )
                for temp in TEMP_GRID:
                    configs.append(
                        (
                            run_type,
                            b["aspi_name"],
                            temp,
                            b["pop_size"],
                            b["max_no_improve_gen"],
                            b["max_row_diff"],
                        )
                    )
    else:
        raise ValueError(f"Unknown SWEEP_MODE: {mode!r} (use 'oat' or 'full')")

    return _dedupe(configs)


def save_results(path: Path, results: dict) -> Path:
    if path.exists():
        raise ValueError(f"File {path} already exists")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    return path


def append_index_row(index_path: Path, row: dict):
    write_header = not index_path.exists()
    fieldnames = [
        "run_type",
        "tag",
        "aspi_name",
        "aspi",
        "temp",
        "pop_size",
        "max_no_improve_gen",
        "max_row_diff",
        "seed",
        "items_seed",
        "n_obj",
        "pkl_path",
        "mode1_generations",
        "mode2_generations",
        "pf_size",
    ]
    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    items = get_items(N_OBJ, ITEMS_SEED)
    base_params = get_params(N_OBJ, items)
    seed = BASE_SEED + ITEMS_SEED

    out_eda = OUTPUT_ROOT / "eda"
    out_hg = OUTPUT_ROOT / "hg_eda"
    ensure_dirs(out_eda, out_hg)
    index_path = OUTPUT_ROOT / "index.csv"

    configs = iter_configs(SWEEP_MODE)
    print(f"Queued {len(configs)} runs (mode={SWEEP_MODE}) → {OUTPUT_ROOT}")

    for i, (run_type, aspi_name, temp, pop_size, max_no_improve_gen, max_row_diff) in enumerate(
        configs, start=1
    ):
        tag = run_tag(
            run_type, aspi_name, temp, pop_size, max_no_improve_gen, max_row_diff, seed
        )
        out_dir = out_eda if run_type == "eda" else out_hg
        pkl_path = out_dir / f"{tag}.pkl"

        if pkl_path.exists():
            print(f"[{i}/{len(configs)}] skip existing {pkl_path.name}")
            continue

        params = dict(base_params)
        params["pop_size"] = pop_size
        params["max_no_improve_gen"] = max_no_improve_gen
        params["max_row_diff"] = max_row_diff

        aspi = resolve_aspi(aspi_name) if run_type == "hg_eda" else None
        aspi_str = "" if aspi is None else np.array2string(aspi, separator=",")

        print(
            f"[{i}/{len(configs)}] {run_type} "
            f"pop={pop_size} noi={max_no_improve_gen} mrd={max_row_diff}"
            + (f" aspi={aspi_name} temp={temp}" if run_type == "hg_eda" else "")
        )

        if run_type == "eda":
            results = run_eda(params, seed)
        else:
            results = run_eda(params, seed, aspi=aspi, temp=temp)

        save_results(pkl_path, results)

        pf = results["converged_pf_table"][-1]
        append_index_row(
            index_path,
            {
                "run_type": run_type,
                "tag": tag,
                "aspi_name": aspi_name or "",
                "aspi": aspi_str,
                "temp": "" if temp is None else temp,
                "pop_size": pop_size,
                "max_no_improve_gen": max_no_improve_gen,
                "max_row_diff": max_row_diff,
                "seed": seed,
                "items_seed": ITEMS_SEED,
                "n_obj": N_OBJ,
                "pkl_path": str(pkl_path),
                "mode1_generations": results.get("mode 1 generations", ""),
                "mode2_generations": results.get("mode 2 generations", ""),
                "pf_size": len(pf),
            },
        )

    print(f"Done. Index: {index_path}")


if __name__ == "__main__":
    main()
