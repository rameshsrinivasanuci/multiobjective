import pandas as pd
import numpy as np
import os
import eda
import eda_distance
import pickle


def ask_for_aspiration(items, stat_columns, n_selected):
    """Print objective ranges and collect comma-separated aspiration values."""

    n_obj = len(stat_columns)
    objective_totals = items[:, :n_obj] * n_selected
    print("Objective order:", stat_columns)
    print("max:", np.around(np.max(objective_totals, axis=0), 2))
    print("min:", np.around(np.min(objective_totals, axis=0), 2))
    print("median:", np.around(np.median(objective_totals, axis=0), 2))
    while True:
        answer = input(
            f"Enter {n_obj} aspiration values separated by commas: "
        )
        try:
            aspiration = np.array(
                [float(value.strip()) for value in answer.split(",")]
            )
        except ValueError:
            print("Invalid input. Enter only numbers separated by commas.")
            continue
        if len(aspiration) != n_obj:
            print(f"Invalid input. Enter exactly {n_obj} values.")
        elif not np.all(np.isfinite(aspiration)):
            print("Invalid input. All values must be finite numbers.")
        else:
            return aspiration


def run_one_year(year, config):
    """Run the EDA pipeline for a single year. Returns True on success,
    False if the year was skipped (e.g. missing file or salary cap)."""

    DATA_DIR = config["DATA_DIR"]
    OUTPUT_DIR = config["OUTPUT_DIR"]
    STAT_COLUMNS = config["STAT_COLUMNS"]
    RESCALE = config["RESCALE"]
    FLIP = config["FLIP"]
    MINUTES_COL = config["MINUTES_COL"]
    PER_MINUTES = config["PER_MINUTES"]
    SALARY_SRC = config["SALARY_SRC"]
    SALARY_CAP = config["SALARY_CAP"]
    n_selected = config["n_selected"]
    pop_size = config["pop_size"]
    generations = config["generations"]
    max_no_improve_gen = config["max_no_improve_gen"]
    max_row_diff = config["max_row_diff"]
    seed = config["seed"]

    FILE = f"nba_fantasy_{year}.csv"
    file_path = os.path.join(DATA_DIR, FILE)

    if year not in SALARY_CAP:
        print(f"[{year}] Skipping: no salary cap entry for this year.")
        return False

    if not os.path.exists(file_path):
        print(f"[{year}] Skipping: file not found ({file_path}).")
        return False

    print(f"[{year}] Running EDA...")

    # Read data
    df = pd.read_csv(file_path)
    df[RESCALE] = df[RESCALE].mul(PER_MINUTES).div(df[MINUTES_COL], axis=0)
    df[FLIP] = -df[FLIP]

    items = df[STAT_COLUMNS + [SALARY_SRC]].to_numpy()

    n_obj = len(STAT_COLUMNS)
    n_con = 1
    capacity = SALARY_CAP[year]

    # ---------- Run unbiased EDA -----------
    eda_run = eda.KnapsackEDA(
        items=items,
        capacity=capacity,
        n_selected=n_selected,
        n_obj=n_obj,
        pop_size=pop_size,
        generations=generations,
        max_no_improve_gen=max_no_improve_gen,
        max_row_diff=max_row_diff,
        seed=seed,
    )
    # ---------------- End ---------------------

    # ----------- Run biased EDA -----------
    # NOTE: if you switch to the biased/aspiration-driven EDA, ask_for_aspiration()
    # will prompt interactively for EACH year in the loop. Consider pre-defining
    # a per-year aspiration dict instead of prompting inside a multi-year run.
    #
    # aspi = ask_for_aspiration(items, STAT_COLUMNS, n_selected)
    # print("Aspiration:", aspi)
    # temp = 0.1
    # eda_run = eda_distance.KnapsackEDA(
    #     items=items,
    #     capacity=capacity,
    #     n_selected=n_selected,
    #     n_obj=n_obj,
    #     pop_size=pop_size,
    #     generations=generations,
    #     max_no_improve_gen=max_no_improve_gen,
    #     max_row_diff=max_row_diff,
    #     seed=seed,
    #     aspi=aspi,
    #     if_rank=True,
    #     temp=temp,
    # )
    # ---------------- End ---------------------

    results = eda_run.run()

    # save all results
    results_path = os.path.join(OUTPUT_DIR, f"results_{year}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # Save pf results
    pf = results["converged_pf_table"][-1]
    pf_indices = results["converged_pf_population_table"][-1]
    pf_df = pd.DataFrame(pf, columns=STAT_COLUMNS)
    pf_indices_df = pd.DataFrame(pf_indices)

    output_file_pf = os.path.join(OUTPUT_DIR, f"pf_{year}.csv")
    output_file_pf_indices = os.path.join(OUTPUT_DIR, f"pf_indices_{year}.csv")

    pf_df.to_csv(output_file_pf, index=False)
    pf_indices_df.to_csv(output_file_pf_indices, index=False)

    print(f"[{year}] Done. Pareto front size: {len(pf_df)}")
    return True


def main():

    # DATA_DIR = "../nba_fantasy/"
    DATA_DIR = "/home/ramesh/Code/nba_fantasy/"
    OUTPUT_DIR = "/home/ramesh/Code/nba_fantasy/eda_results/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Range of years to process (inclusive). Change these to control the loop,
    # or replace with an explicit list, e.g. YEARS = [2005, 2010, 2015].
    YEAR_START = 2000
    YEAR_END = 2016
    YEARS = range(YEAR_START, YEAR_END + 1)

    # Your stat columns, in the FIXED display order you want (keep this order
    # stable across runs so matrices are comparable year to year).
    #   'G' = games played
    # STAT_COLUMNS = ["PTS", "TRB", "AST", "STL","BLK", "3P","G"]
    STAT_COLUMNS = ["STL", "3P", "AST", "TRB", "BLK"]

    # Of the stat columns above, which are per-game counting stats that should
    # be put on a per-36-minute basis:  stat_per36 = stat_per_game / MP * 36
    # (MP and the counting stats here are already per-game averages, so the
    #  "per game" cancels and this yields a clean per-36 number.)
    RESCALE = ["STL", "3P", "AST", "TRB", "BLK"]   # G is left as-is
    FLIP = []
    MINUTES_COL = "MP"
    PER_MINUTES = 36

    SALARY_SRC = "PREDICTED_SALARY"
    SALARY_LABEL = "salary"
    SALARY_CAP = {
        2000: 34.000, 2001: 35.500, 2002: 42.500, 2003: 40.271,
        2004: 43.870, 2005: 43.870, 2006: 49.500, 2007: 53.135,
        2008: 55.630, 2009: 58.680, 2010: 57.700, 2011: 58.044,
        2012: 58.044, 2013: 58.679, 2014: 58.679, 2015: 63.065,
        2016: 70.000, 2017: 94.143, 2018: 99.093, 2019: 101.869,
        2020: 109.140, 2021: 109.140, 2022: 112.414, 2023: 123.655,
        2024: 136.021, 2025: 140.588, 2026: 154.647,
    }

    config = dict(
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        STAT_COLUMNS=STAT_COLUMNS,
        RESCALE=RESCALE,
        FLIP=FLIP,
        MINUTES_COL=MINUTES_COL,
        PER_MINUTES=PER_MINUTES,
        SALARY_SRC=SALARY_SRC,
        SALARY_CAP=SALARY_CAP,
        n_selected=10,
        pop_size=1_000,
        generations=500,
        max_no_improve_gen=3,
        max_row_diff=0.05,  # fraction of the number of Pareto front (depends on number of objectives)
        seed=1123,
    )

    completed, skipped = [], []
    for year in YEARS:
        ok = run_one_year(year, config)
        (completed if ok else skipped).append(year)

    print("\n===== Summary =====")
    print(f"Completed ({len(completed)}): {completed}")
    print(f"Skipped   ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()
