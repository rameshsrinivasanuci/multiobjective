import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

from mrs import get_pf_info, get_query_point, density_filter
from agent import get_aspi, compute_game_score, get_unbiased_pf


# def get_unbiased_pf(block_id: int, trial_id: int):
#     df = pd.read_csv(f"data/unbiased_pf/pf_block_{block_id}_trial_{trial_id}.csv")
#     df_indices = pd.read_csv(f"data/unbiased_pf/pf_indices_block_{block_id}_trial_{trial_id}.csv")
#     return df, df_indices


def main(
    sub_id: int,
    block_id: int,
    trial_id: int,
    submission_id: int,
    slider_values: list[float],
):

    density_threshold = 0.8

    aspi = get_aspi(slider_values)

    df, df_indices = get_unbiased_pf(block_id, trial_id)
    df, kept_idx = density_filter(df, density_threshold)
    info = get_pf_info(df)
    
    query_idx, query_pt = get_query_point(info["data"], aspi, info["iqr"])
    full_idx = int(kept_idx[query_idx])  
    query_items_indices = df_indices.iloc[full_idx]
    
    game_score = compute_game_score(query_pt, info["data"])

    print("recommendation: ", np.round(query_pt, 1))

    return {
        "rec": np.round(query_pt, 1),
        "score": game_score,
        "rec_player_indices": query_items_indices,
    }


if __name__ == "__main__":
    # Example local call; production path passes these from the TCP client.
    raise SystemExit(
        "Call main(trial_id, slider_values) with values from the client "
        "(trial['trial_id'] and slider_values)."
    )
