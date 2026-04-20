import socket
import threading
import eda
from pathlib import Path
import numpy as np
import pandas as pd
import json

HOST = '127.0.0.1'
PORT = 5000


def run_eda(user_input):
    stimuli_dir = Path("stimuli")  
    csv_path = stimuli_dir / f"civ_items_trial_0.csv"
    df = pd.read_csv(csv_path)
    numeric = df.drop(columns=["Name"], errors="ignore")
    items = numeric.to_numpy(dtype=np.int64)
    n_obj = items.shape[1] - 1

    if n_obj == 3:
        n_selected = 6
        max_row_diff = 5 
    elif n_obj == 5:
        n_selected = 10
        max_row_diff = 500
    else:
        raise ValueError(f"Number of objectives {n_obj} not supported")

    capacity = n_selected * 10
    pop_size = 1_000
    generations = 100 
    max_no_improve_gen = 5

    # human input
    aspi_item = np.array(user_input)
    aspi_unit = aspi_item / np.linalg.norm(aspi_item)
    item_scores = items[:, :n_obj] @ aspi_unit
    r = item_scores.argsort().argsort().astype(float)
    s = r / (r.max() + 1e-12)
    logits = s / 0.3
    logits -= logits.max() 
    p_rank = np.exp(logits)
    p_rank /= p_rank.sum()

    eda_process = eda.KnapsackEDA(
        items=items,
        capacity=capacity,
        n_selected=n_selected,
        n_obj=n_obj,
        pop_size=pop_size,
        generations=generations,
        max_no_improve_gen=max_no_improve_gen,
        max_row_diff=max_row_diff,
        seed=1123,
        p_rank=p_rank
    )
    results = eda_process.run()
    return results['converged_pf_table'][-1][:5]  # return the first 5 pareto front solutions


def handle_client(conn, addr):
    print(f"\n[CONNECTED] Unity connected from {addr}")

    try:
        data = conn.recv(4096)
        if not data:
            print("[WARNING] No data received")
            return

        raw_message = data.decode('utf-8').strip()
        print(f"[RAW RECEIVED] {raw_message}")

        # Example incoming message: "10,25.5,8,14"
        user_input = [float(x) for x in raw_message.split(",")]
        print(f"[PARSED] User input: {user_input}")

        pareto_fronts = run_eda(user_input)
        print(f"[UPDATED] Pareto front: {pareto_fronts}")

        pareto_fronts_list = pareto_fronts.tolist()
        response = json.dumps(pareto_fronts_list)
        conn.sendall(response.encode('utf-8'))
        print(f"[SENT] {response}")

    except ValueError as e:
        error_msg = "ERROR: could not parse numeric stats"
        conn.sendall(error_msg.encode('utf-8'))
        print(f"[ERROR] ValueError: {e}")

    except Exception as e:
        error_msg = f"ERROR: {e}"
        conn.sendall(error_msg.encode('utf-8'))
        print(f"[ERROR] {e}")

    finally:
        conn.close()
        print(f"[DISCONNECTED] {addr}")


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()
    print(f"\n[LISTENING] Server running on {HOST}:{PORT}\n")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()


if __name__ == "__main__":
    start_server()