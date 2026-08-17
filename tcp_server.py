# tcp_server.py

import json
import socket
import struct
import threading
import traceback
from typing import Any

import numpy as np

from agent import main


HOST = "127.0.0.1"
PORT = 5000

# Prevent unexpectedly large requests.
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def receive_exactly(connection: socket.socket, num_bytes: int) -> bytes:
    """
    Receive exactly num_bytes from a TCP connection.

    socket.recv() is not guaranteed to return all requested bytes in one call,
    so this function continues receiving until the requested amount arrives.
    """
    chunks: list[bytes] = []
    bytes_received = 0

    while bytes_received < num_bytes:
        chunk = connection.recv(num_bytes - bytes_received)

        if not chunk:
            raise ConnectionError(
                "The client disconnected before the complete message arrived."
            )

        chunks.append(chunk)
        bytes_received += len(chunk)

    return b"".join(chunks)


def receive_json(connection: socket.socket) -> dict[str, Any]:
    """
    Receive one length-prefixed JSON message.

    Protocol:
        1. First 4 bytes: unsigned message length in network byte order.
        2. Remaining bytes: UTF-8 encoded JSON.
    """
    header = receive_exactly(connection, 4)
    message_length = struct.unpack("!I", header)[0]

    if message_length == 0:
        raise ValueError("Received an empty message.")

    if message_length > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Message size {message_length} exceeds the "
            f"{MAX_MESSAGE_SIZE}-byte limit."
        )

    message_bytes = receive_exactly(connection, message_length)

    try:
        message = json.loads(message_bytes.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("The request is not valid UTF-8.") from error
    except json.JSONDecodeError as error:
        raise ValueError("The request is not valid JSON.") from error

    if not isinstance(message, dict):
        raise ValueError("The top-level JSON value must be an object.")

    return message


def send_json(
    connection: socket.socket,
    message: dict[str, Any],
) -> None:
    """
    Send one length-prefixed JSON message.
    """
    message_bytes = json.dumps(
        message,
        ensure_ascii=False,
    ).encode("utf-8")

    if len(message_bytes) > MAX_MESSAGE_SIZE:
        raise ValueError("Response exceeds the maximum message size.")

    header = struct.pack("!I", len(message_bytes))
    connection.sendall(header + message_bytes)


def validate_slider_values(value: Any) -> list[float]:
    """
    Validate and convert the slider values received from the laptop.
    """
    if not isinstance(value, list):
        raise ValueError("'slider_values' must be a list.")

    if len(value) == 0:
        raise ValueError("'slider_values' cannot be empty.")

    slider_values: list[float] = []

    for index, slider_value in enumerate(value):
        if isinstance(slider_value, bool) or not isinstance(
            slider_value,
            (int, float),
        ):
            raise ValueError(
                f"Slider value at index {index} must be numeric."
            )

        slider_values.append(float(slider_value))

    return slider_values


def run_trial_computation(
    trial: dict[str, Any],
    submission_id: int,
    slider_values: list[float],
) -> dict[str, Any]:
    """
    Run agent.main for one trial and return JSON-serializable rec/mrs.
    """
    sub_id = trial.get("sub_id")
    if sub_id is None:
        raise ValueError("'trial' must contain 'sub_id'.")
    block_id = trial.get("block_id")
    if block_id is None:
        raise ValueError("'trial' must contain 'block_id'.")
    trial_id = trial.get("trial_id")
    if trial_id is None:
        raise ValueError("'trial' must contain 'trial_id'.")

    result = main(sub_id, block_id, trial_id, submission_id, slider_values)

    return {
        "rec": np.asarray(result["rec"], dtype=float).tolist(),
        "mrs": np.asarray(result["mrs"], dtype=float).tolist(),
        "score": int(result["score"]),
        "rec_player_indices": np.asarray(result["rec_player_indices"], dtype=int).tolist(),
    }


def process_request(request: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a client request and route it to the appropriate server function.
    """
    request_type = request.get("type")

    if request_type == "ping":
        return {
            "ok": True,
            "type": "pong",
        }

    if request_type != "run_trial":
        raise ValueError(
            "Unknown request type. Expected 'ping' or 'run_trial'."
        )

    trial = request.get("trial")

    if not isinstance(trial, dict):
        raise ValueError("'trial' must be a JSON object.")

    slider_values = validate_slider_values(
        request.get("slider_values")
    )

    submission_id = request.get("submission_id")
    
    if submission_id is None:
        raise ValueError("'request' must contain 'submission_id'.")

    result = run_trial_computation(
        trial=trial,
        submission_id=submission_id,
        slider_values=slider_values,
    )

    return {
        "ok": True,
        **result,
    }


def handle_client(
    connection: socket.socket,
    client_address: tuple[str, int],
) -> None:
    """
    Handle one connected client.

    This implementation processes one request and then closes the connection.
    """
    print(f"Client connected: {client_address[0]}:{client_address[1]}")

    try:
        connection.settimeout(300.0)

        request = receive_json(connection)
        print(
            f"Received request type "
            f"{request.get('type')!r} from {client_address}"
        )

        response = process_request(request)
        send_json(connection, response)

    except Exception as error:
        print(f"Request from {client_address} failed: {error}")
        traceback.print_exc()

        error_response = {
            "ok": False,
            "error": str(error),
            "error_type": type(error).__name__,
        }

        try:
            send_json(connection, error_response)
        except Exception as send_error:
            print(f"Could not send error response: {send_error}")

    finally:
        connection.close()
        print(f"Client disconnected: {client_address}")


def start_server(
    host: str = HOST,
    port: int = PORT,
) -> None:
    """
    Start the TCP server and accept client connections indefinitely.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Allow the server to restart without waiting for the old socket
        # address to time out.
        server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server_socket.bind((host, port))
        server_socket.listen()

        print(f"TCP server listening on {host}:{port}")

        try:
            while True:
                connection, client_address = server_socket.accept()

                client_thread = threading.Thread(
                    target=handle_client,
                    args=(connection, client_address),
                    daemon=True,
                )
                client_thread.start()

        except KeyboardInterrupt:
            print("\nTCP server stopped.")


if __name__ == "__main__":
    start_server()