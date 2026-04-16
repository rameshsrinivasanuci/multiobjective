import socket
import threading

HOST = '127.0.0.1'
PORT = 5000


def add_one_to_stats(stats):
    """
    Add 1 to each number in the list.
    """
    return [x + 1 for x in stats]


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
        stats = [float(x) for x in raw_message.split(",")]
        print(f"[PARSED] Stats: {stats}")

        updated_stats = add_one_to_stats(stats)
        print(f"[UPDATED] Stats after +1: {updated_stats}")

        # Convert back to comma-separated string
        response = ",".join(str(x) for x in updated_stats)
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