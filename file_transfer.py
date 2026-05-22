#!/usr/bin/env python3
"""
Simple file transfer script with server and client modes.
Server shares current directory; client downloads files or folders interactively.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple


DEFAULT_PORT = 9000
DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024
DEFAULT_THREADS = min(8, (os.cpu_count() or 4) * 2)
DEFAULT_MAX_FILE_RETRIES = 3
DEFAULT_MAX_CHUNK_RETRIES = 5
BACKOFF_BASE = 0.5
BACKOFF_MAX = 8.0


def send_json(sock: socket.socket, obj: Dict) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def recvall(sock: socket.socket, size: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def recv_json(sock: socket.socket) -> Optional[Dict]:
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    length = struct.unpack("!I", raw_len)[0]
    payload = recvall(sock, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_index(root: str) -> List[Dict]:
    items: List[Dict] = []
    for base, dirs, files in os.walk(root):
        for d in dirs:
            full = os.path.join(base, d)
            rel = os.path.relpath(full, root)
            if rel == ".":
                continue
            items.append({"path": rel.replace(os.sep, "/"), "type": "dir", "size": 0})
        for f in files:
            full = os.path.join(base, f)
            rel = os.path.relpath(full, root)
            if rel == ".":
                continue
            items.append(
                {
                    "path": rel.replace(os.sep, "/"),
                    "type": "file",
                    "size": os.path.getsize(full),
                }
            )
    items.sort(key=lambda x: (x["type"] != "dir", x["path"]))
    return items


def safe_path(root_real: str, rel_path: str) -> str:
    if not rel_path:
        raise ValueError("Missing path")
    rel_path = rel_path.strip().replace("/", os.sep)
    full = os.path.realpath(os.path.join(root_real, rel_path))
    if not (full == root_real or full.startswith(root_real + os.sep)):
        raise ValueError("Path traversal blocked")
    return full


def handle_client(conn: socket.socket, root_real: str) -> None:
    conn.settimeout(60)
    with conn:
        while True:
            req = recv_json(conn)
            if req is None:
                break
            cmd = req.get("cmd")
            try:
                if cmd == "LIST":
                    items = build_index(root_real)
                    send_json(conn, {"ok": True, "items": items})
                elif cmd == "GET_CHUNK":
                    rel = req["path"]
                    offset = int(req["offset"])
                    length = int(req["length"])
                    full = safe_path(root_real, rel)
                    if not os.path.isfile(full):
                        raise FileNotFoundError("Not a file")
                    if length <= 0:
                        raise ValueError("Invalid length")
                    file_size = os.path.getsize(full)
                    if offset < 0 or offset >= file_size:
                        raise ValueError("Offset out of range")
                    if offset + length > file_size:
                        raise ValueError("Chunk out of range")
                    with open(full, "rb") as f:
                        f.seek(offset)
                        data = f.read(length)
                    checksum = sha256_bytes(data)
                    send_json(conn, {"ok": True, "length": len(data), "checksum": checksum})
                    if data:
                        conn.sendall(data)
                elif cmd == "FILE_HASH":
                    rel = req["path"]
                    full = safe_path(root_real, rel)
                    if not os.path.isfile(full):
                        raise FileNotFoundError("Not a file")
                    checksum = sha256_file(full)
                    send_json(conn, {"ok": True, "checksum": checksum})
                else:
                    send_json(conn, {"ok": False, "error": "Unknown cmd"})
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                send_json(conn, {"ok": False, "error": str(exc)})


def run_server(host: str, port: int) -> None:
    root_real = os.path.realpath(os.getcwd())
    print(f"Serving {root_real} on {host}:{port}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(64)
    srv.settimeout(1.0)
    stop_event = threading.Event()

    def handle_sigint(_signum, _frame) -> None:
        if not stop_event.is_set():
            stop_event.set()
            print("\nShutting down server.")
            try:
                srv.close()
            except OSError:
                pass

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)
    try:
        while not stop_event.is_set():
            try:
                conn, _addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                if stop_event.is_set():
                    break
                raise
            t = threading.Thread(target=handle_client, args=(conn, root_real), daemon=True)
            t.start()
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        srv.close()


def request_list(host: str, port: int) -> List[Dict]:
    with socket.create_connection((host, port), timeout=10) as sock:
        send_json(sock, {"cmd": "LIST"})
        resp = recv_json(sock)
        if not resp or not resp.get("ok"):
            raise RuntimeError(resp.get("error", "LIST failed"))
        return resp["items"]


def request_file_hash(host: str, port: int, path: str) -> str:
    with socket.create_connection((host, port), timeout=10) as sock:
        send_json(sock, {"cmd": "FILE_HASH", "path": path})
        resp = recv_json(sock)
        if not resp or not resp.get("ok"):
            raise RuntimeError(resp.get("error", "FILE_HASH failed"))
        return resp["checksum"]


def download_file_once(
    host: str,
    port: int,
    rel_path: str,
    size: int,
    threads: int,
    chunk_size: int,
    max_chunk_retries: int,
) -> None:
    local_path = rel_path.replace("/", os.sep)
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    with open(local_path, "wb") as f:
        f.truncate(size)
    f = open(local_path, "r+b")

    tasks: List[Tuple[int, int]] = []
    offset = 0
    while offset < size:
        length = min(chunk_size, size - offset)
        tasks.append((offset, length))
        offset += length

    task_index = 0
    task_lock = threading.Lock()
    file_lock = threading.Lock()
    stop_event = threading.Event()
    errors: List[Exception] = []

    def get_task() -> Optional[Tuple[int, int]]:
        nonlocal task_index
        with task_lock:
            if task_index >= len(tasks):
                return None
            task = tasks[task_index]
            task_index += 1
            return task

    def worker() -> None:
        sock: Optional[socket.socket] = None
        try:
            sock = socket.create_connection((host, port), timeout=10)
            sock.settimeout(30)
            while not stop_event.is_set():
                task = get_task()
                if task is None:
                    break
                chunk_offset, length = task
                attempt = 0
                while attempt < max_chunk_retries and not stop_event.is_set():
                    try:
                        send_json(
                            sock,
                            {
                                "cmd": "GET_CHUNK",
                                "path": rel_path,
                                "offset": chunk_offset,
                                "length": length,
                            },
                        )
                        header = recv_json(sock)
                        if not header or not header.get("ok"):
                            raise RuntimeError(header.get("error", "Chunk failed"))
                        data_len = int(header.get("length", 0))
                        if data_len != length:
                            raise RuntimeError("Chunk size mismatch")
                        data = recvall(sock, data_len)
                        if data is None or len(data) != data_len:
                            raise RuntimeError("Incomplete chunk")
                        checksum = header.get("checksum")
                        if checksum != sha256_bytes(data):
                            raise ValueError("Checksum mismatch")
                        with file_lock:
                            f.seek(chunk_offset)
                            f.write(data)
                        break
                    except (OSError, ValueError, RuntimeError, socket.timeout) as exc:
                        attempt += 1
                        if attempt >= max_chunk_retries:
                            stop_event.set()
                            errors.append(exc)
                            break
                        backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)))
                        time.sleep(backoff)
                        try:
                            sock.close()
                        except Exception:
                            pass
                        sock = socket.create_connection((host, port), timeout=10)
                        sock.settimeout(30)
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    workers = []
    num_workers = min(max(1, threads), len(tasks))
    for _ in range(num_workers):
        t = threading.Thread(target=worker)
        t.start()
        workers.append(t)
    for t in workers:
        t.join()
    f.close()

    if stop_event.is_set():
        raise RuntimeError(errors[0] if errors else "Transfer failed")


def download_file_with_retry(
    host: str,
    port: int,
    rel_path: str,
    size: int,
    threads: int,
    chunk_size: int,
    max_file_retries: int,
    max_chunk_retries: int,
) -> None:
    attempt = 0
    while attempt < max_file_retries:
        try:
            download_file_once(
                host, port, rel_path, size, threads, chunk_size, max_chunk_retries
            )
            remote_hash = request_file_hash(host, port, rel_path)
            local_hash = sha256_file(rel_path.replace("/", os.sep))
            if remote_hash != local_hash:
                raise ValueError("File checksum mismatch")
            return
        except Exception:
            attempt += 1
            if attempt >= max_file_retries:
                raise
            try:
                os.remove(rel_path.replace("/", os.sep))
            except OSError:
                pass
            backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)))
            time.sleep(backoff)


def human_size(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def prompt_choice(prompt: str, choices: List[str]) -> str:
    choice_set = {c.lower(): c for c in choices}
    while True:
        val = input(prompt).strip().lower()
        if val in choice_set:
            return choice_set[val]


def prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        val = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not val:
            return default
        if val in {"y", "yes"}:
            return True
        if val in {"n", "no"}:
            return False


def select_item(items: List[Dict]) -> Optional[Dict]:
    if not items:
        print("No files or directories.")
        return None
    for idx, item in enumerate(items, 1):
        label = item["path"] + ("/" if item["type"] == "dir" else "")
        size = human_size(item["size"]) if item["type"] == "file" else ""
        print(f"{idx}. {label} {size}".rstrip())
    while True:
        choice = input("Select by number or path (q to quit): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(items):
                return items[num - 1]
        else:
            for item in items:
                if item["path"] == choice or item["path"] + "/" == choice:
                    return item
        print("Invalid selection.")


def files_under_dir(items: List[Dict], dir_path: str) -> List[Dict]:
    prefix = dir_path.rstrip("/") + "/"
    files = [
        item
        for item in items
        if item["type"] == "file" and item["path"].startswith(prefix)
    ]
    files.sort(key=lambda x: x["path"])
    return files


def run_client(
    host: str,
    port: int,
    threads: int,
    chunk_size: int,
    max_file_retries: int,
    max_chunk_retries: int,
) -> None:
    items = request_list(host, port)
    selection = select_item(items)
    if not selection:
        return

    if selection["type"] == "file":
        local_path = selection["path"].replace("/", os.sep)
        if os.path.exists(local_path):
            if not prompt_yes_no(f"{local_path} exists. Overwrite?", False):
                print("Skipped.")
                return
        download_file_with_retry(
            host,
            port,
            selection["path"],
            selection["size"],
            threads,
            chunk_size,
            max_file_retries,
            max_chunk_retries,
        )
        print("Download complete.")
        return

    files = files_under_dir(items, selection["path"])
    if not files:
        print("Directory is empty.")
        return

    total_size = sum(f["size"] for f in files)
    print(f"{len(files)} files, total {human_size(total_size)}")
    if not prompt_yes_no("Download this directory?", True):
        return

    existing = [f for f in files if os.path.exists(f["path"].replace("/", os.sep))]
    policy = "overwrite"
    if existing:
        policy = prompt_choice(
            f"{len(existing)} files already exist. Overwrite (o), Skip (s), Cancel (c): ",
            ["o", "s", "c"],
        )
        if policy == "c":
            return

    import concurrent.futures

    def download_wrapper(f: Dict) -> None:
        local_path = f["path"].replace("/", os.sep)
        if os.path.exists(local_path) and policy == "s":
            return
        download_file_with_retry(
            host,
            port,
            f["path"],
            f["size"],
            threads,
            chunk_size,
            max_file_retries,
            max_chunk_retries,
        )
        print(f"Downloaded {f['path']}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, threads)) as executor:
        futures = [executor.submit(download_wrapper, f) for f in files]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def prompt_mode() -> str:
    while True:
        mode = input("Mode (server(s)/client(c)): ").strip().lower()
        if mode in {"s", "server"}:
            return "server"
        elif mode in {"c", "client"}:
            return "client"


def prompt_host() -> str:
    while True:
        host = input("Server IP/Host: ").strip()
        if host:
            return host


def main() -> None:
    parser = argparse.ArgumentParser(description="File transfer script.")
    parser.add_argument("--mode", choices=["server", "client"], help="Run as server or client")
    parser.add_argument("--host", help="Server host (client) or bind host (server)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-file-retries", type=int, default=DEFAULT_MAX_FILE_RETRIES)
    parser.add_argument("--max-chunk-retries", type=int, default=DEFAULT_MAX_CHUNK_RETRIES)
    args = parser.parse_args()

    mode = args.mode or prompt_mode()
    if mode == "server":
        host = args.host or "0.0.0.0"
        run_server(host, args.port)
    else:
        host = args.host or prompt_host()
        run_client(
            host,
            args.port,
            args.threads,
            args.chunk_size,
            args.max_file_retries,
            args.max_chunk_retries,
        )


if __name__ == "__main__":
    main()
