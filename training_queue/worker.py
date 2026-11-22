import os
import random
import sqlite3
import time

import psycopg

from .store import configured_store, is_contention

LEASE_MS = 1500
HEARTBEAT_SECONDS = LEASE_MS / 3000
INITIAL_IDLE_SECONDS = 0.030
MAX_IDLE_SECONDS = 0.100


def execute(store, conn, job):
    finish_at = time.monotonic() + job["run_ms"] / 1000
    while True:
        remaining = finish_at - time.monotonic()
        if remaining <= 0:
            store.complete(conn, job)
            return
        time.sleep(min(remaining, HEARTBEAT_SECONDS))
        if time.monotonic() < finish_at and not store.renew(conn, job, LEASE_MS):
            return


def main():
    store = configured_store()
    conn = None
    print(f"worker started pid={os.getpid()}", flush=True)
    idle_delay = INITIAL_IDLE_SECONDS
    try:
        while True:
            try:
                if conn is None:
                    conn = store.connect()
                job = store.claim(conn, LEASE_MS)
                if job is None:
                    time.sleep(random.uniform(idle_delay / 2, idle_delay))
                    idle_delay = min(idle_delay * 2, MAX_IDLE_SECONDS)
                else:
                    idle_delay = INITIAL_IDLE_SECONDS
                    execute(store, conn, job)
            except sqlite3.OperationalError as exc:
                if not is_contention(exc):
                    raise
                print(f"worker database contention pid={os.getpid()}: {exc}", flush=True)
                time.sleep(0.05)
            except psycopg.OperationalError:
                if conn is not None:
                    conn.close()
                    conn = None
                print(f"worker database unavailable pid={os.getpid()}; retrying", flush=True)
                time.sleep(0.5)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
