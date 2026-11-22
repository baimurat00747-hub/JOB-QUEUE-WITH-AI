import os
import sqlite3
from contextlib import closing, contextmanager
from time import perf_counter

def schema(primary_key, job_id_type, milliseconds_type):
    return f"""
CREATE TABLE IF NOT EXISTS jobs (
    id {primary_key},
    idempotency_key TEXT NOT NULL UNIQUE,
    run_ms INTEGER NOT NULL CHECK (run_ms BETWEEN 1 AND 60000),
    outcome TEXT NOT NULL CHECK (outcome IN ('succeeded', 'failed')),
    failure_code TEXT CHECK (failure_code IS NULL OR
        (outcome = 'failed' AND failure_code IN ('TIMEOUT', 'INVALID_INPUT'))),
    state TEXT NOT NULL DEFAULT 'queued'
        CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    lease_until_ms {milliseconds_type},
    CHECK ((state = 'running') = (lease_until_ms IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS queued_jobs ON jobs (id) WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS leased_jobs ON jobs (lease_until_ms, id) WHERE state = 'running';
CREATE TABLE IF NOT EXISTS effects (
    job_id {job_id_type} PRIMARY KEY REFERENCES jobs (id),
    attempt INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS job_events (
    id {primary_key},
    job_id {job_id_type} NOT NULL REFERENCES jobs (id),
    attempt INTEGER NOT NULL,
    kind TEXT NOT NULL,
    failure_code TEXT,
    created_at_ms {milliseconds_type} NOT NULL
);
CREATE INDEX IF NOT EXISTS job_event_history ON job_events (job_id, id DESC);
"""


SCHEMA = schema("INTEGER PRIMARY KEY", "INTEGER", "INTEGER")
NOW = """SELECT CAST(strftime('%s', 'now') AS INTEGER) * 1000
         + CAST(substr(strftime('%f', 'now'), 4, 3) AS INTEGER)"""
# Keep the lease lookup off the job history even when SQLite prefers a rowid scan for min(id).
ELIGIBLE = """
SELECT min(id) FROM (
    SELECT min(id) AS id FROM jobs WHERE state = 'queued'
    UNION ALL
    SELECT min(id) AS id FROM jobs INDEXED BY leased_jobs
    WHERE state = 'running' AND lease_until_ms <= ?
)
"""
OWNER = "id = ? AND state = 'running' AND attempt = ? AND lease_until_ms > ?"


class PayloadConflict(Exception):
    pass


def database_time(conn):
    return conn.execute(NOW).fetchone()[0]


class Store:
    def __init__(self, path=None, clock=database_time):
        self.path = path if path is not None else os.getenv("QUEUE_DB", "queue.sqlite3")
        self.clock = clock

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=2.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA synchronous = FULL")
        except BaseException:
            conn.close()
            raise
        return conn

    def initialize(self):
        with closing(self.connect()) as conn:
            if conn.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
                raise RuntimeError("SQLite WAL mode is required")
            conn.executescript(SCHEMA)
            with self.write(conn):
                columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
                if "failure_code" not in columns:
                    conn.execute(
                        """ALTER TABLE jobs ADD COLUMN failure_code TEXT
                           CHECK (failure_code IS NULL OR
                               (outcome = 'failed' AND failure_code IN ('TIMEOUT', 'INVALID_INPUT')))"""
                    )

    @staticmethod
    def query(conn, sql, params=()):
        return conn.execute(sql, params)

    @contextmanager
    def write(self, conn):
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    @contextmanager
    def timed_write(self, conn, timings):
        if timings is None:
            with self.write(conn):
                yield
            return

        started = perf_counter()
        acquired = None
        try:
            with self.write(conn):
                acquired = perf_counter()
                yield
        finally:
            finished = perf_counter()
            timings["sql-wait"] = (acquired or finished) - started
            if acquired is not None:
                timings["sql-txn"] = finished - acquired

    def create(self, key, run_ms, outcome, *, failure_code=None, timings=None):
        started = perf_counter() if timings is not None else None
        with closing(self.connect()) as conn:
            if timings is not None:
                connected = perf_counter()
                timings["sql-connect"] = connected - started
            job = self.by_key(conn, key)
            if timings is not None:
                timings["sql-key-read"] = perf_counter() - connected
            if job is None:
                with self.timed_write(conn, timings):
                    inserted = self.query(
                        conn,
                        """INSERT INTO jobs (idempotency_key, run_ms, outcome, failure_code)
                           VALUES (?, ?, ?, ?)
                           ON CONFLICT (idempotency_key) DO NOTHING
                           RETURNING *""",
                        (key, run_ms, outcome, failure_code),
                    ).fetchone()
                    if inserted is not None:
                        return inserted, True
                    # A concurrent create won between the lookup and the writer lock.
                    job = self.by_key(conn, key)

            if (job["run_ms"], job["outcome"], job["failure_code"]) != (run_ms, outcome, failure_code):
                raise PayloadConflict
            return job, False

    def by_key(self, conn, key):
        return self.query(
            conn,
            "SELECT * FROM jobs WHERE idempotency_key = ?", (key,)
        ).fetchone()

    def by_id(self, conn, job_id):
        return self.query(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def get(self, job_id):
        with closing(self.connect()) as conn:
            return self.by_id(conn, job_id)

    def page(self, after, limit):
        with closing(self.connect()) as conn:
            return self.query(
                conn,
                "SELECT * FROM jobs WHERE id > ? ORDER BY id LIMIT ?", (after, limit)
            ).fetchall()

    def history(self, job_id):
        with closing(self.connect()) as conn:
            return self.query(
                conn,
                """SELECT jobs.*, e.id AS event_id, e.attempt AS event_attempt,
                          e.kind AS event_kind, e.failure_code AS event_failure_code, e.created_at_ms
                   FROM jobs LEFT JOIN job_events AS e ON jobs.id = e.job_id
                   WHERE jobs.id = ? ORDER BY e.id DESC LIMIT 30""",
                (job_id,),
            ).fetchall()

    def record_event(self, conn, job, kind, now, failure_code=None):
        self.query(
            conn,
            """INSERT INTO job_events (job_id, attempt, kind, failure_code, created_at_ms)
               VALUES (?, ?, ?, ?, ?)""",
            (job["id"], job["attempt"], kind, failure_code, now),
        )

    def cancel(self, job_id):
        with closing(self.connect()) as conn, self.write(conn):
            cancelled = self.query(
                conn,
                """UPDATE jobs SET state = 'cancelled', lease_until_ms = NULL
                   WHERE id = ? AND state IN ('queued', 'running')
                   RETURNING *""",
                (job_id,),
            ).fetchone()
            if cancelled is not None:
                self.record_event(conn, cancelled, "cancelled", self.clock(conn))
                return cancelled
            return self.by_id(conn, job_id)

    def claim(self, conn, lease_ms):
        # Avoid queuing for the writer lock when the queue looks empty.
        if conn.execute(ELIGIBLE, (self.clock(conn),)).fetchone()[0] is None:
            return None
        with self.write(conn):
            now = self.clock(conn)
            job_id = conn.execute(ELIGIBLE, (now,)).fetchone()[0]
            if job_id is None:
                return None
            return self.start_attempt(conn, job_id, now, lease_ms)

    def start_attempt(self, conn, job_id, now, lease_ms):
        job = self.query(
            conn,
            """UPDATE jobs SET state = 'running', attempt = attempt + 1, lease_until_ms = ?
               WHERE id = ? RETURNING *""",
            (now + lease_ms, job_id),
        ).fetchone()
        if job["attempt"] == 1:
            self.record_event(conn, job, "claimed", now)
        else:
            self.record_event(conn, job, "reclaimed", now, "WORKER_LOST")
        return job

    def owner_time(self, conn, job_id):
        return self.clock(conn)

    def renew(self, conn, job, lease_ms):
        with self.write(conn):
            now = self.owner_time(conn, job["id"])
            updated = self.query(
                conn,
                f"UPDATE jobs SET lease_until_ms = ? WHERE {OWNER}",
                (now + lease_ms, job["id"], job["attempt"], now),
            )
            return updated.rowcount == 1

    def complete(self, conn, job):
        with self.write(conn):
            now = self.owner_time(conn, job["id"])
            updated = self.query(
                conn,
                f"UPDATE jobs SET state = ?, lease_until_ms = NULL WHERE {OWNER}",
                (job["outcome"], job["id"], job["attempt"], now),
            )
            if updated.rowcount != 1:
                return False
            if job["outcome"] == "succeeded":
                self.query(
                    conn,
                    "INSERT INTO effects (job_id, attempt) VALUES (?, ?)",
                    (job["id"], job["attempt"]),
                )
            self.record_event(conn, job, job["outcome"], now, job["failure_code"])
            return True


def configured_store():
    if url := os.environ.get("QUEUE_DATABASE_URL"):
        from .postgres import PostgresStore

        return PostgresStore(url)
    return Store()


def is_contention(exc):
    return exc.sqlite_errorcode & 0xff in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) if hasattr(exc, "sqlite_errorcode") else str(exc) in ("database is locked", "database table is locked")
