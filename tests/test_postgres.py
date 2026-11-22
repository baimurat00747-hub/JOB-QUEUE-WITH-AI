import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier
from uuid import uuid4

import psycopg
from psycopg import sql

from training_queue.postgres import PostgresStore
from training_queue.store import PayloadConflict

URL = os.environ.get("TEST_QUEUE_DATABASE_URL")


@unittest.skipUnless(URL, "set TEST_QUEUE_DATABASE_URL to run PostgreSQL tests")
class PostgresQueueTests(unittest.TestCase):
    def setUp(self):
        schema = sql.Identifier(f"queue_test_{uuid4().hex}")
        with psycopg.connect(URL, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(schema))

        def drop_schema():
            with psycopg.connect(URL, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))

        self.addCleanup(drop_schema)

        class IsolatedStore(PostgresStore):
            def connect(self):
                conn = super().connect()
                conn.execute(sql.SQL("SET search_path TO {}").format(schema))
                return conn

        self.now = 1000
        self.store = IsolatedStore(URL, clock=lambda conn: self.now)
        self.store.initialize()

    def test_locked_job_does_not_block_other_worker(self):
        first, _ = self.store.create("first", 1, "succeeded")
        second, _ = self.store.create("second", 1, "succeeded")
        with closing(self.store.connect()) as locked, closing(self.store.connect()) as worker:
            with locked.transaction():
                locked.execute("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (first["id"],))
                self.assertEqual(self.store.claim(worker, 100)["id"], second["id"])
            self.assertEqual(self.store.claim(worker, 100)["id"], first["id"])

    def test_reclaimed_attempt_cannot_commit(self):
        job, _ = self.store.create("recover", 1, "succeeded")
        with closing(self.store.connect()) as worker:
            old = self.store.claim(worker, 100)
            self.now += 100
            new = self.store.claim(worker, 100)
            self.assertFalse(self.store.renew(worker, old, 100))
            self.assertFalse(self.store.complete(worker, old))
            self.assertTrue(self.store.complete(worker, new))
            effects = worker.execute("SELECT job_id, attempt FROM effects").fetchall()
        self.assertEqual(effects, [{"job_id": job["id"], "attempt": 2}])
        history = self.store.history(job["id"])
        self.assertEqual(
            [(event["event_kind"], event["event_failure_code"]) for event in history],
            [("succeeded", None), ("reclaimed", "WORKER_LOST"), ("claimed", None)],
        )

    def test_concurrent_retry_and_cancellation(self):
        barrier = Barrier(2)

        def submit():
            barrier.wait(timeout=3)
            return self.store.create("same-key", 1, "failed", failure_code="TIMEOUT")

        with ThreadPoolExecutor(max_workers=2) as pool:
            calls = [pool.submit(submit) for _ in range(2)]
            pairs = [call.result(timeout=7) for call in calls]
        self.assertEqual({row["id"] for row, _ in pairs}, {pairs[0][0]["id"]})
        self.assertEqual(sum(created for _, created in pairs), 1)
        with self.assertRaises(PayloadConflict):
            self.store.create("same-key", 1, "failed", failure_code="INVALID_INPUT")

        with closing(self.store.connect()) as worker:
            job = self.store.claim(worker, 100)
            self.assertEqual(self.store.cancel(job["id"])["state"], "cancelled")
            self.assertFalse(self.store.complete(worker, job))
            self.assertEqual(worker.execute("SELECT * FROM effects").fetchall(), [])
