import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from training_queue.store import SCHEMA, Store


class ExistingSQLiteTests(unittest.TestCase):
    def test_initialize_preserves_jobs_from_before_failure_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.sqlite3"
            with closing(sqlite3.connect(path)) as legacy:
                legacy.executescript(SCHEMA.replace(
                    "    failure_code TEXT CHECK (failure_code IS NULL OR\n"
                    "        (outcome = 'failed' AND failure_code IN ('TIMEOUT', 'INVALID_INPUT'))),\n",
                    "",
                ))
                legacy.execute(
                    """INSERT INTO jobs (idempotency_key, run_ms, outcome, state)
                       VALUES ('previous', 50, 'failed', 'failed')"""
                )
                legacy.commit()

            store = Store(path)
            store.initialize()
            store.initialize()
            job, created = store.create("previous", 50, "failed")
            self.assertFalse(created)
            self.assertEqual(job["state"], "failed")
            self.assertIsNone(job["failure_code"])
            self.assertIsNone(store.history(job["id"])[0]["event_id"])
