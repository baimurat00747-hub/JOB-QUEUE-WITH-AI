import json
import socket
import tempfile
import threading
import time
import unittest
from contextlib import closing, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from queue_agent import api as agent
from queue_agent.diagnosis import Advice
from training_queue import api as queue
from training_queue.store import Store
from training_queue.worker import LEASE_MS, execute


@contextmanager
def serve(app: FastAPI):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", ws="none"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not server.started:
                raise RuntimeError("Local test service failed to start.")
            yield url
        finally:
            server.should_exit = True
            thread.join(timeout=5)


class AgentAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls.enterClassContext(patch.object(queue, "store", Store(Path(directory) / "queue.sqlite3")))
        queue_url = cls.enterClassContext(serve(queue.app))
        model = FastAPI()
        cls.requests = []

        @model.post("/v1/chat/completions")
        async def complete(body: dict):
            name = body["tools"][0]["function"]["name"]
            prompt = body["messages"][-1]["content"]
            cls.requests.append((name, prompt))
            if prompt == "provider error":
                raise HTTPException(503)
            message = {"role": "assistant"}
            if prompt != "no call":
                if prompt == "unauthorized":
                    name = "submit_simulated_job"
                    arguments = {}
                elif name == "submit_simulated_job":
                    arguments = {"run_ms": int(prompt)}
                else:
                    arguments = {"job_id": int(prompt)}
                message["tool_calls"] = [{"function": {"name": name, "arguments": json.dumps(arguments)}}]
            return {"choices": [{"message": message}]}

        model_url = cls.enterClassContext(serve(model))
        cls.enterClassContext(patch.dict("os.environ", {
            "QUEUE_URL": queue_url,
            "QUEUE_AGENT_MODEL_URL": f"{model_url}/v1",
        }))
        cls.client = cls.enterClassContext(TestClient(agent.app))

    def submit(self, prompt: str, key: str):
        return self.client.post("/agent/jobs", json={"prompt": prompt}, headers={"Idempotency-Key": key})

    def lookup(self, prompt: str):
        return self.client.post("/agent/jobs/lookup", json={"prompt": prompt})

    def test_optional_span_file_contains_correlated_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "logs" / "agent.jsonl"
            with patch.dict("os.environ", {"QUEUE_AGENT_LOG_FILE": str(log_path)}):
                with self.assertLogs("uvicorn.error", level="INFO"):
                    with TestClient(agent.app) as client:
                        response = client.get("/openapi.json")
            logged, = (json.loads(line) for line in log_path.read_text().splitlines())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(logged["trace_id"], response.headers["X-Trace-Id"])
        self.assertEqual(logged["span"], "http.request")
        self.assertEqual(logged["status"], 200)
        self.assertTrue(logged["timestamp"].endswith("+00:00"))

    def test_submit_retry_conflict_and_lookup(self):
        key = str(uuid4())
        start = len(self.requests)
        created = self.submit("1", key)
        self.assertEqual(created.status_code, 201, created.text)
        job_id = created.json()["id"]

        replay = self.submit("1", key)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json()["id"], job_id)
        self.assertEqual(self.submit("2", key).status_code, 409)

        with closing(queue.store.connect()) as conn:
            job = queue.store.claim(conn, LEASE_MS)
            self.assertEqual(job["id"], job_id)
            execute(queue.store, conn, job)

        lookup = self.lookup(str(job_id))
        self.assertEqual(lookup.status_code, 200, lookup.text)
        self.assertEqual(lookup.json()["state"], "succeeded")
        self.assertEqual(self.requests[start:], [
            ("submit_simulated_job", "1"),
            ("submit_simulated_job", "1"),
            ("submit_simulated_job", "2"),
            ("get_job_status", str(job_id)),
        ])

    def test_lookup_cannot_submit_and_missing_job_returns_404(self):
        count = len(queue.store.page(0, 50))
        self.assertEqual(self.lookup("unauthorized").status_code, 422)
        self.assertEqual(self.lookup(str(2**63 - 1)).status_code, 404)
        self.assertEqual(len(queue.store.page(0, 50)), count)

    def test_rejects_invalid_model_responses_and_arguments(self):
        self.assertEqual(self.lookup("no call").status_code, 422)
        self.assertEqual(self.lookup("0").status_code, 422)
        self.assertEqual(self.lookup("provider error").status_code, 502)
        self.assertEqual(self.submit("1", "bad-key").status_code, 422)

    def test_diagnosis_uses_recorded_failure_and_cites_retrieved_instruction(self):
        job, _ = queue.store.create(str(uuid4()), 1, "failed", failure_code="TIMEOUT")
        with closing(queue.store.connect()) as conn:
            execute(queue.store, conn, queue.store.claim(conn, LEASE_MS))
        source = {"url": "https://example.test/runbook", "title": "Timeout", "content": "Check duration."}
        with patch("queue_agent.diagnosis.search_runbooks", AsyncMock(return_value=[source])) as search:
            with patch("queue_agent.diagnosis.recommend", AsyncMock(return_value=Advice(
                recommended_action="Check duration.",
            ))):
                response = self.client.post(f"/agent/jobs/{job['id']}/diagnose")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {
            "job_id": job["id"], "state": "failed", "reason": "TIMEOUT",
            "recommended_action": "Check duration.",
            "sources": [{"title": "Timeout", "url": "https://example.test/runbook"}],
        })
        search.assert_awaited_once_with("TIMEOUT", "What should the operator do now?")

        with patch("queue_agent.diagnosis.search_runbooks", AsyncMock(return_value=[])):
            with patch("queue_agent.diagnosis.recommend", AsyncMock()) as recommender:
                response = self.client.post(f"/agent/jobs/{job['id']}/diagnose")
        self.assertEqual(response.json()["sources"], [])
        self.assertEqual(response.json()["reason"], "TIMEOUT")
        recommender.assert_not_awaited()

    def test_diagnosis_of_unfinished_job_needs_no_model_or_runbook(self):
        job, _ = queue.store.create(str(uuid4()), 1, "succeeded")
        try:
            with patch("queue_agent.diagnosis.search_runbooks", AsyncMock()) as search:
                response = self.client.post(f"/agent/jobs/{job['id']}/diagnose")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["reason"], "in_progress")
            self.assertEqual(response.json()["sources"], [])
            search.assert_not_awaited()
        finally:
            queue.store.cancel(job["id"])
