import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient

from training_queue.api import Job

from .api import app
from .auth import signed_token
from .diagnosis import Diagnosis

PG = os.environ.get("TEST_QUEUE_DATABASE_URL")
REDIS = os.environ.get("TEST_QUEUE_REDIS_URL")


@unittest.skipUnless(PG and REDIS, "set test PostgreSQL and Redis URLs")
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {
            "QUEUE_AGENT_JWT_SECRET": "local-tests-only-signing-key-32-bytes-long",
            "QUEUE_RUNBOOK_DATABASE_URL": PG,
            "QUEUE_AGENT_REDIS_URL": REDIS,
            "QUEUE_AGENT_RATE_LIMIT": "10",
            "QUEUE_AGENT_PROVIDER": "ollama",
        }))
        self.enterContext(patch("queue_agent.api.load_history", AsyncMock()))
        self.submit = self.enterContext(patch("queue_agent.api.submit_retry", AsyncMock(
            return_value=Job(id=77, attempt=0, run_ms=50, outcome="succeeded", state="queued"),
        )))
        self.diagnose = self.enterContext(patch("queue_agent.workflow.diagnose", AsyncMock(
            return_value=Diagnosis(
                job_id=1, state="failed", reason="TIMEOUT", recommended_action="Review the timeout budget.",
                sources=[{"title": "Timeout", "url": "https://example.test/runbook"}],
            ),
        )))
        subject = uuid4().hex
        self.operator = self.headers(subject, "operator")
        self.downgraded = self.headers(subject, "viewer")
        self.other = self.headers(uuid4().hex, "operator")
        self.body = {"job_id": 1, "question": "Why did this time out?", "retry_run_ms": 50}

    @staticmethod
    def headers(subject, role):
        return {"Authorization": "Bearer " + signed_token(subject, role)}

    def test_permission_persistence_and_concurrent_confirmation(self):
        with TestClient(app) as api:
            self.assertEqual(api.post("/agent/runs", json=self.body).status_code, 401)
            readonly = api.post("/agent/runs", json=self.body, headers=self.downgraded)
            self.assertEqual(readonly.json()["status"], "diagnosed")
            planned = api.post("/agent/runs", json=self.body, headers=self.operator)
            self.assertEqual(planned.status_code, 201, planned.text)
            self.assertEqual(planned.json()["status"], "awaiting_approval")
            self.assertTrue(planned.headers["X-Trace-Id"])
            path = f"/agent/runs/{planned.json()['run_id']}"
            self.submit.assert_not_awaited()

        with TestClient(app) as api:
            self.assertEqual(api.get(path, headers=self.other).status_code, 404)
            self.assertEqual(api.post(f"{path}/confirm", json={"approved": True}, headers=self.downgraded).status_code, 403)
            self.assertEqual(api.get(path, headers=self.operator).json()["status"], "awaiting_approval")

            def approve():
                return api.post(f"{path}/confirm", json={"approved": True}, headers=self.operator)

            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = [pool.submit(approve) for _ in range(2)]
                results = [result.result(timeout=15) for result in responses]
            self.assertEqual([result.status_code for result in results], [200, 200])
            self.assertEqual([result.json()["retry_job_id"] for result in results], [77, 77])
            self.assertEqual(api.post(f"{path}/confirm", json={"approved": False}, headers=self.operator).status_code, 409)
        self.submit.assert_awaited_once()

    def test_rate_limit_is_shared_between_gateway_instances(self):
        with patch.dict(os.environ, {"QUEUE_AGENT_RATE_LIMIT": "1"}):
            with TestClient(app) as api:
                self.assertEqual(api.post("/agent/runs", json=self.body, headers=self.operator).status_code, 201)
            with TestClient(app) as api:
                denied = api.post("/agent/runs", json=self.body, headers=self.operator)
                self.assertEqual(denied.status_code, 429, denied.text)
                self.assertGreater(int(denied.headers["Retry-After"]), 0)
        self.diagnose.assert_awaited_once()

    def test_operator_can_decline_without_a_write(self):
        with TestClient(app) as api:
            run = api.post("/agent/runs", json=self.body, headers=self.operator)
            path = f"/agent/runs/{run.json()['run_id']}/confirm"
            closed = api.post(path, json={"approved": False}, headers=self.operator)
            self.assertEqual(closed.status_code, 200, closed.text)
            self.assertEqual(closed.json()["status"], "declined")
        self.submit.assert_not_awaited()

    def test_retry_reuses_key_when_the_submit_response_is_lost(self):
        job = self.submit.return_value
        self.submit.side_effect = [HTTPException(503, "Queue response lost."), job]
        with TestClient(app) as api:
            run = api.post("/agent/runs", json=self.body, headers=self.operator)
            path = f"/agent/runs/{run.json()['run_id']}/confirm"
            first = api.post(path, json={"approved": True}, headers=self.operator)
            self.assertEqual(first.status_code, 503, first.text)
        with TestClient(app) as api:
            replay = api.post(path, json={"approved": True}, headers=self.operator)
            self.assertEqual(replay.status_code, 200, replay.text)
            self.assertEqual(replay.json()["retry_job_id"], job.id)
        self.assertEqual(self.submit.await_args_list[0].args, self.submit.await_args_list[1].args)
