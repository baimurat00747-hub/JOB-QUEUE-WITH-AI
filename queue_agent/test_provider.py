import asyncio
import json
import unittest
from unittest.mock import patch

from fastapi import FastAPI, Request
from mcp.types import Tool

from training_queue.api import Job, JobHistory

from .api import select_arguments
from .diagnosis import recommend
from .observability import usage_fields
from .test_api import serve


class ExternalAdapterTests(unittest.TestCase):
    def test_openai_compatible_tool_call_and_structured_diagnosis(self):
        stub = FastAPI()
        calls = []

        @stub.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            calls.append((request.headers.get("Authorization"), body))
            message = {"role": "assistant", "content": json.dumps({"recommended_action": "Check the logs."})}
            if "tools" in body:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "test-tool", "type": "function",
                    "function": {"name": "get_job_status", "arguments": '{"job_id":1}'},
                }]}
            return {
                "id": "chatcmpl-test", "created": 1, "model": "gpt-test", "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            }

        with serve(stub) as url, patch.dict("os.environ", {
            "QUEUE_AGENT_PROVIDER": "openai", "QUEUE_AGENT_API_KEY": "stub-token",
            "QUEUE_AGENT_MODEL": "gpt-test", "QUEUE_AGENT_MODEL_URL": f"{url}/v1",
            "QUEUE_AGENT_INPUT_USD_PER_MTOK": "2", "QUEUE_AGENT_OUTPUT_USD_PER_MTOK": "5",
        }):
            tool = Tool(name="get_job_status", inputSchema={"type": "object", "properties": {"job_id": {"type": "integer"}}})
            with self.assertLogs("uvicorn.error") as logs:
                self.assertEqual(asyncio.run(select_arguments(tool, "Job 1")), {"job_id": 1})
                history = JobHistory(
                    job=Job(id=1, state="failed", attempt=1, run_ms=50, outcome="failed"), events=[],
                )
                advice = asyncio.run(recommend(history, "Next step?", "Inspect logs."))
            self.assertEqual(advice.recommended_action, "Check the logs.")
            self.assertEqual(usage_fields("openai", 50, 10)["estimated_cost_usd"], .00015)
        self.assertEqual([auth for auth, _ in calls], ["Bearer stub-token"] * 2)
        self.assertEqual(calls[1][1]["response_format"]["type"], "json_schema")
        for stage in ("model.diagnosis", "model.tool_arguments"):
            self.assertTrue(any(stage in row and "estimated_cost_usd" in row for row in logs.output))
