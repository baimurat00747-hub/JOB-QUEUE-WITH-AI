import argparse
import os
from typing import Annotated, Literal
from uuid import UUID

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from training_queue.api import Job, JobHistory, SimulatedFailure

JobID = Annotated[int, Field(ge=1, le=2**63 - 1, strict=True)]
RunTime = Annotated[int, Field(ge=1, le=60_000, strict=True)]


class QueueResult(BaseModel):
    status: int
    job: Job | None = None
    history: JobHistory | None = None
    detail: str | None = None


def create_server(queue_url: str, request_id: UUID | None, history_only: bool = False) -> FastMCP:
    server = FastMCP("Training Queue", log_level="WARNING")
    headers = {
        name: value for name, value in (
            ("X-Queue-Token", os.environ.get("QUEUE_INTERNAL_TOKEN")),
            ("X-Trace-Id", os.environ.get("QUEUE_TRACE_ID")),
        ) if value
    }

    async def request_job(method: str, path: str, **kwargs) -> QueueResult:
        try:
            async with httpx.AsyncClient(base_url=queue_url, timeout=5, headers=headers) as client:
                response = await client.request(method, path, **kwargs)
        except httpx.RequestError:
            return QueueResult(status=503, detail="Queue API is unavailable. Reuse the key for a submission retry.")
        if response.is_success:
            if history_only:
                return QueueResult(status=response.status_code, history=JobHistory.model_validate(response.json()))
            return QueueResult(status=response.status_code, job=Job.model_validate(response.json()))
        if response.status_code == 404:
            return QueueResult(status=404, detail="Job not found.")
        if response.status_code == 409:
            return QueueResult(status=409, detail="Idempotency key already used with different job parameters.")
        if response.status_code == 422:
            return QueueResult(status=422, detail="Queue rejected the job parameters.")
        if response.status_code == 503:
            return QueueResult(status=503, detail="Queue is busy. Reuse the key for a submission retry.")
        return QueueResult(status=502, detail="Queue API returned an unexpected response.")

    if history_only:
        @server.tool()
        async def get_job_history(job_id: JobID) -> QueueResult:
            """Read the job and worker events from the same database snapshot."""
            return await request_job("GET", f"/jobs/{job_id}/history")
    elif request_id is None:
        @server.tool()
        async def get_job_status(job_id: JobID) -> QueueResult:
            """Read the state and attempt number of the specified job ID."""
            return await request_job("GET", f"/jobs/{job_id}")
    else:
        @server.tool()
        async def submit_simulated_job(
            run_ms: RunTime = 50,
            outcome: Literal["succeeded", "failed"] = "succeeded",
            failure_code: SimulatedFailure | None = None,
        ) -> QueueResult:
            """Enqueue a simulated job with a duration in milliseconds and an intended outcome."""
            return await request_job(
                "POST",
                "/jobs",
                headers={"Idempotency-Key": f"mcp-{request_id}"},
                json={"run_ms": run_ms, "outcome": outcome, "failure_code": failure_code},
            )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve training queue tools over MCP stdio.")
    parser.add_argument("--queue-url", default=os.environ.get("QUEUE_URL", "http://127.0.0.1:8000"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submission-id", type=UUID, help="Expose the submission tool for this request.")
    mode.add_argument("--history", action="store_true", help="Expose the job history tool.")
    args = parser.parse_args()
    create_server(args.queue_url, args.submission_id, args.history).run(transport="stdio")


if __name__ == "__main__":
    main()
