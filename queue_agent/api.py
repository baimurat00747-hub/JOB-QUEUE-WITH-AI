import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from redis.asyncio import Redis

from training_queue.api import Job, JobID

from .auth import Principal, auth_secret, authenticate, legacy_read, legacy_write, require_operator
from .diagnosis import Diagnosis, diagnose
from .limits import rate_check
from .models import model_settings
from .observability import TRACE_ID, span, span_log_file, usage_fields
from .runbooks import database_url
from .server import QueueResult
from .workflow import Confirmation, RunRequest, RunResponse, build_graph, config, lock_confirmation, owned_state, public_run

LOGGER = logging.getLogger("uvicorn.error")
SYSTEM_PROMPT = (
    "Extract exactly one queue tool call from the request. "
    "Use the available tool only if the user requested that operation. "
    "Use defaults for omitted optional arguments. Never invent a job ID."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with span_log_file():
        if len(auth_secret()) < 32:
            yield
            return
        async with AsyncPostgresSaver.from_conn_string(database_url()) as saver, Redis.from_url(
            os.environ.get("QUEUE_AGENT_REDIS_URL", "redis://127.0.0.1:56379/0"),
            socket_connect_timeout=3, socket_timeout=3,
        ) as redis:
            await saver.setup()
            app.state.graph = build_graph(saver, load_history, submit_retry)
            app.state.redis = redis
            yield


app = FastAPI(title="Training Queue Agent", lifespan=lifespan)


@app.middleware("http")
async def trace_request(request: Request, call_next):
    trace_id = uuid4().hex
    token = TRACE_ID.set(trace_id)
    try:
        with span("http.request", method=request.method) as trace:
            response = await call_next(request)
            trace["status"] = response.status_code
            trace["route"] = getattr(request.scope.get("route"), "path", None)
        response.headers["X-Trace-Id"] = trace_id
        return response
    finally:
        TRACE_ID.reset(token)


async def meter(request: Request, principal: Principal | None):
    if principal is not None:
        settings = model_settings()
        await rate_check(request.app.state.redis, principal.subject, settings.provider, settings.model)


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: Annotated[str, StringConstraints(strip_whitespace=True)] = Field(min_length=1, max_length=1000)


async def select_arguments(tool: Tool, prompt: str) -> dict:
    settings = model_settings()
    payload = {
        "model": settings.model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "tools": [{
            "type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema},
        }],
    }
    if settings.provider == "ollama":
        payload["temperature"] = 0
    try:
        with span("model.tool_arguments", provider=settings.provider, model=settings.model) as trace:
            async with httpx.AsyncClient(
                base_url=settings.url, headers=settings.headers, timeout=httpx.Timeout(90, connect=5),
            ) as provider:
                response = await provider.post("chat/completions", json=payload)
                response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            trace.update(usage_fields(settings.provider, usage.get("prompt_tokens"), usage.get("completion_tokens")))
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Model did not respond in time.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Model request failed.") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(502, "Model returned an invalid response.") from exc

    try:
        message = data["choices"][0]["message"]
        (call,) = message["tool_calls"]
        function = call["function"]
        arguments = json.loads(function["arguments"])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, "The model did not produce exactly one valid tool call.") from exc
    if function.get("name") != tool.name or not isinstance(arguments, dict):
        raise HTTPException(422, "The requested action is unavailable for this endpoint.")
    if arguments.keys() - tool.inputSchema.get("properties", {}).keys():
        raise HTTPException(422, "The model supplied unexpected tool arguments.")
    return arguments


async def run_tool(
    name: str, prompt: str | None = None, *, request_id: UUID | None = None,
    job_id: int | None = None, arguments: dict | None = None,
) -> QueueResult:
    server_args = ["-m", "queue_agent.server"]
    if request_id is not None:
        server_args.extend(["--submission-id", str(request_id)])
    if name == "get_job_history":
        server_args.append("--history")
    params = StdioServerParameters(
        command=sys.executable,
        args=server_args,
        env={
            "QUEUE_URL": os.environ.get("QUEUE_URL", "http://127.0.0.1:8000"),
            "QUEUE_INTERNAL_TOKEN": os.environ.get("QUEUE_INTERNAL_TOKEN", ""),
            "QUEUE_TRACE_ID": TRACE_ID.get() or "",
        },
    )
    started = perf_counter()
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=10)) as session:
                await session.initialize()
                (tool,) = (await session.list_tools()).tools
                if tool.name != name:
                    raise RuntimeError("MCP server advertised an unexpected tool")
                if arguments is None:
                    arguments = {"job_id": job_id} if prompt is None else await select_arguments(tool, prompt)
                with span("mcp.call_tool", tool=name) as trace:
                    result = await session.call_tool(name, arguments)
                    trace["tool_error"] = result.isError
    except Exception as exc:
        while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        if isinstance(exc, HTTPException):
            raise exc
        LOGGER.warning("MCP request failed for %s (%s)", name, type(exc).__name__)
        raise HTTPException(502, "MCP request failed.") from exc

    LOGGER.info("tool=%s error=%s duration_ms=%.1f", name, result.isError, (perf_counter() - started) * 1000)
    if result.isError:
        raise HTTPException(422, "The tool rejected the model arguments.")
    try:
        queue_result = QueueResult.model_validate(result.structuredContent)
    except ValidationError as exc:
        raise HTTPException(502, "MCP tool returned an invalid result.") from exc
    if queue_result.status >= 400:
        raise HTTPException(queue_result.status, queue_result.detail)
    payload = queue_result.history if job_id is not None else queue_result.job
    if payload is None:
        raise HTTPException(502, "MCP tool returned the wrong result type.")
    return queue_result


@app.post("/agent/jobs", status_code=201, responses={200: {"model": Job}})
async def submit_job(
    body: AgentRequest,
    request: Request,
    response: Response,
    key: Annotated[UUID, Header(alias="Idempotency-Key")],
    principal: Annotated[Principal | None, Depends(legacy_write)],
) -> Job:
    await meter(request, principal)
    result = await run_tool("submit_simulated_job", body.prompt, request_id=key)
    response.status_code = result.status
    return result.job


@app.post("/agent/jobs/lookup")
async def lookup_job(
    body: AgentRequest, request: Request, principal: Annotated[Principal | None, Depends(legacy_read)],
) -> Job:
    await meter(request, principal)
    return (await run_tool("get_job_status", body.prompt)).job


async def load_history(job_id):
    return (await run_tool("get_job_history", job_id=job_id)).history


async def submit_retry(key, run_ms):
    return (await run_tool(
        "submit_simulated_job", request_id=key, arguments={"run_ms": run_ms, "outcome": "succeeded"},
    )).job


@app.post("/agent/jobs/{job_id}/diagnose")
async def diagnose_job(
    job_id: JobID,
    request: Request,
    principal: Annotated[Principal | None, Depends(legacy_read)],
    question: Annotated[str, Query(min_length=1, max_length=400)] = "What should the operator do now?",
) -> Diagnosis:
    await meter(request, principal)
    return await diagnose(await load_history(job_id), question)


@app.post("/agent/runs", status_code=201)
async def start_run(
    body: RunRequest, request: Request, principal: Annotated[Principal, Depends(authenticate)],
) -> RunResponse:
    await meter(request, principal)
    run_id = uuid4()
    with span("workflow.start", run_id=str(run_id)):
        result = await request.app.state.graph.ainvoke({
            **body.model_dump(), "run_id": str(run_id), "subject": principal.subject,
            "retry_run_ms": body.retry_run_ms if principal.role == "operator" else None,
        }, config(run_id), context=principal)
    return public_run(result)


@app.get("/agent/runs/{run_id}")
async def get_run(
    run_id: UUID, request: Request, principal: Annotated[Principal, Depends(authenticate)],
) -> RunResponse:
    return public_run((await owned_state(request.app.state.graph, run_id, principal)).values)


@app.post("/agent/runs/{run_id}/confirm")
async def confirm_run(
    run_id: UUID, body: Confirmation, request: Request, principal: Annotated[Principal, Depends(authenticate)],
) -> RunResponse:
    require_operator(principal)
    graph = request.app.state.graph
    with span("workflow.confirm", run_id=str(run_id)):
        async with lock_confirmation(run_id):
            snapshot = await owned_state(graph, run_id, principal)
            if snapshot.values["status"] == "diagnosed":
                raise HTTPException(409, "This run has no pending retry.")
            if "approved" in snapshot.values and snapshot.values["approved"] != body.approved:
                raise HTTPException(409, "The approval decision has already been recorded.")
            if snapshot.values["status"] in ("submitted", "declined"):
                return public_run(snapshot.values)
            if snapshot.interrupts:
                value = Command(resume=body.approved)
            elif snapshot.next == ("execute",) and snapshot.values.get("approved"):
                value = None
            else:
                raise HTTPException(409, "This run has no pending retry.")
            return public_run(await graph.ainvoke(value, config(run_id), context=principal))
