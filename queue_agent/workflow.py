from contextlib import asynccontextmanager
from typing import Literal, TypedDict
from uuid import NAMESPACE_URL, UUID, uuid5

import psycopg
from fastapi import HTTPException
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from .auth import Principal, require_operator
from .diagnosis import Diagnosis, diagnose
from .runbooks import database_url


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: int = Field(ge=1, le=2**63 - 1, strict=True)
    question: str = Field(min_length=1, max_length=400)
    retry_run_ms: int | None = Field(None, ge=1, le=60_000, strict=True)


class Confirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool = Field(strict=True)


class RunResponse(BaseModel):
    run_id: UUID
    status: Literal["diagnosed", "awaiting_approval", "declined", "submitted"]
    diagnosis: Diagnosis
    retry_run_ms: int | None = None
    retry_job_id: int | None = None


class RunState(TypedDict, total=False):
    run_id: str
    subject: str
    job_id: int
    question: str
    retry_run_ms: int | None
    diagnosis: dict
    status: str
    approved: bool
    retry_job_id: int


def build_graph(saver, load_history, submit_job):
    async def inspect_job(state: RunState) -> dict:
        history = await load_history(state["job_id"])
        result = await diagnose(history, state["question"])
        ready = result.state == "failed" and bool(result.sources) and state["retry_run_ms"] is not None
        return {
            "diagnosis": result.model_dump(),
            "status": "awaiting_approval" if ready else "diagnosed",
            "retry_run_ms": state["retry_run_ms"] if ready else None,
        }

    def approval(state: RunState) -> dict:
        accepted = interrupt({"job_id": state["job_id"], "retry_run_ms": state["retry_run_ms"]})
        return {"approved": accepted, "status": "awaiting_approval" if accepted else "declined"}

    async def execute(state: RunState, runtime: Runtime[Principal]) -> dict:
        principal = runtime.context
        require_operator(principal)
        if principal.subject != state["subject"]:
            raise HTTPException(404, "Agent run not found.")
        key = uuid5(NAMESPACE_URL, f"training-queue/retry/{state['run_id']}")
        job = await submit_job(key, state["retry_run_ms"])
        return {"status": "submitted", "retry_job_id": job.id}

    graph = StateGraph(RunState, context_schema=Principal)
    graph.add_node("inspect", inspect_job)
    graph.add_node("approval", approval)
    graph.add_node("execute", execute)
    graph.add_edge(START, "inspect")
    graph.add_conditional_edges("inspect", lambda state: state["status"] == "awaiting_approval", {
        True: "approval", False: END,
    })
    graph.add_conditional_edges("approval", lambda state: state["approved"], {True: "execute", False: END})
    graph.add_edge("execute", END)
    return graph.compile(checkpointer=saver)


def config(run_id: UUID) -> dict:
    return {"configurable": {"thread_id": str(run_id)}}


async def owned_state(graph, run_id: UUID, principal: Principal):
    snapshot = await graph.aget_state(config(run_id))
    if snapshot.values.get("subject") != principal.subject:
        raise HTTPException(404, "Agent run not found.")
    return snapshot


def public_run(state: RunState) -> RunResponse:
    return RunResponse.model_validate(state)


@asynccontextmanager
async def lock_confirmation(run_id: UUID):
    try:
        async with await psycopg.AsyncConnection.connect(
            database_url(), connect_timeout=5, options="-c lock_timeout=8000",
        ) as conn:
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (str(run_id),))
            yield
    except psycopg.errors.LockNotAvailable as exc:
        raise HTTPException(409, "Confirmation is in progress.", headers={"Retry-After": "1"}) from exc
    except psycopg.OperationalError as exc:
        raise HTTPException(503, "Workflow storage is unavailable.") from exc
