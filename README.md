# Job Queue with AI Diagnostics

A portfolio project that combines a recoverable background job queue with an
AI workflow for investigating failures. The agent reads worker events through
MCP, finds a relevant runbook, and recommends a next step with a source link.
An operator can review the diagnosis and approve a retry.

The workload simulates training jobs. The end-to-end demo runs locally with Ollama.

[Run locally](#run-locally) · [Try the demo](#try-the-demo) · [Inspect the logs](#inspect-the-logs)

## Backend · job execution and recovery

The queue API accepts jobs, exposes their status and event history, and
supports cancellation. PostgreSQL stores the state of each job, its current
attempt, worker events, and the committed result.

- **Idempotent submission.** `POST /jobs` accepts an `Idempotency-Key`.
  Repeating the same request returns the existing job ID. Reusing the key
  with different parameters returns `409 Conflict`.
- **Concurrent workers.** Workers claim jobs using `FOR UPDATE SKIP LOCKED`.
  Each claim increments the attempt number and grants a lease. The worker
  renews that lease while executing the job.
- **Recovery.** When a worker stops renewing its lease, another worker can
  reclaim the job. Completion checks both the attempt number and the lease,
  so a worker that resumes after a reclaim cannot overwrite the newer result.
- **Atomic completion.** The terminal state, completion event, and successful
  result are committed in one transaction. A unique constraint allows one
  committed result per job, even when the work itself runs again after recovery.

[Queue design](DESIGN.md) · [PostgreSQL tests](tests/test_postgres.py) · [Failure tests](tests/test_failure_windows.py)

## AI · diagnosis and operator-approved retry

1. **Read the evidence.** `POST /agent/jobs/{job_id}/diagnose` reads the job
   and its worker events through MCP. The recorded event for the current
   attempt supplies the failure code.
2. **Find an instruction.** The retriever filters runbook sections by failure
   code and combines pgvector similarity with PostgreSQL full-text ranking.
   If no section clears the similarity threshold, the API returns the recorded
   reason and skips generation.
3. **Generate a recommendation.** LangChain sends the question, recent events,
   and the top-ranked section to the configured model. The model returns a
   structured `recommended_action`. The API attaches the source URL from the
   retrieved section. Chat inference can use Ollama or an OpenAI-compatible
   endpoint.
4. **Approve a retry.** `POST /agent/runs` starts a LangGraph workflow that
   saves its state in PostgreSQL and can pause for approval. The run owner
   needs the `operator` role to confirm the retry. A stable idempotency key
   derived from the run ID ensures that repeated approval returns the same
   retry job ID.

The separate natural-language submission and lookup endpoints demonstrate
function calling. The Agent API exposes the MCP tool allowed for each endpoint,
validates the model's arguments, and executes the tool call.

The Agent API uses `async/await` for model, MCP, Redis, and database calls.
The diagnosis endpoint returns the result after inference completes.

[Agent API](queue_agent/api.py) · [Diagnosis](queue_agent/diagnosis.py) · [Workflow](queue_agent/workflow.py) · [Runbooks](runbooks/)

## Run locally

Install Python 3.12 and Docker Compose, and start Ollama. Run these commands
from the repository root.

```sh
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b

mkdir -p .local/queue-agent
if [ ! -f .local/demo.env ]; then
  (
    umask 077
    printf 'QUEUE_AGENT_JWT_SECRET=%s\nQUEUE_INTERNAL_TOKEN=%s\n' \
      "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > .local/demo.env
  )
fi

set -a
. ./.local/demo.env
set +a

docker compose up --build --wait
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r queue_agent/requirements.txt
python -m queue_agent.runbooks

QUEUE_AGENT_LOG_FILE=.local/queue-agent/agent.jsonl \
  python -m uvicorn queue_agent.api:app --host 127.0.0.1 --port 8001
```

Compose runs the queue API, workers, PostgreSQL, and Redis. The Agent API
runs on the host and calls Ollama over HTTP.

## Try the demo

In a second terminal, submit a job with a simulated timeout and generate an
operator token.

```sh
. .venv/bin/activate
set -a
. ./.local/demo.env
set +a

job_id=$(
  curl -fsS http://127.0.0.1:8000/jobs \
    -H "X-Queue-Token: $QUEUE_INTERNAL_TOKEN" \
    -H "Idempotency-Key: $(python -c 'import uuid; print(uuid.uuid4())')" \
    -H 'Content-Type: application/json' \
    -d '{"run_ms":50,"outcome":"failed","failure_code":"TIMEOUT"}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["id"])'
)

curl -fsS "http://127.0.0.1:8000/jobs/$job_id/history" \
  -H "X-Queue-Token: $QUEUE_INTERNAL_TOKEN"
printf '\nJob ID: %s\n' "$job_id"

python -m queue_agent.auth operator demo-operator
```

1. Once the job history shows `failed`, open [Swagger](http://127.0.0.1:8001/docs)
   and paste the operator token into **Authorize**.
2. Call `POST /agent/runs` using the returned job ID, a `retry_run_ms` of `100`,
   and the question `How do I triage this timeout and review the worker logs?`.
   The response contains a diagnosis, a runbook link, and a `run_id` with
   status `awaiting_approval`.
3. Call `POST /agent/runs/{run_id}/confirm` with `{"approved": true}`. Repeat the
   call and verify that both responses contain the same `retry_job_id`.
4. Generate a viewer token with `python -m queue_agent.auth viewer demo-operator`
   and try the same confirmation under that token. The API returns `403`.

## Inspect the logs

The agent records structured JSON events for HTTP requests, MCP calls,
retrieval, and model calls. Model events include latency and token usage when
available. External inference can also log estimated cost based on configured
token prices. Redis applies a rate limit to authenticated model-triggering
requests by user and model.

Start the optional logging stack.

```sh
docker compose -f compose.observability.yaml up -d --wait
```

Grafana Alloy ships the JSON logs to Loki. In
[Grafana Explore](http://127.0.0.1:3000/explore), select **Loki** and use the
`X-Trace-Id` from an Agent API response to find its events.

```logql
{service="queue-agent"} | json | trace_id = "PASTE_TRACE_ID_HERE"
```

## Verification

- **Worker recovery.** In the five-minute SQLite failure test, a replacement
  worker completed the job. The original worker resumed and its completion
  was rejected. PostgreSQL integration tests also cover concurrent claims
  and stale attempts. [Failure tests](tests/test_failure_windows.py) · [PostgreSQL tests](tests/test_postgres.py)
- **Approval and retries.** Integration tests cover role checks, checkpoints
  across an API restart, concurrent confirmations, and retrying after a lost
  submission response. [Workflow tests](queue_agent/test_workflow.py)
- **Observability.** A real Ollama diagnosis produced correlated HTTP, MCP,
  retrieval, and model events. The logs remained queryable after Loki restarted.
- **Retrieval.** On 28 labelled development questions, vector and hybrid
  retrieval both returned the expected top result in 22 of 24 relevant cases
  and abstained on all four out-of-scope cases. The similarity threshold was
  tuned on this set. [Evaluation cases](queue_agent/eval_cases.json)

The OpenAI-compatible adapter was tested against a local HTTP stub.
