import json
import logging
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter

TRACE_ID = ContextVar("queue_trace_id", default=None)
LOGGER = logging.getLogger("uvicorn.error")
SPAN_LOGGER = LOGGER.getChild("queue_agent.spans")


@contextmanager
def span_log_file():
    filename = os.environ.get("QUEUE_AGENT_LOG_FILE")
    if not filename:
        yield
        return

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    SPAN_LOGGER.addHandler(handler)
    try:
        yield
    finally:
        SPAN_LOGGER.removeHandler(handler)
        handler.close()


@contextmanager
def span(name, **fields):
    started = perf_counter()
    outcome = "ok"
    try:
        yield fields
    except BaseException:
        outcome = "error"
        raise
    finally:
        SPAN_LOGGER.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trace_id": TRACE_ID.get(), "span": name, "outcome": outcome,
            "duration_ms": round((perf_counter() - started) * 1000, 2), **fields,
        }))


def usage_fields(provider, input_tokens=None, output_tokens=None):
    fields = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if provider == "openai" and input_tokens is not None and output_tokens is not None:
        input_price = os.environ.get("QUEUE_AGENT_INPUT_USD_PER_MTOK")
        output_price = os.environ.get("QUEUE_AGENT_OUTPUT_USD_PER_MTOK")
        if input_price is not None and output_price is not None:
            try:
                prices = (float(input_price), float(output_price))
                if not all(math.isfinite(price) and price >= 0 for price in prices):
                    raise ValueError("Prices must be finite and non-negative.")
                fields["estimated_cost_usd"] = round((input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000, 9)
            except ValueError:
                LOGGER.warning("Invalid agent token price configuration")
    return fields
