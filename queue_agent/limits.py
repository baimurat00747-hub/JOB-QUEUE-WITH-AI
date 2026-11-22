import hashlib
import os

from fastapi import HTTPException
from redis.exceptions import RedisError

INCREMENT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return {count, redis.call('TTL', KEYS[1])}
"""


async def rate_check(redis, subject, provider, model):
    try:
        quota = int(os.environ.get("QUEUE_AGENT_RATE_LIMIT", "10"))
        seconds = int(os.environ.get("QUEUE_AGENT_RATE_WINDOW_SECONDS", "60"))
    except ValueError as exc:
        raise HTTPException(503, "Invalid agent rate limit configuration.") from exc
    if quota < 1 or seconds < 1:
        raise HTTPException(503, "Invalid agent rate limit configuration.")
    scope = hashlib.sha256(f"{subject}\0{provider}\0{model}".encode()).hexdigest()
    try:
        count, remaining = await redis.eval(INCREMENT, 1, f"queue:agent:limit:{scope}", seconds)
    except RedisError as exc:
        raise HTTPException(503, "The agent rate limiter is unavailable.") from exc
    if count > quota:
        raise HTTPException(429, "Model request limit exceeded.", headers={"Retry-After": str(max(1, remaining))})
