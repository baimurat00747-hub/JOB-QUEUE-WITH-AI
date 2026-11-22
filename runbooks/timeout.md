# Timeout

## Triage

Check the duration and time budget of the failed attempt. Review worker logs for the stage that timed out. If the workload can be split or its budget adjusted, submit the corrected job with a fresh idempotency key.

## Prevention

Separate the worker time budget from HTTP timeouts. Log the last completed stage and measure its latency before raising the execution limit.
