# Invalid input

## Triage

Inspect the input against the job schema and the failing validation message. Fix the invalid values before resubmitting with a fresh idempotency key.

## Prevention

Validate payloads at the API boundary and record a validation failure code together with the failed attempt. Add a regression test using the rejected input.
