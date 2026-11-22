# Training job queue



## Claiming and recovering a job

In PostgreSQL a worker locks one queued or expired job using
`FOR UPDATE SKIP LOCKED`. Other workers can claim different rows while that
transaction is in progress. SQLite uses `BEGIN IMMEDIATE` for the same state
transition. The worker marks the job as running, increments `attempt`,
sets a 1.5-second lease, records the claim event and commits.

While a long job is running, the worker renews its lease every 0.5
seconds. If the worker dies or pauses long enough, another worker
can claim the job. If the original worker later comes back with
attempt 1 while the database says attempt 2, its write updates zero
rows. Both renewal and completion also check that the lease is valid.

Lease deadlines use the database wall clock. PostgreSQL samples the clock
after acquiring the row lock for renewal and completion. A clock jump forward can
expire leases early. A jump backward can delay recovery.

## Cancellation and completion

If cancellation commits first, the job becomes `cancelled`. The
worker can no longer renew the lease or save a result because both
operations require `running`. If completion commits first, the
cancel endpoint returns `409` with the terminal job state.

## Guarantee

A succeeded job has exactly one committed row in `effects`. Failed
and cancelled jobs have none. The worker writes the success state
and the effect in one transaction, and `effects.job_id` is unique.
If the worker dies before commit, both writes roll back.

The computation may run again after a worker failure. The
exactly-once guarantee covers the committed database row. An external side
effect would need its own protection against retries.
