# Worker lost

## Triage

Compare the current attempt number with the reclaimed worker event. If a replacement already holds the lease, wait for that attempt and check its status. Investigate the old worker's logs before submitting another job.

## Prevention

Monitor worker heartbeats and reclaim events. Confirm that completed work is fenced by the current attempt and that any external effect is protected against replay.
