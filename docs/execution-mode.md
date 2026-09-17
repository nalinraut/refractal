# `execution_mode`: defined by what happened to the episodes

Three modes, defined by **run shape** rather than by how the servers happen to be
deployed.

| Mode | What happens | Why you would pick it |
|---|---|---|
| `serial` | All tasks for checkpoint A, then all tasks for checkpoint B. | The only option when VRAM cannot hold both policies at once. |
| `interleaved` | For each task: checkpoint A, then checkpoint B, then the next task. | Arms face the same machine conditions per task rather than per session. |
| `concurrent` | Both checkpoints at once, each against its own server. | Halves wall clock. Arms contend, so durations are not comparable. |

## The error this replaces

The old definition distinguished the modes by whether both servers stayed
resident. That is a property of the deployment, not of the run, and it
distinguishes nothing — a serial run with two servers up has it too.

So `interleaved` was recorded on rows produced by a checkpoint-outer loop:

```
worker -> checkpoint pi0   -> task 1..10
       -> checkpoint pi0.5 -> task 1..10
```

That is serial execution wearing the interleaved label, which is the class of
mislabelling this project exists to catch, produced by this project.

## `execution_mode` is not in `plan_id` — measured, not assumed

```
  execution_mode=serial       plan_id SAME
  execution_mode=interleaved  plan_id SAME
  execution_mode=concurrent   plan_id SAME
```

Pinned by `test_plan_id_ignores_execution_mode`, and deliberate: two runs of one
experiment in different modes belong in one comparison, so the mode is recorded
per episode and `compare` gates latency reporting on it rather than refusing the
join.

Two consequences, and the second is the uncomfortable one:

* **The mislabelling did not fork any identity.** No results are stranded, and
  nothing needs re-running on identity grounds.
* **A joined comparison can silently mix modes.** That is by design for success
  rates and wrong for durations, which is why `compare` warns when `session_id`
  or mode varies. The warning is now carrying more weight than it was, because
  the label it reads was not trustworthy.

## Migration boundary

Rows written before `239e095` carry `execution_mode: interleaved` and were
produced **serially**, under the deployment-shaped definition. They are not
rewritten.

Silently correcting history would be worse than the original error: it would make
every row agree with the current definition while leaving no trace that the
definition changed, which is precisely the failure mode of a positional identity.
The boundary is recorded here instead, and it is checkable — `plan_id` is
unaffected, so the affected rows are exactly those under comparison ids
`sha256:374443dd…` (smoke, 4 episodes), `sha256:1b441de8…` (120 episodes) and
`sha256:4605f47f…` (600 episodes).
