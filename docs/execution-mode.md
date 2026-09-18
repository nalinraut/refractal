# `execution_mode`: defined by what happened to the episodes

Three modes, defined by **run shape** rather than by how the servers happen to be
deployed.

| Mode | What happens | Why you would pick it |
|---|---|---|
| `serial` | All tasks for checkpoint A, then all tasks for checkpoint B. | The only option when VRAM cannot hold both policies at once. |
| `concurrent` | Both checkpoints at once, each against its own server. | Halves wall clock. Arms contend, so durations are not comparable. |

**Two modes, not three.** `interleaved` was to be task-outer ordering, to remove
within-session drift. Drift was then measured at zero — ten tasks, three
replicate positions, pooled rank correlation +0.027 at p=0.839 for pi0 and +0.132
at p=0.230 for pi0.5, contrast +0.046 at p=0.723 (`docs/ten-task-run.md`). A mode
whose only justification does not hold is a default somebody picks for a reason
that is not true, so it was not built and the invocation overhead it would cost
was not estimated — that would have been a number nobody could check, attached to
a mode nobody should use.

A catalog that still says `interleaved` is **refused with that history**, not
aliased to `serial`. Aliasing would be the right guess and would also relabel
someone's catalog without telling them; which of the two they meant is their
call.

## What the modes may not differ in

Ordering, and nothing else. Both modes call one shared `_run_group`, so an
invocation does the same thing either way. If they differed in what an invocation
*does*, the mode would be changing the measurement rather than describing it —
and a test asserts the two write exactly the same episode ids.

`serial` stays checkpoint-outer within a task for the same reason. Reordering it
to alternate would be interleaved execution recorded as serial, which is the
mislabelling this definition exists to remove.

## `concurrent_with`

A column the other mode does not need. It records which other checkpoints were
running while this episode ran, sorted and comma-joined, empty when nothing was.

Provenance, never identity: contention does not change what an episode means, so
it must not gate a join. But `elapsed_sec` sits in the same row, and a duration
measured under contention is not comparable with one measured alone. Without the
column the only way to know is to trust `execution_mode` — which is the field
that was wrong.

The overlap is tested by a barrier that both invocations must reach, so it fails
if the loop runs them sequentially. Routed down the serial path it raises
`BrokenBarrierError`, which is the point: a fake that returned instantly would
have passed either way.

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

## What the first real `concurrent` run found

A harness bug that only this mode could surface.

`Orchestrator._update_progress` writes `<output_dir>/<benchmark_name>.tmp` and
`os.replace`s it into place. The name depends only on the benchmark, so two
orchestrators sharing an `output_dir` race on one temp path: the first replace
succeeds, the second finds nothing, and the run dies with

```
FileNotFoundError: '/tmp/smoke2-harness/LIBEROBenchmark_libero_spatial.tmp'
  -> '/tmp/smoke2-harness/LIBEROBenchmark_libero_spatial.progress'
```

Serial shares that directory too and never collides, because it is never in two
places at once. No amount of testing `serial` would have found it, and the fake
harness in the unit tests does not write progress files — the barrier test proves
the invocations overlap, not that the harness tolerates overlapping.

Fixed on our side: each invocation gets `<output_dir>/<checkpoint>/<task>/seed<N>`
as its scratch directory. `output_dir` is ours to choose, and a harness scratch
directory assuming one orchestrator is a reasonable thing for it to assume.
Refractal's own results never go there; they go to the results URI as Parquet.

Verified after the fix, 8 episodes, one scene, two tasks, two arms in parallel:

```
  task-0  pi0   ok=True  mode=concurrent concurrent_with='pi05'   5.3s
  task-0  pi05  ok=True  mode=concurrent concurrent_with='pi0'    5.2s
  task-1  pi0   ok=True  mode=concurrent concurrent_with='pi05'   6.4s
  task-1  pi05  ok=True  mode=concurrent concurrent_with='pi0'    5.9s
```

Each arm names the other, `execution_mode` is the mode that ran, and the durations
carry the contention flag that says not to compare them with uncontended ones.
