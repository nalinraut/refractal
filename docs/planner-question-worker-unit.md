# Open planner question: what must a worker own whole?

Not a bug. A genuine tension between two correct models, surfaced by building
`--backend compose`, and written down rather than fixed because the fix touches
the thing everything else rests on.

## The tension

Refractal's worker unit is a **scene**, and correctness says LIBERO-Spatial is
**one** scene — ten tasks, identical `:objects` and `:fixtures`, one `MjModel`.
Compose's unit is a **container**, and one worker means one container.

So the correct identity model and the parallel execution model disagree. A
single-suite run has one worker, and `--backend compose` extracts no parallelism
from it. Nothing here is wrong; there is simply nothing to extract.

The demonstration in `compose-backend.md` uses two suites for exactly this
reason, and that is a legitimate shape — but it is a workaround for a
single-suite run, not an answer.

## Why sharding by task is defensible — measured

Scene affinity exists to protect one thing: a compiled model shared across a
batch. **For this harness and classic MuJoCo, that sharing does not span tasks.**

```python
# vla_eval/benchmarks/libero/benchmark.py, reset()
# Only create a new env when the task changes (reuse across episodes)
if self._env is None or self._current_task_id != task_id:
    if self._env is not None:
        self._env.close()
    ...
    env = OffScreenRenderEnv(**env_args)
```

The reuse unit is the **task**, not the scene. Ten tasks in one worker pay ten
environment constructions, exactly as ten workers of one task each would. So a
task-sharded worker gives up nothing real and gains a container boundary.

Which sharpens the original argument rather than restating it: it is not merely
that the harness rebuilds on `task_id` change, it is that its affinity unit is
already the task, so Refractal's scene-level worker unit is protecting an
invariant this benchmark does not have.

## Why the obvious fix is not obviously right

Sharding by task where the engine cannot batch makes the worker unit
**engine-dependent**, and `resolve` must never branch on engine. That rule is
what keeps `refractal plan` runnable on a laptop with no simulator installed, and
it is load-bearing for the whole compiler story.

The clean shape is that the **resource shape declares the number and the planner
reads it without knowing why it is 1** — which is what `envs_per_process` was
reaching for.

### Where that proposal does not quite land, as currently defined

`envs_per_process` today means "how many environments one worker holds at once",
and it drives:

```python
useful_worker_ceiling = ceil(episodes / envs_per_process)
```

With `envs_per_process: 1` that ceiling is *episodes*, so the planner is free to
split a scene maximally — which is precisely what produced twelve workers
splitting a scenario range and 540 of 600 groups refused at render time.

So the existing field answers **"how many can run at once"**. The question the
worker unit needs answered is **"what must one worker own whole"**, and those are
different:

| | question | LIBERO answer |
|---|---|---|
| `envs_per_process` | how many episodes may share one process at once | 1 |
| the missing field | what is the atomic assignment a worker must own entirely | one task's full scenario range |

A shape that said `atomic_unit: task` (or, engine-agnostically, a declared
grouping key the planner partitions by and does not interpret) would let the
planner shard LIBERO ten ways without ever learning what MuJoCo is. `plan` would
then produce ten runnable workers for one scene, `render` would emit ten
containers, and the index contract would hold by construction rather than by
`--workers-per-scene 1`.

## Why not now

* It changes how every plan is partitioned, so every `plan.json` layout moves —
  not `plan_id`, since placement is not identity, but everything downstream that
  reads a worker.
* `--workers-per-scene 1` already makes single-suite runs correct, and the
  two-suite demonstration already shows Compose working.
* The field is a schema addition to `ResourceShape`, which is the most-depended-on
  model in the project, and it should be designed against a second engine rather
  than against LIBERO alone. MJX is the case that would test it, and there is no
  MJX scene here — the same reason scene affinity and sequential packing are on
  the unexercised list in `patterns.md`.

Designing a partitioning primitive against one engine that cannot batch is how it
ends up meaning "task" forever.

## What would settle it

An MJX scene, or any engine where `envs_per_process > 1` is real. At that point
the two questions above stop being confusable, because one scene genuinely does
share a compiled model across a batch and the answers diverge.
