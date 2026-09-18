# Open planner question: what must a worker own whole?

Not a bug. A genuine tension between two correct models, surfaced by building
`--backend compose`, and written down rather than fixed because the fix touches
the thing everything else rests on.

## Not a recovery

Stated first because it decides how the rest reads. The ten-scenes-one-task
catalog gave `--backend compose` twelve services, and the corrected one gives a
single-suite run one. That is **not** parallelism lost: the twelve were an
artifact of a scene model that split one compiled model into ten, and fixing it
removed something that should not have existed.

So this is not a question about recovering a capability. It is the question the
wrong model was hiding — **may a scene's episodes be split at all** — which ten
fake scenes never had to answer, and which one real scene now asks directly.

The parallelism Compose can genuinely exploit is **across scenes**, and a
single-suite LIBERO run genuinely has one scene. Which is why the two-suite
demonstration in [compose-backend.md](compose-backend.md) is the honest one rather
than a workaround.

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

### The name matters more than the mechanism

Proposed: **`partition_unit`**, an enum with values `scenario` and `task`.

An enum rather than a number, and the reason is how it will be read years from
now rather than how it behaves. **A number invites arithmetic; an enum says it is
a boundary.** `envs_per_process` is a number and is divided into things — that is
what `ceil(episodes / envs_per_process)` does — and the moment this field is a
number, somebody will divide by it too, which is exactly the conflation that let
a scenario range be split twelve ways.

| value | means | who says it |
|---|---|---|
| `scenario` | any subset of a scene's scenarios may be split off | MJX — the batch genuinely holds arbitrary same-scene scenarios |
| `task` | a worker must own a task's whole zero-based range | classic MuJoCo — the harness rebuilds per task, and the counter starts at zero |

The planner partitions by it and never interprets it, the same way it already
consumes `sec_per_1k_steps` without knowing what a step is.

### It is a property of the engine-plus-benchmark pair

This is the part the name has to keep honest, and the reason it cannot come from
`envs_per_process`.

`scenario` versus `task` is not a fact about the scene, and not a fact about the
engine alone. It is a fact about **the engine together with the benchmark that
drives it**: MJX could batch arbitrary same-scene scenarios, and classic MuJoCo
plus *this harness* cannot, because the harness rebuilds per `task_id` and counts
episodes from zero within one. A different driver over the same engine could
answer differently.

Which is why it belongs on `ResourceShape` — already keyed by
`(scene, engine, hardware_profile)` and already the place where measured facts
about a specific pairing live — and why it must be declared rather than derived.
Deriving it from `envs_per_process` would be asserting that "how many at once"
determines "what must be owned whole", and the whole point is that they are
independent.

With it, `plan` produces ten runnable workers for one scene, `render` emits ten
containers, and the index contract holds by construction rather than by
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
ends up meaning "task" forever — the same failure as
[the circularity in the statistics](statistics-measurements.md), where a model of
within-scenario correlation was authored to make a fixture discriminate and the
tests were then scored against it. The numbers were real counts; they described
the model rather than the world. A `partition_unit` whose only witness is an
engine that can only answer `task` would be the same shape: a real field that
describes its one example.

## What would settle it — one prerequisite, not two open items

**An MJX scene**, or any engine where `envs_per_process > 1` is real. At that
point the two questions stop being confusable, because one scene genuinely does
share a compiled model across a batch and the answers diverge: `envs_per_process`
becomes greater than 1 *and* `partition_unit` becomes `scenario`, independently.

This is the same prerequisite as the **scene affinity and sequential packing**
entry under "Blocked on a deferred dependency" in
[patterns.md](patterns.md#currently-known-to-be-unexercised). Both wait on a
vectorized scene, and neither is worth touching before one exists — the planner
packs starved scenes with a `startup_sec`-driven threshold that no fixture has
enough scenes to reach, and this field would have only one value anybody could
demonstrate.

So they are **one concrete next thing** rather than two vague deferrals: get an
MJX scene into a catalog, and both become exercisable in the same afternoon. That
is a better-shaped piece of work than either was alone, and it is the argument for
doing it at all — a single fixture unblocks a planner path and settles a schema
question that would otherwise be designed blind.
