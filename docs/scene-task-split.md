# Scene is the compiled model; the goal belongs to the task

A correction to the LIBERO catalog, and a gap in the original design that the
correction exposes.

## What the spec says

> Two scenarios share a scene if they compile to the same `MjModel` and differ
> only in `MjData`.

and the worked example is *vial in slot 4* against *vial in slot 7*: two tasks on
one scene, batching together perfectly, with the note that **tasks are cheap to
vary and scenes are expensive**.

LIBERO-Spatial is that example. Measured: two `libero_spatial` BDDLs differ in
three lines — the `:language` string and two `(On ...)` initial placements —
with identical `:objects` and identical `:fixtures`. One `MjModel`, ten tasks.

The catalog nonetheless declared **ten scenes with one task each**, which turns
the doc's canonical cheap case into its expensive one. That is the error.

## Why it was made, and why the reason does not hold

The reasoning was that `get_task_init_states` is keyed by `task_id`, so
`init_state_index: 3` names a different physical configuration depending on which
task you ask — and a scene-level scenario set is crossed with every task on the
scene, so one `scenario_hash` would cover ten different configurations.

That is true and it is not a problem, because the comparison unit is
`(scene_id, task_hash, scenario_hash)`. Scenario 3 under task 0 and scenario 3
under task 1 are already different units. The join was never on `scenario_hash`
alone. The split was defending against something the identity scheme handles.

The second reason was real: putting each task's BDDL into its `scene_hash` is what
made the task's *goal* content-addressed. That one needs a different fix.

## The gap in the original design

`task_hash` covers `instruction`, `predicate`, `predicate_args`, `max_steps` and
`phases`. That is complete for a task whose goal is *defined* by those fields — a
predicate over observations, with arguments.

It is empty for a task whose goal is opaque to all three. `from_benchmark` is a
passthrough: LIBERO decides whether a LIBERO task succeeded. So for these tasks
`predicate` is a constant, `predicate_args` is `{}`, and the only thing
distinguishing two goals is `instruction` — a human-readable string that a
benchmark release could keep while moving the goal region underneath it.

The spec never anticipated a benchmark whose goal is not expressible in its Task
table. So this is a gap in the original design rather than something the
BDDL-in-`scene_hash` workaround was contradicting. The workaround was solving a
real problem in the wrong place.

## The fix

A content hook on `Task`, folded into `task_hash`, supplied by the probe the same
way an external scene's facts already are:

* **`Scene`** keeps what defines the `MjModel`: suite, objects and fixtures,
  engine version, provider, `libero` version.
* **`Task`** gains probe-supplied facts, hashed into `task_hash`: the BDDL digest
  **and the init-state array digest**.

The init-state digest moves to the task for the same reason the BDDL does — the
array is keyed by `task_id`, so it is task content, not scene content. It is
`MjData`, and `MjData` was never what a scene hash was for. Putting it in
`scene_hash` was the same category error as splitting the scenes.

### A named field, not a reserved `predicate_args` key

`predicate_args` is already immutable and hashed, so a reserved key would ride in
for free. Rejected: `from_benchmark` takes no arguments, so a goal digest sitting
in its argument dict would look like an argument to a predicate that has none,
and a reader would reasonably wonder what it was doing there. The saving is a
schema field; the cost is that the most load-bearing digest in the catalog is
disguised as something else.

## What it buys beyond correctness

One scene with ten tasks means one worker holds ten tasks' episodes, and
`worker_selection` refuses a worker spanning two tasks — the harness's work-item
loop can only be constrained to one at a time. So the loop must gain a task
level:

```
scene -> worker -> task -> checkpoint -> seed
```

Which is exactly the nesting `interleaved` needs. The restructure and the
execution-mode work are the same change.

## Cost

Every `plan_id` moves: `scene_hash` loses the BDDL and init-state digests,
`task_hash` gains them. Results recorded under the ten-scene catalog do not join
with results recorded after. They remain readable and their catalog is copied
into their results directory, which is what that copy is for.
