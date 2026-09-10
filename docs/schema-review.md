# Spec review — things I think are wrong, found while building `schema`

Against `refractal-design.md` (Parts 1–12), the implementation prompt, and the
API reference. Ordered by how expensive they are to fix later.

Everything here is implemented as described in `src/refractal/schema/`, marked
`DEVIATION` at the site and tested. Reverse any of them and the tests will say so.

---

## 1. Task identity is positional — the project's own bug, one level up

`episode_id` is specified as

    sha256(scenario_hash + seed + checkpoint_id + scene_hash + task_id)

`scene_hash` is the *content* of the scene. `task_id` is a string a human chose.
Those are not symmetric, and the asymmetry is exactly the failure Refractal exists
to fix.

Edit `predicate_args` from `{slot: 4}` to `{slot: 5}`, leave the id as
`vial-slot-4`, and you get an episode with an unchanged identity and a changed
meaning. Resume skips it. `compare` joins it against the old result. Nothing
errors. This is `(task_name, 7)` wearing a hash.

**Fix.** `task_hash` over `{instruction, predicate, predicate_args, max_steps,
phases}`, and `episode_id` uses that. `id` and `description` are excluded, so
renaming a task is free and changing its meaning is not.

**Consequence:** `max_steps` has to be part of task identity, though the
reference leaves it unmarked. Raising the limit from 400 to 800 changes
outcomes, so it changes comparability.

**Not fixed, related:** `checkpoint_id` is also a label. Overwrite
`./checkpoints/gr00t-n16-47` in place and the identity is unchanged. Hashing
weights at plan time is too expensive to do casually; the cheap version is for
`refractal build` to record a `checkpoint_hash` when the path is local, and warn
when it can't.

## 2. The join key cannot be `scenario_hash`

Two independent reasons:

- A scenario set is crossed with **every task on its scene**, so one
  `scenario_hash` covers `vial-slot-4` and `vial-slot-7` alike. Joining on it
  alone merges two different goals into one 2×2 — and the reference specifies
  the 2×2 as being *per task*, so the two statements contradict each other.
- `scenario_hash` is over parameter values only, so two scenes that both
  parameterize `x` and `y` collide.

**Fix.** The comparison unit is `(scene_id, task_hash, scenario_hash)`.

`scene_hash` is deliberately **not** in the key. If it were, editing a mesh
would produce an empty join rather than a loud one. `compare` should join on the
key above and then assert `scene_hash` agreement across checkpoints — "you are
comparing results from two different geometries" is a sentence someone needs to
read, not a silent zero-row result.

## 3. `workers = ceil(episodes / envs_per_process)` cannot be right

Under `mujoco`, `envs_per_process` is 1. The reference's own worked example has
204 episodes on `vial-rack-v1` and prints **8 workers total** — the formula gives
204.

`envs_per_process` is a *batch width*, not a capacity. It sets what one worker
holds at once and therefore that worker's CPU and VRAM footprint. It does not
set how many workers exist.

Worker count has to come from host capacity, with episodes sharded across the
workers you can afford:

    workers_for_scene = min(episodes, budget_from_host_capacity)
    episodes_per_worker = ceil(episodes / workers_for_scene)

This is `resolve`'s problem, not `schema`'s, but it changes `schema`: see next.

## 4. Nothing in the catalog says what the host has — added `hardware.yaml`

`--hardware rtx5090` selects a **resource shape**, which describes what one
worker *needs*. Nothing anywhere describes what the machine *provides*. Yet the
reference's example output prints `VRAM: cuda:0 16384 / 32768 MB` and a worker
count, both of which require a capacity to divide into. The "refuse to emit a
plan that oversubscribes VRAM" invariant has no number to compare against.

**Fix.** A fifth catalog file, `hardware.yaml`: `HardwareProfile{id, cpu_cores,
memory_mb, devices[{id, vram_mb}], max_workers}`. Optional — its absence is what
makes `refractal plan` fail with "no capacity declared for profile X" instead of
producing a plan built on a guess.

**Excluded from `plan_id`.** Hardware decides placement, not experiment
identity. The same experiment planned for a 5090 and for an A100 is the same
experiment and its results must join.

## 5. `plan_id` contradicts D9, and depends on the output directory

Specified as `sha256(catalog_bytes + sorted(checkpoints) + plan_schema)`. Three
problems:

- **`plan_schema` is in it.** D9 argues at length that format compatibility and
  experiment identity are independent, and that conflating them "would
  invalidate comparability with old results. That is wrong." Then the formula
  does it. Bumping the plan format would orphan every recorded comparison.
- **`results_uri` is in it**, via `catalog_bytes` → `run.yaml`. So the
  comparison id depends on where the output goes: the same experiment written
  to `./results` and to `s3://…` yields two ids that never join, and the output
  directory ends up naming itself.
- **Raw bytes.** A reordered key, a changed comment or trailing whitespace
  churns the id. That is the same objection D9 raises against deriving
  `plan_schema` structurally, applied to the field that actually is derived.

**Fix.** `plan_id = H(experiment_identity)` over the *parsed* models, excluding
`results_uri`, `execution_mode`, `max_concurrent_checkpoints`, resource shapes,
hardware, and all descriptions.

`execution_mode` is excluded so interleaved and serial runs of one experiment
land in one comparison; the mode is on every episode row, which is what gates
latency reporting.

`tier` **is** included, since it changes which scenarios exist. Cost worth
naming: a smoke run and a full run of one catalog get different ids, so smoke
results can't be reused as a head start on the full run even though the smoke
set is a strict subset. Revisit if tier-promotion turns out to matter.

## 6. "Import strings resolved at load time" breaks invariant 1

`Task.predicate` resolves into the adapter, which imports the simulator.
`Checkpoint.server` resolves into a model server, which imports torch. Resolving
those during `refractal plan` means `plan` no longer runs on a bare laptop —
the property the compiler architecture rests on.

**Fix.** Two tiers. `schema` validates **syntax only** and never imports
anything (`tests/test_boundaries.py` asserts this). `resolve` imports generators
and filters, because it must call them. `execute` imports predicates and servers.

**Contract this creates, and it needs documenting for users:** generators and
filters must be importable without a simulator, because `resolve` runs where no
simulator exists. `filter: so101_eval.filters:reachable` is fine if reachability
is pure kinematics; it is not fine if it opens an `MjModel`. If a filter really
needs the engine, it belongs in `refractal build`, baking its output into the
catalog.

## 7. `phase_outcomes` as a fixed Parquet struct will break

`episodes.parquet` is partitioned by `checkpoint` and `scene` only, so two tasks
on one scene write to the same file. `vial-slot-4` declares four phases;
`vial-slot-7` declares none. A struct has one schema per file.

**Fix.** `map<string, bool>`, or a list of `{name, reached}`. Costs nothing;
avoids a schema conflict that only appears once you have two tasks per scene,
which is step 4.

## 8. Float determinism: the stated risk isn't the real one

The spec frames it as `0.1` versus `0.10`. Those already parse to the same
double — free.

The real hazard is arithmetic. `0.08 + 2*0.02` is `0.12000000000000001`;
`linspace(0.08, 0.16, 5)[2]` may be `0.12`, depending on library, platform and
summation order. numpy has changed its `linspace` summation between releases. A
generator that drifts one ulp forks scenario identity silently, and the symptom
is an empty join, which is a legal join that raises nothing.

**Fix.** Quantize to 12 significant digits before hashing, store the quantized
value, and fold `-0.0` to `0.0`. Also fold int→float **in scenario params only**:
editing `vial_x: 0` to `vial_x: 0.0` is a no-op to a physicist and must be a
no-op to the hash. Authored data like `predicate_args: {slot: 4}` keeps the
distinction, because there `4` is an index and `4.0` reads differently.

Also: the built-in generators take no numpy dependency, for exactly this reason.

## 9. `scene_hash` needs an engine version that no field carries

`model_hash` is defined as covering "the engine version string", and `resolve`
runs where no engine is installed to be asked for one.

**Fix.** `Scene.engine_version`, declared, written by `refractal build` where the
engine does exist. Its absence is a warning: without it, two engine versions
produce the same `scene_hash` and their results join as if comparable.

**Naming, while here:** `scenes.yaml` calls it `model_hash`; `plan.json` and
`episodes.parquet` call the same value `scene_hash`. Suggest `scene_hash`
everywhere. Minor supporting evidence: `model` and `model_hash` both collide
with pydantic's reserved `model_*` namespace and need
`protected_namespaces=()` to declare.

## 10. Smaller things

- **The determinism gate can't pass as written.** "Same catalog → byte-identical
  plan" versus `plan.json` carrying `created_at`. Either exclude it from the
  comparison or drop it. `tests/test_loader.py` compares `plan_id` and the
  scenario hash list across two interpreters instead.
- **`failure_reason` is ambiguous.** "Null means a genuine policy failure" — but
  it is also null on success. Make it null iff success, with an explicit
  `policy_failure` value.
- **Hash concatenation.** `sha256(a + b)` is ambiguous for variable-length
  parts: `("a","b1")` and `("ab","1")` collide. Fixed-width hashes mostly hide
  it; `checkpoint_id` is free-form and doesn't. Every identity here hashes a
  keyed document instead. Costs nothing.
- **Infra failures break the pairing.** Dropping `is_infra_failure` rows from the
  denominator is right, but if ckpt-46's episode crashed and ckpt-47's didn't,
  that scenario is no longer paired. `compare` needs an explicit rule — drop the
  unit when any checkpoint has too few valid episodes — and should report how
  many units it dropped that way, in the same block as the overlap counts.
- **Tier subsampling needs stratifying.** A hash-prefix threshold on
  `scenario_hash` is stable and gives a strict subset, which is right. But a
  scenario set with three scenarios can get zero in smoke. Stratify per
  `(scene, scenario_set)` with a floor of one.
- **The reserved `faults` hook only works if it's hashed now.** Reserving the
  field is not enough: if faults are simply *absent* from the hashed document
  today and added later, every recorded `scenario_hash` changes the day Transect
  lands, and the reservation bought nothing. `scenario_identity` folds an empty
  `faults` list into the canonical form from the first commit.

---

## 11. `plan.json` workers need episode *records*, not episode ids

Found while writing `resolve`. The API reference gives each worker
`episodes: []string` — a list of `episode_id`s — and puts the scenario params on
`PlannedScene.scenarios`. Nothing anywhere maps an `episode_id` back to
`(task, scenario, seed, checkpoint)`.

An `episode_id` is a hash, so that mapping is not recoverable. `execute` would
receive a worker assignment it cannot act on.

**Fix.** `PlannedWorker.episodes` holds `PlannedEpisode` records carrying
`episode_id`, `task_id`, `task_hash`, `scenario_hash`, `seed`, `checkpoint_id`.
The plan becomes self-contained, which is what lets a worker be handed its
assignment and nothing else — and it is also what makes explicit assignment
possible at all, which is the basis of resume.

## Decided: filters are a build-time concern (closes the §6 hole)

The split in §6 keeps `resolve` pure but leaves a hole: if filters must be
simulator-free and `reachable` needs forward kinematics, either the filter
cannot do its job, or `resolve` gains a simulator, or filtering moves to
`execute` and `plan` can no longer state an episode count — which was the point
of planning.

**Resolved: `refractal build` owns filter evaluation, alongside everything else
that needs the engine.** It runs where the engine exists, applies the filter,
and records the surviving scenarios. `resolve` reads the record and **refuses**
if it is stale, exactly as it refuses to oversubscribe VRAM.

This consolidates all four drift-prone facts into the one command that already
requires the engine, under one staleness rule:

| Fact | Why it needs `build` |
|---|---|
| `model_hash` | reads sources; verified against the compiled model in-container |
| `engine_version` | only knowable where the engine is installed |
| `resource_shape` | probed by a calibration episode |
| filter survivors | may need FK, collision checks, penetration tests |

**Cost, stated plainly:** generated content in the catalog. Contained by putting
all of it in `catalog/build.lock` and never rewriting hand-authored YAML —
machine-written and human-written data in one file means lost comments and
merge conflicts. This supersedes the API reference's "`build` overwrites
`scenes.yaml`".

Analytic IK for the SO-101 stays available later, if planning from a fresh
checkout with no lock ever matters enough to justify writing it. It is an
optimisation of this design, not an alternative to it.

Lock entries are keyed by what they depend on, so staleness is detectable rather
than assumed: filter survivors by `(scenario_set identity, scene_hash, filter
import string)`, resource shapes by `(scene_hash, engine_version,
hardware_profile)`. A key mismatch names the stale entry and the command to
refresh it.

## Verified against the harness

All four §7 claims hold, at `4aeb436` and at HEAD, as do `max_batch_size`
defaulting to 1 and `_build_task_result` counting errors in the denominator.
Details, three wrinkles the spec does not mention, and two spec edits for
`execute` and the adapter are in [harness-integration.md](harness-integration.md).

One correction lands on `refractal-design.md` rather than here: the gcd shard
collision in Part 12 was fixed upstream in `0865d42`, after the v0.5.0 pin. It
is history now, not a differentiator.
