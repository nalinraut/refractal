# Writing a catalog

A catalog is five YAML files. `refractal plan` compiles them into `plan.json`,
and everything after that reads the plan.

```
catalog/
  scenes.yaml       what the simulator loads
  tasks.yaml        what counts as success on it
  scenarios.yaml    what varies between episodes
  run.yaml          which checkpoints, how many seeds, which tier
  hardware.yaml     what machine you are planning for
```

`refractal init` writes a working one. This page is what to change in it.

## Deciding what is a scene

**A scene is one compiled model.** Two configurations share a scene if the
simulator would load the same bodies, geometry and collision pairs, and differ
only in where things start.

```yaml
scenes:
  - id: bench-v1
    engine: mujoco
    engine_version: "3.2.0"
    model: assets/bench.xml
```

Scenes are expensive and tasks are cheap. A scene is constructed once per worker
and reused across every episode that runs on it, so **put as much as you can on
one scene** and vary the rest with tasks and scenarios.

If you find yourself writing two scenes whose model files are identical, they are
one scene with two tasks.

### Scenes that live in someone else's package

When the geometry comes from an installed benchmark rather than a file you
control, declare it `external` and let `refractal build` fetch the facts:

```yaml
  - id: bench-v1
    engine: mujoco
    engine_version: "3.2.0"
    external:
      provider: some_package.benchmark:SomeBenchmark   # which class defines it
      ref:    {suite: kitchen}                          # which scene within it
      params: {send_state: true, num_steps_wait: 10}    # constructor arguments
```

`ref` says *which* scene; `params` is passed to the provider's constructor. Both
are hashed into the scene's identity, and a key appearing in both must have the
same value in both. Keep them separate: a key that identifies the scene is
usually not a constructor argument, and passing one as the other raises.

An external scene needs `refractal build --probe module:Class`, run somewhere the
provider is importable. `refractal plan` then works anywhere.

## Deciding what is a task

**A task is a goal on a scene.** Many tasks per scene is the normal case.

```yaml
tasks:
  - id: cup-on-shelf
    scene: bench-v1
    instruction: "put the cup on the shelf"
    predicate: my_package.predicates:object_on
    predicate_args: {object: cup, surface: shelf}
    max_steps: 300
```

Four of those fields decide whether two runs are comparable (`instruction`,
`predicate`, `predicate_args`, `max_steps`), so changing any of them makes a
different task, and results recorded before the change will not join results
recorded after. `id` and `description` do not: rename freely.

`max_steps` being part of that set is the one that surprises people. Two runs at
different step budgets are two experiments, deliberately, because a task you can
complete in 300 steps and one you can complete in 100 are different questions.

### When the benchmark decides success

If the wrapped benchmark already knows whether its own task succeeded, use the
passthrough predicate rather than re-deriving it:

```yaml
    predicate: refractal.predicates:from_benchmark
```

That leaves `instruction` as the only thing distinguishing two goals, which is
not enough: a release could keep the string and move the goal. Give the task a
`provider_ref` so `refractal build` can record what the goal actually *is*:

```yaml
    provider_ref: {task_id: 3}
```

The scene's `ref` says which model; the task's `provider_ref` says which goal
within it.

## Deciding what is a scenario

**A scenario is what varies between episodes on one scene.** Scenario sets are
crossed with every task on their scene.

```yaml
scenario_sets:
  - id: cup-positions
    scene: bench-v1
    generator: refractal.generators:linspace_grid
    generator_seed: 0
    params:
      cup_x: {range: [-0.15, 0.15], steps: 12}
      cup_y: {range: [0.30, 0.60], steps: 12}
    perturbations: []
```

The parameter names are **yours**. Refractal never looks inside the dict. It
hashes it and hands it to your adapter's `reset`. A catalog naming `cup_x` and an
adapter reading `cube_x` produces episodes that all start from the adapter's
default and a comparison that means nothing, with no error anywhere. Name them
once and grep for both.

Four parameter forms are available: `{value: x}`, `{choices: [...]}`,
`{range: [lo, hi], steps: n}`, and `{range: [lo, hi]}` for samplers. Integer
ranges that divide evenly stay integers.

### `perturbations`

Timed physical changes to the simulator during an episode: cut gripper torque at
step 200, displace an object mid-transport. **Reserved.** The key is validated
and a non-empty list is refused, because nothing executes one yet and a spec
recorded as though it had fired, having never fired, is worse than no feature.

The schema and the identity are settled ahead of the machinery because identity
ossifies the moment results exist. A scenario gets two hashes:

| | covers |
|---|---|
| `scenario_hash` | the parameters **and** the perturbation |
| `base_scenario_hash` | the parameters alone |

A perturbed episode is genuinely a different experiment, so it must not join its
unperturbed counterpart as though it were the same — and a sweep still needs
something to hold fixed while the level varies. `compare` joins on the base and
groups by level, which is the join it already does across checkpoints, one axis
over. `episode_id` derives from `scenario_hash`, so two episodes differing only
in torque scale are different episodes.

For an unperturbed scenario the two are **the same string**. So every result
recorded before the field existed already carries a valid base, and a sweep run
later joins it on the unperturbed arm of its own curve.

The key was `faults` and is not. Reducing gripper torque by 20% is a weaker
gripper, not a thing that went wrong, and the old name presumed the outcome being
measured. `is_infra_failure` and `failure_reason` keep their meaning: a dropped
object under perturbation is an outcome, a crashed worker is a failure. Renaming
the key in your catalog moves no identity — the hashed document keeps its
original key deliberately, so that the rename costs nothing.

### If your parameters are indices

Some benchmarks expose a fixed array of start states, and the only thing you can
vary is which one. That works, with one constraint: **the range must start at
zero and be contiguous**, because a wrapped benchmark counts its own episodes
from zero.

```yaml
      start_state_index: {range: [0, 9], steps: 10}
```

`[5, 14]` would run start states 0–9 while every row claimed 5–14. `refractal
run` refuses this rather than recording it.

## Declaring the machine

`hardware.yaml` names the machine; `resource_shape` on each scene says what one
worker of that scene needs on it.

```yaml
    resource_shape:
      - hardware_profile: workstation
        envs_per_process: 1     # how many environments share one process
        partition_unit: task    # what one worker must own whole
        vram_per_env_mb: 0
        cpu_cores: 1
        sec_per_1k_steps: 40.6  # wall clock for the WORKER, including inference
        startup_sec: 3
        measured_at: "2026-09-19"
```

**Every figure here is per environment, except `memory_mb`.** A worker may hold
more than one environment, and the planner multiplies. Under `execution_mode:
concurrent` a worker runs one thread per checkpoint, each with its own
environment, so comparing two checkpoints doubles what a worker occupies:
`cpu_cores: 1` above reserves two cores per worker, and the container Compose
renders is pinned to two.

`memory_mb` is the exception and is per worker. It says so in the schema; the
declared values were measured that way.

Two of these are easy to confuse and are independent:

| field | question |
|---|---|
| `envs_per_process` | how many episodes may share one process at once |
| `partition_unit` | what one worker must own entirely |

An engine that vectorises answers `4096` and `scenario`. One that does not, driven
by a benchmark that counts episodes from zero per task, answers `1` and `task`.
Those point in opposite directions, so neither can be derived from the other.

`partition_unit` defaults to `task`, the conservative value: splitting a worker
that should not be split runs the wrong start states and records them as right,
which is silent, while refusing to split only costs parallelism.

`sec_per_1k_steps` is the wall clock a **worker** experiences, which includes
waiting on the policy. Timing the simulator alone will give you a number several
times too small, and it will look measured.

Three fields in this block have now been wrong in use before they were wrong in
name -- `sec_per_1k_steps` (worker or simulator), `startup_sec` (per invocation
or per episode), and `cpu_cores` (per worker or per environment). Each survived
because the two readings agree in the ordinary case and diverge silently outside
it. When you add a field here, say what it is per.

Set `measured_at` when you have measured. A comment saying "measured" beside an
empty `measured_at` fails `scripts/lint_provenance_claims.py`, because that
combination has shipped a value ten times too large before.

## Declaring the run

```yaml
run:
  checkpoints:
    - {id: baseline,  path: org/model-a, server: my_package.servers:MyServer}
    - {id: candidate, path: org/model-b, server: my_package.servers:MyServer}
  seeds: 3
  tier: full
  execution_mode: serial
  results_uri: ./results
```

`seeds` is repeats per scenario, and you want more than one: with a single repeat
there is no within-scenario variance to measure, and `compare` cannot tell a
clustered difference from an unclustered one.

`tier` subsamples scenarios (`smoke` 10%, `regression` 50%, `full` 100%), and is
part of the experiment's identity, so a smoke run and a full run do not join by
default. `refractal compare --promote-from` pools them when you ask.

`server_args` on a checkpoint is part of the identity too. A parameter that
changes what the policy sees belongs there, so two runs that disagree about it
cannot be compared by accident.

## What invalidates what

| change | effect |
|---|---|
| rename a task or scene `id` | nothing; ids are labels |
| edit a `description` | nothing |
| change `instruction`, `predicate`, `predicate_args`, `max_steps`, `phases` | new task; old results do not join |
| change a scene's model, engine version, or external `ref`/`params` | new scene; old results do not join |
| change scenario parameters or `generator_seed` | new scenarios |
| change `tier`, `seeds`, or the checkpoint set | new experiment |
| change `results_uri`, `execution_mode`, or any `resource_shape` | nothing; these are placement |

When two plans differ and you cannot see why, `refractal explain before.json
after.json` names the field.
