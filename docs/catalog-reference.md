# Catalog reference

Every file, every field. [Writing a catalog](writing-a-catalog.md) explains the
decisions; this is for looking things up.

**A key not listed here is rejected.** Every model sets `extra="forbid"`, so a
typo fails at `refractal plan` naming the exact path — `scenes.0.resource_shape.0.vram_per_env_md`
— rather than being silently ignored. That is the single highest-value line in
the schema.

`refractal init .` writes a working example of all five.

## The five files

| file | answers | hashed into |
|---|---|---|
| `scenes.yaml` | what world, and what one environment of it costs | `scene_hash` |
| `tasks.yaml` | what counts as success, and the step budget | `task_hash` |
| `scenarios.yaml` | what varies between episodes | `scenario_hash` |
| `run.yaml` | which checkpoints, how many seeds | `plan_id` |
| `hardware.yaml` | what machine to plan against | nothing |

`hardware.yaml` is deliberately outside identity: the same experiment planned
for a laptop and for three H100s is the same experiment, and `plan_id` does not
move. Only the worker layout does.

### The wrapper every file shares

| field | required | what it does |
|---|---|---|
| `apiVersion` | yes | `refractal.dev/v1alpha1`. Present in all five files. |

and then exactly one list, named for the file: `scenes`, `tasks`,
`scenario_sets`, `run`, `hardware_profiles`.

---

## `scenes.yaml`

```yaml
apiVersion: refractal.dev/v1alpha1
scenes:
  - id: cube-bowl-v1
    engine: mujoco
    model: assets/cube_bowl.xml
    resource_shape:
      - hardware_profile: laptop
        envs_per_process: 1
        vram_per_env_mb: 0
        cpu_cores: 1
        sec_per_1k_steps: 12.0
        startup_sec: 2
```

### `scenes[]`

| field | required | default | what it does |
|---|---|---|---|
| `id` | yes | | Names the scene. Referenced by `tasks[].scene` and `scenario_sets[].scene`. |
| `engine` | yes | | `mujoco`, `mjx` or `isaac`. Decides which image a rendered deployment uses. |
| `model` | one of | `None` | Path to the model file, relative to the catalog. **Exactly one of `model` and `external`** — a required choice rather than an optional field, because an optional one is the one people forget. |
| `external` | one of | `None` | For a scene defined by someone else's benchmark. See below. |
| `engine_version` | no | `None` | Written by `refractal build`, not by hand. Part of the scene's identity. |
| `model_hash` | no | `None` | Written by `refractal build`. Never hand-written. |
| `assets` | no | `[]` | Extra files folded into `scene_hash`. A texture the model references is part of the scene. |
| `resource_shape` | no | `[]` | One entry per hardware profile. See below. |
| `description` | no | `""` | Free text. Not hashed. |

### `scenes[].external`

For a scene that lives in an installed package rather than a file here.

| field | required | default | what it does |
|---|---|---|---|
| `provider` | yes | | Import string of the benchmark class, e.g. `vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark`. |
| `ref` | no | `{}` | How that provider identifies the scene, e.g. `{suite: libero_spatial}`. |
| `params` | no | `{}` | Passed to the provider's constructor. **Not name-validated** — Refractal does not know the constructor's signature. They are hashed into `scene_hash` and recorded in `build.lock`, so *changing* one contradicts the lock and `plan` refuses until you rebuild. A typo present from the start is caught later, by the constructor. |

### `scenes[].resource_shape[]`

What one worker of this scene needs, per hardware profile. **The planner divides
the machine by this and refuses a plan that will not fit** — so these numbers
decide whether it can refuse at all.

| field | required | default | what it does |
|---|---|---|---|
| `hardware_profile` | yes | | Which profile in `hardware.yaml` this shape describes. |
| `envs_per_process` | yes | | Environments one worker holds at once. `1` for classic MuJoCo, which cannot batch. |
| `vram_per_env_mb` | yes | | GPU memory **per environment**. Measure it. A wrong number gives a wrong plan; **a zero removes the constraint entirely** and the planner will never refuse anything. |
| `vram_base_mb` | no | `0` | GPU memory the worker needs **regardless of batch width**. Total is `vram_base_mb + vram_per_env_mb × envs`. Put a fixed per-process cost here, not in the per-env figure — they are identical at one env and diverge the moment a scene batches. |
| `cpu_cores` | yes | | Cores **per environment**. The worker's cpuset is this times the concurrent environments. |
| `memory_mb` | no | `2048` | Host RAM per worker. Becomes the container memory limit. |
| `sec_per_1k_steps` | yes | | Seconds per 1000 simulated steps, **including inference** if the worker blocks on a model server. Drives the duration bound. |
| `startup_sec` | yes | | Cost of constructing the scene once. Above `--pack-below-startup-sec`, starved scenes are packed sequentially rather than given their own worker. |
| `partition_unit` | no | `task` | The indivisible unit a worker owns whole: `task` or `scene`. `task` is the finer grain, and the ceiling on parallelism is the number of these. |
| `max_envs` | no | `None` | A hard ceiling on batch width the planner cannot infer — a driver limit, a licence, a known instability. **Validated but not acted on**: it is checked against `envs_per_process` and otherwise only documents the limit. |
| `measured_at` | no | `None` | A date. Read by nothing; a note to the next person that these were measured, and when. |

---

## `tasks.yaml`

| field | required | default | what it does |
|---|---|---|---|
| `id` | yes | | Names the task. Must be unique across the catalog — merging two catalogs that both number from `task-0` collides. |
| `scene` | yes | | Which scene this task runs on. |
| `instruction` | yes | | The language goal. For a wrapped benchmark this **is the selector** — the harness filters its task list on this string, so it must match exactly. |
| `predicate` | yes | | Import string deciding success, e.g. `refractal.predicates:from_benchmark` to let the benchmark decide. |
| `predicate_args` | no | `{}` | Passed to the predicate. |
| `phases` | no | `[]` | Sub-goals, for tasks scored in stages. Each has a `name`. |
| `max_steps` | no | `400` | The step budget, and **it is identity**: two budgets are two experiments. A failure at the cap is a timeout, not an inability — say which you intended. |
| `provider_ref` | no | `{}` | How the provider identifies this task, e.g. `{task_id: 3}`. |
| `description` | no | `""` | Free text. Not hashed. |

---

## `scenarios.yaml`

Scenarios are **generated, not enumerated**: you describe a grid once and the
description is what gets stored as provenance.

| field | required | default | what it does |
|---|---|---|---|
| `id` | yes | | Names the set. |
| `scene` | yes | | Which scene these scenarios vary. |
| `generator` | yes | | Import string, e.g. `refractal.generators:linspace_grid` or `:latin_hypercube`. |
| `generator_seed` | yes | | Makes a sampling generator reproducible. |
| `params` | yes | | Map of parameter name to a spec — see below. |
| `filter` | no | `None` | A predicate dropping generated scenarios. Evaluated by `refractal build`, which needs the engine. |
| `tasks` | no | `None` | Restrict this set to named tasks. `None` means every task on the scene. |
| `perturbations` | no | `[]` | Timed changes applied during the episode. Hashed into the scenario, so a perturbed episode is a **different experiment** from its unperturbed counterpart. See below, and [perturbations](perturbations/index.md). |

### `perturbations[]`

| field | required | default | what it does |
|---|---|---|---|
| `type` | yes | | Which effect to run, e.g. `scale_actuator`, `drop_observation`. |
| `target` | no | `None` | What it acts on — a name the adapter resolves for a world effect, a field of the observation for a transform. |
| `at_step` | yes | | When it fires, counted from the episode's first step. `0` for anything measuring robustness at a level: an episode that ends before its trigger never experienced the perturbation, and that bias favours whichever checkpoint finishes sooner. |
| `until_step` | no | `None` | When it ends. Absent means it lasts to the end of the episode. Not every effect can carry one — an effect with no inverse is refused rather than given semantics that only hold when nothing else is happening. |
| `args` | no | `{}` | That effect's parameters, e.g. `{factor: 0.1}`. |

### `params[]` — the parameter spec

Which fields you use depends on the generator. A grid takes `range` + `steps`;
a sampler takes `range` + `samples` + `distribution`.

| field | default | what it does |
|---|---|---|
| `range` | `None` | `[low, high]`. |
| `steps` | `None` | How many evenly spaced points across `range`. Grid generators. |
| `choices` | `None` | An explicit list of values. |
| `value` | `None` | A constant — held fixed while other parameters vary. |
| `samples` | `None` | How many points to draw. Sampling generators. |
| `distribution` | `None` | `uniform` or `normal`. How `samples` are drawn from `range`. |
| `mean` | `None` | Centre, for `normal`. |
| `std` | `None` | Spread, for `normal`. |

---

## `run.yaml`

| field | required | default | what it does |
|---|---|---|---|
| `checkpoints` | yes | | What is being compared. Two or more — one checkpoint is not a comparison. |
| `results_uri` | yes | | Where rows are written. A local path or an object-store URI. |
| `seeds` | no | `3` | Replicates per scenario. With a stochastic policy these are genuine replicates, not a way to reproduce a run. |
| `seed_base` | no | `0` | First seed value; seeds run `seed_base … seed_base + seeds - 1`. Part of `plan_id`. |
| `tier` | no | `full` | Which tier of the catalog to run. |
| `scenario_sets` | no | `None` | Restrict the run to named sets. `None` means all of them. |
| `execution_mode` | no | `serial` | `serial` runs every task for checkpoint A, then for checkpoint B — one policy resident at a time. `concurrent` runs them together, one thread per checkpoint. **This is about checkpoints within a worker, not about whether workers are parallel.** Workers always are. |

### `checkpoints[]`

| field | required | default | what it does |
|---|---|---|---|
| `id` | yes | | Names the arm. Used by `--server id=URL` at run time. |
| `path` | yes | | The checkpoint itself — a hub id or a path. |
| `server` | yes | | Import string of the model-server class. |
| `server_args` | no | `{}` | Passed to that class. **In `plan_id`**, and they change what the policy does: `chunk_size`, `state_key`, a normalisation key. One set of weights with two `server_args` is two checkpoints, correctly. |
| `vram_mb` | no | `8192` | What this server reserves on its device. A **fixed** cost, however many workers connect — and it comes out of the pool per-worker environments draw on. |

**No server address appears in the catalog.** Where a server listens is
placement, supplied at run time as `--server id=ws://host:port`, and `plan_id`
cannot see it. Which server and how it is configured *is* identity; where it is
running is not.

---

## `hardware.yaml`

Not hashed into anything. The same experiment on a different machine is the same
experiment.

| field | required | default | what it does |
|---|---|---|---|
| `id` | yes | | Profile name, passed as `--hardware`. |
| `cpu_cores` | yes | | Cores available to workers. |
| `memory_mb` | yes | | Host RAM available. |
| `devices` | no | `[]` | GPUs — `{id, vram_mb}` each. |
| `max_workers` | no | `None` | A ceiling on workers regardless of what fits. |

### `devices[]`

| field | required | what it does |
|---|---|---|
| `id` | yes | e.g. `cuda:0`. |
| `vram_mb` | yes | **Allocatable, not the card's total.** `nvidia-smi` reports a nameplate figure of which a few hundred MiB is driver and display context no allocation can reach. Declaring the nameplate hands the planner memory it cannot give out, and that margin decides worker counts. |

---

## What invalidates what

| change this | and this moves |
|---|---|
| a scene's `model`, `assets`, `external.params` | `scene_hash`, every episode on it |
| a task's `instruction`, `predicate`, `max_steps` | `task_hash`, every episode of it |
| a scenario's generated values or `perturbations` | `scenario_hash` |
| `checkpoints`, `seeds`, `seed_base`, `tier`, `execution_mode` | `plan_id` |
| `resource_shape`, `hardware.yaml`, `results_uri`, the backend | **nothing** — placement, not identity |
