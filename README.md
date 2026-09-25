# Refractal

[![PyPI](https://img.shields.io/pypi/v/refractal)](https://pypi.org/project/refractal/)
[![Python](https://img.shields.io/pypi/pyversions/refractal)](https://pypi.org/project/refractal/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/nalinraut/refractal/blob/main/LICENSE)
[![CI](https://github.com/nalinraut/refractal/actions/workflows/ci.yml/badge.svg)](https://github.com/nalinraut/refractal/actions/workflows/ci.yml)

**A closed-loop evaluation compiler.**

Refractal turns a declared set of scenarios into a plan you can inspect before running, 
executes it on your chosen backend with the policy acting in simulation, and tells you 
whether the difference between two checkpoints is real.

Backends: local, Docker Compose, Kubernetes.

Run the same evaluation twice and know whether the difference is real.

You have a policy. You changed something. You want to know if the new version is
better.

Today you run a benchmark, get 78%, run the old one, get 74%, and guess. That
number hides everything that matters: *which* situations got better, whether four
points is real or noise, whether some of those "failures" were your container
crashing, and whether the new version got worse at something while getting better
overall.

Refractal turns "run a benchmark" into "run an experiment."

```console
$ pip install "refractal[execute,compare]"
$ mkdir demo && cd demo
$ refractal init .
$ refractal plan catalog --hardware laptop -o plan.json
  cube-bowl-v1        30 scenarios     360 episodes    2 worker(s)  <=37 min
  ------------------------------------------------------------------
  360 episodes across 1 scene(s), 2 worker(s), at most 37 min
  that is an upper bound: every episode is costed at its full step limit,
  and episodes that succeed finish sooner.
  wrote plan.json  (plan_schema 3, plan_id sha256:82b73f180a7a...)

$ refractal run plan.json -o results --catalog catalog
$ refractal compare results $(python -c "import json;print(json.load(open('plan.json'))['plan_id'])")
```

About a minute end to end, with no GPU, no simulator and no checkpoints: the
example runs against a built-in fake benchmark, so the machinery is real and only
the robot is not.

**[Start here →](https://github.com/nalinraut/refractal/blob/main/docs/getting-started.md)**

## Documentation

| | |
|---|---|
| [Getting started](https://github.com/nalinraut/refractal/blob/main/docs/getting-started.md) | Install to verdict, start to finish. |
| [Writing a catalog](https://github.com/nalinraut/refractal/blob/main/docs/writing-a-catalog.md) | Describing your own experiment, and which edits invalidate existing results. |
| [Catalog reference](https://github.com/nalinraut/refractal/blob/main/docs/catalog-reference.md) | Every file and every field, for looking things up. A key not listed there is rejected. |
| [Reading a comparison](https://github.com/nalinraut/refractal/blob/main/docs/reading-a-comparison.md) | Every number in the output and when not to trust it. |
| [Perturbations](https://github.com/nalinraut/refractal/blob/main/docs/perturbations/index.md) · [declaring](https://github.com/nalinraut/refractal/blob/main/docs/perturbations/declaring.md) · [adapters](https://github.com/nalinraut/refractal/blob/main/docs/perturbations/adapters.md) · [sweeps](https://github.com/nalinraut/refractal/blob/main/docs/perturbations/sweeps.md) | Changing the world mid-episode, when success rates alone cannot separate two checkpoints. |
| [Adapter contract](https://github.com/nalinraut/refractal/blob/main/docs/adapter-contract.md) · [example](https://github.com/nalinraut/refractal/blob/main/docs/adapter-example.md) | Connecting your own simulator. |
| [Against vla-eval](https://github.com/nalinraut/refractal/blob/main/docs/backend-vla-eval.md) · [In containers](https://github.com/nalinraut/refractal/blob/main/docs/backend-compose.md) | Running at scale. |
| [Execution modes](https://github.com/nalinraut/refractal/blob/main/docs/execution-mode.md) · [Releasing](https://github.com/nalinraut/refractal/blob/main/docs/releasing.md) | |

## What it does

**Declares experiments, not benchmark invocations.** You describe scenes, tasks
and scenario grids in YAML. Refractal expands that into a concrete list of
episodes, each with a content-addressed identity, works out how many workers the
run needs and whether they fit in memory, and writes a plan you can read, diff and
commit before spending anything.

**Compares episode by episode, with statistics that fit the design.** The same
scenarios face every checkpoint, so the data is *paired*. Refractal reports the
full 2×2 per task, runs McNemar on scenario-level outcomes, and gates on a
clustered bootstrap that resamples whole scenarios, because five seeds at one
pose are one observation of that pose rather than five.

**Refuses rather than guessing.** A plan that oversubscribes VRAM is not emitted.
A comparison across two scene geometries is blocked, not reported. A worker
assignment the backend cannot execute is refused at planning time rather than
discovered forty minutes in.

## Three things it gets right that aggregate scores cannot

**Identity is content, never position.** An episode is a hash of
`(scene, task, scenario, seed, checkpoint)`. Rename a task and results still
join; change what the task *means* and they correctly stop. Reordering a list
cannot silently change what an attempt was.

**Infra failures leave the denominator.** A crashed worker is its own column, not
a policy failure. And because dropping it breaks the pairing a paired test needs,
the seed is dropped from *every* checkpoint, with the loss reported rather than
absorbed.

**Multiplicity is corrected across the whole family.** The gate fires if any
contrast trips, so every contrast is one family. Three tasks at a nominal 5% is a
family-wise error rate near 14%; four checkpoints across three tasks is eighteen
contrasts and about 60%. Holm-adjusted, with the uncorrected number printed so
the correction does not read as pedantry.

## Comparing more than two checkpoints

Two is the common case, not a special one. A single checkpoint is a comparison of
size one; three is a training sweep, and produces something a leaderboard cannot:

```
scene-v1 / task-4   (119 scenarios)
  rates:  ckpt-46: 49.4%   ckpt-47: 49.8%   ckpt-48: 37.5%   (baseline ckpt-46)
  outcome patterns:
       35  none solve
       18  only ckpt-46, ckpt-47        <- what ckpt-48 lost
       18  all solve
       17  only ckpt-47
        3  only ckpt-48
  Cochran Q (screen):  Q=19.73 p=0.0005  (66 of 119 scenarios disagree)
  ckpt-46 -> ckpt-47: +0.005 [-0.059, +0.066]  p=0.8874 holm=1.0000  -> no change
  ckpt-46 -> ckpt-48: -0.118 [-0.179, -0.059]  p=0.0008 holm=0.0040  -> REGRESSED
```

"Eighteen scenarios that ckpt-46 and ckpt-47 both solve and ckpt-48 lost" is a
hypothesis you can go replay. Three lost scenarios would be noise.

## Exit codes

`refractal compare` is meant for CI:

| code | meaning |
|---|---|
| 0 | no regression |
| 1 | regression detected |
| 2 | **cannot be answered**: different scene geometry, duplicated episodes, or a change to the code that ran them |

The third is not a regression and is never reported as one. "The two runs used
different geometry" is a different sentence from "the policy got worse", and
conflating them teaches people to ignore the gate.

## Architecture

Refractal is a compiler, not a runtime. A catalog goes in, a `plan.json` comes
out, and the plan executes. The schedule is decided once and recorded, which makes
placement part of the provenance rather than an accident of the day.

| package | does | needs infrastructure? |
|---|---|---|
| `refractal.schema` | catalog, validation, identity | no |
| `refractal.resolve` | catalog → `plan.json` | no |
| `refractal.build` | facts that need the engine → `build.lock` | yes |
| `refractal.execute` | runs a plan; three backends | yes |
| `refractal.perturbations` | timed changes to world, observation or action | no |
| `refractal.render` | plan → deployment file | no |
| `refractal.compare` | Parquet → verdict | no |

`refractal plan` runs on a laptop with no simulator, no GPU and no Docker. That is
enforced by a test rather than by convention: it is the property that lets you
inspect a plan and its cost before committing to it.

Results are Parquet at an fsspec URI, so `./results` and `s3://bucket/results` are
the same code path. Every write is atomic, which is what makes resume safe, and
resume works by episode identity rather than by a counter.

## Backends

Two things are called a backend and it is worth keeping them apart. `refractal
run --backend X` executes a plan. `refractal render --target Y` writes a
deployment file that something else executes.

| `refractal run --backend` | |
|---|---|
| `local` | simulates outcomes; no infrastructure |
| `vla-eval` | drives the harness against running model servers |
| `compose` | one container per worker, over the same entrypoint |

| `refractal render --target` | |
|---|---|
| `compose` | a Compose file; `docker compose up` runs it |
| `k8s` | one Job per worker; `kubectl apply` runs it |

Kubernetes is reached through `render`, not through `run`: Refractal writes the
Jobs and your cluster schedules them. Both targets render **without Docker or a
cluster installed**, so you can read the file before running it.

## Install

```console
pip install refractal                    # plan and build
pip install "refractal[execute]"         # + run
pip install "refractal[compare]"         # + compare
pip install "refractal[vla-eval]"        # + the harness backend
```

Comparison needs no numerical stack: exact McNemar is a binomial tail, the
bootstrap is resampling, and Cochran's Q is a permutation test.

## Status

Alpha. Every stage is implemented and tested, and the identity-bearing fields are
still able to move between versions, which is what the `a` in `0.1.0a1` is for.

Validated against real model servers as well as the synthetic backend: **15,490
episodes across 13 runs**, including a 600-episode comparison whose success rate
was predicted from an earlier run's data before it was run and came back within
two points.

The largest is a 2,700-episode comparison of three checkpoints across three
LIBERO suites, executed twice under one `plan_id` — once locally, once as one
container per worker:

| | wall clock | episode work | overall success |
|---|---|---|---|
| local, workers in sequence | 317 min | 314 min | 72.6% |
| compose, workers in parallel | **171 min** | 462 min | 72.0% |

Placement moved the wall clock by 1.86× and the result by 0.6 points. The two
placements agreed on **92.0%** of episodes — while re-running a *single*
placement against itself agreed on only **89.7%**, so two deployments differ
less than one deployment differs from its own rerun. That is the claim the
content-addressed identity exists to support, and it is measured rather than
asserted.

The 462 minutes of episode work against 314 is the honest other half:
parallelism bought real wall clock and was not free, because four workers
contend where one had the machine to itself.

## What is not here yet

- **Trajectory quality metrics.** The `steps.parquet` schema is declared and
  nothing writes it, deliberately: writing it against synthetic data would bake in
  guesses about what a real adapter can record.
- **`--backend k8s`.** `refractal render --target k8s` writes the Jobs and they
  run against the same `--worker` entrypoint Compose uses, but nothing supervises
  them from inside Refractal: no `run` subcommand submits them, watches them, or
  collects their exit codes. Rendered and applied, not driven.
- **Resource-shape probing.** `refractal build` carries hand-declared shapes
  forward and records whether they were measured. A lint refuses a comment
  claiming a measurement when nothing recorded one.

## Built on vla-evaluation-harness

Refractal is a layer over
[**`allenai/vla-evaluation-harness`**](https://github.com/allenai/vla-evaluation-harness),
from the Allen Institute for AI, and does not fork it. The harness runs the
episodes: it owns the model-server protocol, the benchmark adapters, the
observation and action specs, and the episode loop. Refractal decides which
episodes to run, gives them content-addressed identities, and compares the
results.

Everything `--backend vla-eval` does is the harness doing it. Refractal subclasses
two of its classes and otherwise stays out of the way, which is why the
integration is a pinned dependency rather than a vendored copy, and why
`scripts/verify_harness_claims.py` exists: the assumptions Refractal makes about
someone else's code are checked rather than assumed.

The harness is Apache-2.0, as is Refractal. No code is copied from it.

If you use Refractal for published work, cite the harness as well. The evaluation
is theirs; the comparison is ours.

## Acknowledgements

- [`allenai/vla-evaluation-harness`](https://github.com/allenai/vla-evaluation-harness)
  (Allen Institute for AI, Apache-2.0), which runs every episode.
- The benchmark suites and model servers it wraps, each under its own licence and
  each the work of its own authors.

## Licence

Apache-2.0. See [LICENSE](https://github.com/nalinraut/refractal/blob/main/LICENSE)
and [NOTICE](https://github.com/nalinraut/refractal/blob/main/NOTICE).
