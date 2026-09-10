# Refractal

**Run the same evaluation twice and know whether the difference is real.**

You have a robot policy. You changed something. You want to know if the new
version is better.

Today you run a benchmark, get 78%, run the old one, get 74%, and guess. That
number hides everything that matters: *which* situations got better, whether
four points is real or noise, whether some of those "failures" were your
container crashing, and whether the new version got worse at something while
getting better overall.

Refractal turns "run a benchmark" into "run an experiment."

```console
$ pip install refractal
$ refractal init .
$ refractal plan catalog --hardware laptop
  cube-bowl-v1        30 scenarios     360 episodes    4 worker(s)  ~18 min
  ------------------------------------------------------------------
  360 episodes across 1 scene(s), 4 worker(s), est. 18 min
  wrote plan.json  (plan_schema 1, plan_id sha256:dc5e1755911f...)

$ refractal run plan.json --catalog catalog
$ refractal compare results <plan_id>
```

> **Status: alpha.** The planning, execution, comparison and build stages are
> implemented and tested. The simulator adapter, the Compose backend and
> resource-shape probing are not — see [What is not here yet](#what-is-not-here-yet).

## What it does

**Declares experiments, not benchmark invocations.** You describe scenes, tasks
and scenario grids in YAML. Refractal expands that into a concrete list of
episodes, each with a content-addressed identity, works out how many workers the
run needs and whether they fit in memory, and writes a plan you can read, diff
and commit before spending anything.

**Compares episode-by-episode, with statistics that fit the design.** The same
scenarios face every checkpoint, so the data is *paired*. Refractal reports the
full 2×2 per task, runs McNemar on scenario-level outcomes, and gates on a
clustered bootstrap that resamples whole scenarios — because five seeds at one
pose are one observation of that pose, not five.

**Refuses rather than guessing.** A plan that oversubscribes VRAM is not
emitted. A comparison across two scene geometries is blocked, not reported. A
filter whose body changed since it was last evaluated invalidates the cached
result instead of answering from it.

## Three things it gets right that aggregate scores cannot

**Identity is content, never position.** An episode is a hash of
`(scene, task, scenario, seed, checkpoint)`. Rename a task and results still
join; change what the task *means* and they correctly stop. Reordering a list
cannot silently change what an attempt was.

**Infra failures leave the denominator.** A crashed worker is its own column,
not a policy failure. And because dropping it breaks the pairing a paired test
needs, the seed is dropped from *every* checkpoint, with the loss reported
rather than absorbed.

**Multiplicity is corrected across the whole family.** The gate fires if any
contrast trips, so every contrast is one family. Three tasks at a nominal 5% is
a family-wise error rate near 14%; four checkpoints across three tasks is
eighteen contrasts and about 60%. Holm-adjusted, with the uncorrected number
printed so the correction does not read as pedantry.

## Comparing more than two checkpoints

Two is the common case, not a special one. A single checkpoint is a comparison
of size one; three is a training sweep, and produces something a leaderboard
cannot:

```
vial-rack-v1 / vial-slot-4   (119 scenarios)
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
| 2 | **cannot be answered** — different scene geometry, or a harness change that may have altered what was measured |

The third is not a regression and is never reported as one. "The two runs used
different geometry" is a different sentence from "the policy got worse", and
conflating them teaches people to ignore the gate.

## Architecture

Refractal is a compiler, not a runtime. A catalog goes in, a `plan.json` comes
out, and the plan executes. The schedule is decided once and recorded, which
makes placement part of the provenance rather than an accident of the day.

| package | does | touches infrastructure? |
|---|---|---|
| `refractal.schema` | catalog, validation, identity | no |
| `refractal.resolve` | catalog → `plan.json` | no |
| `refractal.build` | facts that need the engine → `build.lock` | yes |
| `refractal.execute` | runs a plan; backends | yes |
| `refractal.compare` | Parquet → verdict | no |

`refractal plan` runs on a laptop with no simulator, no GPU and no Docker. That
is enforced by a test, not by convention: it is the property that lets you
inspect a plan and its cost before committing to it.

Results are Parquet at an fsspec URI, so `./results` and `s3://bucket/results`
are the same code path. Every write is atomic, which is what makes resume safe.

## Install

```console
pip install refractal              # plan and build
pip install refractal[execute]     # + run
pip install refractal[compare]     # + compare
```

Comparison needs no numerical stack: exact McNemar is a binomial tail, the
bootstrap is resampling, and Cochran's Q is a permutation test.

## What is not here yet

- **The simulator adapter.** Nothing has touched MuJoCo. `refractal run
  --backend local` executes plans against a synthetic benchmark, which is what
  the schema and the statistics are tested against.
- **`--backend compose` and `--backend k8s`.**
- **Resource-shape probing.** `refractal build` carries forward hand-declared
  shapes and records that they were declared rather than measured.
- **Trajectory quality metrics.** The `steps.parquet` schema is declared and
  nothing writes it, deliberately: writing it against synthetic data would bake
  in guesses about what a real adapter can record.

Refractal sits on top of [`allenai/vla-evaluation-harness`][harness] and does
not fork it.

[harness]: https://github.com/allenai/vla-evaluation-harness

## Licence

Apache-2.0.
