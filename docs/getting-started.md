# Getting started

Five commands, about a minute, and you end on a real comparison verdict. No GPU,
no simulator and no checkpoints: the example runs against a built-in fake
benchmark that produces outcomes directly, so the machinery is real and only the
robot is not.

```console
$ pip install refractal[execute,compare]
```

## 1. Make a catalog

```console
$ mkdir demo && cd demo
$ refractal init .
  wrote 7 file(s) under .
  next:  refractal plan catalog --hardware laptop
```

You now have `catalog/` with five YAML files and a `README.md`. One scene, two
tasks on it, two scenario sets, two checkpoints called `baseline` and
`candidate`.

## 2. Compile it

```console
$ refractal plan catalog --hardware laptop -o plan.json
  cube-bowl-v1        30 scenarios     360 episodes    2 worker(s)  <=37 min
  ------------------------------------------------------------------
  360 episodes across 1 scene(s), 2 worker(s), at most 37 min
  that is an upper bound: every episode is costed at its full step limit,
  and episodes that succeed finish sooner.
  tier=full  seeds=[0, 1, 2]  checkpoints=['baseline', 'candidate']  mode=serial
  wrote plan.json  (plan_schema 1, plan_id sha256:82b73f180a7a...)
```

`plan` is the compiler. It expands two scenario sets into 30 distinct scenarios,
crosses them with tasks, seeds and checkpoints to get 360 episodes, assigns each
one to a worker, and writes the result. It ran on a laptop with no simulator
installed, which is the point: you can see what a run will cost before spending
anything.

**`plan_id` is the experiment's identity.** It is a hash of everything that
decides whether two runs are comparable. Change a step budget or a predicate and
it moves; change the output directory or the worker layout and it does not.

The estimate above is an upper bound from the catalog's declared figures. Once
you have actually run something, `--expect-from RESULTS` prints an expected
duration learned from it alongside the bound, which is the more useful number:

```console
$ refractal plan catalog --hardware laptop --expect-from ./results -o plan.json
  ...
  360 episodes across 1 scene(s), 2 worker(s), at most 37 min
  that is an upper bound: every episode is costed at its full step limit,
  and episodes that succeed finish sooner.
  expected 21 min, from 360 prior episode(s) of ['baseline', 'candidate']
```

Add `--expect-plan PLAN_ID` when the prior results are from a different
experiment than the one being planned. If nothing usable is found it says
`no usable prior episodes; expected duration not computed` and prints the bound
alone, rather than quietly falling back to it.

| | |
|---|---|
| `--hardware PROFILE` | which profile in `hardware.yaml` to plan against |
| `--expect-from DIR` | prior results to learn an expected duration from |
| `--expect-plan ID` | the `plan_id` of those results, if not this plan |
| `--pack-below-startup-sec N` | pack starved scenes sequentially when scene construction costs more than the parallelism saves; default 30 |

`refractal init .` takes `--force` to overwrite files that are already there.

## 3. Run it

```console
$ refractal run plan.json -o results --catalog catalog
  session 09b49614  360 episode(s) written, 0 already done
  next:  refractal compare results sha256:82b73f180a7a...
```

The default backend simulates. `--catalog` copies the catalog into the results
directory so the run can explain itself later.

Run it again:

```console
$ refractal run plan.json -o results --catalog catalog
  session a2b57450  0 episode(s) written, 360 already done
```

**Resume is by episode identity, not by counter.** It read the episode ids
already on disk and subtracted them. Interrupt a real run at any point and this
is what happens.

## 4. Compare

```console
$ refractal compare results $(python -c "import json;print(json.load(open('plan.json'))['plan_id'])")
```

```
  Scenarios in all 2 checkpoints: 60
  Statistics computed on the 60 scenarios in common.

  cube-bowl-v1 / cube-in-bowl   (30 scenarios)
    rates:  baseline: 65.6%   candidate: 62.2%    (baseline baseline)
                        candidate fails   candidate succeeds
      baseline fails                2             7
      baseline succeeds             7            14
    design effect on the paired difference: 1.53
      (observed Var 0.20958 vs binomial 0.13704, 30 scenarios, 3.0 seeds each)
      baseline: within-cell ICC 0.041, between-scenario Var 0.00626
      candidate: within-cell ICC 0.284, between-scenario Var 0.04994
      clustering is load-bearing here: an unclustered test would be
      anti-conservative by roughly this factor in variance.
    baseline -> candidate: -0.033 [-0.200, +0.122]  p=0.7463 holm=0.7463
      McNemar p=1.0000  -> no change
      note: 30 scenarios is below 200, where the percentile interval measures
      ~7% false positives against a nominal 5%.
```

Reading it:

- **`Scenarios in all 2 checkpoints: 60`**: the comparison is paired. Only
  scenarios both checkpoints attempted are used, and this line tells you how many
  survived. If it is far below what you planned, something did not run.
- **The 2×2** is scenarios, not episodes. `7` and `7` are the scenarios that
  flipped, and they are what McNemar tests. A rate that moved with few flips is a
  uniform shift rather than a set of scenarios breaking.
- **The design effect** is how much more variable the paired difference is than
  independent coin flips would be. Above ~1.25 and clustering matters: treating
  each episode as independent would understate the uncertainty.
- **The interval** is a clustered bootstrap. It resamples whole scenarios, not
  episodes, because five repeats of one scenario are not five independent facts.
- **`-> no change`** is the verdict. `compare` exits 0 for no regression, 1 for a
  regression, 2 when the question cannot be answered.

The exit code is the gate; the text is for the human deciding what to do about it.

## 5. Look at what was written

```
results/comparison_id=82b73f18…/
  plan.json                          the experiment, copied in
  catalog/                           the source, copied in
  render/<digest>.json               what was supplied at RUN time
  harness/<surface>.json             digests of the code that ran
  checkpoint=baseline/scene=cube-bowl-v1/episodes/part-….parquet
  checkpoint=candidate/scene=cube-bowl-v1/episodes/part-….parquet
```

Partitioned by comparison, checkpoint and scene, so several workers write
concurrently without coordinating and a later run can list what already exists.
One row per episode, with the identity it was run under and the provenance of the
process that ran it.

The four provenance entries answer different questions, and the split is the
point:

| | |
|---|---|
| `plan.json` | what the experiment **is** — the identity `comparison_id` digests |
| `catalog/` | where it came from, so the run can explain itself later |
| `render/` | server addresses, backend, results URI — what was supplied when it ran |
| `harness/` | per-file digests of the harness code that actually loaded |

`render/` is recorded and deliberately **not** hashed. A server address is a
property of a machine, not of an experiment; folding it into `plan_id` would
make the same experiment on two machines two experiments — the same reason
`results_uri` is excluded, since otherwise writing to `./results` and to `s3://`
produces ids that never join.

It is keyed by a digest of those settings and appended rather than overwritten,
so a comparison resumed on a second machine keeps both. **More than one entry
means the runs did not agree about how they were deployed**, which is what
somebody needs to see before explaining a difference in the numbers.

## What to change next

- **Point it at a real simulator.** You write an adapter (see
  [the adapter contract](adapter-contract.md) and
  [the worked example](adapter-example.md)), and Refractal drives it.
- **Change the catalog.** [Writing a catalog](writing-a-catalog.md) covers what
  goes in each file and which edits invalidate existing results.
- **Run the checkpoints against real model servers** with `--backend vla-eval`, or one
  container per worker with `--backend compose`.

## What the example is not

The fake benchmark produces outcomes from a hash of the episode id. They are
deterministic, they have realistic within-scenario correlation, and they are not
a robot. Every number above is real arithmetic over synthetic outcomes. The
statistics are exercised, the conclusion is about nothing.
