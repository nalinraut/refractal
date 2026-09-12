# Unexercised invariants

Three bugs in this project had the same shape. Three is where a coincidence stops
being one, so it is written down.

In each case the invariant was **already correct and already tested**. What was
missing was any fixture in a configuration that could make it fire. The test
passed because the situation it guards against had never been constructed.

## The three

| invariant | already tested by | what was missing |
|---|---|---|
| A recorded `scene_hash` must match the sources it was recorded from | nothing — filters had two freshness checks, scenes had none | a lock that *had* to win, which only appeared when an externally-defined scene made recomputation impossible |
| Every episode is assigned exactly once | `test_every_episode_assigned_exactly_once` | a catalog with overlapping scenario sets. Deduplication was per set, so two sets producing one scenario gave 24 episodes with 12 distinct ids |
| The distribution contains all five subpackages | the whole suite, which imported them fine | a *wheel*. `.gitignore`'s unanchored `build/` hid `src/refractal/build/` from git and from hatchling, while `git status` stayed clean |

The failure mode is identical every time: **correct-looking output, no error, wrong
denominator.** A doubled episode count reads as a suspiciously even sample size
nobody questions. A stale scene hash reads as a comparison. A missing subpackage
reads as a successful build.

## The standing practice

For each invariant, ask a question separate from "is it correct":

> **What configuration would make this fire, and is any fixture in that
> configuration?**

Where none is, either put one there or record that the invariant is currently
unexercised. Recording it is a real answer — an unexercised invariant you know
about is different in kind from one you do not.

It is not more tests. All three invariants already had tests. It is a question
about the *inputs*, asked once per invariant.

## What put each of them in position

Worth noting, because it was the same thing three times: **looking at a real
artifact rather than at the test suite.**

- A lock that had to be authoritative, because an external scene has no local
  sources.
- A catalog with two scenario sets on one scene.
- A wheel, rather than the repo it was built from.

Which is why `examples/catalog` now contains a deliberately overlapping pair of
scenario sets, and why CI's `seconds-test` job installs the built wheel instead of
an editable checkout. Two of these were found by the artifact, not the assertion.

## The fixture that exercised nothing

The best evidence for the practice is that applying it failed on the first
attempt, in exactly the way it exists to catch.

Adding an overlapping scenario set to `examples/catalog` meant putting a second
set on `cube-bowl-v1` alongside the existing `cube-grid`. The first version was:

```yaml
  - id: cube-grid-extra
    scene: cube-bowl-v1
    generator: refractal.generators:linspace_grid
    params:
      cube_x:   {range: [0.05, 0.20], steps: 2}
      cube_y:   {range: [-0.10, 0.10], steps: 2}
      bowl_yaw: {value: 0.0}
```

All 199 tests passed. It added four scenarios and **overlapped nothing** —
`cube-grid` uses `latin_hypercube`, whose continuous draws never land on a grid
point. Zero dedup warnings. The invariant remained exactly as unexercised as
before, and the suite reported success.

That is the same failure one level up: a thing that looks like it covers the case,
produces no error, and covers nothing. Which is the argument for the practice
being a *check* rather than a box ticked by having written a fixture —

> having added a fixture is not evidence that the invariant is exercised. The
> evidence is the invariant firing.

The version that works uses two explicit `choices` grids that share
`(0.05, -0.10)` and `(0.05, +0.10)`, and is verified by the warnings appearing:

```
scene 'cube-bowl-v1': scenario set 'cube-edges' produces a scenario already
produced by 'cube-corners' ({'cube_x': 0.05, 'cube_y': -0.1, ...}); kept once.
```

## Enforced where it can be

`tests/test_resolve.py::TestTheDefaultFixtureExercisesItsInvariants` asserts the
*fixture*, not the code. If someone tidies the overlapping scenario sets out of
`examples/catalog`, or removes the second task on `vial-rack-v1`, the
corresponding invariant silently returns to being unexercised — so those tests
fail with a message saying so.

`docs/releasing.md` carries the artifact-versus-repo version of the same rule:
`dist/` is deleted and rebuilt rather than reused, because a stale wheel has a
byte-identical filename and nothing in `twine check`, `git status` or the test
suite looks at it.

## Currently known to be unexercised

"Unexercised" is not one condition, and the list is only useful if it says which
kind each one is. Four entries, three kinds, and the last is the interesting one.

### Blocked on a deferred dependency — nothing to do

**Scene affinity and sequential packing.** The planner packs starved scenes
sequentially with a `startup_sec`-driven threshold, and no fixture has enough
scenes for the path to matter. It needs an MJX scene to be worth exercising at
all, and LIBERO is classic MuJoCo. Deliberate, and revisiting it before there is
a vectorized scene would be inventing a fixture to satisfy a checklist.

### Blocked on scheduled work

**Every placement field in `plan.json`.** `device`, `cpuset` and `packed_scenes`
are written and never read, because `--backend local` is the only backend. They
get exercised by `--backend compose`, which is step 5. No action beyond doing
step 5.

### A gap, not a deferral — so it was closed

**`lock_schema` refusal.** ~~Nothing has ever bumped it.~~ Blocked on nothing: the
refusal was written and correct, and a fixture carrying a future version number
cost two lines and needed nothing that did not already exist. Now covered by
`tests/test_build.py::TestLockSchemaRefusal`, in both directions and including a
missing field, plus a check that `resolve` surfaces the refusal rather than
degrading into "no lock, plan anyway".

That one mattered more than its size: the LIBERO bridge lives in a separate repo,
so two distributions have to agree on `build.lock` across a version boundary. The
first time the gate fires for real should not be the first time it fires at all.

### Real evidence at stake

**Multi-seed clustering against genuinely stochastic outcomes.** Everything
`compare` claims about clustering rests on `FakeBenchmark`'s `interaction_spread`,
which is a *model* of within-scenario correlation rather than the thing itself.
The false-positive and power tables in
[statistics-measurements.md](statistics-measurements.md) are measured against
that model.

pi0's flow matching is the first thing available that produces genuine
within-scenario stochasticity. **If the measured clustering behaves differently
from the modelled one, that is a finding about the statistics doc rather than
about pi0** — and it is the only item on this list that could contradict
something already believed. The other three would confirm what is expected.

Which is the real reason the LIBERO bridge is next, rather than because it is the
next line in a build order.
