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

Recorded rather than fixed, per the practice above:

- **Scene affinity and bin-packing across a long tail of small scenes.** The
  planner packs starved scenes sequentially and the threshold is
  `startup_sec`-driven, but no fixture has enough scenes for the packing path to
  matter. Deliberate: it needs an MJX scene to be worth exercising, and LIBERO is
  classic MuJoCo.
- **`lock_schema` version refusal.** The field exists and the reader refuses an
  unrecognised value, but nothing has ever bumped it. First real use will be the
  LIBERO bridge, which lives in a separate repo and so has to agree with this one
  across a version boundary.
- **`--backend compose` and `k8s`.** No backend but `local` exists, so every
  placement decision in `plan.json` — `device`, `cpuset`, `packed_scenes` — is
  written and never read.
- **Multi-seed clustering against genuinely stochastic outcomes.** The clustered
  bootstrap is exercised against `FakeBenchmark`'s injected
  `interaction_spread`, which is a model of the effect rather than the effect.
