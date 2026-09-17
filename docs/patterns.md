# Patterns

Things that went wrong more than twice, and what they generalise to. Three is
where a coincidence stops being one, so each of these is written down at the point
it became a third instance.

Five so far: invariants that were correct but unexercised, containers that did not
match their question, assertions about code we do not control, rules applied past
the edge of their reason, and digests recorded without their inputs.

---

## Invariants that were correct but unexercised

Three bugs in this project had the same shape.

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

## Post-conditions, for the same reason

The practice above asks whether a fixture puts an invariant in position. The
complement is to make the check run on every real execution, so it does not
depend on anyone being in a position to ask.

`ResultWriter.verify_written` is the first of these: at the end of every worker,
**episodes were planned, therefore rows must exist**. It reads the episode ids
back out of the part files rather than trusting the in-memory count, because a
count is produced by the same code path that did the writing — the artifact is
the evidence, which is the week's other recurring lesson.

It exists for a specific failure and is written against the symptom rather than
the cause. The specific failure: the harness's `_build_recorder` returns
`NullEpisodeRecorder` whenever `self._store is None`, so a bridge that overrides
the recorder and not the store runs every episode, succeeds on every episode, and
records nothing. A run that completes, reports success and writes nothing is
indistinguishable from a correct one until someone tries to compare.

Written against the symptom, it also catches the variants nobody has thought of:
a writer pointed at the wrong prefix, a filesystem that accepts a write and drops
it, a backend that forgets its last flush. `tests/test_execute.py::TestAWorkerThatWritesNothingIsCaught`
breaks the writer in two ways that have nothing to do with `_store`.

Partial loss is treated as an error rather than a warning, because a partial
result silently shrinks a denominator — which is the same failure as every entry
in the table above.

**The general form:** a check that only runs when someone suspects a problem is a
check that is not there. Print the design effect on every comparison; assert
non-empty output on every worker; put the fixture in the configuration rather
than remembering to test it.

## A recurring shape: the container not matching the question

Duplication has now slipped past a correctness check four times, and every time
the invariant was about *identity* while the container discarded *multiplicity*.

| where | the container | the question it could not answer |
|---|---|---|
| `resolve` scenario expansion | dedup scoped to one scenario set | do two sets produce the same scenario? |
| `execute` output check | `set` of written episode ids | did any row arrive twice? |
| `compare` pairing | `dict` keyed by seed | did two rows claim the same episode? |
| `schema` scene hashing | list of file paths | *(the inverse — see below)* |

The first three collapsed a duplicate silently. In `compare` the collapse was the
worst of them: two rows for one episode with *opposite* outcomes, and the second
simply overwrote the first.

**But the rule is not "always preserve duplicates."** The fourth is the same shape
inverted. `hash_file_set` preserved multiplicity over a question that is
set-valued — *which files define this scene* — so a scene whose `assets` glob
happened to match its own `model` hashed differently from one whose glob did not.
Same geometry, two identities, decided by how a glob was written. There
deduplication was the fix.

So the rule is: **match the container to the question.** A set answers "did
everything arrive"; it cannot answer "did anything arrive twice". A list answers
"how many"; it cannot answer "which distinct". Writing down which question is
being asked usually makes the right container obvious, and it is the step that
was skipped all four times.

Worth a scan whenever one of these turns up, because two instances of a shape are
rarely two. The scan that found the last two took one grep over id-keyed
assignments and `set()` construction.

## Assertions about code you do not control

A guard fails when it is wrong. An **assertion** — a claim written into a design
document about someone else's code — just sits there being wrong.

`refractal-design.md` said the harness hardcodes the model server address to
localhost. It appeared in the design doc, the implementation prompt, the API
reference and the Kubernetes discussion, and it decided that multi-host was
blocked upstream. It was wrong:

```python
url: str = "ws://localhost:8000"     # a default
url=data.get("url", cls.url)         # overridable from YAML
```

How it survived a review that caught a dozen subtler things: it came from a grep
whose hits were **a docstring example and `--network host` help text**, and
nothing ever tested it because nothing needed multi-host. Same shape as the
unexercised invariants above — correct-looking, never exercised — except it was an
assertion rather than a guard, so there was no test that *could* have failed.

The practice that follows is the same question in a different place:

> For each claim about an external dependency, what would falsify it, and is
> anything positioned to notice?

Three answers, and picking the right one is the work:

- **The digest notices.** True for the hooks in `SURFACE` — a change to the
  `_store` gate moves `harness_surface` and blocks comparisons.
- **Nothing notices, and that is the right trade.** `db_path` having no caller
  lives in `model_servers/`, which changes on every commit; gating on it would
  produce an alarm nobody reads. Recorded as a documented assumption with the
  one-line re-check instead.
- **Nothing notices, and that is a gap.** Which is what the localhost claim was
  for nine days. Re-checked by hand when the pin moves, and the results written
  down with a date.

The cheap version of all three: **re-verify on every pin move, and record the
date**. A claim with a date attached degrades visibly; one without looks equally
true forever.

## A rule is a compressed reason

Three times now a rule has been correctly *declined* rather than followed, and
each time the decline produced something better than compliance would have.

| the rule | what it compressed | why it did not apply |
|---|---|---|
| quantise numbers before hashing | *arithmetic accumulates error* | LIBERO's init states are **loaded**, not computed. Nothing accumulated. Rounding would have discarded precision LIBERO chose. |
| preserve multiplicity in identity containers | *identity questions do not care how many times a thing was listed* | `hash_file_set` asks *which files define this scene*, which is set-valued. A glob overlapping its own `model` hashed differently from one that did not. |
| adapters do not import Refractal | *adapters are loaded by the harness, not by us* | A probe is loaded by `refractal build` — a plugin pointing the other way, which the rule never covered. |

**A rule is a compressed reason, and compression is lossy at the boundary.** All
three compressions were correct; each one dropped the condition that made it true,
and the condition is exactly what you need at the edge.

The recovery move is one question: **what was this rule for?** Not *does it
apply* — that invites a verdict — but *what was it protecting*, which invites the
thing that actually happens next.

Because checking the reason does more than adjudicate. It tells you what the rule
was protecting, and sometimes there is a cleaner way to protect it:

- Quantisation was protecting identity against drift. Loaded values have no drift,
  so the protection was already there.
- Deduplication was protecting against a question the container could not answer.
  Naming the question picked the container.
- The import rule was protecting against Refractal-specific knowledge leaking into
  a benchmark package. **Not hashing in the probe at all** protects that better
  than importing `hash_obj` — one implementation of the canonicalisation instead
  of two — and it is not reachable by deciding the rule does not cover probes.

That last one is the argument for the question over the verdict. Deciding "the
rule does not apply here" ends the thought. Asking what it was for continued it.

## Record what a digest was computed from

A digest answers *did something change*. It cannot answer *what changed*, and the
second question is the one somebody actually has. Three instances:

| digest | recorded beside it | the question it makes answerable |
|---|---|---|
| `harness_surface` | a per-file manifest, `harness/<surface>.json` | which harness file moved — `runners/live_runner.py`, as it turned out |
| an external `scene_hash` | the facts the probe reported, in `build.lock` | which fact about the scene moved: init states, BDDL, a version |
| `plan_id` | the experiment identity document, in `plan.json` | which field makes two runs different experiments |

The third was added before it bit, because the first two predicted it. `refractal
explain` now answers it directly:

```console
$ refractal explain before.json after.json
  run.checkpoints[1].server_args.max_batch_size: 8 -> 32
  run.seeds: 3 -> 5

  Results under these two ids are separate comparisons and will not join.
```

Note what the first line is. `max_batch_size` is nested two levels inside
`server_args`, and it is the class of parameter their own paper measures at 55
percentage points. A digest comparison would have said "these differ"; this says
which knob.

**The cost is small and the alternative is a bisect.** A manifest is a few hundred
bytes per session; an identity document is smaller than the scenario list already
in the plan. What they replace is somebody holding two comparison directories and
a question, months after both machines are gone.

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
