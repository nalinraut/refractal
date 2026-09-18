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

## The discriminator the code under test destroys

Two more instances of the fixture-that-exercised-nothing shape turned up while
writing the vla-eval loop, and both are worth recording because the *mechanism*
was new. In neither case was the fixture missing the case. In both, the fixture
had the case and something downstream erased it.

**The truncated discriminator.** Resume in the vla-eval backend is
group-granular — the harness counts episodes from zero, so a partly-done group
re-runs in full. That makes part-file naming load-bearing: a session-stamped name
leaves the old file beside the new one and puts two rows under one `episode_id`,
which `compare` blocks on as a duplicate. So part names are deterministic per
`(worker, checkpoint, seed)` and a re-run replaces the file.

The test for it re-ran a completed group and asserted three rows, not six. Then,
following the practice of breaking the thing a check checks, the part name was
reverted to the session-stamped version it exists to reject —

```
--- with a session-stamped part name (the version I rejected) ---
Ran 2 tests in 0.103s
OK
```

The two sessions were named `session-one` and `session-two`. The rejected scheme
appends `session_id[:8]`. Both truncate to `session-`, so the part names collided
anyway and the second run overwrote the first exactly as the correct scheme does.
The fixture *had* two distinct sessions; the code under test truncated the
distinction away. Renamed to `aaaa-first` / `bbbb-second`, the break produces what
it should:

```
AssertionError: 6 != 3 : expected 3 rows, got 6:
  ['...-0-0-pi0', '...-0-0-pi0', '...-1-0-pi0', '...-1-0-pi0', ...]
```

**The mutation inside a per-item filter.** The same test file's helper selected
which servers to pass:

```python
{k: v for k, v in all_servers.items() if k in kwargs.pop("only", all_servers)}
```

`pop` is in the comprehension's `if`, so it runs once per server. The first
iteration consumed `only` and every later one fell back to the default — the
full map. The filter passed everything through, and the missing-server test ran
against a complete server map and reported that no error was raised. Which was
true, and not what it claimed to be testing.

The generalisation, which the earlier entry gets only half of:

> A fixture can carry the distinction and still exercise nothing, because
> something between the fixture and the assertion collapses it. Truncation,
> rounding, a `set` where a list was needed, a mutation inside a loop. So the
> check is not "does the fixture differ" but "does the assertion fail when the
> implementation is wrong" — which is only answerable by making it wrong.

Both were found by the same move, applied to the checks rather than to the code:
break it, and require the failure to be the one you predicted. The first break
returning `OK` is the finding.

## A check that reads a downstream symptom

The recorder receipt, one commit after it was written, on first contact with two
real model servers.

The reasoning was: if `_build_recorder` is never consulted, the run completes,
reports success and records nothing — indistinguishable from a correct run until
somebody reads the results. So the step buffer being empty is the symptom, and
the loop refuses on it.

What happened: every episode errored with `TimeoutError (act timeout=30.0s)`
during the server's `torch.compile` warm-up, before its first step. The buffer
was empty. The check reported that the recorder had never been consulted. It had
been consulted twice.

The error in the reasoning is one word:

> a silent recorder implies an empty buffer — **not the reverse**.

The check was written from the implication and used as though it were an
equivalence. Everything downstream of a failure is a symptom of that failure and
of every other failure that shares the symptom, so a check on a symptom fires on
a class, not on a cause. The receipt now counts recorder *constructions*, which
is the thing it was always trying to observe and which nothing else can produce.

The general form, which is not the same as the earlier entries — those were about
fixtures that failed to exercise a property, this one is about a check that
exercised the wrong property, correctly:

> When a check observes X to conclude Y, ask what else produces X. If the answer
> is "anything that fails earlier", the check reports on the wrong thing and will
> be loudest exactly when something else is already wrong.

Worth noting where it *did* work: the failure surfaced immediately, loudly, and
before anything was written. A check aimed at the wrong thing still beat no check
— it just accused the wrong component, which cost the time it took to read the
server log.

## A test that restates the implementation

The worst one in the project, found on the day the loop first ran against real
servers, and it had been green since the day it was written.

`scene_hash` for an externally-defined scene was computed in `build` as
`hash_obj(facts)` — the probe's facts and nothing else. The test:

```python
self.assertEqual(
    plan.scenes[0].scene_hash,
    hash_obj({"provider": "stand-in", "digest": self.Probe().digest}),
)
```

That is not a test. It is the implementation, written twice, and two copies of a
formula agree with each other whatever the formula leaves out.

**This is the canonical form, and it is canonical because it survives review.**
The other entries in this document describe tests that were weak — a fixture that
overlapped nothing, a discriminator truncated away, a `pop` in the wrong place.
Each of those is findable by reading carefully enough, and each was found that
way eventually.

`assertEqual(value, recompute_the_formula())` is not weak. It is *structurally
incapable* of the thing it appears to do, and no amount of care in reading
catches that — because the reading it invites is exactly the one that confirms
it. A reviewer checks the expected value against the implementation, sees that
they agree, and concludes the code is correct. They are right that the two agree.
Agreement was never in question. What is in question is whether the formula
covers what it needs to cover, and a copy of the formula cannot speak to that.

So it is not a mistake to be more careful about. It is a shape to refuse
outright.

What it left out was `external.params` — the benchmark's constructor arguments,
which decide what the policy observes. Measured against the real thing:
`quat_no_antipodal` moves pi0 on one LIBERO task from **0/8 to 2/4**, because
`LIBEROBenchmark` ships two quaternion-to-axis-angle conversions for the
proprioceptive state and chooses between them on that flag. Under the facts-only
digest, a run with the flag and a run without it produced **the same
`plan_id`** — so they would have joined into one comparison and been averaged.

The failure this project exists to make unrepresentable, sitting inside the
identity scheme, protected by a passing test.

It was found by changing the catalog and noticing `plan_id` did not move. Not by
the suite — and there was a second test, added an hour earlier, asserting exactly
this property on `external_scene_ref_key`. That test was correct and passed. It
tested the helper; `build` hashed something else. Which is the earlier lesson
arriving again from a new direction:

> assert the property on the thing that consumes it, not on the helper it is
> supposed to consume.

Both halves are now tested as properties — "a changed probe fact moves the hash",
"a changed param moves the hash" — with no formula restated anywhere. The general
rule:

> If an assertion recomputes what the code computes, it can only detect a change,
> never an omission. Ask instead what the value must *distinguish*, and vary those
> things.

The stub probe in the new test deliberately returns facts that do not depend on
`params`, because a stub that varied its facts with params would let the
facts-only digest pass too — the fixture failing to exercise the property, one
layer down, which is the trap the rest of this document is about.

## A statistical property pinned to one pseudo-random draw

Found while adding a field to `task_identity`. Three statistics tests failed:

```
AssertionError: 0.06565060092190661 not less than 0.05
AssertionError: -0.0026400560224089514 not greater than 0.01066666666666667
AssertionError: False is not true
```

Nothing was wrong with the statistics. `FakeBenchmark` seeds each episode's
outcome from its `episode_id`, which derives from `task_hash` — so adding
`provider_ref` to task identity reshuffled every synthetic outcome in the suite.
The tests were asserting properties of one particular pseudo-random dataset,
selected by a hardcoded salt.

Two different problems wearing one costume, and they need different fixes.

**A demonstration of a specific phenomenon** — "here is a draw where the
unclustered test declares a false finding", "here is a draw where McNemar and the
bootstrap disagree" — is *inherently* about a particular dataset. The fixture
must therefore **search for a qualifying draw and fail loudly if none exists**:

```python
for salt in candidate_salts:
    units = build(salt)
    if wrong_test_fires(units) and correct_test_stays_null(units):
        break
else:
    self.fail("no draw exhibits the anti-conservatism this class demonstrates: ...")
```

"No draw disagrees" is a real finding — it would mean the two tests had stopped
being distinguishable, which is the entire reason the gate chooses between them.
"0.0657 is not less than 0.05" is a maintenance chore wearing a finding's
clothes, and the difference between those two failure messages is the whole
value.

**A calibration claim** — "the measured excess tracks the modelled variance" —
is not about a draw at all. A single draw's excess is itself a random variable;
one reshuffle produced −0.0026 against a modelled 0.0107, which is one sample of
a noisy quantity rather than a calibration failure. Asserting it on a pinned draw
was a claim about that draw. Averaging over twelve fixes it, and the failure
message now prints the individual values so a real drift is distinguishable from
one unlucky sample.

The generalisation, which is not the same as the earlier entries — those are
about fixtures that fail to exercise a property. This one is about a fixture that
exercised the property *by coincidence*:

> If a test would break when an unrelated identity field changes, it is asserting
> on the draw rather than on the property. Ask whether the claim is "this dataset
> has property P" or "datasets like this have property P". The first must search
> and report when it cannot find; the second must average.

The tell is cheap to check: change something upstream that should be irrelevant
and see what breaks. Here the upstream change was adding a field to a hash, and
it found three.

## Comparing at the wrong aggregation level

The step-budget run was predicted before it: from the previous run's
steps-to-success distribution, pi0 ≈ 32% and pi0.5 ≈ 49% at a 100-step cap,
written down first so it could be wrong.

Two groups in, pi0 came back at 70% and 60%. Flagged as a possible miss against
the predicted 32%.

It was not a miss. Those were **task 0 only**, and the 32% was **pooled across ten
tasks**. Checked against the matching task, the prediction was 63.3% and the
result was 63.3% at n=30. Task 0 is one of the faster ones — median 78 steps — so
it loses least to a 100-step cap, and a pooled prediction says nothing about it.

Same family as the fixture that exercised nothing and the formula-restating test,
and a different tell. Those are about a test that cannot detect what it appears
to. This is about **a comparison between two things that are not comparable** —
neither number was wrong, and putting them side by side produced a finding out of
nothing.

The part that matters is the direction it could have gone. A fast task matching a
pooled prediction would have read as **confirmation**, and nobody checks a
confirmation. The error is symmetric and only one half of it is loud.

> State what a prediction is a prediction *of* — which population, at which
> aggregation level — before looking at any result. A number is not a prediction
> until it says what would count as comparing against it.

**The guard was not enough, and the demonstration is this entry's own document.**
One paragraph after writing "state what a prediction is a prediction *of*", the
run write-up contained the sentence *"At 100 steps every task should be a two-arm
observation."* That is a per-task claim derived from a pooled prediction, made
while the per-task numbers were sitting in the same dataset. They said tasks 4, 7
and 9 would be 0% for both arms. They were.

So the guard as stated protects the comparison you are *about* to make and not
the sentence you wrote two paragraphs earlier. The sharper version:

> The aggregation level is a property of every claim in the document, not of the
> moment you compare. If a number was computed by pooling, every sentence it
> licenses is about the pool.

The live consequence, recorded before the run finished rather than after: it
produces **ten per-task comparisons and one pooled one**, and only the pooled one
tests the prediction. Most of the ten will differ from 32% for reasons that have
nothing to do with whether the model was right — task difficulty and step
distribution vary, which is exactly what pooling averages over. Reading them
individually would be doing the same thing again, ten times, with a menu of
answers to pick from.

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

## Naming a limitation is not respecting it

`tests/test_bridge.py` installs a module shaped like the parts of the harness the
bridge touches, because the harness is not installed on most machines that run the
suite. The claim boundary was stated in the test class name from the start: green
means *the override is self-consistent against the gate as we understand it*, not
that the real gate is unchanged.

The blind spot was named at the time — "if a future harness moves the gate rather
than changing it, `harness_surface` fires and the stand-in keeps passing." It did
not take a future harness. **The gate had never been where the stand-in put it.**

The stand-in exposed an `_init_store` method. The harness has no such method; it
assigns the store inside `run()`:

```python
async def run(self):
    if not self.no_save:
        self._store = RecordingStore(db_path_for_eval(...))
```

`ParquetOrchestrator` overrode `_init_store`, the stand-in called it, every test
passed, and against the real harness the override would have been dead code —
producing the exact silent failure it was written to prevent. Two further things
fell out of reading the real source:

- **The gates are coupled.** `no_save=True` makes the recording config `None`,
  which shuts the *other* gate; `no_save=False` makes `run()` build a real SQLite
  store. No flag combination opens both, so `_store` has to become a property
  whose setter swallows both `run()`'s assignment and the `finally` block's
  `None`.
- **The null store cannot be a bare `object()`.** The harness calls
  `upsert_eval_metadata` before the loop and `close` in the `finally`. The
  stand-in never called either, so a sentinel that would have raised on the first
  benchmark passed every test.

### The part worth keeping

The mechanics above are specific to one harness. This is not:

**The limitation was written into the test class name, and writing it down
produced the feeling of having handled it.** The class is called
`TestOverrideLogicAgainstAReproducedGate`, its docstring says green means the
override is self-consistent *against the gate as we understand it* and not that
the real gate is unchanged, and it names the exact failure mode — a gate that
moved rather than changed. Then the reading that sentence was warning about did
not happen.

That is worse than not having named it. An unnamed assumption is an oversight
somebody might trip over. A named one carries a receipt, and the receipt is what
stops anybody looking again — including its author, who now remembers *having
thought about it* rather than what was concluded.

So the test for a named limitation is not "is it documented" but **what would
have to happen for this to be checked, and has that happened?** For a stand-in:
somebody has to read the dependency. Writing "this does not verify the real gate"
is not that reading, and is easily mistaken for it.

Secondary, and still general: a stand-in inherits every assumption its author
had. It cannot disagree with you, which is exactly what makes it cheap and
exactly what makes it blind.

Made mechanical where it can be: `SURFACE_DEPENDENCIES` in
`execute/harness.py` records what Refractal assumes about each surface file, and
`compare` prints those assumptions when a surface change blocks a comparison —
with the `_store` property flagged as the most intrusive one, to be checked
first. A hash says something moved; this says what is at risk.

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

**It shares its prerequisite with an open schema question**, which makes the two
of them one piece of work rather than two deferrals.
[planner-question-worker-unit.md](planner-question-worker-unit.md) asks what a
worker must own *whole* — as against how many episodes may share a process at
once, which is what `envs_per_process` answers — and the two questions are only
confusable while no engine can batch. An MJX scene separates them and exercises
this path in the same afternoon, so the concrete next thing is a vectorized scene
in a catalog, not either item on its own.

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
