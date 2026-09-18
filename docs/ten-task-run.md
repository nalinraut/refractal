# The ten-task run: 600 episodes, and what it settles

`sha256:4605f47f…`, 600 episodes, 60 harness invocations, 0 infra failures,
`compare` exit 0, ~47 minutes wall clock. Two warm servers, `serial` execution
(recorded as `interleaved` — see `execution-mode.md`).

| task | pi0 | pi0.5 | design effect | pi0 ICC | pi0.5 ICC | verdict |
|---|---|---|---|---|---|---|
| 0 | 73.3% | 100% | **4.30** | 0.622 | 0.000 | no change |
| 1 | 73.3% | 100% | 1.04 | 0.018 | 0.000 | improved |
| 2 | 93.3% | 100% | **4.00** | 0.600 | 0.000 | no change |
| 3 | 70.0% | 100% | 1.98 | 0.328 | 0.000 | improved |
| 4 | 56.7% | **86.7%** | 2.35 | 0.458 | **0.239** | no change |
| 5 | 80.0% | **96.7%** | 1.89 | 0.280 | 0.000 | no change |
| 6 | 76.7% | 100% | 0.76 | 0.000 | 0.000 | improved |
| 7 | 83.3% | 100% | 1.25 | 0.111 | 0.000 | no change |
| 8 | 60.0% | 100% | **0.44** | 0.000 | 0.000 | improved |
| 9 | 50.0% | 100% | 0.56 | 0.000 | 0.000 | improved |

Design effect: min 0.44, median 1.89, max 4.30. Five of ten above the 1.25
threshold, three below 1.0.

## 1. Drift does not exist at this timescale

`interleaved` exists to remove within-session drift. It does not survive the
check.

The test is a pooled rank correlation between replicate position and cell rate,
with a permutation p-value that shuffles replicate labels **within each task** —
so task difficulty is held fixed by construction and only position can
contribute. Thirty cells per arm, ten tasks, three positions.

| arm | rep0 | rep1 | rep2 | ρ | permutation p |
|---|---|---|---|---|---|
| pi0 | 71.0% | 73.0% | 71.0% | +0.027 | 0.839 |
| pi0.5 | 97.0% | 99.0% | 99.0% | +0.132 | 0.230 |
| contrast | | | | +0.046 | 0.723 |
| both-interior only (n=6) | | | | +0.194 | 0.665 |

Nothing. The spread across replicate positions is 2 points for both arms.

**The two-task run said otherwise, and it was wrong.** It showed pi0 going
85% → 75% → 65%, per-task 80/60/50 and 90/90/80, both non-increasing. That is
what a small sample does: the correct test on those six cells already gave
ρ = −0.554 at p = 0.108, which is not significant, and thirty cells put it at
+0.027. "Each task's sequence is monotone" is the wrong question, because
non-increasing across three points happens by accident often enough that two
tasks agreeing is two observations rather than six.

**A direction error worth recording.** Before the data arrived I said the serial
loop's ordering "would flatter pi0.5". That is backwards. Checkpoints are sorted,
so pi0 runs entirely first and gets the *fresher* machine; monotone degradation
would disadvantage the later arm and make pi0.5's measured advantage an
underestimate. There is also a structural argument against drift being the
explanation at all, available before any statistics: the decline sat inside pi0's
own block while pi0.5 ran after all of it and scored higher. Session-wide drift
predicts the later arm is worst. It was not.

### So: two modes, not three

`interleaved`'s justification was drift removal, and there is no drift to remove
at this length of run. Building it would pay per-invocation overhead — sixty
orchestrator setups instead of two, each with a connection handshake, render-mode
setup and spec cross-validation — for nothing, and it would become a default
somebody picks for a reason that does not hold.

**Recommendation: `serial` and `concurrent`.** `serial` is a rename of what the
loop already does. `concurrent` is the one nothing currently uses and the one the
two warm servers already make possible, and it needs a column the others do not —
which checkpoints were running alongside, per episode — because a success rate
measured under contention is fine and a duration measured under it is not.

This conclusion is scoped: no drift **in a 47-minute run on this machine**. A
longer run, a thermally tighter box, or a shared host could all differ. The check
is a script, so re-running it is cheap, and it should be re-run before anyone
concludes drift is absent somewhere else.

## 2. The ceiling is the headline, not a caveat

**pi0.5 is pinned at 100% on 8 of 10 tasks.** Only tasks 4 and 5 have both arms
interior.

That is the finding rather than a limitation of it: LIBERO-Spatial is too easy for
pi0.5, so eight of these ten contrasts can only ever detect pi0 being worse. More
scenarios per task would not help — the ceiling is not a sample-size problem. The
comparison wants a harder suite: LIBERO-Long, or LIBERO-Object at a shorter step
budget.

It also means the eight pinned tasks contribute a design effect that is not
really about a paired difference. With one arm constant, `d = 1 − pi0`, so the
number measures pi0's homogeneity with the subtraction contributing nothing. Two
of these ten are genuine two-arm observations.

## 3. What the run does establish about the statistics

Across ten tasks the estimator produced **0.44 to 4.30** and gave the correct
reading at both ends.

* **4.30 on task 0**, with pi0's within-cell ICC at 0.622 and between-scenario
  variance 0.110. Strong clustering, correctly flagged as load-bearing.
* **0.44 on task 8**, with both ICCs at 0.000. Below 1, so the naive test is
  conservative, and `compare` says clustering is not load-bearing rather than
  warning.

The two extremes come from one run, one estimator and one mechanism. A single
design effect above 1 would have been consistent with the estimator being biased
upward — the failure already found once in this project, the `s` versus `s−1`
one. A spread from 0.44 to 4.30 that tracks the measured ICC is much harder to
explain that way.

**Holm is doing real work here.** Ten contrasts, and the note says the
uncorrected chance of at least one false positive is about 40%. Four contrasts
are significant uncorrected and not after adjustment — task 0 at p=0.0116 →
holm=0.0580, task 4 at p=0.0260 → holm=0.0780. Those are the ones that would have
been reported as wins by a per-task test.

## LIBERO cannot discriminate these two checkpoints

The cheap check before committing compute to a harder suite: pi0.5 alone, three
tasks from each of LIBERO-Object and LIBERO-Long (`libero_10`), five episodes
each, against the already-warm server.

| suite | max_steps | pi0.5 |
|---|---|---|
| `libero_object` | 280 | **15/15 (100%)** |
| `libero_10` (Long) | 520 | **15/15 (100%)** |

Thirty episodes, thirty successes, including the three-stage
`turn on the stove and put the moka pot on it`.

So the ceiling is not a LIBERO-Spatial problem to be fixed by changing suite. **Of
the three LIBERO suites sampled, none can discriminate pi0 from pi0.5**, because
pi0.5 saturates all of them. Running LIBERO-Long instead would have cost an hour
and produced the same eight-of-ten shape.

Stated plainly because this is the kind of benchmark people keep running anyway:
a suite on which the candidate scores 100% cannot answer "did my change help".
It can only answer "did my change break something", and only for the arm that is
not at the ceiling.

Scoped honestly: three tasks per suite, five episodes each, first three task ids,
one checkpoint. That is enough to rule the suites *out* — 30/30 leaves no room
for a contrast — and not enough to characterise them.

### What would discriminate

Not more scenarios, and not another LIBERO suite. The options are different in
kind:

* **A harder benchmark.** Something where pi0.5 is genuinely interior.
* **A tighter step budget.** Both arms succeed given 220 steps; at 60 they may
  not. That measures efficiency rather than capability, and it is a real question
  with an existing knob — `max_steps` is already in `task_hash`, so two budgets
  are already two experiments and cannot silently join.
* **Closer checkpoints.** pi0 against pi0.5 is a generational gap. Two adjacent
  checkpoints of one training run are the case the paired machinery is actually
  for, and the case where a design effect of 4.30 versus 0.44 changes what you
  conclude.

The third is the one this project was built for: "did my change help" is a
question about two nearby things, and a 27-point gap does not need a clustered
bootstrap to detect.

## The step-budget run: a prediction made before it

The finding above — that no LIBERO suite discriminates these two checkpoints —
has an answer that costs a catalog edit rather than a new benchmark. `max_steps`
is already in `task_hash`, so two budgets are already two experiments and cannot
silently join.

**The budget was chosen from the 600-episode run's own steps-to-success
distribution**, not guessed:

```
  pi0   215/300 succeeded; steps p10=80 p25=90 p50=103 p75=117 p90=127 max=213
  pi05  295/300 succeeded; steps p10=80 p25=89 p50=101 p75=117 p90=124 max=165
```

Counting a win only if it finished within a tighter budget:

| budget | pi0 | pi0.5 |
|---|---|---|
| 220 (the harness default) | 71.7% | **98.3%** — ceiling |
| 120 | 57.7% | 82.3% |
| **100** | **32.0%** | **48.7%** |
| 80 | 7.3% | 10.7% |

100 puts both arms nearest 50%, where per-scenario variance is greatest and the
design effect has the most to measure.

### The prediction

At `max_steps: 100`, **pi0 ≈ 32% and pi0.5 ≈ 49%**, a gap of about 17 points with
both arms interior.

This is a real prediction and it can be wrong, because it assumes a truncated
rerun reproduces the step distribution of the untruncated one. The policies are
not budget-aware — nothing about a 100-step cap changes what the policy does at
step 40 — but they *are* stochastic, so the rerun draws new trajectories rather
than replaying the old ones. The prediction is therefore about the **rate**, not
about which episodes succeed.

If the measured rates come back near these, the step distribution is stable and
the budget knob is a reliable way to move a saturated comparison into range. If
they come back far off, that is more interesting: it would mean truncation
interacts with the policy in some way not visible in the step histogram.

### What the run is for

The thing the last run could not provide: **a design effect where both arms
contribute variance.** At the default budget, eight of ten tasks had pi0.5 pinned
at 100%, so `d = 1 − pi0` by arithmetic and the design effect measured one arm's
homogeneity. Two genuine two-arm observations out of ten. At 100 steps every task
should be a two-arm observation.

It also changes the question from capability to **efficiency** — both policies can
do these tasks, and the question becomes which does them in fewer steps. That is a
different question and a real one, and it is the one this benchmark can still
answer about these checkpoints.
