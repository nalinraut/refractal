# Which test, and why — measured rather than argued

D6 chose McNemar plus a clustered bootstrap because the data is paired and five
seeds at one pose are one observation. Both conclusions survive. One of the
*reasons* does not, the condition under which clustering matters is narrower and
more specific than the design doc implies, and the power cost of McNemar's
dichotomisation is large enough to decide which test gates CI.

All experiments: 120 scenarios × 5 seeds × 2 checkpoints, 3% infra failure rate,
α = 0.05, 300 trials per cell unless noted.

- *scenario spread* — per-scenario difficulty **shared** by every checkpoint.
  Some poses are simply hard.
- *interaction spread* — per-(scenario, checkpoint) difficulty. A scenario hard
  for one checkpoint and easy for another, correlated across that scenario's
  seeds.

---

## 1. Type I error: what breaks the naive tests

Rejection rate at a **true difference of zero**. A calibrated test sits near 5%.

| scenario | interaction | McNemar | bootstrap | seeds-as-pairs | two-proportion z |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 3.7% | 8.0% | 5.3% | 7.3% |
| 0.45 | 0.00 | 5.0% | 8.3% | 6.0% | **2.7%** |
| 0.00 | 0.35 | 4.7% | 4.7% | **13.7%** | **14.0%** |
| 0.30 | 0.35 | 5.0% | 6.3% | **14.0%** | **13.7%** |

### A shared scenario effect makes the naive test *safer*, and here is why

Row 2 is the surprise: strong scenario correlation, no clustering correction,
and the z-test's error rate falls to 2.7%.

The mechanism is **correlation between the arms**, not anything about the
marginals. A scenario effect common to both checkpoints induces positive
correlation between them, and

    Var(rate_B − rate_A) = Var(rate_A) + Var(rate_B) − 2·Cov(rate_A, rate_B)

so that covariance *reduces* the variance of the difference. The unpaired
z-test computes its standard error as though the arms were independent —
dropping the `−2·Cov` term — so its interval is too wide and it under-rejects.

This is the textbook reason paired tests have more power, arriving here as the
reason the wrong test accidentally becomes conservative.

So the familiar warning — *five seeds are not five samples, you will call noise
significant* — is **not unconditionally true for a paired difference**. It holds
for a single checkpoint's rate and its interval, where nothing cancels.

### A checkpoint × scenario interaction is what actually breaks them

Rows 3 and 4: both naive tests roughly triple their false-positive rate to ~14%,
while both correct tests stay near 5%.

This is the case that matters, because it is what a regression *is*. A worse
checkpoint does not lower every scenario slightly; it destroys a handful and
leaves the rest alone. Those per-scenario differences carry variance an
episode-level test cannot see, having already assumed independence within a
scenario.

**The clustered bootstrap is not insurance against a textbook error. It is
load-bearing for exactly the effect the tool exists to detect.**

### D6's stated reason for rejecting the two-proportion test is wrong

D6 says it "throws the pairing away and makes you *less* likely to detect a real
difference." That is right only in row 2. In rows 3 and 4 it is *more* likely —
to detect differences that are not there.

The conclusion stands; the justification should be **it is uncalibrated in both
directions depending on a correlation structure nobody checks, and you do not
know in advance which regime you are in.**

---

## 2. Power: what the dichotomisation costs

Type I error alone cannot choose a test — a test that never rejects scores
perfectly on every row above. Rejection rate at a **true difference**, same grid:

| true Δ | interaction | McNemar (majority) | McNemar (all) | bootstrap | seeds-as-pairs | 2-prop z |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 4.3% | 1.3% | 6.0% | 5.0% | 4.3% |
| 0.00 | 0.35 | 4.0% | 3.7% | 4.3% | 11.0% | 9.0% |
| 0.05 | 0.00 | 22.7% | 10.0% | **43.3%** | 39.7% | 39.7% |
| 0.05 | 0.35 | 13.0% | 8.0% | **29.3%** | 42.3% | 41.3% |
| 0.07 | 0.00 | 39.0% | 22.0% | **73.3%** | 69.7% | 66.7% |
| 0.07 | 0.35 | 24.7% | 19.3% | **47.3%** | 61.7% | 60.0% |
| 0.10 | 0.00 | 71.3% | 44.0% | **92.0%** | 92.0% | 91.0% |
| 0.10 | 0.35 | 46.0% | 38.0% | **75.3%** | 85.7% | 85.0% |
| 0.15 | 0.35 | 79.7% | 75.7% | **97.3%** | 99.3% | 99.0% |

**The dichotomisation costs about half the power, at the same false-positive
rate.** At a real 7-point difference with an interaction present, McNemar detects
it 24.7% of the time and the bootstrap 47.3%. Collapsing 3/5 versus 2/5 to one
bit discards most of the signal in a uniform shift, and a uniform shift is what a
slightly worse checkpoint produces.

`all`-pass dichotomisation is worse again — 19.3% where majority gets 24.7% —
which is the argument for majority being the default.

The naive tests look more powerful still (61.7%), but they are buying it with an
11–14% false-positive rate. They are not better tests, they are further along the
same trade-off at an error rate nobody agreed to.

---

## 3. The gate: the bootstrap decides

"Report both" is a display decision. A regression gate needs one bit, and when
McNemar says no and the bootstrap says yes, something has to break the tie or the
CLI's exit code makes the choice by accident.

**The clustered bootstrap gates. McNemar is reported and never gates.** Same
false-positive rate, roughly double the power; gating on McNemar would miss a
real 7-point regression three times in four. A rate regression is a regression
whether or not any scenario flipped its majority.

**The failure mode of this choice, stated plainly.** The bootstrap can go red
when nothing flipped from failing to passing — a rate moved, no scenario broke —
and a user reading "REGRESSED" will picture scenarios that broke. So the 2×2 is
printed next to every verdict, and when the two disagree the report says so
explicitly:

> note: the rate moved but few scenarios flipped their majority (McNemar
> p=0.659). A uniform shift, not a set of scenarios breaking — see the 2×2
> before acting.

Exit codes: **0** no regression, **1** regression, **2** cannot be answered. The
third is not a regression and must not be reported as one — "the two runs used
different geometry" is a different sentence from "the policy got worse", and
conflating them teaches people to ignore the gate.

---

## 4. The bootstrap's 8%: not percentile bias, and BCa would not have helped

Two cheap checks before reaching for BCa. 250 trials at a true difference of
zero:

| scenarios | seeds | interaction | percentile | basic (reverse) | McNemar |
|---:|---:|---:|---:|---:|---:|
| 120 | 5 | 0.00 | 7.2% | 8.0% | 3.2% |
| 120 | 5 | 0.35 | 6.8% | 7.2% | 4.4% |
| 300 | 5 | 0.00 | 4.8% | 4.8% | 2.4% |
| 300 | 5 | 0.35 | 2.8% | 2.4% | 2.8% |
| 500 | 5 | 0.00 | 4.0% | 4.0% | 1.6% |
| 500 | 5 | 0.35 | 4.4% | 3.6% | 3.2% |
| 494 | 3 | 0.35 | 5.2% | 6.0% | 4.4% |

**The basic interval tracks the percentile one at every size.** Reflecting about
the observed statistic moves nothing, which rules out the bias percentile
intervals are known for — so BCa, which corrects that bias, would not have fixed
it.

**It shrinks toward nominal with scenario count:** ~7% at 120, 4.8% at 300, 4.0%
at 500. Small-cluster behaviour, as expected.

So the honest response is not more machinery but a stated threshold.
`MIN_UNITS_FOR_CALIBRATED_CI = 200`; below it every task verdict carries a note
that a marginal interval should be treated as marginal.

Also visible here: McNemar is conservative throughout (1.6–4.4%), a consequence
of the exact binomial's discreteness — and part of why its power is so much
lower.

---

## 4b. The circularity, and how a correction gets stated

Everything above was measured against `FakeBenchmark`'s `interaction_spread`.
That knob is a *model* of within-scenario correlation, it was authored to make the
fixture able to discriminate between the clustered and unclustered tests, and then
the tests were scored against it.

That does not invalidate the conclusions — the false-positive rates are real
counts of real rejections — but it does mean the numbers describe a model rather
than the world. The first real policy is the first independent source of the
effect, and pi0's flow matching is genuinely stochastic within a scenario.

So a correction to this document should be a **comparison**, not a replacement.
`refractal.compare.measure_variance` reports the structure in units comparable
with `modelled_variance`, which states what the knob implies:

| shared | interaction | ICC (A) | design effect | modelled Var(d) | measured excess | table above says |
|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.00 | 0.000 | 0.84 | 0.00000 | −0.01642 | naive calibrated (5.3%) |
| 0.45 | 0.00 | 0.424 | 1.10 | 0.00000 | 0.00687 | naive **conservative** (2.7%) |
| 0.00 | 0.35 | 0.164 | 1.83 | 0.08167 | 0.06896 | anti-conservative (13.7%) |
| 0.30 | 0.35 | 0.313 | 1.80 | 0.08167 | 0.05770 | anti-conservative (14.0%) |

**Both numbers are needed and the pair is the finding.** Row 2 has the highest
ICC and the lowest design effect: a shared scenario effect cancels in a paired
difference, so a high ICC alone does not imply clustering is load-bearing. Reading
ICC alone inverts the answer.

### The estimator manufactured the conclusion until it was calibrated

The first version divided the binomial term by `s`. Since
`E[p̂(1−p̂)] = (s−1)/s · p(1−p)`, that understates the sampling variance — and
understates it *most* when `p` is extreme, which is exactly what a strong shared
effect produces. Row 2 read **1.38** and tripped the "clustering matters" flag on
the one row where the unclustered test is conservative.

An estimator built to test a conclusion, producing that conclusion from a bias.
Caught by requiring it to reproduce the false-positive column — independent
evidence, since those were measured by counting rejections rather than by reading
a variance. Fixed by dividing by `s−1`.

Asserted in `tests/test_compare.py::TestVarianceEstimatorIsCalibrated`, including
the row-2 case specifically, so the bias cannot return quietly.

### Is a design effect below 1.0 a finding or an artefact?

Two of the five rows read below 1.0, which is worth settling before a real policy
reads below 1.0 and nobody knows how to take it. 200 trials per row:

| shared | interaction | mean DE | sd | % below 1.0 | true value |
|---:|---:|---:|---:|---:|---|
| 0.00 | 0.00 | 1.014 | 0.136 | 50% | 1.0 |
| 0.45 | 0.00 | 1.000 | 0.146 | 53% | 1.0 |
| 0.00 | 0.20 | 1.285 | 0.175 | 4% | >1 |

**Noise, not construction.** The estimator is centred on 1.0 with half the draws
either side.

The analytic reason, which is the part worth keeping: the two arms' *sampling*
noises are independent, because they are different episodes, so no covariance term
enters the sampling estimate. A shared scenario effect couples the true
probabilities `π_iA` and `π_iB`, and that coupling shrinks `Var(Δπ_i)` toward zero
— driving the design effect to exactly 1 **from above**, never below. So the true
value is ≥ 1 always, and a reading below 1 is estimation error.

**Correction, 2026-09-17: "below 1.0 is noise" is only true *near* 1.0.** Two arms
drawn from one source give a design effect of **0.000 with sd 0.000**, against
0.973 ± 0.117 for a true null — roughly eight sigma down. That is a signature, not
a low reading, and it is what a checkpoint compared against itself looks like. The
usual cause is two checkpoints pointed at one model server.

`compare` now blocks on it (`DEGENERATE_DESIGN_EFFECT = 0.25`, about five sigma
below 1 at this scale). It catches a *deterministic* policy behind one server,
where every per-scenario difference is exactly zero; a stochastic policy behind
one server draws fresh noise per episode and looks exactly like a true null, which
this cannot distinguish. The reliable guard is `check_server_assignment` in the
bridge, before the run — this is the second, independent one.

Practical consequences:

- A real policy reading a little below 1.0 is not evidence of anything. A reading
  meaningfully above 1.0 is. A reading *near zero* is a wiring fault.
- At 119 scenarios the sd is ~0.14, so the **1.25 threshold sits about 1.8σ above
  1.0** — roughly a 4% false-alarm rate for "clustering matters" when it does not.
  Close to the α it is used alongside, by coincidence rather than design, and
  stated so nobody reads the threshold as exact.

### The baseline is recorded unconditionally

`refractal compare` prints the design effect and per-checkpoint ICC for every
task, always, not behind a flag:

```
design effect on the paired difference: 0.99 (observed Var 0.08989 vs binomial 0.09094,
                                             24 scenarios, 4.8 seeds each)
  ckpt-46: within-cell ICC 0.082, between-scenario Var 0.01668
  ckpt-47: within-cell ICC 0.278, between-scenario Var 0.05920
  clustering is not load-bearing for the difference at this design effect ...
```

Behind a flag it would be skipped on exactly the runs where nothing seemed
notable — which are the runs that establish what normal is. **An agreement
recorded is what makes a later disagreement legible**, and there is no way to have
been interesting against a baseline that was never written down.

## 5. What the pairing rule costs, and how many checkpoints you can afford

The rule drops a **seed slot** — one `(unit, seed)` pair — whenever any
checkpoint failed on infrastructure there. So a slot survives only if all `k`
checkpoints produced a clean episode, and at infra rate `p` the expected loss is

    slot loss = 1 − (1 − p)^k        ≈ k·p  for small p

Linear in checkpoint count, not explosive:

| infra rate | k=2 | k=3 | k=4 | k=8 |
|---:|---:|---:|---:|---:|
| 1% | 2.0% | 3.0% | 3.9% | 7.7% |
| 3% | 5.9% | 8.7% | 11.5% | 21.6% |
| 5% | 9.8% | 14.3% | 18.5% | 33.7% |
| 10% | 19.0% | 27.1% | 34.4% | 57.0% |
| 20% | 36.0% | 48.8% | 59.0% | 83.2% |

Three or four checkpoints at a 3% infra rate is fine. The rule only bites when
the infra rate is itself bad, and then it bites hard — which is the right
incentive, since a 20% infra rate means the numbers were not worth trusting
anyway.

Removing one slot costs `k` episodes, one per checkpoint, and **the report gives
both numbers** because one of them alone is ambiguous:

    Seed slots dropped to keep pairs matched: 94 (7.1% of slots),
    removing 188 episodes across 2 checkpoints

That distinction was originally collapsed into a single number labelled "seeds
dropped", which read as slots and counted episodes. The two differ by exactly
`k`, so nothing about the value reveals which one it is — a reader checking the
arithmetic against `1 − (1−p)^k` gets an implied scenario count twice the truth.

## 6. A bug worth keeping in the record

`evaluate` blocks a comparison when one `scene_id` carries two different
`scene_hash` values — the geometry changed between runs, so the two sets of
results were never the same experiment. It set `blocking`, returned exit code 2,
and then **went on to compute per-task verdicts anyway**, so `Verdict.regressed`
read `True` on a comparison it had just refused to make. Anyone reading the
object rather than the exit code got a regression report derived from
incomparable data.

Which is the exact failure this project exists to prevent, in its own code,
found by its own distinction. It is the most convincing available argument that
the distinction is real: a tool built to stop people comparing things that
should not be compared did it to itself, and only the rule it had just written
down caught it.

Fixed by returning before any per-task work once `blocking` is non-empty.
`TestGate.test_changed_geometry_blocks_rather_than_reporting_a_regression`.

## 7. What the fixture needed before any of this was measurable

`FakeBenchmark` originally drew every episode independently. With no
scenario-level correlation there is nothing for a clustered analysis to correct,
so clustered and unclustered agreed to three decimals and the fixture could not
tell a correct test from a broken one.

Three knobs make it able to: `scenario_spread`, `interaction_spread`, and `salt`
(which redraws outcomes without touching identity, so an experiment can be
repeated enough times to estimate an error rate).

Row 1 of the first table is the original fixture. The table shows exactly what it
could see: nothing.
