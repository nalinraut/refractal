# Reading a comparison

What `refractal compare` prints, what each number means, and when not to trust it.

```console
$ refractal compare results <plan_id>
```

Exit codes: **0** no regression, **1** regression detected, **2** the question
cannot be answered. The exit code is the gate; the text is for deciding what to
do about it.

## The header

```
  Scenarios in all 2 checkpoints: 60
  Statistics computed on the 60 scenarios in common.
```

The comparison is paired. Only scenarios both checkpoints attempted are used.

If this number is far below what you planned, episodes are missing: a worker
died, or a resume never finished. Compare it against `total_episodes` in
`plan.json` before reading anything else.

## Per task

```
  bench-v1 / cup-on-shelf   (30 scenarios)
    rates:  baseline: 65.6%   candidate: 62.2%    (baseline baseline)
                        candidate fails   candidate succeeds
      baseline fails                2             7
      baseline succeeds             7            14
```

**The 2×2 is scenarios, not episodes.** Each scenario is reduced to one outcome
per checkpoint by majority vote across its seeds, then cross-tabulated.

The off-diagonal cells are what moved. `7` and `7` here means fourteen scenarios
flipped, seven each way: a rate that barely moved and a lot of churn underneath.

A rate that moves while the off-diagonals stay small is a uniform shift: the same
scenarios succeed, slightly more or less often. A rate that holds while the
off-diagonals are large means one set of scenarios broke and another started
working. Those call for different actions, and only the 2×2 distinguishes them.

## The design effect

```
    design effect on the paired difference: 1.53
      (observed Var 0.20958 vs binomial 0.13704, 30 scenarios, 3.0 seeds each)
      baseline: within-cell ICC 0.041, between-scenario Var 0.00626
      candidate: within-cell ICC 0.284, between-scenario Var 0.04994
      clustering is load-bearing here: an unclustered test would be
      anti-conservative by roughly this factor in variance.
```

**How much more variable the paired difference is than independent coin flips
would be.**

| value | meaning |
|---|---|
| **> 1.25** | Clustering matters. Treating each episode as independent understates the uncertainty by roughly this factor. |
| **≈ 1.0** | Episodes behave independently. |
| **< 1.0** | The paired difference is *less* variable than independent flips. A test that ignored clustering would be conservative, not wrong. |

**ICC** is how much repeats of one scenario agree with each other. High ICC and a
low design effect happen together and are not a contradiction: an effect shared
by both checkpoints cancels in a paired difference. Reading ICC alone inverts the
answer, which is why both are printed.

## The verdict line

```
    baseline -> candidate: -0.033 [-0.200, +0.122]  p=0.7463 holm=0.7463
      McNemar p=1.0000  -> no change
```

| | |
|---|---|
| `-0.033` | the difference in success rate |
| `[-0.200, +0.122]` | 95% interval from a clustered bootstrap; it resamples whole scenarios, not episodes |
| `p=` | from the bootstrap |
| `holm=` | corrected across every contrast in this run |
| `McNemar p=` | whether the *set* of scenarios solved changed |
| `-> no change` | the gate |

**The gate is the bootstrap**, not McNemar. They answer different questions and
disagree often: McNemar asks whether different scenarios are being solved, the
bootstrap asks whether the rate moved. A change in rate with no change in which
scenarios succeed is real and McNemar will not see it.

`holm` is the number to act on. With ten contrasts, an uncorrected 5% threshold
gives about a 40% chance of at least one false positive.

## Notes that appear underneath

Each fires on a condition and each means something specific.

**`N scenarios is below 200`**: the bootstrap interval is optimistic at small
scenario counts, measuring roughly 7% false positives against a nominal 5%.
Treat a marginal result as marginal.

**`the rate moved but few scenarios flipped their majority`**: a uniform shift
rather than a set of scenarios breaking. Read the 2×2 before acting.

**`X is at 100% on every episode of this task`**: that checkpoint is at a ceiling. A
change that improved it could not be measured here, and the interval's bound on
that side comes from arithmetic rather than from data. Widen the task set before
reading it as a limit.

**`clustering is not load-bearing for the difference`**: the design effect is
low. A shared scenario effect cancels in a paired contrast, so a high ICC alone
does not imply clustering matters.

**`episode(s) were PROMOTED from run …`**: rows from another run were pooled in.
See `--promote-from`.

## When compare refuses

Exit 2, with the reason. The common ones:

- **Fewer than two checkpoints** in the results.
- **Conflicting scene hashes** for one scene id: two different worlds recorded
  under one name.
- **Duplicate episode ids**: one episode counted twice, which would weight a
  scenario double.
- **A harness surface mismatch**: the code driving the episodes changed between
  checkpoints. Override with `--allow-harness-mismatch` after reading what moved.
- **Degenerate checkpoints**: the two checkpoints produced byte-identical outcomes,
  usually meaning both were pointed at the same server.

## Options

| | |
|---|---|
| `--baseline ID` | which checkpoint is the reference |
| `--checkpoints A B` | restrict to these |
| `--min-seeds N` | drop scenarios with fewer than N repeats |
| `--dichotomy majority\|any\|all` | how seeds reduce to one outcome per scenario |
| `--resamples N` | bootstrap resamples, default 5000 |
| `--no-correction` | skip Holm; the per-contrast p-values stand alone |
| `--allow-harness-mismatch` | proceed despite a surface change |
| `--promote-from PLAN_ID` | pool another run's episodes into this comparison |

## Pooling two runs

A smaller run nests inside a larger one when every episode it contains is one the
larger plan also contains. Changing only `tier` produces that; changing anything
else does not.

```console
$ refractal compare results <full_plan_id> --promote-from <smoke_plan_id>
  promoted 36 episode(s) in 36 row(s) from sha256:1b441de82c5d... (324 still unrun)
```

Refused if the runs do not nest, naming how many episodes did not fit. The two
runs keep their separate identities; only their rows are pooled, and the verdict
records that it happened.
