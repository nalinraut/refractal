# Running a torque sweep

A sweep varies one physical parameter across levels and measures where behaviour
changes. The numbers on this page come from a real run: LIBERO-Spatial, two
checkpoints, six torque levels, 120 episodes.

## The shape of a sweep

One level per scenario set, all sharing the same base scenarios. Six levels is
six sets on one scene, identical except for the factor:

```yaml
scenario_sets:
  - id: torque-1p0
    scene: libero-spatial
    generator: refractal.generators:linspace_grid
    generator_seed: 0
    params:
      init_state_index: {range: [0, 4], steps: 5}
    perturbations:
      - {at_step: 0, type: scale_actuator,
         target: gripper0_gripper_finger_joint1, args: {factor: 1.0}}

  - id: torque-0p1
    # ... identical, args: {factor: 0.1}
```

The `params` are the same in every set. That repetition **is** the curve: the
levels share a base identity, which is what lets them be plotted against each
other.

### Each level is its own run

Six levels are six plans and six runs, joined afterwards on the base identity.

That is not a limitation to work around — it follows from two requirements that
cannot both hold in one plan:

- The harness identifies an episode by its own counter, which must run
  contiguously from zero within a worker.
- A sweep repeats the same base scenarios at every level.

Six levels of five scenarios in one plan presents the counter with
`[0,0,0,0,0,0,1,1,…]` where it will run `[0,1,2,…,29]`, and every row would claim
a start state that never executed. That is refused before anything runs.

So: six treatments over one shared base. The relationship between levels is a
**join**, not a shared identity — which is what the base identity is for.

Run each plan as you would any other, pointing the run at the benchmark class
that can execute a timeline.

## Coarse pass first

**Do not guess the band.** Two predictions were made before this run. Both were
wrong, from opposite directions:

| Guess | Reasoning | How wrong |
|---|---|---|
| 1.0 → 0.3 | a plausible-sounding range | ~430× too high |
| 0.01 → 0.05 | a 5.6 g bowl needs 0.014 N to hold | ~20× too low |

The measured band is roughly **0.05 to 1.0** — 1 N to 20 N on a 20 N limit.

The static calculation failed for an interesting reason: **holding an object is
not what demands torque.** Closing the fingers onto it, and accelerating it during
transport, demand far more than supporting its weight. The arithmetic was correct
and answered the wrong question.

Neither intuition nor calculation found the band. A coarse pass did, in four
minutes.

## The result

Per task, as measured. `spatial-task-1` is included to show something important.

| level | limit | spatial-task-0 pi0 / pi0.5 | spatial-task-1 pi0 / pi0.5 |
|---|---|---|---|
| 1.0 | 20 N | 4/5 · 5/5 | 0/5 · 0/5 |
| **0.1** | **2 N** | **1/5 · 0/5** | 0/5 · 0/5 |
| 0.05 | 1 N | 0/5 · 0/5 | 0/5 · 0/5 |
| 0.02 | 0.4 N | 0/5 · 0/5 | 0/5 · 0/5 |
| 0.01 | 0.2 N | 0/5 · 0/5 | 0/5 · 0/5 |
| 0.005 | 0.1 N | 0/5 · 0/5 | 0/5 · 0/5 |

### What this does not support

**No verdict about the checkpoints.** One usable task, five scenarios, one seed.
The single interesting cell — pi0 at 1/5 against pi0.5 at 0/5 at 2 N — is **one
episode of difference**. That is not a finding, and reporting it as one would be
the error this tool exists to prevent.

What the run does establish is the **band**, and that the method works. Those are
findings about where to look and about the instrument, not about which checkpoint
is better.

### Say what each number is a number of

`spatial-task-1` scores **0/10 unperturbed**. It was at the floor before anything
was perturbed, so its six rows measure nothing about torque.

Pooled across both tasks, the first reading of this table looked like "everything
fails below 20 N". Per task, one of them never worked at all. A torque curve can
be per task or pooled across tasks, and which one you are looking at has to be
stated every time.

For the dense pass: drop the task that cannot move, sample inside 0.05–1.0, and
use enough seeds to separate the checkpoints.

## Two anchors at the top

Factor 1.0 is a no-op scale — it multiplies the limit by one. It should be
identical to not perturbing at all, and it is:

| | perturbed @ 1.0 | unperturbed, on disk |
|---|---|---|
| spatial-task-0, pi0 | 4/5 | 4/5 |
| spatial-task-0, pi0.5 | 5/5 | 5/5 |
| spatial-task-1, pi0 | 0/5 | 0/5 |
| spatial-task-1, pi0.5 | 0/5 | 0/5 |

Across the same five scenarios. Two things follow.

The perturbation path is **inert when it should be** — firing a no-op changes
nothing measurable, so the machinery is not quietly perturbing something else.

And the curve has **two independent anchors** at its top end rather than one. The
unperturbed runs were recorded before the base identity column existed at all, and
they joined anyway, because the base of an unperturbed scenario is just its
ordinary scenario identity.

## Reading the receipt

**Check the receipt before you read the rates.**

A sweep where nothing actually fired looks exactly like a working one. That
happened here, twice:

| column | what it said |
|---|---|
| level | 0.02 ✓ |
| count | 1 ✓ |
| benchmark class | the perturbing one ✓ |
| episodes | completing, 85 steps ✓ |
| success | pi0 3/10, pi0.5 5/10 — plausible |

Every column said it was working. At 0.02 the gripper would hold 0.4 N, and those
rates are indistinguishable from unperturbed runs because they *were* unperturbed
runs. Read without the receipt, the headline is "pi0.5 tolerates low torque
better" from a sweep where no torque was ever reduced.

The only thing that caught it was the receipt recording what the simulator **did**
rather than what was **requested**.

So, per episode, the results carry each perturbation with the values either side
of it:

- **a fired step, with before and after** — it happened, and by how much
- **no fired step, with a reason** — the episode ended before its trigger. Real,
  expected, and excluded from that level's curve
- **no fired step, no reason** — a mapping failure. This is refused at the write,
  because a record that is present, well-formed and empty passes every check that
  asks whether a record *arrived*

A sweep in which **no** episode experienced its level is refused outright. Those
results read as clean unperturbed runs, and would be reported as such.

## Related

- [Declaring](declaring.md) — the syntax, and why `at_step: 0`
- [Adapters](adapters.md) — making this work in your own scene
- [Reading a comparison](../reading-a-comparison.md) — what the per-level numbers
  mean once you have them
