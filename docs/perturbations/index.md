# Perturbations

A perturbation is a timed physical change to the simulator during an episode:
reduce the gripper's torque limit, displace an object, apply a force. It is
declared as data in the scenario, so it is hashed, reproducible and sweepable.

If you have run a comparison and want to vary a physical parameter, this is the
mechanism. Start with [getting started](../getting-started.md) if you have not
run one yet.

## Why it exists

Success rates alone often cannot separate two checkpoints.

On LIBERO-Spatial, pi0.5 scores 100% on eight of ten tasks. Once a checkpoint is
at the ceiling, varying the starting pose has nothing left to discriminate —
every scenario comes back a success, and the comparison reports no difference
because there is none left to see at that resolution.

A continuous physical axis does have something to say. Torque scale from 1.0
downward is guaranteed to have a boundary somewhere: at full torque the task is
possible, at zero it is not, and the crossing is a measurement. *Where* each
checkpoint crosses is a number that a saturated success rate cannot give you.

## Scripted cause, emergent effect

Reducing the gripper's torque limit is an input **you** chose. Whether the object
slips, when it slips, and whether the policy recovers is the **outcome**.

Keeping those apart is the whole design. A scripted change counted as a failure
makes the success rate meaningless — you would be measuring your own instruction.
So a perturbation is recorded as what was asked for, separately from what
happened, and the two are never conflated.

## What is in scope

Two protocols, and the line between them is worth keeping sharp:

> Did simulator state change?

A real object placed in front of the camera is **world-side**. A black rectangle
drawn on the camera image is **observation-side**. They look the same to the
policy and are entirely different mechanisms.

**World-side** is the original scope: timed physical events. Published
robustness work is almost entirely stationary statistical corruption — noise on
every step. "Cut the gripper's torque at step 200" is a different question, and
the answer depends on *when*.

**Observation-side** was added for a narrower reason than image corruption, and
it is worth being precise about it. Generic corruption is well covered
elsewhere: Robust-Gymnasium supplies the taxonomy, and RobustVLA has benchmarked
five image corruptions and three action-noise levels against pi0 and OpenVLA on
LIBERO. Repeating that is not the point.

What is not covered anywhere is **configuration that silently changes what the
policy sees**. Which rotation convention the state vector uses, which source it
is read from — settings that produce an observation of the right shape, the
right units and the wrong meaning. The published figure for that class of
mistake is 55 points of success rate, and it is nobody's experiment: it is an
argument somebody set once and no results table states.

Recording such a setting prevents the wrong comparison. Only making it a
perturbation enables the right one — declared, hashed, joined on a base, and
compared like any other axis. See
[perturbing the observation](declaring.md#perturbing-the-observation).

**Action-side is not built.** The protocol is designed for three and two exist.

## What does not exist yet

Stated plainly so you do not plan around it:

- **State-based triggers.** You cannot say "when the object clears 5 cm".
  Triggers are step numbers only. This is additive rather than a restructure —
  it would not change how anything here is declared or recorded — but it is not
  built.
- **Ramps.** A perturbation is a step change. There is no way to fade one in.

- **Action perturbations.** The design carries three protocols and two are
  built. Nothing about declaring or recording changes when the third arrives.

Sustained perturbations **are** supported: see
[`until_step`](declaring.md#sustained-perturbations). Not every effect can carry
one, and the page says which and why.

## Where to go next

| | |
|---|---|
| [Declaring](declaring.md) | the catalog syntax, the effects, what gets refused |
| [Adapters](adapters.md) | making perturbations work in your own scene |
| [Sweeps](sweeps.md) | a torque sweep end to end, with the real numbers |
