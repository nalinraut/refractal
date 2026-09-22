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

**World-side only.** The test for which is which:

> Did simulator state change?

A black rectangle drawn on the camera image is **channel-side**. A real object
placed in front of the camera is **world-side**. They look the same to the policy
and are entirely different mechanisms.

Channel-side perturbation — image corruption, action noise — is not here and is
not planned. Robust-Gymnasium supplies the taxonomy for it, and RobustVLA has
already benchmarked five image corruptions and three action-noise levels against
pi0 and OpenVLA on LIBERO. That work exists; this is the part that does not.

What is specific here is **timed** physical events. Published robustness work is
almost entirely stationary statistical corruption — noise on every step. "Cut the
gripper's torque at step 200" is a different question, and the answer depends on
*when*.

## What does not exist yet

Stated plainly so you do not plan around it:

- **Sustained perturbations.** A perturbation fires at a step. There is no
  duration, no `until_step`, no ramp. A weakened gripper stays weakened for the
  rest of the episode.
- **State-based triggers.** You cannot say "when the object clears 5 cm". Triggers
  are step numbers only.

Both are additive rather than restructures — neither would change how anything
here is declared or recorded. They are simply not built.

## Where to go next

| | |
|---|---|
| [Declaring](declaring.md) | the catalog syntax, the effects, what gets refused |
| [Adapters](adapters.md) | making perturbations work in your own scene |
| [Sweeps](sweeps.md) | a torque sweep end to end, with the real numbers |
