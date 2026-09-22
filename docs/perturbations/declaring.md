# Declaring a perturbation

Perturbations go in `scenarios.yaml`, on the scenario set. Every scenario the set
generates carries them.

```yaml
scenario_sets:
  - id: weak-grip
    scene: libero-spatial
    generator: refractal.generators:linspace_grid
    generator_seed: 0
    params:
      init_state_index: {range: [0, 4], steps: 5}
    perturbations:
      - at_step: 0
        type: scale_actuator
        target: gripper0_gripper_finger_joint1
        args: {factor: 0.1}
```

That is one perturbation, firing at the start of every episode in the set, which
cuts the gripper's torque limit to a tenth of what the model declares.

Four fields:

| field | |
|---|---|
| `at_step` | when it fires, counted from the episode's first step |
| `type` | which effect to run |
| `target` | a name your adapter resolves — an actuator or a body |
| `args` | that effect's parameters |

`target` is whatever the scene calls the thing. The names above are LIBERO's, and
`refractal build` records the ones your scene actually has, so a typo is caught
when the plan is made rather than mid-episode.

## Effects

| type | what it does | `args` | the adapter must supply |
|---|---|---|---|
| `scale_actuator` | multiplies an actuator's **torque limit** by `factor` | `factor` | `scale_actuator`, `get_actuator_limit` |
| `displace_body` | moves a body once, by a fixed offset | `delta`, a list | `get_body_pose`, `set_body_pose` |
| `apply_force` | applies a wrench to a body | `wrench`, a list | `apply_force`, `get_applied_wrench` |

Two points of precision on each.

**`scale_actuator` is a torque limit, not a current limit.** MuJoCo has no
current. You are scaling the force range the actuator is allowed to exert, and
the mapping from that to motor current depends on the motor constant, which the
simulator does not model. Call it torque.

It multiplies the **current** limit, not the originally declared one. Two
perturbations on the same actuator therefore compose multiplicatively — 0.5 then
0.6 leaves 0.3 of what it started with — and each step's resulting value is
recorded, so composition is visible in the results rather than only in your head.

**`apply_force` is sustained, not an impulse.** The wrench persists until
something changes it, because that is what the underlying field does. It is named
for what it does. A true one-off impulse — a single instantaneous change of
momentum — is not implemented.

## `at_step: 0`, and why it is usually what you want

This is a correctness matter, not a preference.

An episode that **ends before its trigger** never experienced the perturbation.
It succeeded unperturbed. Counting that success at the declared level credits the
policy with robustness it never demonstrated.

**The bias is not random.** Fast episodes escape a late trigger more often, so
leaving those episodes in favours whichever checkpoint finishes sooner. A quicker
checkpoint looks more torque-robust purely by outrunning the perturbation — and
that confound moves with the very quantity you are comparing.

Worse, such an episode is **unusable in both directions**. Its scenario identity
covers the perturbation, so it cannot join the unperturbed baseline either. It is
not misfiled and cannot be reclassified; it measured baseline behaviour under a
perturbed identity. The only fix is not creating it.

So: **for anything measuring robustness at a level, fire at step 0.** Every
episode is then perturbed before it can finish, and the gap between assigned and
experienced closes.

A mid-episode trigger is for experiments where *when* is the variable — measuring
how recovery depends on the moment a disturbance arrives. There, the distinction
is the point rather than a nuisance, and the results record both the step you
asked for and the step it actually fired at.

`compare` reports how many episodes were excluded for never experiencing their
level, and refuses outright a comparison in which none did.

## What gets refused

All three are checked **when the plan is made**. They cost nothing and they
happen before any container starts.

**An actuator the scene does not have.**

```
error: scenario_set 'weak-grip' perturbs actuator 'gripper0_thumb', which scene
'libero-spatial' does not have. It has ['gripper0_gripper_finger_joint1',
'gripper0_gripper_finger_joint2', 'robot0_torq_j1', ...]; did you mean
'gripper0_gripper_finger_joint1'?
```

Fix the name. The list is what the scene actually reports.

**An actuator with no torque limit to scale.**

```
error: scenario_set 'weak-grip' scales the torque limit of 'robot0_torq_j2' on
scene 'libero-spatial', but that actuator has no limit to scale -- MuJoCo's
forcerange [0, 0] means unlimited, and scaling unlimited by any factor leaves it
unlimited. The episode would run, the log would say the perturbation fired, and
nothing would have changed.
```

This is the one that catches people. On LIBERO's Panda, the two gripper actuators
declare a limit of 20 N; all seven arm joints declare `[0, 0]`, which MuJoCo reads
as *unlimited*. Scaling unlimited by 0.3 is still unlimited — the call succeeds,
the episode completes, and nothing happened. Perturb an actuator that has a limit,
or give that one an absolute limit in the scene.

**A scene whose capabilities were never recorded.**

```
error: scenario_set 'weak-grip' scales the torque limit of
'gripper0_gripper_finger_joint1' on scene 'libero-spatial', but nothing has
recorded which of that scene's actuators have a torque limit. Run 'refractal
build' where the benchmark is importable.
```

Run `refractal build` with your probe. This refuses rather than assuming, because
*nothing looked* and *it said yes* must not be the same answer when the failure is
silent and costs a whole run.

## Identity: what changes

A perturbed episode is a **different experiment** from its unperturbed
counterpart, so its scenario identity covers the perturbation. Scale 0.5 and scale
0.3 are different scenarios and will not join as though they were the same.

Every episode also carries a **base** identity covering only the unperturbed
scenario. That is what a sweep joins on: hold the base fixed, vary the level, get
a curve.

The consequence worth stating plainly:

> **Results you have already recorded become the baseline end of any future
> sweep.**

The base of an unperturbed scenario is just its ordinary scenario identity, so a
run written before perturbations existed at all joins a curve measured later. This
is measured rather than asserted — a sweep joined against runs that predate the
base column entirely, all five base scenarios matching.

One more thing the results carry, because it decides how a sweep is grouped:

| | |
|---|---|
| no perturbations | the **baseline**, and it belongs on the curve at its top |
| one perturbation | its level, which is what the curve groups by |
| two or more | no single level, so it is off any single axis |

The count is recorded alongside, because the second and third cases both have no
level and mean opposite things.
