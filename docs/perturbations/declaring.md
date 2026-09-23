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

Five fields:

| field | |
|---|---|
| `at_step` | when it fires, counted from the episode's first step |
| `until_step` | optional: when it ends. Absent means it lasts to the end |
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
| `drop_observation` | blanks a field of the observation | `fill` | `transform_observation` |
| `substitute_observation` | replaces a field with another representation of the same thing | `kind`, `name` | `transform_observation`, `alternative` |

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

## Perturbing the observation

The first two effects change the world. The last two change **what the policy
is shown of it** — same physics, different picture.

`substitute_observation` is the interesting one, and it exists for a specific
class of problem: **configuration that silently changes what the policy sees.**

A proprioceptive state vector encodes the end-effector's rotation in some
convention. Pick a different convention and the vector is still the right
length, still the right units, still the same field — and describes a rotation
the policy was not trained to read. Nothing downstream can tell. The published
measurement of that class of mistake is **55 points of success rate**, which
is larger than most differences anyone is trying to measure between
checkpoints.

Today that is a setting somebody chose once. As a perturbation it is declared,
hashed, and joined on a base like any other axis:

```yaml
    perturbations:
      - at_step: 0
        type: substitute_observation
        target: states
        args: {kind: state, name: robosuite}
```

`target` names the field to replace. `kind` and `name` say what to put there —
what sort of thing is wanted, and which one. Which names exist is your scene's
business; the refusal lists them when you get one wrong.

**It is categorical, not sweepable.** A representation is this one or that one;
there is nothing in between and no ordering. So it is a matched comparison at
each value rather than a curve, which `compare` already does.

### Dosing: `probability`

The example above substitutes on every step. Add a rate and it applies on a
declared fraction of them instead:

```yaml
    perturbations:
      - at_step: 0
        probability: 0.08
        type: drop_observation
        target: states
        args: {fill: zeros}
```

**This is the field that lets you ask what a policy tolerates**, and it is
worth being clear about why the obvious alternative does not.

You might expect to sweep the *effect* and read how much of it the episode
received. The trouble is that for many transforms, how much lands is decided
by the trajectory — and the trajectory is the policy's. Two rotation
conventions differ only over part of the space of orientations, so an episode
is perturbed only on the steps it spends there. Measured on LIBERO, the
trajectory moves that fraction about **six times more than the start state
does**.

So at one declared setting you get a spread of realized exposures across
episodes, and the spread is mostly the policy's own doing. Worse, it can run
backwards: a policy already going wrong may wander into the region where the
perturbation bites, raising its own exposure. High exposure is then a symptom
of failing rather than a cause of it, and a curve drawn through those points
reads the wrong way round.

`probability` has none of that. **The coin does not know what the policy did**,
so the dose cannot be caused by the outcome. A result reads "fails above 8%
corrupted observations" — a property of the policy, which transfers to any
perturbation that corrupts observations at some rate, rather than a property of
this scene's geometry.

Three things follow.

**The rate is the level a curve groups on.** Declare it and it becomes the
sweep axis, whatever the effect's own arguments are called.

**It is drawn from the episode's seed**, so the same episode is perturbed on
the same steps every run, and different episodes get different draws. A dose
that moved between runs would be unreproducible in exactly the way everything
here exists to prevent.

**Only transforms can carry one.** A state mutation persists, so applying it on
a fraction of steps would compound rather than dose — it is refused with the
list of effects that can.

#### The declared rate is not the realized one, and the gap is not small

A rate is a coin per step, so what an episode actually receives varies around
it — more so the shorter the episode. Measured through the real gate, 200
episodes per cell:

| declared | over 100 steps | over 220 steps |
|---|---|---|
| 5% | 2.0% – 9.0% | 2.7% – 7.3% |
| 10% | 5.0% – 15.0% | 7.3% – 13.2% |
| 25% | 18.0% – 33.0% | 20.9% – 29.5% |

*(central 90% of episodes; LIBERO-Spatial caps at 220 steps)*

At 5% over 100 steps an episode receives anywhere from 0.4× to 1.8× the
declared rate. **Do not read a per-episode declared rate as exact.** The
receipt's realized exposure is what says what that episode got.

**Group on the declared rate anyway.** The spread is per episode; the mean over
a cell converges to the declared value, so it is within-cell noise that episode
count handles — the ordinary reason to run more episodes, not a reason to
change the axis.

Grouping on realized exposure instead is worse, and not obviously so. Episode
length is itself an outcome — a task that succeeds stops early — and a shorter
episode has a wider spread of realized rates. So the extreme bins fill with
short episodes and the middle with long ones. Simulating a case where exposure
has **no effect at all** and success is a coin flip, binning by realized
exposure produces an apparent swing from 72% to 96%. Every point of it is
artifact.

That is a different confound from the one `probability` was introduced to
remove, arriving by a different route, and it is why the declared rate is the
axis and the measured one is the check.

Check the realized rate against the declared one in the receipt. For an effect
that changes the observation every time it applies, the two should agree within
that spread; where they do not, the difference is the geometry described above,
and that gap is the thing worth knowing.

### What the receipt says for these

An observation effect applies on **every step it is active**, so its receipt is
one entry per window rather than one per step, carrying:

- a digest of the observation before and after, from the first application
- how many steps it applied on
- **on how many of those the observation actually changed**

The last is the one that matters, and the first is deliberately *not* a
verdict. A representation swap is frequently **identity**: two rotation
conventions agree exactly over half the orientations there are and differ by a
full turn over the rest, so the same correctly applied effect changes nothing
on some steps and everything on others.

Measured on LIBERO, that is not a rare edge. The wrist sits within a few
ten-thousandths of the boundary between the two at the pose every episode
starts from, and 19 of 50 episodes begin on the side where they agree. A
receipt judging the effect by its first application would call a working
perturbation a no-op on roughly two episodes in five.

Being that close to a discontinuity is **reproducible, not random**: repeated
resets give identical values and changing the seed does not move them, because
the pose comes from the benchmark's pinned start states. What it is not is
meaningful — nothing anyone configured decides it, and an unrelated change to
the simulator could move it.

So: **zero changes across the whole window** is the failure, and it is refused
at the write. Fewer changes than applications is ordinary.

## Sustained perturbations

Add `until_step` and the perturbation is in force for a window rather than the
rest of the episode:

```yaml
    perturbations:
      - at_step: 50
        until_step: 100
        type: scale_actuator
        target: gripper0_gripper_finger_joint1
        args: {factor: 0.1}
```

That is the realistic version of a weak gripper — a servo that browns out and
recovers — rather than one that is weak from the moment it matters onward.

**Ending applies the effect's inverse, not a remembered value.** The distinction
decides whether sustained perturbations compose. If one perturbation scales an
actuator by 0.5 and another by 0.6, the first one ending must leave 0.6 of the
original. Putting back the value it read before the second existed would clobber
the second entirely.

So the end is `× 1/factor`, and the arithmetic works whatever order things start
and finish in.

### Which effects can be sustained

| effect | how it ends | |
|---|---|---|
| `scale_actuator` | × 1/factor | yes |
| `displace_body` | − delta | yes |
| `apply_force` | — | **no** |
| `drop_observation` | you stop applying it | yes |
| `substitute_observation` | you stop applying it | yes |

The two observation effects end for free, and for a reason worth stating: they
change nothing that persists. A world mutation has to be undone because the
simulator is still holding the change. An observation is built fresh every
step, so ceasing to transform it *is* the undo.

That inverts which case is which. For a world effect, `at_step` is the instant
it fires and it stays in force by itself. For an observation effect, the window
is the whole point — leaving `until_step` off means every observation from
`at_step` to the end, and a single transformed frame has to be asked for
explicitly with `until_step: at_step + 1`.

`apply_force` **assigns** its wrench rather than adding to it, so undoing it
would mean restoring a previous value and clobbering anything applied since. It
is refused rather than given semantics that only hold when nothing else is
happening:

```
error: scenario_set 'brownout' gives 'apply_force' an until_step, but that
effect cannot be ended: undoing it would mean restoring a value rather than
applying an inverse, which clobbers anything else acting on the same target.
Effects that can be sustained: ['displace_body', 'scale_actuator'].
```

Drop `until_step` to have it last to the end of the episode.

The results record the start and the end as **two events**, each with the values
either side of it, and the end marked as an end — so a receipt showing a limit
going down and then back up says which was which rather than showing two
indistinguishable entries.

An episode that ends inside the window leaves the end unfired, with a reason,
exactly as an unfired start does.

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

| the episode has | its level |
|---|---|
| no perturbations | none — it is the **baseline**, and belongs on the curve at its top |
| one **sweepable** perturbation | its level, which is what the curve groups by |
| one **categorical** perturbation | none, correctly — see below |
| two or more | none — no single level, so it is off any single axis |

The count is recorded alongside, because three of those four have no level and
mean different things.

### Not every perturbation has a level

Torque scale is continuous, so a sweep is natural: 1.0, 0.1, 0.05, and a curve
through them.

Some perturbations are **categorical**. A state source is correct, wrong, or
some third thing — there is no ordering and nothing between the values. You
cannot sweep it, and an axis with two points is a comparison rather than a
curve. That is fine: a matched comparison at each value is what `compare`
already does, and it needs no sweep machinery at all.

Which kind an effect is, the effect declares. It is not inferred from what its
arguments are called — an effect naming its level something unexpected would
otherwise silently become categorical, losing the column a curve groups by with
nothing to say so.

`scale_actuator` is sweepable, with `factor` as its level. `displace_body` and
`apply_force` are not.
