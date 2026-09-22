# Making perturbations work in your scene

Perturbation code never talks to a simulator. It talks to a handful of calls,
which whoever owns the scene supplies. Implement them and every effect works;
the timeline, the ordering and the recording are already done.

You implement only the ones the perturbations you care about need. Six cover
the world protocol; observation effects need one or two more.

This also means the perturbation machinery is testable with no simulator at all —
which is worth knowing when you come to test your own adapter.

## The world primitives

```
resolve(name) -> handle
get_body_pose(name)
set_body_pose(name, pose)
apply_force(name, wrench)
scale_actuator(name, factor)
get_actuator_limit(name) -> float | None
```

`resolve` is the one necessary leak. A spec saying `target: "gripper"` needs
somebody to know what that means, and a readable name through a map beats a raw
index that is portable, unreadable, and wrong the moment the model changes.

### Why `get_actuator_limit` exists

It looks redundant beside `scale_actuator` and is not.

**Every perturbation is verified by reading the world back.** An effect reads the
simulator, writes, and reads again; both values go into the results. You cannot
read back what you cannot read, so the getter is what makes the record true rather
than merely present.

This is not hypothetical. On LIBERO's Panda, scaling an *unlimited* actuator
succeeds and changes nothing — the call returns cleanly, the episode completes,
and a record of the call alone would say it fired. Reading the limit back shows
unlimited before and unlimited after, and the effect refuses rather than reporting
a success.

`None` means the actuator has no limit. That is also how the planner learns that
scaling that target would do nothing, so it can refuse before the run instead of
during it.

Return the engine's own "is this limited" flag rather than inferring it from the
range. An actuator can legitimately be declared limited to a range of zero, and
inferring would call that unlimited — a second implementation of the engine's
rule, which is how these things go wrong.

## The observation primitives

```
transform_observation                  # a promise about WHERE, see below
alternative(kind, name) -> value       # something you can still build
```

`transform_observation` carries no arguments because it is not really a call:
it is how you declare that your adapter hooks the observation at the right
place. The section below says where that is.

### `alternative`, and why the effect cannot do this itself

An observation effect hooks **after** your own transforms — which is correct,
and costs it reach. Anything your transforms consumed is gone by the time the
effect sees the result.

A state vector is the case that bites. If you assemble it from selected raw
fields, the raw fields are no longer there. *Blanking* the vector needs only
the vector. Substituting the **source it was built from** needs what has
already been discarded.

Moving the hook earlier would fix the reach and break the evidence, reporting
changes the policy never saw. So the effect asks instead:

> **The effect selects. The adapter produces.**

`alternative(kind, name)` is that question. You are the only thing that still
holds the raw observation and knows how to rebuild it another way.

```python
def alternative(self, kind, name):
    if kind != "state":
        raise KeyError(f"this scene builds no {kind!r} alternatives")
    raw = self._raw_observation()
    return self._build_state(raw, convention=name)
```

Two rules for writing one.

**Rebuild it exactly as you build the real one — dtype included.** You are
producing something that will replace part of the observation, so any
difference you introduce is a difference the policy receives. Returning a
32-bit vector where the observation carries 64-bit changes every byte, and the
receipt then reports a change on every step regardless of whether the thing
you were actually varying changed anything. The measurement carries a silent
passenger.

There is an exact check for this, and it costs one run: **ask for the variant
your adapter is already configured with.** The observation must come back
byte-identical, and the run must then be refused as a no-op. If it is not
identical, the difference is yours.

**Keyed by `kind` so one call serves every such need.** A later effect wanting
a different camera projection or an unfiltered sensor reading asks the same
primitive with a different `kind`, rather than growing the protocol once per
effect.

Declaring the observation protocol without this is refused when the plan is
made, naming what is missing — so an adapter that cannot build alternatives
fails loudly rather than quietly producing a sweep of no-ops.

## Two rules that will catch you

### Resolve after every reset. Never cache a handle.

A simulator reset typically replaces the model and data objects outright. Measured
on LIBERO: after `reset`, the model, the data and the simulation object are all
different objects, and every value written into the old ones is gone.

So a handle taken in episode one addresses an orphan by episode two. The write
lands somewhere nothing reads, nothing raises, and the perturbation silently stops
happening.

**A single-episode test cannot see this.** It passes either way. The fixture needs
**two episodes**, asserting the perturbation fired in the second:

```python
for episode in (0, 1):
    bench.reset(task(episode))
    assert limit(bench) == 20.0          # reset restored it
    for _ in range(4):
        bench.step(action)
    assert limit(bench) == 5.0, f"episode {episode} was not perturbed"
```

Episode one passes with a cached handle. Episode two is the test.

If your own fake simulator keeps one model across resets, it will hide this bug
rather than catch it — so make the fake replace its model too.

### Every effect reads the world back

Implement the getter beside the setter. An effect that cannot be verified is one
whose record says *fired* when nothing happened, and that is the failure the whole
design exists to prevent.

When you add an effect, check what field its result actually lands in. Choosing
the wrong observable looks like an unverifiable effect and usually is not one —
a wrench, for instance, does not move a pose during the call, but it does appear
immediately in the applied-force field, which reads back fine.

## If you support the observation protocol

An observation effect transforms what the policy sees. Declaring support for it
is a promise about **where you put the hook**, and it is the one thing a user is
most likely to get wrong.

> **The hook goes at the last point you control before the observation leaves
> for the policy — after every content transform you perform.**

Only you know where that is. Most adapters do not hand the simulator's raw
output straight to the policy: they pick cameras, resize images, convert
conventions, and build a state vector from selected fields. Every one of those
is a content transform, and the hook belongs after all of them.

### The failure mode, which is worse than it sounds

Put the hook **before** your own transforms and the receipt reports a change the
policy never saw.

The digest pair will show the observation changing, because it did — you changed
it. Then your transform runs, and what reaches the policy is something else. The
receipt is not lying about what it measured; it is measuring the wrong thing,
and nothing downstream can tell.

**It is worst exactly where it matters most.** A state vector is usually *built*
from selected raw fields rather than passed through. Perturb a field your
transform does not read and you get: raw observation changed, digest changed,
receipt says the wrapper applied, count non-zero — and the policy received a
byte-identical observation. Every guard passes. The only thing that would catch
it is noticing the result did not move, which is the thing you were trying to
measure.

So: after your transforms, not before.

### What the receipt can then honestly say

*This is what was sent.* One hop remains beyond it — whether the model server
decodes to what was encoded is across a process boundary and outside anything
observable from here. That gap is named rather than papered over, and it is one
hop rather than an unknown number.

If serialisation in your stack *does* alter content — quantising, resizing,
reordering — then the hook after your transforms is not sufficient, and the
honest place is wherever the last alteration happens. Check rather than assume.

## A worked adapter

Against a small MuJoCo scene. Illustrative — the names are your scene's.

```python
class MySceneAdapter:
    def __init__(self, sim):
        self._sim = sim

    # Looked up every call, never stored. `reset` replaces these objects, and a
    # cached index addresses the previous episode's model.
    @property
    def _model(self):
        return self._sim.model

    @property
    def _data(self):
        return self._sim.data

    def resolve(self, name):
        model = self._model
        for i in range(model.nu):
            if model.actuator_id2name(i) == name:
                return ("actuator", i)
        # Raise rather than return None: an unknown target is a catalog mistake,
        # and it should surface before the episode runs, not as a no-op during it.
        return ("body", model.body_name2id(name))

    def get_actuator_limit(self, name):
        _, i = self.resolve(name)
        # The engine's own flag. A range of [0, 0] with the flag set is legal and
        # means limited-to-zero; inferring from the range would call it unlimited.
        if not bool(self._model.actuator_forcelimited[i]):
            return None
        return float(self._model.actuator_forcerange[i][1])

    def scale_actuator(self, name, factor):
        _, i = self.resolve(name)
        if not bool(self._model.actuator_forcelimited[i]):
            return          # the effect reads the limit back and refuses
        low, high = self._model.actuator_forcerange[i]
        # The CURRENT limit, not the declared one: this is what makes two
        # perturbations on one actuator compose multiplicatively.
        self._model.actuator_forcerange[i] = [low * factor, high * factor]

    def get_body_pose(self, name):
        _, i = self.resolve(name)
        return list(self._data.xpos[i])

    def set_body_pose(self, name, pose):
        _, body = self.resolve(name)
        joint = int(self._model.body_jntadr[body])
        if joint < 0:
            # A body with no joint cannot be moved. Say so rather than writing
            # into an address that belongs to something else.
            raise KeyError(f"body {name!r} has no joint")
        address = int(self._model.jnt_qposadr[joint])
        for offset, value in enumerate(list(pose)[:3]):
            self._data.qpos[address + offset] = float(value)

    def apply_force(self, name, wrench):
        _, i = self.resolve(name)
        self._data.xfrc_applied[i] = [float(v) for v in wrench]

    def get_applied_wrench(self, name):
        # Readable at the write, before any physics step -- which is what lets
        # the wrench be verified like every other effect.
        _, i = self.resolve(name)
        return list(self._data.xfrc_applied[i])
```

## Recording capabilities

The planner refuses a perturbation targeting something the scene cannot support,
and it needs to know which actuators have a limit to do that. Report it from your
probe, and `refractal build` records it:

```python
def scene_capabilities(self, scene):
    model = open_scene(scene).sim.model
    return {
        "actuators": {
            model.actuator_id2name(i): {
                "force_limited": bool(model.actuator_forcelimited[i]),
                "forcerange": [float(v) for v in model.actuator_forcerange[i]],
            }
            for i in range(model.nu)
            if model.actuator_id2name(i)
        }
    }
```

This is **not** hashed into the scene's identity. It constrains which plans are
legal; it does not change what an episode is. Folding it into the scene hash would
move every plan identity to record something that discriminates nothing — and
would strand results you have already recorded.

If your probe cannot report capabilities — no display, no GPU on the build
machine — the build warns rather than failing, and only a plan that actually
perturbs that scene is refused later.

## Wrapped benchmarks

If the scene comes from a benchmark you do not own, the primitives come from a
subclass that reaches into whatever it holds, and the catalog names your subclass
at run time rather than in the scene.

Keep that subclass **inert when nothing is perturbed**, and check it rather than
assuming: run the same unperturbed episode through the original class and through
yours, and compare the simulator's own state. For LIBERO's the trajectories are
bit-identical over 20 steps, which is what makes it safe to treat the choice of
class as deployment rather than as part of the experiment's identity.
