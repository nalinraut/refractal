"""Timed physical changes to the simulator, during an episode.

Cut gripper torque at step 200. Displace an object once, mid-transport. The
thing nobody publishes: everything in the literature is stationary statistical
corruption, noise every step, and the interesting question is *when*.

**World-side only.** Channel-side corruption -- noise on the observation or the
action array -- is done by others and is out of scope. The test for which is
which: did simulator state change?

Nothing here imports a simulator, and that is the point of the protocol below.
The whole module is exercised against a fake backend that records calls and
holds a dictionary of state: no MuJoCo, no GPU, no robot. If that stops being
possible the boundary has moved to the wrong place.

The receipt is the hard part
----------------------------

The obvious design logs that a perturbation fired, and it is wrong. Measured on
LIBERO: every one of the Panda's seven arm actuators declares
``forcerange = [0, 0]``, which in MuJoCo means *unlimited*. Scaling an unlimited
limit by 0.3 leaves it unlimited. The primitive is called, it returns cleanly,
the log says fired -- and nothing happened. A receipt that records the call
records the request a second time and calls it evidence.

So an effect reads the simulator before it writes, writes, and reads again. The
log carries both values. A no-op reads as unlimited-before, unlimited-after, and
is visible in the artifact rather than inferred from the absence of an error.
That is the same rule as reading results back from Parquet instead of trusting a
counter kept by the code doing the writing, applied to model state.

This holds for every effect, with no exemptions. The one that looked like it
needed one -- a wrench, whose result shows up in the pose only over the steps
that follow -- turned out to be looking at the wrong field: MuJoCo puts it in
``data.xfrc_applied``, readable at the write. When an effect really cannot be
read back, the honest move is to say so and name it as the unguarded one, not to
invent a check. So far none is.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from ..schema.errors import RefractalError

__all__ = [
    "Fired",
    "Primitives",
    "PerturbationError",
    "Timeline",
    "effect",
    "PROTOCOLS",
    "declaration",
    "effects",
    "protocol_of",
    "has_inverse",
    "invertible",
    "rng_for_episode",
]


class PerturbationError(RefractalError):
    """A perturbation could not be applied, or was applied and did nothing.

    Both are errors. "Did nothing" is the one worth having a class for: it
    completes, it looks like success, and it silently turns a sweep into a set
    of identical unperturbed runs.
    """


@runtime_checkable
class Primitives(Protocol):
    """What the simulator's owner supplies. Six calls, and no more.

    The perturbation code depends on this and never on an adapter, which is what
    keeps it testable with no simulator present.

    ``resolve`` is the one necessary leak: a spec saying ``target: "gripper"``
    needs somebody to know what that means, and a readable name through a map
    beats a raw index that is portable, unreadable, and wrong the moment the
    model changes.

    **``resolve`` must be called per episode and its result never cached.**
    Measured on LIBERO: ``env.reset()`` replaces the sim, the model and the data
    objects outright -- verified by object identity -- and restores every value
    written into them. A handle taken in episode one addresses an orphan by
    episode two, and the write lands where nothing reads it. The failure is
    silent and a single-episode test cannot see it, which is why ``Timeline``
    resolves at construction and is constructed per episode.

    ``get_actuator_limit`` is the sixth, added to the five the design called for,
    and it is what makes the receipt true rather than merely present. It returns
    the current torque limit, or ``None`` when the actuator is unlimited.

    ``scale_actuator`` multiplies the **current** limit, not the originally
    declared one. That is what makes two perturbations on one actuator compose
    multiplicatively without anything having to track them: 0.5 then 0.6 leaves
    0.3 of what it started with, and each step's resulting value is logged.
    """

    def resolve(self, name: str) -> Any: ...

    def get_body_pose(self, name: str) -> Sequence[float]: ...

    def set_body_pose(self, name: str, pose: Sequence[float]) -> None: ...

    def apply_force(self, name: str, wrench: Sequence[float]) -> None: ...

    def get_applied_wrench(self, name: str) -> Sequence[float]: ...

    def get_actuator_limit(self, name: str) -> float | None: ...

    def scale_actuator(self, name: str, factor: float) -> None: ...


@dataclass(frozen=True)
class Fired:
    """One perturbation, and what the simulator held either side of it.

    ``specified_at`` and ``fired_at`` are separate because they can differ. A
    trigger at step 200 fires at the first step that reaches 200, which is 200
    for a loop that visits every step and later for one that does not -- and
    state-based triggers, when they arrive, will differ by design. Recording
    only the specified step would make a correct firing look like a wrong one;
    recording only the actual step would lose what was asked for.
    """

    type: str
    target: str
    specified_at: int
    fired_at: int
    before: Any
    after: Any
    #: True when this event ENDED a sustained perturbation rather than starting
    #: one. Both are real events with real before and after values, and a receipt
    #: showing two indistinguishable entries would not say which was which.
    ends: bool = False
    @property
    def changed(self) -> bool:
        return self.before != self.after


#: name -> callable. Each returns ``(before, after)`` read from the simulator.
#: Ten lines each, no state, no branching: sophistication belongs in sweep
#: design and analysis, never in the effect.
Effect = Callable[[Primitives, str, dict, random.Random], tuple[Any, Any]]
#: Where an effect acts, and therefore what hook it needs. The place is the
#: protocol: a world effect needs the model, an observation effect needs what the
#: policy is about to see, an action effect needs what the simulator is about to
#: receive. Those cannot share a protocol -- an observation is not a property of
#: the world and has no handle to resolve.
PROTOCOLS = ("world", "observation", "action")

_EFFECTS: dict[str, Effect] = {}
#: name -> how to undo it, as ARGUMENTS for the same effect. Absent means the
#: effect cannot be ended, and ``until_step`` on it is refused.
_INVERSES: dict[str, Callable[[dict], dict]] = {}
#: name -> (protocol, primitives it calls). Declared, never inferred.
#:
#: This is the only thing the planner has. It cannot inspect an unfamiliar effect
#: and work out that it needs a camera, so refusal comes from what the effect
#: SAYS it needs against what the adapter SAYS it supplies. In a closed registry
#: the planner could hard-code that; in an open one the declaration is all there
#: is, which is why it is required rather than optional.
_DECLARED: dict[str, tuple[str, tuple[str, ...]]] = {}


def effect(
    name: str,
    *,
    protocol: str,
    needs: tuple[str, ...],
    inverse: Callable[[dict], dict] | None = None,
) -> Callable[[Effect], Effect]:
    """Register an effect, and optionally how to undo it.

    ``inverse`` returns the ARGUMENTS that reverse this effect -- not the value
    to restore. That distinction is what makes ``until_step`` compose.

    Restoring a remembered value is the obvious implementation and it is wrong.
    If one perturbation scales an actuator by 0.5 and a second scales it by 0.6,
    the first one ending must leave 0.6 of the original, not the value it
    happened to read before the second existed. Restoring would clobber the
    second -- the last-write-wins failure that multiplicative composition was
    built to avoid, returning through the back door.

    Applying the inverse OPERATION composes correctly: x0.5 then x0.6 then
    x2.0 leaves 0.6, whatever order they end in.

    An effect with no inverse is one that cannot be ended. ``apply_force``
    assigns the wrench rather than adding to it, so "undo" would have to restore
    a previous value and would clobber anything applied since. It is refused
    rather than given semantics that only work when nothing else is happening.
    """

    if protocol not in PROTOCOLS:
        raise ValueError(
            f"effect {name!r} declares protocol {protocol!r}; known protocols "
            f"are {list(PROTOCOLS)}"
        )
    if not needs:
        raise ValueError(
            f"effect {name!r} declares no primitives. An effect that needs "
            "nothing cannot be refused before it runs, and refusing before it "
            "runs is the only protection an open registry has."
        )

    def register(fn: Effect) -> Effect:
        _EFFECTS[name] = fn
        _DECLARED[name] = (protocol, tuple(needs))
        if inverse is not None:
            _INVERSES[name] = inverse
        return fn

    return register


def declaration(name: str) -> tuple[str, tuple[str, ...]] | None:
    """What this effect says it is and what it says it calls."""
    return _DECLARED.get(name)


def protocol_of(name: str) -> str | None:
    declared = _DECLARED.get(name)
    return declared[0] if declared else None


def has_inverse(name: str) -> bool:
    """Whether this effect can carry ``until_step``."""
    return name in _INVERSES


def invertible() -> list[str]:
    return sorted(_INVERSES)


def effects() -> dict[str, Effect]:
    return dict(_EFFECTS)


@effect(
    "scale_actuator",
    protocol="world",
    needs=("scale_actuator", "get_actuator_limit"),
    inverse=lambda args: {"factor": 1.0 / float(args["factor"])},
)
def _scale_actuator(
    primitives: Primitives, target: str, args: dict, rng: random.Random
) -> tuple[Any, Any]:
    """Multiply an actuator's torque limit.

    Refuses an unlimited actuator instead of succeeding at nothing. MuJoCo says
    unlimited with ``forcerange = [0, 0]``, and every arm actuator on LIBERO's
    Panda is declared that way, so this is the common case rather than a corner.
    The planner should have refused it earlier from the probe's record; this is
    the same refusal one layer down, where the model is actually in hand.

    It is a **torque limit**. MuJoCo has no current, and the mapping to current
    depends on the motor constant.
    """
    factor = float(args["factor"])
    before = primitives.get_actuator_limit(target)
    if before is None:
        raise PerturbationError(
            f"cannot scale the torque limit of {target!r}: it is unlimited, so "
            f"scaling it by {factor} leaves it unlimited and the episode would "
            "record a perturbation that did nothing. Declare an absolute limit "
            "on this actuator, or perturb one that has a limit."
        )
    primitives.scale_actuator(target, factor)
    return before, primitives.get_actuator_limit(target)


@effect(
    "displace_body",
    protocol="world",
    needs=("get_body_pose", "set_body_pose"),
    inverse=lambda args: {"delta": [-float(x) for x in args["delta"]]},
)
def _displace_body(
    primitives: Primitives, target: str, args: dict, rng: random.Random
) -> tuple[Any, Any]:
    """Move a body once, by a fixed delta."""
    delta = [float(x) for x in args["delta"]]
    before = list(primitives.get_body_pose(target))
    moved = list(before)
    for i, d in enumerate(delta):
        moved[i] = moved[i] + d
    primitives.set_body_pose(target, moved)
    return before, list(primitives.get_body_pose(target))


@effect("apply_force", protocol="world", needs=("apply_force", "get_applied_wrench"))
def _apply_force(
    primitives: Primitives, target: str, args: dict, rng: random.Random
) -> tuple[Any, Any]:
    """Apply a wrench to a body, and read the applied wrench back.

    Not named ``apply_impulse``, and the measurement is why. In MuJoCo this
    lands in ``data.xfrc_applied``, which is readable at the write with no
    physics step -- so the receipt is true here, the pose was simply the wrong
    thing to look at.

    But ``xfrc_applied`` **persists**: MuJoCo does not clear it between steps.
    Verified on LIBERO -- written once, still set after a step, and qvel moving
    because of it. One call is therefore a force that continues until something
    changes it, which is a sustained perturbation wearing the name of a single
    event. The name now says which it is.

    A true impulse -- one instantaneous change of momentum -- is a write to
    ``qvel``, also readable back. It needs the body's degree-of-freedom
    addresses, and it is not needed by the torque-margin experiment, so it waits
    for one that does.
    """
    before = list(primitives.get_applied_wrench(target))
    primitives.apply_force(target, [float(x) for x in args["wrench"]])
    return before, list(primitives.get_applied_wrench(target))


def rng_for_episode(episode_id: str) -> random.Random:
    """The perturbation stream's own RNG, seeded off the episode's identity.

    Never the global ``random``, and never the policy's seed. A probabilistic
    perturbation drawing from a shared stream injects differently on a re-run of
    the same episode, and reproducibility is gone **with no error anywhere** --
    the run completes, the numbers differ, and nothing says why.

    Seeded from a digest of ``episode_id`` rather than from ``hash()``, which is
    salted per process and would make this non-reproducible in exactly the way
    it exists to prevent.

    The asymmetry is deliberate and useful: this stream is controlled, and the
    policy's sampling inside the model server is not. Scripted cause, emergent
    effect, enforced by where the seeds live rather than by intention.
    """
    digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass
class Timeline:
    """Sorted triggers, one pointer, checked each step.

    Not a scheduler. O(1) per step in the ordinary case: the pointer only moves
    when something fires, so an episode with no perturbations pays one integer
    comparison per step.

    Constructed **per episode**, because ``resolve`` runs here and its results
    must not outlive a ``reset``.
    """

    specs: Sequence[dict]
    primitives: Primitives
    rng: random.Random
    _pending: list[dict] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False)
    _fired: list[Fired] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # A sustained perturbation is TWO triggers: the effect at `at_step`, and
        # its inverse at `until_step`. Expanded here rather than special-cased in
        # `step`, so the loop stays one sorted list and one pointer -- ending a
        # perturbation is the same machinery as starting one.
        expanded: list[dict] = []
        for spec in self.specs:
            expanded.append(dict(spec))
            until = spec.get("until_step")
            if until is None:
                continue
            if not has_inverse(spec["type"]):
                raise PerturbationError(
                    f"{spec['type']!r} cannot carry until_step: it has no inverse, "
                    f"so there is no defined way to end it. Effects that can be "
                    f"ended: {invertible()}."
                )
            if int(until) <= int(spec["at_step"]):
                raise PerturbationError(
                    f"until_step {until} is not after at_step {spec['at_step']}; "
                    "a perturbation that ends before it starts is a mistake, not "
                    "a zero-length one."
                )
            ending = dict(spec)
            ending["at_step"] = int(until)
            ending["args"] = _INVERSES[spec["type"]](dict(spec.get("args", {})))
            ending.pop("until_step", None)
            #: Marks this as the END of a sustained perturbation, so the receipt
            #: says which it is rather than showing two indistinguishable events.
            ending["_ends"] = True
            expanded.append(ending)

        self._pending = sorted(expanded, key=lambda s: (s["at_step"], s["type"]))
        for spec in self._pending:
            if spec["type"] not in _EFFECTS:
                raise PerturbationError(
                    f"no effect named {spec['type']!r}; this build knows "
                    f"{sorted(_EFFECTS)}"
                )
            # Resolve now, once, for this episode. Names are validated before the
            # episode runs rather than at step 200 of 400, and no handle survives
            # the next reset because this object does not either.
            self.primitives.resolve(spec["target"])

    def step(self, index: int) -> list[Fired]:
        """Fire whatever is due at or before ``index``. Returns what fired."""
        fired: list[Fired] = []
        while self._index < len(self._pending):
            spec = self._pending[self._index]
            if spec["at_step"] > index:
                break
            before, after = _EFFECTS[spec["type"]](
                self.primitives, spec["target"], dict(spec.get("args", {})), self.rng
            )
            fired.append(
                Fired(
                    ends=bool(spec.get("_ends")),
                    type=spec["type"],
                    target=spec["target"],
                    specified_at=spec["at_step"],
                    fired_at=index,
                    before=before,
                    after=after,
                )
            )
            self._index += 1
        self._fired.extend(fired)
        return fired

    @property
    def fired(self) -> list[Fired]:
        return list(self._fired)

    def unfired(self) -> list[dict]:
        """Specs that never came due.

        Not an error on its own: a trigger at step 200 in an episode that ended
        at 150 legitimately did not fire, and the episode is still valid. It is
        an error for a *sweep* in which nothing fired anywhere, the same way a
        filter that rejects every scenario is -- and that check needs this list
        from every episode, which is why it is reported rather than raised.
        """
        return [dict(s) for s in self._pending[self._index :]]
