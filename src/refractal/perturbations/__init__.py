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

Where the effect is emergent rather than a state write -- an impulse changes a
pose over the following steps, not during the call -- the before/after pair
cannot show it, and the effect says so rather than implying a check it does not
perform.
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
    "effects",
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
    #: False when this effect's result cannot be seen in a before/after pair --
    #: an impulse acts over the following steps. Says so rather than reporting
    #: an unchanged pose as though it were a detected no-op.
    observable: bool = True

    @property
    def changed(self) -> bool:
        return self.observable and self.before != self.after


#: name -> callable. Each returns ``(before, after)`` read from the simulator.
#: Ten lines each, no state, no branching: sophistication belongs in sweep
#: design and analysis, never in the effect.
Effect = Callable[[Primitives, str, dict, random.Random], tuple[Any, Any]]
_EFFECTS: dict[str, Effect] = {}
_UNOBSERVABLE: set[str] = set()


def effect(name: str, *, observable: bool = True) -> Callable[[Effect], Effect]:
    def register(fn: Effect) -> Effect:
        _EFFECTS[name] = fn
        if not observable:
            _UNOBSERVABLE.add(name)
        return fn

    return register


def effects() -> dict[str, Effect]:
    return dict(_EFFECTS)


@effect("scale_actuator")
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


@effect("displace_body")
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


@effect("apply_impulse", observable=False)
def _apply_impulse(
    primitives: Primitives, target: str, args: dict, rng: random.Random
) -> tuple[Any, Any]:
    """Apply a wrench once.

    Marked unobservable: the wrench changes the pose over the steps that follow,
    not during this call, so the pose either side of it is normally identical.
    Reporting that as a detected no-op would be a false alarm, and reporting it
    as a change would be a lie. The pair is recorded for provenance and
    ``changed`` stays false.
    """
    before = list(primitives.get_body_pose(target))
    primitives.apply_force(target, [float(x) for x in args["wrench"]])
    return before, list(primitives.get_body_pose(target))


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
        self._pending = sorted(self.specs, key=lambda s: (s["at_step"], s["type"]))
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
                    type=spec["type"],
                    target=spec["target"],
                    specified_at=spec["at_step"],
                    fired_at=index,
                    before=before,
                    after=after,
                    observable=spec["type"] not in _UNOBSERVABLE,
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
