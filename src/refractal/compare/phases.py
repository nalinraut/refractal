"""Per-phase conditional rates, and what chaining costs.

Two policies with the same episode success rate can need completely
different work. Four phases, and:

    A: 0.80 0.80 0.80 0.80 -> 0.410
    B: 1.00 1.00 1.00 0.41 -> 0.410

At episode level those are indistinguishable AT ANY N, because the number
they produce is the same number. Per phase, A needs a better model and B
needs one skill added. For a deployment decision that is the whole answer,
and the episode rate cannot carry it.

Conditional, never marginal
---------------------------

Episode success is a product of conditionals:

    P(all) = P(p1) * P(p2 | p1) * P(p3 | p1, p2) * ...

Reporting a MARGINAL rate for phase 4 mixes "failed phase 4" with "never
reached phase 4", which are facts about different things -- the second is a
statement about phases 1 to 3. Conditioning on having been attempted is what
separates them, and it is why `attempted` is recorded per phase rather than
inferred from the outcome being absent.

Informative censoring, and the one place it is real
---------------------------------------------------

In a CHAINED run a phase after a failure is never attempted, and the reason
it is missing is the policy's own failure. That is informative censoring --
the genuine article, unlike the time-to-success case where nothing is lost
to follow-up.

It also shrinks n at every successive phase, and the conditioning set differs
from phase to phase, so a Holm family ACROSS phases is not well formed: the
contrasts are not on comparable denominators and the hypotheses are nested.

ISOLATED runs eliminate it. Every episode reaches every phase by
construction, because each phase starts from a canonical state injected
rather than inherited. That restores full n at the last phase and makes a
per-phase family well formed.

Hardware cannot run the isolated condition at all: once the object is on the
floor the operation is over. This is a measurement simulation can make and
hardware cannot -- a capability, not a fidelity concession.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .report import Finding


class PhaseError(ValueError):
    """A per-phase question the data cannot answer."""


@dataclass(frozen=True)
class PhaseRate:
    """One phase's conditional success rate, and its denominator."""

    phase_id: str
    attempted: int
    succeeded: int

    @property
    def rate(self) -> float | None:
        """Conditional on having been attempted. None when never reached."""
        return self.succeeded / self.attempted if self.attempted else None

    def render(self) -> str:
        if self.rate is None:
            return f"{self.phase_id}: never attempted"
        return (f"{self.phase_id}: {self.rate:.3f} "
                f"({self.succeeded}/{self.attempted} attempted)")


def _credited(rec: Mapping[str, object], credit: str) -> bool:
    """Whether this record counts as the phase having been achieved.

    Two readings of one record, and the choice is the caller's because there
    is no default answer: a policy that puts the item in the bin and later
    knocks it out demonstrated the skill and did not deliver. `at_end` asks
    whether the predicate still held when the episode finished;
    `first_satisfied` asks whether it ever held.
    """
    if credit == "first_satisfied":
        return rec.get("first_satisfied_step") is not None
    if credit == "at_end":
        return rec.get("outcome") == "SUCCESS"
    raise PhaseError(
        f"unknown phase_credit {credit!r}. It is 'at_end' or "
        "'first_satisfied'; defaulting would bury a choice that changes every "
        "rate it touches."
    )


def conditional_rates(
    episodes: Sequence[Mapping[str, object]], *, credit: str = "at_end"
) -> tuple[PhaseRate, ...]:
    """Per-phase rates, conditioned on reaching the phase.

    Keyed on ``phase_id``, never on position: under a free order "phase 2"
    is a different phase in each arm, and pairing on position would compare
    unlike things.
    """
    attempted: dict[str, int] = {}
    succeeded: dict[str, int] = {}
    order: list[str] = []
    for ep in episodes:
        for rec in (ep.get("phase_records") or ()):
            pid = rec["phase_id"]
            if pid not in attempted:
                attempted[pid] = succeeded[pid] = 0
                order.append(pid)
            if not rec["attempted"]:
                continue
            attempted[pid] += 1
            if _credited(rec, credit):
                succeeded[pid] += 1
    return tuple(PhaseRate(pid, attempted[pid], succeeded[pid]) for pid in order)


@dataclass(frozen=True)
class Traversal:
    """The order one arm actually attempted its phases in.

    Read from `start_step`, not from the declaration: under a free order the
    declaration constrains which orders are legal and the policy picks one
    from those, so the order a run took is a measurement.
    """

    arm: str
    order: tuple[str, ...]

    def render(self) -> str:
        return f"{self.arm}: {' -> '.join(self.order) or '(no phases recorded)'}"


def traversal_of(arm: str, episodes: Sequence[Mapping[str, object]]) -> Traversal:
    """The modal order across an arm's episodes, by first attempt.

    Ordered by `start_step`, with unattempted phases left out: a phase never
    reached has no position, and giving it one would invent an order the run
    did not take.
    """
    first: dict[str, int] = {}
    for ep in episodes:
        for rec in (ep.get("phase_records") or ()):
            if not rec.get("attempted"):
                continue
            start = rec.get("start_step")
            if start is None:
                continue
            pid = rec["phase_id"]
            if pid not in first or start < first[pid]:
                first[pid] = int(start)
    return Traversal(arm=arm, order=tuple(
        pid for pid, _ in sorted(first.items(), key=lambda kv: (kv[1], kv[0]))))


def orders_differ(traversals: Sequence[Traversal]) -> str | None:
    """Rule 4, and it is narrow on purpose: the CHAINED per-phase comparison.

    Order affects the three readings differently, so one refusal for all
    three would be wrong in two of them:

    - **Episode level**: order does not matter. Both arms got the same scene,
      task and budget; the order is how each arm chose to spend it, which is
      part of what is being evaluated rather than a confound.
    - **Per phase, chained**: order matters, and this is the refusal.
      ``P(phase 3 | prefix)`` conditions on the phases that came before it, so
      two arms with different prefixes are not estimating the same quantity.
      Averaging them produces a number that is about neither arm.
    - **Per phase, isolated**: order does not matter. Every phase starts from
      the canonical state by construction, so there is no prefix to condition
      on.

    Returns the refusal for the chained reading, or None. Episode-level and
    isolated callers do not consult it.
    """
    seen = {t.arm: t.order for t in traversals}
    distinct = {t.order for t in traversals}
    if len(distinct) <= 1:
        return None
    shown = "; ".join(f"{arm}: {' -> '.join(order)}"
                      for arm, order in sorted(seen.items()))
    return (
        "the chained per-phase comparison is refused: the arms attempted their "
        f"phases in different orders ({shown}). A chained rate is "
        "P(phase | the phases before it), so arms with different prefixes are "
        "not estimating the same quantity and averaging them describes neither. "
        "The episode-level comparison is unaffected, and so is the isolated "
        "per-phase comparison, where every phase starts from the canonical "
        "state."
    )


def compounding(
    isolated: Sequence[PhaseRate], chained_episode_rate: float
) -> float:
    """``prod(P(isolated_k)) - P(chained)``.

    POSITIVE means the policy leaves the scene worse than canonical, and
    later phases pay for a state earlier ones produced. Near ZERO means the
    phases are genuinely independent and can be reasoned about separately.

    NEGATIVE is possible and worth surfacing rather than clamping: the
    policy's own end state suits it better than the canonical one does,
    which says the canonical state is not neutral.

    A product over phases, so order does not enter it and the metric
    survives a free ordering unchanged.
    """
    missing = [r.phase_id for r in isolated if r.rate is None]
    if missing:
        raise PhaseError(
            f"no isolated rate for {missing}: the product is over every phase, "
            "and a phase never attempted even in isolation means the canonical "
            "start states do not cover it. Report that rather than dropping the "
            "phase, which would silently compute a product over a different "
            "task."
        )
    product = 1.0
    for r in isolated:
        product *= r.rate
    return product - chained_episode_rate


@dataclass(frozen=True)
class CompoundingReading:
    """What a set of compounding values says, and about what.

    The sign is read per arm; the PATTERN across arms is read about the
    scene. One arm negative on one scene is a fact about that arm -- its own
    end state suits it better than the canonical one. Most arms negative on
    the SAME scene is not a fact about any of them: it says the canonical
    state for that scene is harder than what policies actually produce,
    which means it is mis-authored.

    So the metric does double duty, and the two readings are distinguished
    by whether the sign is consistent across arms. Surfaced here rather than
    left for a reader to notice, because the mis-authored case looks exactly
    like a run of unlucky policies.
    """

    scene_id: str
    by_arm: tuple[tuple[str, float], ...]
    #: Which of the three methods produced the canonical state this number is
    #: measured against -- hand-authored, replayed from a successful episode,
    #: or solver-generated. Compounding is a comparison WITH that state, so
    #: the number does not mean the same thing under a different method, and a
    #: reader who is not told which one cannot read it. It is in the scene
    #: hash already; this puts it where the number is.
    method: str | None = None

    @property
    def negative_arms(self) -> tuple[str, ...]:
        return tuple(a for a, v in self.by_arm if v < 0)

    @property
    def suspects_catalog(self) -> bool:
        if len(self.by_arm) < 2:
            return False
        return len(self.negative_arms) * 2 > len(self.by_arm)

    def as_finding(self) -> Finding:
        return Finding(about="catalog" if self.suspects_catalog else "arm",
                       text=self.render())

    def render(self) -> str:
        shown = ", ".join(f"{a}={v:+.3f}" for a, v in self.by_arm)
        against = (f" Measured against a canonical state authored by "
                   f"{self.method}." if self.method else
                   " The canonical state's method is unrecorded, so what this "
                   "is measured against is unknown.")
        if self.suspects_catalog:
            return (
                f"scene {self.scene_id}: compounding is negative for "
                f"{len(self.negative_arms)} of {len(self.by_arm)} arms "
                f"({shown}). A single arm below zero says its own end state "
                "suits it better than the canonical one; most arms below zero "
                "on one scene says the canonical state for that scene is "
                "harder than what policies actually produce, and is "
                "mis-authored. This reads as a catalog finding, not a policy "
                "one." + against
            )
        if self.negative_arms:
            return (
                f"scene {self.scene_id}: {', '.join(self.negative_arms)} below "
                f"zero ({shown}) -- that arm's own end state suits it better "
                "than the canonical one." + against
            )
        return (f"scene {self.scene_id}: compounding non-negative for all "
                f"arms ({shown})." + against)


def read_compounding(
    scene_id: str, by_arm: Mapping[str, float], *, method: str | None = None
) -> CompoundingReading:
    return CompoundingReading(scene_id=scene_id,
                              by_arm=tuple(sorted(by_arm.items())),
                              method=method)


def require_canonical_states(
    scene_id: str, phases: Sequence[str],
    canonical: Mapping[tuple[str, str], object],
) -> None:
    """Isolated mode needs a canonical post-phase state per (SCENE, phase).

    Per scene, not per phase, and the difference is not bookkeeping. One
    canonical state shared across scenes erases scene difficulty in the
    isolated condition while leaving it present in the chained one, so
    `compounding` -- the difference between them -- silently absorbs it and
    reports a number about the scenes as though it were about the policy.

    Keyed on the pair so that cannot happen by omission: a scene with no
    state of its own is missing, not inheriting.

    Refuses rather than synthesising. A state invented here would be a start
    condition nobody declared, and every isolated rate measured from it would
    be measuring this function's guess.
    """
    missing = [p for p in phases if (scene_id, p) not in canonical]
    if missing:
        raise PhaseError(
            f"isolated mode needs canonical post-phase start states for scene "
            f"{scene_id!r} and the catalog has none for {missing}. They are "
            "keyed per (scene, phase) and not inherited across scenes: one "
            "state shared between scenes would erase scene difficulty in the "
            "isolated condition while leaving it in the chained one, and "
            "compounding would absorb the difference. They are also not "
            "synthesised -- an invented state is a start condition nobody "
            "declared, and the per-phase rates would be measuring it."
        )


__all__ = [
    "PhaseError",
    "PhaseRate",
    "CompoundingReading",
    "Traversal",
    "compounding",
    "orders_differ",
    "read_compounding",
    "traversal_of",
    "conditional_rates",
    "require_canonical_states",
]
