"""Assembling a curve from several comparisons.

``compare`` reads one comparison directory and answers one question: did these
checkpoints differ on this experiment. A sweep asks a different one -- how does
a difference move as a treatment varies -- and that needs several comparisons
read together.

**A separate verb, deliberately.** ``compare`` refusing incomparable things is
its whole character: mismatched scene hashes, mismatched harness surfaces, two
task hashes under one task id. Making it read across directories would blur
what it refuses, because "these do not join" is its answer and this verb's
premise is that several plans *do* join. Two verbs, two questions.

Why a sweep is several plans
----------------------------

A level sweep repeats the same base scenarios at each level, and that
repetition is what makes the curve joinable. The vla-eval harness identifies an
episode by its own counter, so a single plan holding six levels of the same five
init states presents ``[0,0,0,0,0,0,1,1,...]`` where the harness will run
``[0,1,2,...]`` -- every row would claim a start state that never executed, and
the index contract refuses it before anything runs.

The alternative was making the level a scenario parameter so the indices stay
distinct. That puts the level in two places -- in ``params`` and in the spec
that references it -- which can disagree, and nothing would catch it: the
planner does not interpret scenario parameters and the perturbation path does
not read them. Same shape as a catalog naming ``cube_x`` against an adapter
reading ``vial_x``.

So: six treatments over one shared base, related by a join rather than by a
shared identity. Which is what ``base_scenario_hash`` was defined for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .schema.errors import RefractalError

__all__ = ["SweepError", "SweepPoint", "Sweep", "assemble"]


class SweepError(RefractalError):
    """These comparisons cannot be assembled into one curve."""


@dataclass
class SweepPoint:
    """One level, and what the checkpoints did at it."""

    level: float | None
    plan_id: str
    #: checkpoint -> (successes, attempts) over episodes that EXPERIENCED the
    #: level. An episode that outran its trigger is excluded and counted below.
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    episodes: int = 0
    excluded_unfired: int = 0

    def rate(self, checkpoint: str) -> float | None:
        wins, of = self.counts.get(checkpoint, (0, 0))
        return wins / of if of else None


@dataclass
class Sweep:
    checkpoints: list[str]
    points: list[SweepPoint]
    bases: set[str]
    notes: list[str] = field(default_factory=list)

    @property
    def band(self) -> tuple[float | None, float | None]:
        """The levels between which behaviour actually changes.

        Reported BEFORE any verdict, and it is the finding in its own right.
        Below some level everything fails, which measures the object's weight.
        Above some level nothing changes. If every level gives the same answer
        there is no band, and saying so is the result rather than a failed run.
        """
        moving = [
            p for p in self.points
            if p.level is not None
            and any(0.0 < (p.rate(c) or 0.0) < 1.0 for c in self.checkpoints)
        ]
        if not moving:
            return (None, None)
        levels = sorted(p.level for p in moving)
        return (levels[0], levels[-1])

    @property
    def degenerate(self) -> str | None:
        """Why this sweep has no band, when it has none."""
        rates = [
            p.rate(c) for p in self.points for c in self.checkpoints
            if p.rate(c) is not None
        ]
        if not rates:
            return "no level had a single episode that experienced it"
        if all(r == 0.0 for r in rates):
            return "every level scored 0% -- at and below this range the task is impossible"
        if all(r == 1.0 for r in rates):
            return "every level scored 100% -- the whole range is above where it matters"
        return None


def assemble(
    comparisons: Iterable[tuple[str, Iterable[Mapping[str, Any]]]],
    *,
    checkpoints: list[str],
) -> Sweep:
    """Join several comparisons into one curve, or refuse.

    ``comparisons`` is ``(plan_id, rows)`` per comparison directory.

    Refuses when the plans do not share a base, the same way ``compare`` refuses
    a mismatched ``scene_hash``. Joining six runs assumes they are six
    treatments over one set of scenarios, and nothing else checks that -- a
    curve assembled from plans whose bases differ is six unrelated experiments
    plotted on one axis, which looks exactly like a result.
    """
    from .compare.pairing import experienced_its_level

    points: list[SweepPoint] = []
    bases_per_plan: dict[str, set[str]] = {}
    notes: list[str] = []

    for plan_id, rows in comparisons:
        rows = list(rows)
        if not rows:
            raise SweepError(f"comparison {plan_id[:19]}... has no episodes")
        bases_per_plan[plan_id] = {r["base_scenario_hash"] for r in rows}
        levels = {r.get("perturbation_level") for r in rows}
        if len(levels) > 1:
            raise SweepError(
                f"comparison {plan_id[:19]}... holds {len(levels)} levels "
                f"({sorted(l for l in levels if l is not None)}). A point on a "
                "curve is one level; a comparison holding several is not a point."
            )
        point = SweepPoint(level=next(iter(levels)), plan_id=plan_id)
        for row in rows:
            if row["is_infra_failure"]:
                continue
            if experienced_its_level(row) is False:
                point.excluded_unfired += 1
                continue
            wins, of = point.counts.get(row["checkpoint_id"], (0, 0))
            point.counts[row["checkpoint_id"]] = (
                wins + int(bool(row["success"])), of + 1)
            point.episodes += 1
        points.append(point)

    distinct = {frozenset(b) for b in bases_per_plan.values()}
    if len(distinct) > 1:
        sizes = {p[:19]: len(b) for p, b in bases_per_plan.items()}
        raise SweepError(
            "these comparisons do not share a base. A sweep is several "
            "treatments over ONE set of scenarios, joined on "
            "base_scenario_hash; plans whose bases differ are unrelated "
            "experiments plotted on one axis, which looks exactly like a "
            f"result. Base counts per plan: {sizes}"
        )

    excluded = sum(p.excluded_unfired for p in points)
    if excluded:
        notes.append(
            f"{excluded} episode(s) were assigned a level they never "
            "experienced and are not on the curve. Fast episodes escape a late "
            "trigger more often, so including them would favour whichever "
            "checkpoint finishes sooner."
        )

    points.sort(key=lambda p: (p.level is None, -(p.level or 0.0)))
    return Sweep(
        checkpoints=checkpoints,
        points=points,
        bases=set(next(iter(bases_per_plan.values()))),
        notes=notes,
    )
