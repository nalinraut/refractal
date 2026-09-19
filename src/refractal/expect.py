"""Expected duration from a prior run's step distribution.

The companion to the makespan bound, and deliberately a different kind of number.

``plan.estimated_seconds`` is a **bound**: every episode costed at its full step
limit. Correct for deciding whether a night's compute is enough, and
systematically high, because episodes that succeed stop early. Measured once: a
220-step plan bounded at 89 minutes ran in 49, because episodes averaged 121
steps.

This computes the other number -- what it will probably take -- from how long
episodes actually ran last time.

Why it is not in ``plan.json``
------------------------------

A plan is portable: the same experiment on any machine. This number is fitted to
a particular prior run on particular hardware with particular checkpoints, so it
is **render-time by the project's own rule** -- anything that differs between two
people running the same experiment. It is printed and never stored, and
``plan_id`` cannot see it.

Why it can be optimistic exactly when that hurts
------------------------------------------------

Episode length is a property of the **policy**, not of the task. A worse
checkpoint times out more often and therefore runs *longer*, so a distribution
borrowed from a better one under-estimates -- in precisely the case where
somebody is waiting on a slow run and wants to know when it ends.

That is the difference from the step-budget prediction, which used the same move
and held to 2.5 percentage points: a step *budget* is a property of the task and
transfers across checkpoints; a step *distribution* is a property of the policy
and does not. So this reports which checkpoints it learned from and says so when
they are not the ones being planned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Mapping, Sequence

from .schema.plan import Plan


@dataclass
class Expectation:
    """An expected duration, and everything needed to distrust it."""

    seconds: int
    #: Episodes the estimate was learned from.
    sample: int
    #: Tasks in this plan that found a match in the prior run.
    matched_tasks: int
    total_tasks: int
    #: Checkpoints the prior run used, and whether they are this plan's.
    learned_from: list[str] = field(default_factory=list)
    planning_for: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def checkpoints_differ(self) -> bool:
        return sorted(self.learned_from) != sorted(self.planning_for)


def expected_seconds(
    plan: Plan, prior: Sequence[Mapping[str, Any]]
) -> Expectation | None:
    """Expected wall clock for ``plan``, learned from ``prior`` episode rows.

    Matched **by ``task_id``**, which is a label rather than an identity, and that
    is a deliberate compromise worth naming: ``task_hash`` would be the honest
    key, and it moves when ``max_steps`` moves -- so it never matches across
    exactly the change this estimate is most useful for, a re-plan at a different
    step budget. A label is the only join available, and a task renamed between
    the two runs silently falls back to the pooled figure.

    Every prior episode is capped at this plan's ``max_steps`` **individually,
    before averaging**. A prior run under a looser budget contains episodes longer
    than this plan permits; truncating each one is what this plan would actually
    do to it. Capping the mean instead gives a different and wrong answer --
    measured on the LIBERO run, 100 against the correct 95 -- because it lets long
    episodes pull the average up before the cap that would have stopped them is
    applied. Transform then aggregate, not the reverse.
    """
    usable = [r for r in prior if r.get("steps") and not r.get("is_infra_failure")]
    if not usable:
        return None

    by_task: dict[str, list[int]] = {}
    for row in usable:
        by_task.setdefault(str(row["task_id"]), []).append(int(row["steps"]))

    episodes = [
        e for scene in plan.scenes for worker in scene.workers for e in worker.episodes
    ]
    if not episodes:
        return None

    shape_by_scene = {s.scene_id: s.resource_shape for s in plan.scenes}
    scene_of = {
        e.episode_id: s.scene_id
        for s in plan.scenes
        for w in s.workers
        for e in w.episodes
    }

    all_steps = [s for v in by_task.values() for s in v]
    total = 0.0
    matched = set()
    for episode in episodes:
        prior_steps = by_task.get(episode.task_id)
        if prior_steps is not None:
            matched.add(episode.task_id)
        else:
            prior_steps = all_steps
        # Cap EACH prior episode at this plan's limit, then average. Capping the
        # average instead lets episodes this plan would have stopped pull it up.
        steps = mean(min(s, episode.max_steps) for s in prior_steps)
        shape = shape_by_scene[scene_of[episode.episode_id]]
        total += steps / 1000.0 * shape.sec_per_1k_steps

    planned_tasks = {e.task_id for e in episodes}
    expectation = Expectation(
        seconds=int(total),
        sample=len(usable),
        matched_tasks=len(matched),
        total_tasks=len(planned_tasks),
        learned_from=sorted({str(r["checkpoint_id"]) for r in usable}),
        planning_for=sorted({e.checkpoint_id for e in episodes}),
    )

    if expectation.checkpoints_differ:
        expectation.notes.append(
            f"learned from {expectation.learned_from} but planning "
            f"{expectation.planning_for}. Episode length is a property of the policy: "
            "a worse checkpoint times out more often and runs LONGER, so this will "
            "under-estimate exactly when somebody is waiting on it."
        )
    if expectation.matched_tasks < expectation.total_tasks:
        expectation.notes.append(
            f"only {expectation.matched_tasks} of {expectation.total_tasks} task(s) "
            "matched by task_id; the rest fall back to the pooled mean. Task ids are "
            "labels, so a rename since the prior run looks like a miss."
        )
    return expectation


__all__ = ["Expectation", "expected_seconds"]
