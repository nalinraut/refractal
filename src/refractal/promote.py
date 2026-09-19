"""Reusing a smaller run's results inside a larger one, when they nest.

``tier`` is in ``plan_id``, so a smoke run and a full run of one catalog are
different experiments with different ids, and their results do not join. That is
the correct reading -- every other field in ``experiment_identity`` was decided
by "was this the same experiment" rather than "may these rows be averaged" -- and
it has a cost: the smoke set is a strict subset of the full set, every episode in
it is identical to its counterpart, and re-running it is pure waste.

Promotion is that reuse, made explicit. It does not weaken the identity: the two
plans keep their different ids, `compare` is told to pool them, and the pooling
is recorded in the verdict. Somebody asks for it out loud rather than getting it
because two ids happened to collide.

Why the nesting check is exact rather than approximate
------------------------------------------------------

``episode_id`` is derived from ``(scene_hash, task_hash, scenario_hash, seed,
checkpoint_id)``. **Tier is not in it.** So a smoke episode and the full run's
counterpart already carry the same id -- the subsetting changes which episodes
exist, not what any of them is.

That makes the condition exact: promotion is legal when every promoted episode id
is one the target plan actually contains. No tolerance, no heuristic, and nothing
to get subtly wrong. If the catalog changed between the runs, the hashes moved,
the ids do not match, and the refusal is immediate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .schema.errors import RefractalError
from .schema.plan import Plan


class PromotionError(RefractalError):
    """The promoted results do not nest inside the target plan."""


@dataclass
class Promotion:
    """What was reused, recorded so a verdict can say so."""

    from_plan_id: str
    #: ROWS pooled, not episodes. The first version called this ``episodes`` and
    #: counted rows, which is how a duplicate got past review: the name asserted
    #: the two were the same and nothing checked.
    rows: int
    #: Distinct episode ids among those rows. Equal to ``rows`` unless the
    #: promoted run contains a doubled episode, which is now refused -- kept
    #: separate so the two can be compared rather than assumed equal.
    episodes: int
    #: Episodes the target plan still has to run after this.
    remaining: int
    checkpoints: list[str] = field(default_factory=list)


def check_nesting(plan: Plan, promoted: Sequence[Mapping[str, Any]]) -> set[str]:
    """Episode ids in ``promoted`` that the target plan does not contain.

    Empty means every promoted id is a planned id. That is necessary and **not
    sufficient** -- see ``duplicated_in``, which asks the other question a set
    cannot.
    """
    planned = {
        e.episode_id for s in plan.scenes for w in s.workers for e in w.episodes
    }
    return {str(r["episode_id"]) for r in promoted} - planned


def duplicated_in(promoted: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Episode ids appearing more than once in ``promoted``, and how often.

    A set answers "which ids are here". It cannot answer "is any id here twice",
    and those are different questions -- which is why ``episode_ids_in`` returns
    a list and ``verify_written`` compares multisets.

    Missing it here admitted a doubled row into a pooled comparison, where it
    would be counted twice and weight one scenario double. ``compare`` blocks on
    duplicates downstream, so nothing corrupts -- but it blocks naming "duplicate
    rows" rather than "the run you promoted from has them", which sends the reader
    to the wrong place.
    """
    counts: dict[str, int] = {}
    for row in promoted:
        key = str(row["episode_id"])
        counts[key] = counts.get(key, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def promote(
    plan: Plan,
    target_rows: Sequence[Mapping[str, Any]],
    promoted_rows: Sequence[Mapping[str, Any]],
    from_plan_id: str,
) -> tuple[list[Mapping[str, Any]], Promotion]:
    """Pool ``promoted_rows`` into ``target_rows``, or refuse with the reason."""
    if not promoted_rows:
        raise PromotionError(
            f"no episodes found for {from_plan_id}. Promotion needs a run that "
            "actually happened; check the plan id and the results URI."
        )

    doubled = duplicated_in(promoted_rows)
    if doubled:
        worst = sorted(doubled.items(), key=lambda kv: -kv[1])[:2]
        raise PromotionError(
            f"the run being promoted contains {len(doubled)} duplicated episode id(s), "
            f"for example {[f'{k[:22]}... x{v}' for k, v in worst]}. Pooling them would "
            "count one scenario twice and weight it double in the comparison.\n\n"
            "This is a defect in that run rather than in the nesting: a resumed run "
            "that wrote a second part file for a group it had already written. "
            "`compare` would block on it downstream, naming 'duplicate rows' rather "
            "than the run that supplied them."
        )

    stray = check_nesting(plan, promoted_rows)
    if stray:
        example = sorted(stray)[:2]
        raise PromotionError(
            f"{len(stray)} of {len(promoted_rows)} promoted episode(s) are not in this "
            f"plan, for example {example}. The runs do not nest.\n\n"
            "episode_id covers scene_hash, task_hash, scenario_hash, seed and "
            "checkpoint -- and NOT tier. So a genuine subset matches exactly, and a "
            "mismatch means something other than the tier differs: a changed task, a "
            "regenerated scenario set, a different seed count, a moved catalog. "
            "Pooling them would average two experiments."
        )

    # The target's own rows win: if an episode was run under both plans, the one
    # belonging to this experiment is the one to keep. Promotion fills gaps, it
    # does not overwrite.
    have = {str(r["episode_id"]) for r in target_rows}
    fresh = [r for r in promoted_rows if str(r["episode_id"]) not in have]

    planned = {
        e.episode_id for s in plan.scenes for w in s.workers for e in w.episodes
    }
    pooled = list(target_rows) + fresh
    return pooled, Promotion(
        from_plan_id=from_plan_id,
        rows=len(fresh),
        episodes=len({str(r["episode_id"]) for r in fresh}),
        remaining=len(planned - have - {str(r["episode_id"]) for r in fresh}),
        checkpoints=sorted({str(r["checkpoint_id"]) for r in fresh}),
    )


__all__ = ["Promotion", "PromotionError", "check_nesting", "promote"]
