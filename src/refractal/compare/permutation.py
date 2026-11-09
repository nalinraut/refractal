"""Within-pair permutation over scenes, and the sweeps that carry no p-value.

Pairing here is exact -- two episodes are comparable only when scene, task,
scenario and seed all match -- so the null distribution can be BUILT rather
than assumed. Flip which arm is which, recompute the statistic, and the
spread of those recomputations is what the observed value is read against.
No normal approximation, no asymptotics, no distributional assumption about
the statistic at all.

Why the pairing unit is the CELL, not the episode
-------------------------------------------------

There IS a per-episode paired statistic available: the difference in
``step_to_success`` on seeds where both arms succeeded. It would be more
powerful than the cell grain, and it is deliberately not used.

The reason is SELECTION, not arithmetic. An episode-grain difference only
exists where both arms succeeded, so the pairs available are chosen by the
very thing being measured: a fast-but-unreliable arm contributes only its
easy seeds, and the test then compares the two arms on the subset one of them
found tractable. That is the prefix-censoring problem arriving in a new place.

Cell grain avoids it because the functional absorbs the failures rather than
dropping them -- a terminal takes the penalty, and a median over a cell where
fewer than half succeeded is undefined and the cell says so. Every episode
reaches the statistic one way or another.

The episode-grain test remains available for later. If it is ever built, the
guard it needs is a declared minimum overlap between the arms' success sets,
with the realized overlap reported beside the result.

Why the permutation unit is the SCENE
-------------------------------------

Episodes cluster within a scene: a scenario's difficulty is shared by every
seed run on it, and two seeds of one scene are not independent draws. Flipping
arm labels pair-by-pair would assume they are, and would produce a null
narrower than the truth -- an anti-conservative test, which is the failure
mode that matters.

``clustered_bootstrap`` already resamples whole scenes for exactly this
reason. The permutation has to match it, so a scene's pairs all flip together.

What that costs, stated rather than discovered
----------------------------------------------

With ``c`` scenes there are ``2**c`` sign assignments, so the smallest
two-sided p-value reachable is ``2**(1-c)``:

======  ==========
scenes  floor
======  ==========
4       0.125
5       0.0625
6       0.031
8       0.008
======  ==========

Below six scenes the test cannot produce ``p < 0.05`` however many episodes
ran. That is refused rather than warned about: a test that cannot reach the
threshold it will be read against is not a weak result, it is an unanswerable
question, and returning 0.0625 invites someone to read it as "nearly".
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .distribution import (
    Crossings,
    DistributionError,
    GhostPenalty,
    across_cells,
    estimate,
)
from .pairing import Unit

#: Scenes needed before a two-sided permutation p can fall under 0.05.
#: ``2**(1-c) < 0.05`` first holds at c = 6.
MIN_SCENES_FOR_ALPHA_05 = 6

#: Above this many scenes, enumerating every assignment stops being free and
#: the test samples instead. 2**20 is about a million recomputations of a
#: mean, which is seconds; 2**21 is not.
MAX_EXACT_SCENES = 20


class PermutationError(ValueError):
    """The permutation test was asked something it cannot answer."""


@dataclass(frozen=True)
class PermutationResult:
    """What the test concluded, and by which route."""

    statistic: str
    observed: float
    p_value: float
    #: Scenes, which is what bounds resolution -- not episodes, and not pairs.
    scenes: int
    #: Cells contributing, after dropping those where either arm had no value.
    cells: int
    #: "exact" or "sampled". Never inferred by the reader from the numbers.
    path: str
    #: Set only on the sampled path, and then always.
    samples: int | None = None
    seed: int | None = None
    alternative: str = "two-sided"
    #: Pairs dropped because one arm had no non-APPARATUS episode there.
    dropped_pairs: int = 0
    #: Cells dropped because the functional was undefined for an arm, which
    #: for the median means fewer than half its episodes succeeded.
    undefined_cells: int = 0

    @property
    def floor(self) -> float:
        """The smallest p this design could have produced."""
        return 2.0 ** (1 - self.scenes)


def permutation_test(
    units: Sequence[Unit],
    arm_a: str,
    arm_b: str,
    *,
    statistic: str,
    ghost_penalty: GhostPenalty | None = None,
    bases: Mapping[tuple, object] | None = None,
    alternative: str = "two-sided",
    samples: int = 10000,
    seed: int | None = None,
    **kwargs,
) -> PermutationResult:
    """Permute arm labels within scenes; read the observed statistic against it.

    The arms are ``arm_a``/``arm_b`` rather than ``a``/``b`` because the
    functional's own keyword arguments are forwarded through ``**kwargs``, and
    ``success_at`` takes a parameter called ``b``. Short arm names would make
    ``permutation_test(units, "A", "B", statistic="success_at", b=300)`` a
    duplicate-argument TypeError -- a collision between two unrelated
    vocabularies, found the first time the test was pointed at success_at.

    ``statistic`` is any Phase 2 functional by name -- median, rmst,
    success_at -- rather than a hardcoded difference. The test does not know
    what it is testing, which is what lets the same machinery answer "is the
    median different" and "is the restricted mean different" without a second
    implementation of the permutation itself.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise PermutationError(
            f"alternative={alternative!r}; expected two-sided, greater or less.")

    # Per-cell values once. Macro-averaging is a mean over cells, so the
    # statistic is the mean of per-cell DIFFERENCES -- and flipping a scene
    # negates the differences of its cells. That turns 2**c recomputations of
    # the estimator into 2**c sums over a short vector.
    diffs: list[float] = []
    scenes_of: list[str] = []
    dropped = undefined = 0
    for unit in units:
        if arm_a not in unit.outcomes or arm_b not in unit.outcomes:
            dropped += 1
            continue
        cell_bases = {
            seed_: (bases or {}).get((unit.key, seed_))
            for seed_ in set(unit.outcomes[arm_a]) | set(unit.outcomes[arm_b])
        }
        try:
            ea = estimate(unit, arm_a, ghost_penalty=ghost_penalty, bases=cell_bases)
            eb = estimate(unit, arm_b, ghost_penalty=ghost_penalty, bases=cell_bases)
        except DistributionError:
            raise
        from .distribution import FUNCTIONALS

        fn = FUNCTIONALS[statistic]
        va, vb = fn(ea, **kwargs), fn(eb, **kwargs)
        if va is None or vb is None:
            # The median of a cell where fewer than half succeeded is
            # undefined, not large. Dropping the cell and SAYING SO is the
            # only honest move: coercing it to the budget would invent an
            # observation at the extreme of the very distribution under test.
            undefined += 1
            continue
        diffs.append(float(va) - float(vb))
        scenes_of.append(unit.key.scene_id)

    if not diffs:
        raise PermutationError(
            f"no cell yielded {statistic!r} for both {arm_a!r} and {arm_b!r}: "
            f"{undefined} cell(s) undefined, {dropped} pair(s) dropped.")

    scenes = sorted(set(scenes_of))
    c = len(scenes)
    if c < MIN_SCENES_FOR_ALPHA_05:
        raise PermutationError(
            f"{c} scene(s) cannot support a permutation test read against "
            f"alpha = 0.05. The permutation unit is the scene, so the smallest "
            f"two-sided p this design can produce is 2**(1-{c}) = "
            f"{2.0 ** (1 - c):.4f}; {MIN_SCENES_FOR_ALPHA_05} scenes are needed "
            f"to reach 0.05. This is not a weak result, it is a question the "
            f"design cannot answer -- add scenes rather than episodes, because "
            f"episodes within a scene are not independent and do not raise the "
            f"floor."
        )

    index = {s: i for i, s in enumerate(scenes)}
    by_scene = [index[s] for s in scenes_of]
    observed = sum(diffs) / len(diffs)

    def stat_for(signs: Sequence[int]) -> float:
        return sum(d * signs[s] for d, s in zip(diffs, by_scene)) / len(diffs)

    def more_extreme(value: float) -> bool:
        if alternative == "two-sided":
            return abs(value) >= abs(observed) - 1e-12
        if alternative == "greater":
            return value >= observed - 1e-12
        return value <= observed + 1e-12

    if c <= MAX_EXACT_SCENES:
        hits = total = 0
        for bits in itertools.product((1, -1), repeat=c):
            total += 1
            if more_extreme(stat_for(bits)):
                hits += 1
        # Exact: the observed assignment IS one of the enumerated ones, so no
        # correction is needed or correct here. hits >= 1 always.
        return PermutationResult(
            statistic=statistic, observed=observed, p_value=hits / total,
            scenes=c, cells=len(diffs), path="exact",
            alternative=alternative, dropped_pairs=dropped,
            undefined_cells=undefined)

    if seed is None:
        raise PermutationError(
            f"{c} scenes exceeds the {MAX_EXACT_SCENES} this will enumerate, so "
            "the test must sample -- and a sampled p-value without a recorded "
            "seed cannot be reproduced. Pass seed=.")
    rng = random.Random(seed)
    hits = 0
    for _ in range(samples):
        bits = [1 if rng.random() < 0.5 else -1 for _ in range(c)]
        if more_extreme(stat_for(bits)):
            hits += 1
    # PLUS ONE, on both halves. Without it the smallest reportable p is 0,
    # which is a claim no finite resampling supports: B draws can only bound
    # the tail from above. The observed assignment is itself a valid
    # permutation and is what the extra count represents.
    p = (1 + hits) / (1 + samples)
    return PermutationResult(
        statistic=statistic, observed=observed, p_value=p, scenes=c,
        cells=len(diffs), path="sampled", samples=samples, seed=seed,
        alternative=alternative, dropped_pairs=dropped,
        undefined_cells=undefined)


def contingency_at(
    units: Sequence[Unit], arm_a: str, arm_b: str, rule: str, budget: int
):
    """McNemar's 2x2, with each episode's bit recomputed at step ``budget``.

    Mechanically this changes nothing about the test: same discordant pairs,
    same exact binomial, same clustered bootstrap for the interval. What
    changes is where the bit comes from -- ``step_to_success <= budget``
    rather than the recorded ``success``.

    Which is exactly why the two forms have different reach, and the
    difference is load-bearing:

    ``contingency(...)``
        reads the recorded ``success`` bit. WORKS on pre-v2 rows, because
        that column has always existed. The published comparison is pinned to
        a catalog_hash and must keep running unchanged, so this path must
        never acquire a dependency on the new columns.

    ``contingency_at(..., budget=b)``
        needs ``step_to_success`` to know whether the predicate was satisfied
        BY b. Pre-v2 rows do not have it, and the answer is unknowable rather
        than approximable: an episode recorded as a success says nothing about
        when. So this refuses.
    """
    from .stats import Discordance

    both_fail = only_a = only_b = both_pass = 0
    for unit in units:
        verdicts = {}
        for arm in (arm_a, arm_b):
            seeds = unit.outcomes[arm]
            bits = []
            for seed, o in sorted(seeds.items()):
                if o.is_apparatus:
                    continue
                if o.step_budget is None:
                    raise PermutationError(
                        f"{unit.key.scenario_hash}: {arm!r} has an episode with no "
                        "step_budget, so a bit cannot be recomputed at a chosen "
                        "budget. Rows written before schema version 2 record "
                        "whether an episode succeeded but not when, and the "
                        "no-argument form is the one that works on them."
                    )
                if budget > o.step_budget:
                    raise PermutationError(
                        f"{unit.key.scenario_hash}: budget={budget} exceeds this "
                        f"episode's step_budget ({o.step_budget}). Past it nothing "
                        "was observed."
                    )
                bits.append(o.step_to_success is not None
                            and o.step_to_success <= budget)
            if not bits:
                break
            verdicts[arm] = (all(bits) if rule == "all"
                             else sum(bits) * 2 > len(bits))
        if len(verdicts) < 2:
            continue
        a_ok, b_ok = verdicts[arm_a], verdicts[arm_b]
        if a_ok and b_ok:
            both_pass += 1
        elif a_ok:
            only_a += 1
        elif b_ok:
            only_b += 1
        else:
            both_fail += 1
    return Discordance(both_fail=both_fail, only_b_passes=only_b,
                       only_a_passes=only_a, both_pass=both_pass)


@dataclass(frozen=True)
class BudgetCurve:
    """success_at(b) across b, for one arm. DESCRIPTIVE.

    There is no p-value field, and that is the design rather than an
    omission: the type cannot carry one without being changed, so nobody can
    add one at a call site and nobody can read one that was never computed.

    The reason it must not have one: tests at adjacent b are near-perfectly
    correlated -- the same episodes, reclassified by a threshold that moved
    slightly -- so a Holm correction across the sweep is both wrong in spirit
    and brutally conservative in practice. One pre-registered b carries the
    inferential claim; this curve shows the shape around it.
    """

    arm: str
    #: (b, fraction succeeding by b). Ascending in b.
    points: tuple[tuple[int, float], ...]
    cells: int

    def at(self, b: int) -> float | None:
        for step, value in self.points:
            if step == b:
                return value
        return None


def budget_curve(
    units: Sequence[Unit], arm: str, budgets: Sequence[int],
    *, ghost_penalty: GhostPenalty | None = None,
) -> BudgetCurve:
    """success_at over a range of b. Refuses on pre-v2 rows, like any
    b-parameterised call: without step_to_success there is no curve to draw,
    only the single recorded bit repeated."""
    points = []
    cells = 0
    for b in sorted(budgets):
        got = across_cells(units, arm, "success_at",
                           ghost_penalty=ghost_penalty, b=b)
        cells = len(got.per_cell)
        points.append((b, float(got.value)))
    return BudgetCurve(arm=arm, points=tuple(points), cells=cells)


@dataclass(frozen=True)
class Stability:
    """Whether the ranking survives the scoring choices. DESCRIPTIVE.

    Also carries no p-value, for a different reason than the curve: this
    describes where a SCALAR'S VERDICT changes as tau and the penalty move.
    That is a property of the scoring convention, not a hypothesis about the
    policies, and attaching a p to it would invite reading a convention as a
    finding.
    """

    arm_a: str
    arm_b: str
    #: One entry per penalty convention tried.
    by_penalty: tuple[tuple[str, "Crossings"], ...]

    @property
    def invariant(self) -> bool:
        return all(c.invariant for _, c in self.by_penalty)

    def describe(self) -> str:
        if self.invariant:
            names = ", ".join(n for n, _ in self.by_penalty)
            first = self.by_penalty[0][1]
            lead = first.describe(self.arm_a, self.arm_b)
            return f"{lead}, under all penalties tested ({names})"
        return "; ".join(
            f"under {name}: {c.describe(self.arm_a, self.arm_b)}"
            for name, c in self.by_penalty)


def stability(
    units: Sequence[Unit], arm_a: str, arm_b: str, *,
    taus: Sequence[int], penalties: Mapping[str, GhostPenalty],
    statistic: str = "rmst",
) -> Stability:
    """Sweep tau under several penalty conventions; find ALL sign changes.

    Not the first. The gap between two arms is not monotone in tau -- an arm
    still accumulating slow successes grows faster than its asymptotic
    ghost-share slope and flattens once they are exhausted -- so a quantity
    that falls and then rises can cross zero more than once. Naming only the
    first would quote an arbitrary member of a set as though it were the
    answer.
    """
    from .distribution import crossings

    out = []
    for name, penalty in penalties.items():
        va, vb = {}, {}
        for tau in taus:
            pen = float(tau) if penalty == "tau" else penalty
            va[tau] = across_cells(units, arm_a, statistic,
                                   ghost_penalty=pen, tau=tau).value
            vb[tau] = across_cells(units, arm_b, statistic,
                                   ghost_penalty=pen, tau=tau).value
        out.append((name, crossings(va, vb)))
    return Stability(arm_a=arm_a, arm_b=arm_b, by_penalty=tuple(out))


__all__ = [
    "BudgetCurve",
    "MAX_EXACT_SCENES",
    "Stability",
    "budget_curve",
    "stability",
    "contingency_at",
    "MIN_SCENES_FOR_ALPHA_05",
    "PermutationError",
    "PermutationResult",
    "permutation_test",
]
