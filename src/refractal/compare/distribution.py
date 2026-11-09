"""Time-to-success as a distribution, and the scalars read off it.

One estimate per cell, then every scalar is a functional of that estimate.
The alternative -- a list of metrics each walking the episodes itself -- puts
the censoring rule in as many places as there are metrics, and they drift.

No p-values here. This module produces distributions and scalars; the tests
that compare them are Phase 3, and keeping the two apart is what stops a
descriptive sweep from being read as a family of tests.

Why the ECDF and not Kaplan-Meier
---------------------------------

There is no censoring here in the statistical sense, so KM has nothing to do.

Evaluated at a horizon ``tau <= step_budget``, every episode has an OBSERVED
``min(T, tau)``: one that succeeded contributes its ``step_to_success``, and
one that did not contributes exactly ``tau``, because the run is known to have
continued to at least there. Nothing is lost to follow-up, which is the
condition KM exists to handle. With no censored observations the
Kaplan-Meier estimator is algebraically identical to ``1 - ECDF``, so it would
be the same numbers behind a dependency and a vocabulary that invites
reviewers to ask about censoring assumptions that do not apply.

The one genuinely unobserved case is a TERMINAL episode, where ``T`` is not
large-but-unseen, it is INFINITE: success did not happen late, it became
impossible. That is not censoring either, and KM would handle it wrongly by
treating the episode as still at risk. It is handled by ``ghost_penalty``,
which the caller chooses, because what a never-finishing episode contributes
to a mean is a convention rather than a measurement.

Do not add lifelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from ..schema.outcome import Outcome, TerminalBasis
from .pairing import EpisodeOutcome, Unit, UnitKey

#: What a TERMINAL episode contributes, when the caller chooses to drop it
#: rather than charge it. Named rather than None, because None already means
#: "no penalty given" and the two are opposite instructions.
EXCLUDE = "EXCLUDE"

GhostPenalty = float | str | Mapping[TerminalBasis, float | str]


class DistributionError(ValueError):
    """A functional was asked for something the data cannot support."""


@dataclass(frozen=True)
class Estimate:
    """One cell's time-to-success, and what was dropped to get it."""

    key: UnitKey
    checkpoint: str
    #: Observed success times, ascending. One entry per succeeding episode.
    times: tuple[int, ...]
    #: Episodes that did not succeed and are still IN the distribution: budget
    #: exhaustions, plus terminals the caller chose to penalise rather than
    #: exclude. Each carries what it contributes under the chosen penalty.
    ghosts: tuple[float, ...]
    #: The smallest budget in the cell. Every functional is bounded by it.
    budget: int
    #: Dropped and why, reported rather than quietly shrinking the denominator.
    apparatus: int = 0
    excluded: int = 0
    excluded_bases: tuple[str, ...] = ()
    #: The convention this estimate was built under, recorded rather than
    #: passed around.
    #:
    #: It has to be READABLE, not merely supplied: refusing "an RMST
    #: comparison where the arms used different ghost_penalty conventions"
    #: means comparing the two conventions, and a value forwarded by
    #: signature inspection lives nowhere either side can read. Storing it
    #: makes the convention part of the estimate's identity, which is the
    #: thing that rule compares.
    #:
    #: None only when the estimate was built with no penalty at all, which
    #: is legitimate for functionals that need none -- the median does not.
    ghost_penalty: GhostPenalty | None = None

    @property
    def n(self) -> int:
        """Episodes the distribution is over, after exclusions."""
        return len(self.times) + len(self.ghosts)


def _penalty_for(basis: TerminalBasis | None, penalty: GhostPenalty,
                 key: UnitKey) -> float | str:
    """What one TERMINAL contributes, by its basis.

    A single value applies to every terminal. A mapping lets the two bases be
    treated differently, which is the point of recording the basis at all: a
    simulator ending an episode is not the same claim as a task becoming
    impossible, and whether the ranking survives excluding the first instead
    of charging it is a question the functional layer should be able to
    answer rather than assume.
    """
    if isinstance(penalty, Mapping):
        if basis is None:
            raise DistributionError(
                f"{key.scenario_hash}: a TERMINAL episode has no terminal_basis, "
                "and ghost_penalty was given per basis. Either the adapter's "
                "termination map is missing a basis or the penalty should be a "
                "single value."
            )
        if basis not in penalty:
            covered = ", ".join(sorted(b.value for b in penalty))
            raise DistributionError(
                f"{key.scenario_hash}: ghost_penalty covers [{covered}] and this "
                f"cell contains a {basis.value}-basis TERMINAL. A partial mapping "
                "would silently charge one kind and drop the other."
            )
        return penalty[basis]
    return penalty


def estimate(
    unit: Unit,
    checkpoint: str,
    *,
    ghost_penalty: GhostPenalty | None = None,
    bases: Mapping[int, TerminalBasis | None] | None = None,
) -> Estimate:
    """One cell's distribution.

    The cell is the UnitKey, unchanged: scene, task_hash and scenario_hash is
    already the right grain, and it is what keeps a perturbed scenario out of
    the same distribution as its own baseline -- they differ in difficulty by
    construction, so pooling them would blur the thing a sweep is measuring.
    """
    seeds = unit.outcomes[checkpoint]
    budgets = [o.step_budget for o in seeds.values()
               if o.step_budget is not None and not o.is_apparatus]
    missing = [s for s, o in seeds.items()
               if o.step_budget is None and not o.is_apparatus]
    if missing:
        raise DistributionError(
            f"{unit.key.scenario_hash}: {len(missing)} episode(s) of {checkpoint!r} "
            "record no step_budget, so there is no horizon to evaluate against. "
            "Rows written before schema version 2 have none, and it is not "
            "reconstructed from the plan: every functional here is bounded by that "
            "budget, and a bound taken from a file beside the rows is an assumption "
            "rather than a measurement. Re-run to compare distributionally."
        )
    if not budgets:
        raise DistributionError(
            f"{unit.key.scenario_hash}: {checkpoint!r} has no non-APPARATUS episode "
            "in this cell. Every episode was an apparatus failure, so there is no "
            "policy behaviour here to describe."
        )

    times: list[int] = []
    ghosts: list[float] = []
    apparatus = excluded = 0
    excluded_bases: set[str] = set()
    for seed, o in sorted(seeds.items()):
        if o.is_apparatus:
            apparatus += 1
            continue
        if o.outcome is Outcome.SUCCESS:
            times.append(int(o.step_to_success))
            continue
        if o.outcome is Outcome.BUDGET_EXHAUSTED:
            ghosts.append(float(o.step_budget))
            continue
        # Only here does a penalty become necessary. A cell with no TERMINAL
        # needs no convention at all, and the median over such a cell is
        # fully determined without one -- so requiring it up front would make
        # callers invent a value they never use, which is how a convention
        # gets chosen by accident.
        if ghost_penalty is None:
            raise DistributionError(
                f"{unit.key.scenario_hash}: {checkpoint!r} has a TERMINAL episode "
                "and no ghost_penalty. What an episode that can never succeed "
                "contributes is a convention the caller has to choose; there is no "
                "measurement to fall back on."
            )
        basis = (bases or {}).get(seed)
        value = _penalty_for(basis, ghost_penalty, unit.key)
        if value == EXCLUDE:
            excluded += 1
            excluded_bases.add(basis.value if basis else "unknown")
            continue
        ghosts.append(float(value))

    return Estimate(
        key=unit.key, checkpoint=checkpoint,
        times=tuple(sorted(times)), ghosts=tuple(ghosts),
        budget=min(budgets), apparatus=apparatus,
        excluded=excluded, excluded_bases=tuple(sorted(excluded_bases)),
        ghost_penalty=ghost_penalty,
    )


#: name -> functional. A registry rather than a hardcoded list, so a new
#: scalar is a registration and not an edit to every caller -- and so nothing
#: can reach back into the episode rows, because a functional only ever
#: receives an Estimate.
FUNCTIONALS: dict[str, Callable[..., object]] = {}


def functional(name: str):
    def register(fn):
        FUNCTIONALS[name] = fn
        return fn
    return register


def _check_tau(est: Estimate, tau: int | float) -> None:
    """Refuse a horizon past the budget.

    Evaluating beyond it reintroduces censoring by hand: past the budget an
    episode's non-success is no longer an observation, it is an absence of
    one, and the ECDF would read "did not succeed by tau" for episodes that
    were never run that far.
    """
    if tau > est.budget:
        raise DistributionError(
            f"{est.key.scenario_hash}: tau={tau} exceeds the smallest step_budget "
            f"in this cell ({est.budget}). Past the budget nothing was observed, so "
            "a reading there is an assumption about episodes that were stopped."
        )


@functional("success_at")
def success_at(est: Estimate, b: int | float) -> float:
    """Fraction succeeding by step ``b``."""
    _check_tau(est, b)
    if est.n == 0:
        raise DistributionError(f"{est.key.scenario_hash}: no episodes to read.")
    return sum(1 for t in est.times if t <= b) / est.n


@functional("ecdf")
def ecdf(est: Estimate) -> tuple[tuple[int, float], ...]:
    """The curve: ``(step, fraction succeeded by then)`` at each success."""
    if est.n == 0:
        raise DistributionError(f"{est.key.scenario_hash}: no episodes to read.")
    out, seen = [], 0
    for t in est.times:
        seen += 1
        out.append((t, seen / est.n))
    return tuple(out)


@functional("quantile")
def quantile(est: Estimate, q: float) -> float | None:
    """The ``q``-th quantile of time-to-success, or None when undefined.

    NO GHOST PENALTY, and that is the property worth knowing. If more than
    ``q`` of the episodes succeeded, the answer is fixed by observed successes
    alone and nothing a terminal contributes can move it. If fewer did, the
    quantile is genuinely undefined -- it lies somewhere past the budget, or
    at infinity -- and None says so.

    It is NOT coerced to the budget or to tau. Coercing would make a cell
    where 40% succeeded report the same number as one where 49% did, which is
    exactly the distinction the caller came for.
    """
    if not 0 < q < 1:
        raise DistributionError(f"quantile q={q} must be in (0, 1).")
    if est.n == 0:
        raise DistributionError(f"{est.key.scenario_hash}: no episodes to read.")
    need = q * est.n
    if len(est.times) < need:
        return None
    idx = math.ceil(need) - 1
    return float(est.times[max(0, idx)])


@functional("median")
def median(est: Estimate) -> float | None:
    """The 50th percentile, or None when fewer than half succeeded.

    The only convention-free distributional scalar here: it needs no
    ghost_penalty and therefore no defence of one. That makes it a better
    headline than RMST, which always carries a chosen convention into the
    number. Both are exposed; neither is privileged.
    """
    return quantile(est, 0.5)


@functional("rmst")
def rmst(est: Estimate, tau: int | float,
         ghost_penalty: GhostPenalty | None = None) -> float:
    """Restricted mean time to success, over ``[0, tau]``.

    Both arguments are required and neither is defaulted. ``tau`` because a
    restricted mean is meaningless without saying restricted to what, and
    ``ghost_penalty`` because what a never-finishing episode contributes is a
    convention: the paper maps it to tau, which is a choice a caller should
    make rather than inherit.
    """
    # From the estimate when it has one, because that is the value its ghosts
    # were actually built with. Passing a different one here would describe a
    # distribution that does not exist: the ghosts are already fixed.
    if ghost_penalty is None:
        ghost_penalty = est.ghost_penalty
    if ghost_penalty is None:
        raise DistributionError(
            f"{est.key.scenario_hash}: rmst needs ghost_penalty, and this estimate "
            "records none. What an episode that never succeeds contributes to a "
            "mean is a convention, not a measurement, and defaulting it would bury "
            "the choice in this function."
        )
    if (est.ghost_penalty is not None
            and ghost_penalty != est.ghost_penalty):
        raise DistributionError(
            f"{est.key.scenario_hash}: rmst was given ghost_penalty "
            f"{ghost_penalty!r} but this estimate's ghosts were built under "
            f"{est.ghost_penalty!r}. The ghosts are already fixed, so the number "
            "would describe a distribution that was never estimated. Re-estimate "
            "under the penalty you want."
        )
    _check_tau(est, tau)
    if est.n == 0:
        raise DistributionError(f"{est.key.scenario_hash}: no episodes to read.")
    total = sum(min(t, tau) for t in est.times)
    total += sum(min(g, tau) for g in est.ghosts)
    return total / est.n


@dataclass(frozen=True)
class CellResult:
    """One functional, across the cells of one arm."""

    name: str
    per_cell: tuple[tuple[UnitKey, object], ...]
    value: float | None
    #: Cells whose answer was undefined, which is a count rather than a
    #: silence: a macro-average over "the cells that had an answer" is a
    #: different quantity from one over all cells, and the reader needs to
    #: know which they are looking at.
    undefined_cells: int = 0
    apparatus: int = 0
    excluded: int = 0
    excluded_bases: tuple[str, ...] = ()


def across_cells(
    units: Sequence[Unit],
    checkpoint: str,
    name: str,
    *,
    ghost_penalty: GhostPenalty,
    bases: Mapping[tuple[UnitKey, int], TerminalBasis | None] | None = None,
    **kwargs,
) -> CellResult:
    """Estimate per cell, read the functional, macro-average across cells.

    MACRO, never pooled. Pooling episodes from different cells into one
    distribution would weight a cell by how many episodes it happened to get,
    so a scenario run at five seeds would count five times a scenario run at
    one -- and the cells differ in difficulty by design, which is the whole
    reason they are separate cells.
    """
    if name not in FUNCTIONALS:
        raise DistributionError(
            f"unknown functional {name!r}; registered: {sorted(FUNCTIONALS)}")
    fn = FUNCTIONALS[name]

    per_cell, values = [], []
    undefined = apparatus = excluded = 0
    bases_seen: set[str] = set()
    for unit in units:
        cell_bases = {
            seed: (bases or {}).get((unit.key, seed))
            for seed in unit.outcomes.get(checkpoint, {})
        }
        est = estimate(unit, checkpoint, ghost_penalty=ghost_penalty,
                       bases=cell_bases)
        apparatus += est.apparatus
        excluded += est.excluded
        bases_seen.update(est.excluded_bases)
        # No forwarding and no signature inspection. The estimate carries the
        # penalty it was built under, so `rmst` reads it from there -- which
        # survives a decorator, a partial or a rename, none of which an
        # inspection-based forward does, and which gives Phase 6 somewhere to
        # read the convention FROM when it refuses two arms that used
        # different ones.
        got = fn(est, **kwargs)
        per_cell.append((unit.key, got))
        if got is None:
            undefined += 1
        elif isinstance(got, (int, float)):
            values.append(float(got))

    return CellResult(
        name=name,
        per_cell=tuple(per_cell),
        value=(sum(values) / len(values)) if values else None,
        undefined_cells=undefined,
        apparatus=apparatus,
        excluded=excluded,
        excluded_bases=tuple(sorted(bases_seen)),
    )


def clustered_interval(
    units: Sequence[Unit],
    checkpoint: str,
    name: str,
    *,
    ghost_penalty: GhostPenalty | None = None,
    bases: Mapping[tuple, object] | None = None,
    resamples: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
    **kwargs,
) -> tuple[float, float] | None:
    """A per-arm interval, resampling SCENES rather than cells.

    ``clustered_bootstrap`` in stats.py gives the interval on the paired
    DIFFERENCE, which is the right object for a contrast and the wrong one
    for a single arm's rate. This is the one-sample companion, and it
    clusters the same way for the same reason: a scenario's difficulty is
    shared by every seed on it, so resampling cells would treat correlated
    observations as independent and return an interval narrower than the
    truth.

    Returns None when the functional is undefined on too many cells to
    resample meaningfully -- an interval over two surviving cells is not an
    interval.
    """
    import random as _random

    by_scene: dict[str, list[float]] = {}
    for unit in units:
        if checkpoint not in unit.outcomes:
            continue
        cell_bases = {
            s_: (bases or {}).get((unit.key, s_))
            for s_ in unit.outcomes[checkpoint]
        }
        est = estimate(unit, checkpoint, ghost_penalty=ghost_penalty,
                       bases=cell_bases)
        value = FUNCTIONALS[name](est, **kwargs)
        if value is None:
            continue
        by_scene.setdefault(unit.key.scene_id, []).append(float(value))

    scenes = sorted(by_scene)
    if len(scenes) < 3:
        return None
    rng = _random.Random(seed)
    draws = []
    for _ in range(resamples):
        picked = [by_scene[rng.choice(scenes)] for _ in scenes]
        flat = [v for group in picked for v in group]
        draws.append(sum(flat) / len(flat))
    draws.sort()
    lo = draws[max(0, int((alpha / 2) * resamples) - 1)]
    hi = draws[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (lo, hi)


@dataclass(frozen=True)
class Crossings:
    """Where one arm overtakes the other across a range of tau.

    DESCRIPTIVE. No p-value, and the type carries none: this says where a
    ranking changes, not whether the change is significant. The inferential
    half is Phase 3.

    Three shapes, and the third is the interesting one:

    ``invariant``
        No sign change. One arm is ahead across the whole range.

    ``one crossing``
        The ranking inverts once, at a tau the caller can name.

    ``two or more``
        Neither arm dominates anywhere, and "the crossing tau" is not a
        well-defined quantity. Reporting the first one would name an
        arbitrary member of a set.

    The third is not hypothetical. The gap between two arms under
    ``ghost_penalty = tau`` is NOT monotone in tau: an arm is still
    accumulating slow successes early, so its RMST grows faster than its
    asymptotic ghost-share slope, and only flattens once those successes are
    exhausted. A quantity that rises and then falls can cross zero more than
    once, and a finder that stops at the first would silently pick one.
    """

    taus: tuple[int, ...]
    #: ``a - b`` at each tau. Negative means ``a`` is ahead on a
    #: lower-is-better functional.
    gaps: tuple[float, ...]
    #: Each tau at which the sign of the gap changes.
    at: tuple[int, ...]

    @property
    def invariant(self) -> bool:
        return not self.at

    def _sign_before(self, tau: int | None) -> int:
        """The last non-zero sign strictly before ``tau``, or overall.

        NOT ``gaps[0]``. At a small tau every episode contributes exactly
        tau -- successes have not landed and ghosts are charged the horizon
        -- so the first gap is reliably ZERO and reading the lead from it
        reports whichever arm the tie-break happened to favour. Found by the
        end-to-end witness naming the wrong winner.
        """
        for t, g in zip(reversed(self.taus), reversed(self.gaps)):
            if tau is not None and t >= tau:
                continue
            if g:
                return (g > 0) - (g < 0)
        return 0

    def describe(self, a: str, b: str) -> str:
        if self.invariant:
            sign = self._sign_before(None)
            if sign == 0:
                return (f"the two arms are identical over tau in "
                        f"[{self.taus[0]}, {self.taus[-1]}]")
            lead = a if sign < 0 else b
            return (f"ranking invariant over tau in "
                    f"[{self.taus[0]}, {self.taus[-1]}]: {lead} ahead throughout")
        if len(self.at) == 1:
            t = self.at[0]
            before, after = (a, b) if self._sign_before(t) < 0 else (b, a)
            return (f"ranking inverts at tau = {t}. Below it {before} wins, "
                    f"above it {after} wins")
        named = " and again at ".join(str(t) for t in self.at)
        return (f"ranking inverts at tau = {named}. Neither arm dominates over "
                f"[{self.taus[0]}, {self.taus[-1]}], so there is no single "
                "horizon-free answer and no 'the' crossing to quote")


def crossings(
    a_values: Mapping[int, float], b_values: Mapping[int, float]
) -> Crossings:
    """Every tau where ``a - b`` changes sign, not merely the first.

    Ties are carried rather than counted: a gap of exactly zero is not a
    crossing until the sign on the far side differs from the sign before it,
    so a flat touch at zero does not register as two.
    """
    taus = tuple(sorted(set(a_values) & set(b_values)))
    gaps = tuple(a_values[t] - b_values[t] for t in taus)
    at: list[int] = []
    last = 0
    for t, g in zip(taus, gaps):
        sign = (g > 0) - (g < 0)
        if sign == 0:
            continue
        if last and sign != last:
            at.append(t)
        last = sign
    return Crossings(taus=taus, gaps=gaps, at=tuple(at))


__all__ = [
    "EXCLUDE",
    "Crossings",
    "crossings",
    "CellResult",
    "DistributionError",
    "Estimate",
    "FUNCTIONALS",
    "across_cells",
    "clustered_interval",
    "ecdf",
    "estimate",
    "functional",
    "median",
    "quantile",
    "rmst",
    "success_at",
]
