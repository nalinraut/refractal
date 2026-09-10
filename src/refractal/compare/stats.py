"""The tests, and the wrong ones kept alongside on purpose.

Three estimators live here. Two are what `compare` reports; the third is
deliberately incorrect and exists so the difference can be *demonstrated*
rather than asserted in a docstring.

``mcnemar_exact``
    The right test for paired binary outcomes. Reads only the discordant pairs,
    because agreements carry no information about direction.

``clustered_bootstrap``
    Resamples **whole scenarios**, never individual episodes. Five seeds at one
    pose are one observation of that pose.

``two_proportion_z`` and ``mcnemar_unclustered``
    The two ways to get this wrong, in opposite directions. The first throws the
    pairing away and loses power against a real difference. The second treats
    each (scenario, seed) as an independent pair, inflating n and calling noise
    significant. Both are exported so tests can show a case where they disagree
    with the correct answer -- a fixture that only contains obvious cases cannot
    tell a correct test from a broken one.

No scipy. An exact McNemar is a binomial tail, the bootstrap is resampling, and
the normal CDF the (wrong) z-test needs is in ``statistics``. Adding a numerical
stack for three functions would buy nothing and cost every install.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .pairing import Dichotomy, Unit


@dataclass
class Discordance:
    """The 2x2. All four cells, not only the two the test consumes."""

    both_fail: int
    only_b_passes: int
    only_a_passes: int
    both_pass: int

    @property
    def discordant(self) -> int:
        return self.only_a_passes + self.only_b_passes

    @property
    def total(self) -> int:
        return self.both_fail + self.only_a_passes + self.only_b_passes + self.both_pass


@dataclass
class TestResult:
    statistic: float | None
    p_value: float
    n: int
    detail: str = ""


def contingency(units: Sequence[Unit], a: str, b: str, rule: Dichotomy) -> Discordance:
    """The full table.

    The diagonal is often the more actionable half -- both-fail is where to aim
    training next, both-pass is the regression-safety set -- and it is what gives
    the test its context: 12 flips out of 15 disagreements is a strong effect,
    12 flips when 400 scenarios agreed is a small one at the same p-value.
    """
    cells = Discordance(0, 0, 0, 0)
    for unit in units:
        pa, pb = unit.passed(a, rule), unit.passed(b, rule)
        if pa and pb:
            cells.both_pass += 1
        elif pa and not pb:
            cells.only_a_passes += 1
        elif pb and not pa:
            cells.only_b_passes += 1
        else:
            cells.both_fail += 1
    return cells


def _binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p=0.5, by doubling the smaller tail."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def mcnemar_exact(cells: Discordance) -> TestResult:
    """Exact McNemar on the discordant pairs.

    Exact rather than the chi-square approximation, because the discordant count
    is routinely small -- and the small-n regime is exactly where a regression
    first shows up, so an approximation that needs n>25 is wrong where it
    matters most.
    """
    b, c = cells.only_a_passes, cells.only_b_passes
    n = b + c
    return TestResult(
        statistic=float(min(b, c)),
        p_value=_binomial_two_sided(min(b, c), n),
        n=n,
        detail=f"{b} scenarios only {'A'} solves, {c} only B solves, {cells.total - n} agree",
    )


def mcnemar_unclustered(units: Sequence[Unit], a: str, b: str) -> TestResult:
    """**Wrong on purpose.** Treats every (scenario, seed) as an independent pair.

    Kept so a test can show what it costs: n inflates by the seed count, so the
    same data yields a much smaller p-value and noise gets called a finding.
    """
    only_a = only_b = agree = 0
    for unit in units:
        for seed in unit.seeds_kept:
            pa, pb = unit.outcomes[a][seed], unit.outcomes[b][seed]
            if pa and not pb:
                only_a += 1
            elif pb and not pa:
                only_b += 1
            else:
                agree += 1
    n = only_a + only_b
    return TestResult(
        statistic=float(min(only_a, only_b)),
        p_value=_binomial_two_sided(min(only_a, only_b), n),
        n=n,
        detail="seeds treated as independent pairs -- do not report this",
    )


def two_proportion_z(units: Sequence[Unit], a: str, b: str) -> TestResult:
    """**Wrong on purpose.** Assumes two independent groups.

    That is not this design: the same scenarios face both checkpoints. Throwing
    the pairing away discards the information about *which* scenarios changed
    and makes a real difference harder to detect.
    """
    sa = sum(sum(u.outcomes[a].values()) for u in units)
    sb = sum(sum(u.outcomes[b].values()) for u in units)
    na = sum(len(u.outcomes[a]) for u in units)
    nb = sum(len(u.outcomes[b]) for u in units)
    if not na or not nb:
        return TestResult(None, 1.0, 0, "no data")
    pa, pb = sa / na, sb / nb
    pooled = (sa + sb) / (na + nb)
    se = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    if se == 0:
        return TestResult(0.0, 1.0, na + nb, "degenerate")
    z = (pb - pa) / se
    p = 2 * (1 - statistics.NormalDist().cdf(abs(z)))
    return TestResult(z, p, na + nb, "unpaired -- do not report this")


@dataclass
class BootstrapResult:
    difference: float
    low: float
    high: float
    resamples: int
    #: The basic (reverse-percentile) interval, ``2*theta - percentiles``,
    #: reflected about the observed statistic. Free to compute alongside the
    #: percentile interval, and the two disagreeing is the signature of the bias
    #: percentile intervals are known for -- so it is cheaper to look at this
    #: than to reach for BCa on a hunch.
    basic_low: float = 0.0
    basic_high: float = 0.0
    #: Achieved significance level: twice the smaller tail of the resample
    #: distribution about zero. The interval is what a human reads, but a
    #: multiplicity correction needs a p-value it can rank, and inventing a
    #: second procedure for intervals would put the gate and the report on
    #: different footings.
    p_value: float = 1.0

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    @property
    def basic_excludes_zero(self) -> bool:
        return self.basic_low > 0 or self.basic_high < 0


def clustered_bootstrap(
    units: Sequence[Unit],
    a: str,
    b: str,
    *,
    resamples: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Percentile CI for the rate difference, resampling whole scenarios.

    Resampling episodes instead would treat five seeds at one pose as five
    independent observations, which is the single most common way to make a
    robotics eval look more certain than it is. The unit of independence is the
    scenario, so the scenario is what gets resampled.

    Seeded, because a confidence interval that moves between two runs of the
    same analysis is not something anyone can act on.
    """
    if not units:
        return BootstrapResult(0.0, 0.0, 0.0, resamples)

    per_unit = [(unit.rate(a), unit.rate(b)) for unit in units]
    observed = sum(rb - ra for ra, rb in per_unit) / len(per_unit)

    diffs = [rb - ra for ra, rb in per_unit]
    rng = random.Random(seed)
    n = len(diffs)
    choices = rng.choices
    draws = sorted(sum(choices(diffs, k=n)) / n for _ in range(resamples))

    lo_index = max(0, int(math.floor((alpha / 2) * resamples)))
    hi_index = min(resamples - 1, int(math.ceil((1 - alpha / 2) * resamples)) - 1)
    low, high = draws[lo_index], draws[hi_index]
    # +1 in both parts so a p-value is never exactly zero: with B resamples the
    # most that can honestly be claimed is 1/(B+1).
    below = sum(1 for d in draws if d <= 0)
    above = sum(1 for d in draws if d >= 0)
    asl = min(1.0, 2 * (min(below, above) + 1) / (resamples + 1))
    return BootstrapResult(
        difference=observed,
        low=low,
        high=high,
        resamples=resamples,
        p_value=asl,
        # Reflected about the observed value: 2*theta - upper, 2*theta - lower.
        basic_low=2 * observed - high,
        basic_high=2 * observed - low,
    )


__all__ = [
    "BootstrapResult",
    "cochran_q",
    "holm",
    "outcome_vectors",
    "Discordance",
    "TestResult",
    "clustered_bootstrap",
    "contingency",
    "mcnemar_exact",
    "mcnemar_unclustered",
    "two_proportion_z",
]


# ---------------------------------------------------------------------------
# N checkpoints
# ---------------------------------------------------------------------------


def outcome_vectors(
    units: Sequence[Unit], checkpoints: Sequence[str], rule: Dichotomy
) -> dict[tuple[bool, ...], int]:
    """How many scenarios show each pass/fail pattern across all k checkpoints.

    The k-checkpoint generalisation of the 2x2, and the reason it is worth
    running three checkpoints together rather than three pairs: "scenarios only
    ckpt-48 solves" is directly readable here and cannot be recovered from three
    pairwise comparisons, because a pairwise table never says which *other*
    checkpoint was also failing.
    """
    counts: dict[tuple[bool, ...], int] = {}
    for unit in units:
        pattern = tuple(unit.passed(c, rule) for c in checkpoints)
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def _cochran_q_statistic(rows: Sequence[tuple[bool, ...]], k: int) -> float:
    """Q for a scenarios x checkpoints table of pass/fail."""
    col = [sum(row[j] for row in rows) for j in range(k)]
    row_totals = [sum(row) for row in rows]
    denom = k * sum(row_totals) - sum(t * t for t in row_totals)
    if denom == 0:
        return 0.0
    mean_col = sum(col) / k
    return k * (k - 1) * sum((c - mean_col) ** 2 for c in col) / denom


def cochran_q(
    units: Sequence[Unit],
    checkpoints: Sequence[str],
    rule: Dichotomy,
    *,
    permutations: int = 4000,
    seed: int = 0,
) -> TestResult:
    """The k-sample extension of McNemar: do *any* of these checkpoints differ?

    Used as a gate before pairwise testing, which is standard practice and keeps
    the family of pairwise comparisons from being run at all when there is
    nothing to find.

    The p-value is by permutation rather than the chi-square approximation.
    Under the null the k outcomes within one scenario are exchangeable, so
    permuting labels within each row gives an exact reference distribution. The
    approximation needs a large table, and the small-table regime is exactly
    where a regression first shows up -- the same reason McNemar here is exact
    rather than chi-square. It also avoids depending on an incomplete gamma
    function for one call site.
    """
    k = len(checkpoints)
    rows = [tuple(unit.passed(c, rule) for c in checkpoints) for unit in units]
    observed = _cochran_q_statistic(rows, k)

    rng = random.Random(seed)
    # Rows that are all-pass or all-fail are invariant under permutation and
    # contribute nothing; skipping them is free and makes the loop honest about
    # how much information there actually is.
    variable = [list(row) for row in rows if 0 < sum(row) < k]
    if not variable:
        return TestResult(0.0, 1.0, 0, "every scenario agrees across all checkpoints")

    fixed_contrib = [row for row in rows if not (0 < sum(row) < k)]
    at_least = 1
    for _ in range(permutations):
        shuffled = []
        for row in variable:
            copy = list(row)
            rng.shuffle(copy)
            shuffled.append(tuple(copy))
        stat = _cochran_q_statistic(shuffled + fixed_contrib, k)
        if stat >= observed - 1e-12:
            at_least += 1
    return TestResult(
        observed,
        at_least / (permutations + 1),
        len(variable),
        f"{len(variable)} of {len(rows)} scenarios disagree across checkpoints",
    )


def holm(p_values: Mapping[Any, float]) -> dict[Any, float]:
    """Holm-Bonferroni adjusted p-values, controlling family-wise error.

    Holm rather than plain Bonferroni because it is uniformly more powerful and
    no harder to implement, and neither needs a dependency.

    This matters more than it looks. Four checkpoints is six pairwise
    comparisons; across three tasks that is eighteen, and at a nominal 5% each
    you expect a false positive about 60% of the time. Correction is not
    bookkeeping, it is the difference between a finding and noise -- and a gate
    that fires when *any* comparison trips is exactly the family this corrects.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[Any, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        # Monotone: an adjusted p can never fall below one for a smaller raw p.
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted
