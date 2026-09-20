"""Measuring the correlation structure the clustered bootstrap exists for.

Every number in the measurement write-up (development record) -- the false-positive tables,
the power tables, the case for gating on the bootstrap rather than McNemar -- was
measured against ``FakeBenchmark``'s ``interaction_spread``. That is a *model* of
within-scenario correlation, and it was authored to make the fixture able to
discriminate between the clustered and unclustered tests, then the tests were
scored against it.

Circular. It does not invalidate the conclusions, but it does mean the numbers
describe a model rather than the world, and the first real policy is the first
independent source of the effect.

So this module measures the same structure from real episode rows, in a form
directly comparable to the modelled value. The point is that a correction to the
statistics doc should be a *comparison* between the two, not the model quietly
being replaced by whatever the first real run happened to show.

Two numbers, and they are informative in different ways
-------------------------------------------------------

``design_effect``
    ``Var(d_i) / E[binomial Var(d_i)]`` where ``d_i`` is the paired difference on
    scenario *i*. This is the quantity that decides whether ignoring clustering
    is anti-conservative, because the clustered bootstrap resamples exactly these
    ``d_i``. At 1.0 the naive test is calibrated; above 1.0 it is not.

``icc``
    Intraclass correlation of outcomes within one (scenario, checkpoint) cell, per
    checkpoint. How much a scenario's difficulty dominates the seed noise.

**Both are needed, and the pair is the finding.** A scenario effect *shared* by
every checkpoint produces a high ICC and a design effect near 1, because the
paired difference cancels it -- that is the row-2 result in the statistics doc,
the one that was surprising. A checkpoint x scenario interaction produces a high
ICC *and* a design effect above 1. Reading either number alone cannot tell those
apart, and they have opposite implications for whether the bootstrap is
load-bearing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from .pairing import Unit


@dataclass
class VarianceReport:
    """Correlation structure measured from episode rows."""

    units: int
    mean_seeds: float
    #: Var of the paired difference across scenarios, observed.
    observed_var: float
    #: What that variance would be if seeds within a cell were independent.
    binomial_var: float
    #: Per checkpoint: intraclass correlation within a (scenario, checkpoint) cell.
    icc: dict[str, float] = field(default_factory=dict)
    #: Per checkpoint: between-scenario variance of the cell proportion.
    between_var: dict[str, float] = field(default_factory=dict)

    @property
    def design_effect(self) -> float:
        """>1 means clustering is load-bearing for the paired difference."""
        if self.binomial_var <= 0:
            return 1.0
        return self.observed_var / self.binomial_var

    @property
    def clustering_matters(self) -> bool:
        # 1.25 is a judgement, not a threshold with a derivation. Stated as one
        # rather than hidden: below it the naive and clustered intervals agreed
        # to within noise in every fixture run so far.
        return self.design_effect >= 1.25

    def summary_lines(self) -> list[str]:
        lines = [
            f"design effect on the paired difference: {self.design_effect:.2f} "
            f"(observed Var {self.observed_var:.5f} vs binomial {self.binomial_var:.5f}, "
            f"{self.units} scenarios, {self.mean_seeds:.1f} seeds each)"
        ]
        for checkpoint in sorted(self.icc):
            lines.append(
                f"  {checkpoint}: within-cell ICC {self.icc[checkpoint]:.3f}, "
                f"between-scenario Var {self.between_var[checkpoint]:.5f}"
            )
        if self.clustering_matters:
            lines.append(
                "  clustering is load-bearing here: an unclustered test would be "
                "anti-conservative by roughly this factor in variance."
            )
        else:
            lines.append(
                "  clustering is not load-bearing for the difference at this design "
                "effect -- a shared scenario effect cancels in a paired contrast, so a "
                "high ICC alone does not imply it is."
            )
        return lines


def _cell_proportions(units: Sequence[Unit], checkpoint: str) -> list[float]:
    return [unit.rate(checkpoint) for unit in units]


def measure_variance(units: Sequence[Unit], a: str, b: str) -> VarianceReport:
    """Decompose the variance of the paired difference, and per-cell ICC.

    Estimators are deliberately simple and stated rather than cited: the point is
    a number comparable with the modelled one, and an estimator nobody can read
    is worse than a crude one everybody can.
    """
    usable = [u for u in units if len(u.seeds_kept) >= 2]
    if len(usable) < 2:
        return VarianceReport(units=len(usable), mean_seeds=0.0, observed_var=0.0, binomial_var=0.0)

    mean_seeds = statistics.fmean(len(u.seeds_kept) for u in usable)

    differences = [u.rate(b) - u.rate(a) for u in usable]
    observed_var = statistics.variance(differences)

    # Under within-cell independence a cell proportion has variance p(1-p)/s,
    # and the two checkpoints of a paired difference are independent, so the difference
    # has the sum.
    #
    # The divisor is s-1, not s, and this is not a detail. E[p_hat(1-p_hat)] is
    # (s-1)/s * p(1-p), so p_hat(1-p_hat)/s understates the sampling variance --
    # and it understates it *most* when p is extreme, which is exactly what a
    # strong shared scenario effect produces (scenarios become reliably easy or
    # reliably hard). Using /s made the design effect read 1.38 on a fixture with
    # zero interaction, where the statistics doc measured the unclustered test as
    # conservative. The estimator was manufacturing the conclusion it was built
    # to test.
    #
    # Caught by calibrating against known injected values before trusting it on
    # real data, which is the whole reason to calibrate.
    binomial_var = statistics.fmean(
        sum(
            unit.rate(c) * (1 - unit.rate(c)) / max(1, len(unit.seeds_kept) - 1)
            for c in (a, b)
        )
        for unit in usable
    )

    report = VarianceReport(
        units=len(usable),
        mean_seeds=mean_seeds,
        observed_var=observed_var,
        binomial_var=binomial_var,
    )

    for checkpoint in (a, b):
        proportions = _cell_proportions(usable, checkpoint)
        observed = statistics.variance(proportions)
        within = statistics.fmean(p * (1 - p) for p in proportions)
        # Subtract the sampling component to leave the between-scenario part.
        # Clamped at zero: a negative variance estimate is an artefact of a small
        # sample, not a finding, and reporting it as negative invites nonsense.
        between = max(0.0, observed - within / max(1.0, mean_seeds - 1))
        report.between_var[checkpoint] = between
        report.icc[checkpoint] = between / (between + within) if (between + within) > 0 else 0.0

    return report


def modelled_variance(
    *, scenario_spread: float, interaction_spread: float, base_rate: float = 0.5
) -> dict[str, float]:
    """What ``FakeBenchmark``'s knobs imply, for comparison with a measurement.

    Exists so a correction to the statistics doc is a comparison rather than a
    replacement. If a real policy's design effect differs from the value its
    ``interaction_spread`` equivalent would produce, that difference is the
    finding -- and it can only be stated if the modelled number is written down
    in the same units as the measured one.

    Both spreads are uniform offsets on a success probability, so each
    contributes ``(2 * spread)^2 / 12`` to the variance of that probability.
    Only the interaction term survives a paired difference; the shared term
    cancels, which is why they are reported separately.
    """
    shared = (2 * scenario_spread) ** 2 / 12
    interaction = (2 * interaction_spread) ** 2 / 12
    return {
        "shared_var": shared,
        "interaction_var": interaction,
        # A paired difference sees the interaction term from both checkpoints and none
        # of the shared one.
        "expected_difference_var": 2 * interaction,
        "cancels_in_paired_difference": shared,
    }


__all__ = ["VarianceReport", "measure_variance", "modelled_variance"]
