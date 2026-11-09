"""How a comparison was READ, as an identifier beside what it ran.

``plan_id`` answers what ran. Nothing answered how it was read, and the gap
is not cosmetic: two runs of the same plan over identical episodes produce
different verdicts under different flags. The conclusion is a function of the
episodes AND the choices made about them, and only the first half had a name.

The compare-time surface, every item of which moves the answer:

* the dichotomy rule, which collapses seeds to the bit McNemar consumes
* the pre-registered budget ``b``
* ``tau``
* ``ghost_penalty``, including a per-basis mapping
* which bases were excluded rather than penalised
* the declared Holm family, whose SIZE scales every adjusted p
* the permutation functional, and its seed on the sampled path

None of these was in any hash. ``analysis_hash`` is.

    plan_id        what ran
    analysis_hash  how it was read

Two verdicts are comparable only under the same pair, and a difference in
either is a refusal rather than a footnote.

This also turns pre-registration from a promise into an artifact. Commit to an
analysis_hash before the results exist and the verdict proves which analysis
produced it; the phrase "pre-registered budget" stops being an assurance and
becomes something a reader can check.

One deliberate exception. The stability report VARIES tau and the penalty --
that is what it is for -- so it spans many analysis hashes by construction. It
therefore belongs to the swept set rather than to one hash, and the verdict
records the range it covered instead of pretending a single identifier
describes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..schema.canonical import hash_obj


class FamilyError(ValueError):
    """The Holm family was changed after it could still be changed safely."""


@dataclass(frozen=True)
class AnalysisSpec:
    """Every compare-time choice that moves the conclusion.

    Frozen, so a spec cannot drift between being hashed and being used.
    """

    #: How seeds collapse to the bit McNemar consumes.
    dichotomy: str = "majority"
    #: The ONE budget carrying the inferential claim. None means "each
    #: episode's own recorded budget", which is the pre-v2-safe path.
    budget: int | None = None
    #: Horizon for restricted functionals.
    tau: int | None = None
    #: A scalar, or a per-basis mapping, or None where no functional needs one.
    ghost_penalty: object = None
    #: Bases dropped rather than penalised, as declared.
    excluded_bases: tuple[str, ...] = ()
    #: The functional the permutation test reads, and its seed when sampled.
    permutation_statistic: str | None = None
    permutation_seed: int | None = None
    #: Pairwise contrasts, declared up front. See `Family`.
    family: tuple[tuple[str, str], ...] = ()
    #: Significance level the p-values will be read against.
    alpha: float = 0.05
    #: Which of a phase record's two step facts a per-phase functional reads:
    #: "at_end" credits the phase only if its predicate still held when the
    #: episode finished, "first_satisfied" credits it if the predicate ever
    #: held.
    #:
    #: A real choice with no default answer. A policy that puts the item in
    #: the bin and later knocks it out demonstrated the skill and did not
    #: deliver, and both readings are legitimate. So it is recorded here and
    #: hashed: two per-phase reports of one run under different credit are not
    #: comparable, and nothing else in the identity would say why they differ.
    phase_credit: str = "at_end"

    def document(self) -> dict:
        """The canonical form that gets hashed."""
        penalty = self.ghost_penalty
        if isinstance(penalty, Mapping):
            penalty = {str(getattr(k, "value", k)): v
                       for k, v in sorted(penalty.items(),
                                          key=lambda kv: str(kv[0]))}
        return {
            "dichotomy": self.dichotomy,
            "budget": self.budget,
            "tau": self.tau,
            "ghost_penalty": penalty,
            "excluded_bases": sorted(self.excluded_bases),
            "permutation_statistic": self.permutation_statistic,
            "permutation_seed": self.permutation_seed,
            "family": sorted(tuple(sorted(pair)) for pair in self.family),
            "alpha": self.alpha,
            "phase_credit": self.phase_credit,
        }

    def analysis_hash(self) -> str:
        return hash_obj(self.document())


@dataclass
class Family:
    """The Holm family, composed before any test in it runs.

    Holm tests the i-th smallest p against ``alpha / (m - i + 1)``, so ``m``
    scales every threshold. Adding a contrast after a test has run changes
    the bar that test was already judged against -- retroactively, and
    invisibly in the output. The runner therefore refuses it.

    Composed as the ``C(k, 2)`` pairwise contrasts among the declared arms,
    which is also what keeps the budget sweep and the stability report OUT:
    neither is a contrast between two arms, so neither can enter, and neither
    can inflate ``m``. That is enforced by the family being built from the arm
    list rather than accumulated as tests are run.

    ``m`` is recorded on the verdict because a Holm-adjusted p cannot be
    interpreted without it: 0.04 against m = 2 and 0.04 against m = 15 are
    different claims and look identical.
    """

    arms: tuple[str, ...]
    contrasts: tuple[tuple[str, str], ...] = field(init=False)
    _sealed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        arms = tuple(self.arms)
        object.__setattr__ if False else None
        self.contrasts = tuple(
            (a, b) for i, a in enumerate(arms) for b in arms[i + 1:]
        )

    @property
    def m(self) -> int:
        return len(self.contrasts)

    def seal(self) -> None:
        """Called by the runner before the first test. Nothing may be added after."""
        self._sealed = True

    def add(self, a: str, b: str) -> None:
        if self._sealed:
            raise FamilyError(
                f"cannot add the contrast {a!r} vs {b!r}: tests in this family "
                f"have already run against m = {self.m}. Holm reads the i-th "
                "smallest p against alpha/(m-i+1), so growing m now would change "
                "the threshold every result already cleared, without any of them "
                "being recomputed. Declare the full arm set before testing."
            )
        self.contrasts = self.contrasts + ((a, b),)

    def prune_duplicates(self, identical: Sequence[tuple[str, str]]) -> list[str]:
        """Drop contrasts involving an arm declared identical to another.

        This is the point of the declared-identity refusal doing more than
        printing a line: five distinct arms deserve m = 10, and six arms with
        one duplicate give m = 15, so every genuine contrast is held to a
        tighter threshold for no reason.
        """
        drop = {b for _, b in identical}
        before = self.m
        self.contrasts = tuple(
            c for c in self.contrasts if not (set(c) & drop)
        )
        return [f"m reduced {before} -> {self.m}: dropped contrasts involving "
                f"{sorted(drop)}, declared identical to another arm"] if drop else []


__all__ = ["AnalysisSpec", "Family", "FamilyError"]
