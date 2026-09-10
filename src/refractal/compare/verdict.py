"""The verdict: N checkpoints, one gate, and a correction across the whole family.

**Two checkpoints is the common case, not a special one.** A single checkpoint is
a comparison of size one; three is a training sweep. The shape of the report
changes with k, but nothing structural does.

What the gate is
----------------

**The clustered bootstrap gates. McNemar and Cochran's Q are reported and never
gate.** Measured at 120 scenarios x 5 seeds, 300 trials per cell, with a
checkpoint x scenario interaction present -- both at the same false-positive
rate:

===========  =========  =============
true delta   McNemar    bootstrap
===========  =========  =============
0.00          4.0%       4.3%
0.05         13.0%      29.3%
0.07         24.7%      47.3%
0.10         46.0%      75.3%
===========  =========  =============

Same type I error, roughly double the power. Gating on McNemar would miss a real
7-point regression three times in four, because collapsing 3/5 versus 2/5 into a
single bit discards most of the signal in a uniform shift -- and a uniform shift
is what a slightly worse checkpoint produces.

Multiplicity, which is where two checkpoints was quietly wrong
--------------------------------------------------------------

The gate fires if *any* comparison trips, so every comparison is in one family
and the family has to be corrected as a whole. Uncorrected, three tasks at a
nominal 5% is a family-wise rate near 14%; four checkpoints across three tasks
is eighteen comparisons and roughly 60%.

That is the same error the design doc criticises in D7, so it is worth being
exact about the family: **(k-1) contrasts against the baseline x T tasks**, with
Holm-Bonferroni applied across all of them at once. Contrasts against a baseline
rather than all k(k-1)/2 pairs, because the question is "did my change help",
which has a reference point. Comparing every pair would triple the family for
answers nobody asked for.

Cochran's Q runs per task as a screen -- do *any* of these checkpoints differ --
and is reported alongside. It does not gate, because a screen that gates would
make the gate's calibration depend on a second test's power.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .pairing import Dichotomy, Eligibility, Unit
from .stats import (
    BootstrapResult,
    Discordance,
    TestResult,
    clustered_bootstrap,
    cochran_q,
    contingency,
    holm,
    mcnemar_exact,
    outcome_vectors,
)

#: Below this many scenarios the percentile interval is optimistic: measured at
#: a true difference of zero, 120 scenarios rejects ~7% of the time against a
#: nominal 5%, 300 rejects 4.8%, 500 rejects 4.0%. The basic (reverse
#: percentile) interval tracks it at every size, which rules out the bias BCa
#: corrects -- so this is small-cluster behaviour and more machinery would not
#: have fixed it. Saying so beats pretending otherwise.
MIN_UNITS_FOR_CALIBRATED_CI = 200


@dataclass
class Contrast:
    """One checkpoint measured against the baseline, for one task."""

    task_id: str
    scene_id: str
    baseline: str
    candidate: str
    cells: Discordance
    mcnemar: TestResult
    bootstrap: BootstrapResult
    #: Holm-adjusted across every contrast in the run, not just this task's.
    adjusted_p: float = 1.0
    gate: str = "ok"  # ok | regressed | improved
    notes: list[str] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        return self.gate == "regressed"


@dataclass
class TaskVerdict:
    task_id: str
    scene_id: str
    units: int
    rates: dict[str, float]
    #: Pattern of pass/fail across all k checkpoints -> how many scenarios.
    #: "Only ckpt-48 solves these" is readable here and is not recoverable from
    #: pairwise tables.
    patterns: dict[tuple[bool, ...], int]
    cochran: TestResult
    contrasts: list[Contrast] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        return any(c.regressed for c in self.contrasts)


@dataclass
class Verdict:
    eligibility: Eligibility
    checkpoints: list[str]
    baseline: str
    tasks: list[TaskVerdict] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    family_size: int = 0
    #: Pairs that exist in the data and were never tested, because contrasts run
    #: against a baseline. Disclosed for the same reason the overlap counts are:
    #: a reader looking at three rate columns forms a view about every pair,
    #: including the ones no test covered.
    untested_pairs: list[tuple[str, str]] = field(default_factory=list)
    #: Set when a precondition was overridden on purpose. Echoed in the report,
    #: so nobody reads a number without knowing what was waived to get it.
    overrides: list[str] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        return any(t.regressed for t in self.tasks)

    @property
    def exit_code(self) -> int:
        """0 clean, 1 regression, 2 cannot be answered.

        The third is not a regression and must not be reported as one: "the two
        runs used different geometry" is a different sentence from "the policy
        got worse", and conflating them teaches people to ignore the gate.
        """
        if self.blocking:
            return 2
        return 1 if self.regressed else 0


def evaluate(
    eligibility: Eligibility,
    checkpoints: Sequence[str] | None = None,
    *,
    baseline: str | None = None,
    rule: Dichotomy = "majority",
    alpha: float = 0.05,
    resamples: int = 5000,
    permutations: int = 2000,
    seed: int = 0,
    correct: bool = True,
    allow_harness_mismatch: bool = False,
) -> Verdict:
    """Compare k checkpoints against a baseline, correcting across the family."""
    checkpoints = list(checkpoints or eligibility.checkpoints)
    if len(checkpoints) < 2:
        raise ValueError(f"need at least two checkpoints to compare, got {checkpoints}")
    baseline = baseline or checkpoints[0]
    if baseline not in checkpoints:
        raise ValueError(f"baseline {baseline!r} is not among {checkpoints}")

    verdict = Verdict(
        eligibility=eligibility, checkpoints=checkpoints, baseline=baseline
    )

    for scene_id, hashes in sorted(eligibility.scene_hash_conflicts.items()):
        verdict.blocking.append(
            f"scene {scene_id!r} has {len(hashes)} different scene_hash values across the "
            "checkpoints being compared: the geometry changed between runs, so these "
            "results are not comparable. Re-run, or compare within one geometry."
        )
    if len(eligibility.harness_surfaces) > 1:
        # A precondition, not a note. vla-eval's own paper reports a harness-side
        # integration parameter -- the proprioceptive state source fed to the
        # policy -- moving a LIBERO success rate from 97.8% to 42%. Two runs from
        # different harness versions may simply not be measuring the same thing,
        # and joining them silently is the failure this project exists to stop.
        #
        # It is not in `plan_id`, because pinning it there would invalidate every
        # historical comparison on a dependency bump -- and most harness commits
        # change no behaviour at all. Three of the five commits in the range this
        # was written against were a docs edit, a pin bump and a data refresh.
        # Hence a gate with an explicit override rather than an identity.
        surfaces = ", ".join(sorted(eligibility.harness_surfaces))
        versions = ", ".join(sorted(eligibility.harness_versions)) or "unknown"
        if allow_harness_mismatch:
            verdict.overrides.append(
                f"harness surface mismatch waived (--allow-harness-mismatch): {surfaces}"
            )
        else:
            verdict.blocking.append(
                f"episodes were produced by {len(eligibility.harness_surfaces)} different "
                f"harness surfaces ({surfaces}; versions {versions}). This digest covers only "
                "the harness modules Refractal depends on, so it did not move for a docs "
                "edit or a data refresh -- something behavioural changed. Re-run under one "
                "surface, or pass --allow-harness-mismatch if you know the difference does "
                "not affect these results."
            )

    if verdict.blocking:
        # Return before any per-task work. Data just declared incomparable must
        # not also yield a verdict -- `regressed` would then be True on a
        # comparison we refused to make.
        return verdict

    if len(eligibility.harness_versions) > 1:
        # Provenance: the surfaces agree, so the comparison stands; the reader
        # should still know two builds were involved.
        verdict.notes.append(
            f"episodes came from {len(eligibility.harness_versions)} harness versions "
            f"({', '.join(sorted(eligibility.harness_versions))}), but the modules Refractal "
            "depends on are identical across them, so the comparison stands."
        )

    if len(eligibility.sessions) > 1:
        verdict.notes.append(
            f"this comparison spans {len(eligibility.sessions)} sessions, so warmup and "
            "thermal conditions differ across episodes. Success rates are unaffected; "
            "latency comparisons from it are not."
        )
    by_task: dict[tuple[str, str], list[Unit]] = {}
    for unit in eligibility.units:
        by_task.setdefault((unit.key.scene_id, unit.key.task_id), []).append(unit)

    candidates = [c for c in checkpoints if c != baseline]
    raw_p: dict[tuple[str, str, str], float] = {}

    for (scene_id, task_id), units in sorted(by_task.items()):
        task = TaskVerdict(
            task_id=task_id,
            scene_id=scene_id,
            units=len(units),
            rates={c: sum(u.rate(c) for u in units) / len(units) for c in checkpoints},
            patterns=outcome_vectors(units, checkpoints, rule),
            cochran=cochran_q(
                units, checkpoints, rule, permutations=permutations, seed=seed
            ),
        )
        for candidate in candidates:
            bootstrap = clustered_bootstrap(
                units, baseline, candidate, resamples=resamples, seed=seed, alpha=alpha
            )
            contrast = Contrast(
                task_id=task_id,
                scene_id=scene_id,
                baseline=baseline,
                candidate=candidate,
                cells=contingency(units, baseline, candidate, rule),
                mcnemar=mcnemar_exact(contingency(units, baseline, candidate, rule)),
                bootstrap=bootstrap,
            )
            raw_p[(scene_id, task_id, candidate)] = bootstrap.p_value
            task.contrasts.append(contrast)
        verdict.tasks.append(task)

    adjusted = holm(raw_p) if correct else dict(raw_p)
    verdict.family_size = len(raw_p)

    for task in verdict.tasks:
        for contrast in task.contrasts:
            key = (task.scene_id, task.task_id, contrast.candidate)
            contrast.adjusted_p = adjusted[key]
            if contrast.adjusted_p < alpha:
                contrast.gate = (
                    "regressed" if contrast.bootstrap.difference < 0 else "improved"
                )
            if task.units < MIN_UNITS_FOR_CALIBRATED_CI:
                contrast.notes.append(
                    f"{task.units} scenarios is below {MIN_UNITS_FOR_CALIBRATED_CI}, where the "
                    "percentile interval measures ~7% false positives against a nominal 5%."
                )
            if contrast.gate != "ok" and contrast.mcnemar.p_value >= alpha:
                contrast.notes.append(
                    "the rate moved but few scenarios flipped their majority "
                    f"(McNemar p={contrast.mcnemar.p_value:.3f}). A uniform shift, not a set "
                    "of scenarios breaking -- read the 2x2 before acting."
                )

    tested = {(baseline, c) for c in candidates}
    verdict.untested_pairs = [
        (x, y)
        for i, x in enumerate(checkpoints)
        for y in checkpoints[i + 1 :]
        if (x, y) not in tested and (y, x) not in tested
    ]
    if verdict.untested_pairs:
        pairs = ", ".join(f"{x} vs {y}" for x, y in verdict.untested_pairs)
        best = max(checkpoints, key=lambda c: sum(t.rates[c] for t in verdict.tasks))
        extra = ""
        if best != baseline and any(best in pair for pair in verdict.untested_pairs):
            extra = (
                f" {best} has the highest mean rate here and is in an untested pair, so "
                "shipping it on this report would mean shipping on a contrast nobody ran."
            )
        verdict.notes.append(
            f"contrasts are against the baseline {baseline!r}, so these pairs were NOT "
            f"tested: {pairs}. Choosing a baseline is a modelling decision.{extra}"
        )

    if verdict.family_size > 1:
        verdict.notes.append(
            f"{verdict.family_size} contrasts "
            f"({len(candidates)} checkpoint(s) x {len(verdict.tasks)} task(s)); "
            f"p-values are Holm-adjusted across all of them. Uncorrected, the chance of at "
            f"least one false positive here would be about "
            f"{1 - (1 - alpha) ** verdict.family_size:.0%}."
        )

    return verdict


def _pattern_label(pattern: tuple[bool, ...], checkpoints: Sequence[str]) -> str:
    solved = [c for c, ok in zip(checkpoints, pattern) if ok]
    if not solved:
        return "none solve"
    if len(solved) == len(checkpoints):
        return "all solve"
    return "only " + ", ".join(solved)


def render(verdict: Verdict) -> str:
    """Overlap first, then what changed, then the tests, then the gate."""
    lines: list[str] = []
    lines.extend(f"  {line}" for line in verdict.eligibility.summary_lines())
    lines.append("")

    if verdict.blocking:
        for blocker in verdict.blocking:
            lines.append(f"  BLOCKED: {blocker}")
        return "\n".join(lines) + "\n"

    for override in verdict.overrides:
        lines.append(f"  OVERRIDDEN: {override}")
    if verdict.overrides:
        lines.append("")

    checkpoints = verdict.checkpoints
    for task in verdict.tasks:
        lines.append(f"  {task.scene_id} / {task.task_id}   ({task.units} scenarios)")
        lines.append(
            "    rates:  "
            + "   ".join(f"{c}: {task.rates[c]:.1%}" for c in checkpoints)
            + f"    (baseline {verdict.baseline})"
        )

        if len(checkpoints) > 2:
            lines.append("    outcome patterns:")
            for pattern, count in sorted(
                task.patterns.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                lines.append(
                    f"      {count:>5}  {_pattern_label(pattern, checkpoints)}"
                )
            lines.append(
                f"    Cochran Q (screen):  Q={task.cochran.statistic:.2f} "
                f"p={task.cochran.p_value:.4f}  ({task.cochran.detail})"
            )
        else:
            cells = task.contrasts[0].cells
            a, b = verdict.baseline, task.contrasts[0].candidate
            lines.append(f"                        {b} fails   {b} succeeds")
            lines.append(f"      {a} fails        {cells.both_fail:>9}   {cells.only_b_passes:>11}")
            lines.append(f"      {a} succeeds     {cells.only_a_passes:>9}   {cells.both_pass:>11}")

        for contrast in task.contrasts:
            marker = {"regressed": "REGRESSED", "improved": "improved", "ok": "no change"}
            lines.append(
                f"    {verdict.baseline} -> {contrast.candidate}: "
                f"{contrast.bootstrap.difference:+.3f} "
                f"[{contrast.bootstrap.low:+.3f}, {contrast.bootstrap.high:+.3f}]  "
                f"p={contrast.bootstrap.p_value:.4f} "
                f"holm={contrast.adjusted_p:.4f}  "
                f"McNemar p={contrast.mcnemar.p_value:.4f}  "
                f"-> {marker[contrast.gate]}"
            )
            for note in contrast.notes:
                lines.append(f"      note: {note}")
        lines.append("")

    for note in verdict.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


__all__ = [
    "MIN_UNITS_FOR_CALIBRATED_CI",
    "Contrast",
    "TaskVerdict",
    "Verdict",
    "evaluate",
    "render",
]
