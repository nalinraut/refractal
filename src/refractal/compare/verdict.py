"""The gate: one bit, decided deliberately rather than by whatever the CLI returns.

"Report both" is a display decision. A regression gate needs a single answer,
and when McNemar says no and the bootstrap says yes something has to break the
tie. Left implicit, that decision gets made by an exit code somebody wrote
without thinking about it.

**The clustered bootstrap gates. McNemar is reported and never gates.**

Measured, at 120 scenarios x 5 seeds, 300 trials per cell, with a checkpoint x
scenario interaction present:

===========  =========  =============  ==================
true delta   McNemar    bootstrap      what this means
===========  =========  =============  ==================
0.00          4.0%       4.3%          both calibrated
0.05         13.0%      29.3%          bootstrap 2.3x the power
0.07         24.7%      47.3%          McNemar misses a real 7pp regression 3 times in 4
0.10         46.0%      75.3%
0.15         79.7%      97.3%
===========  =========  =============  ==================

Same false-positive rate, roughly double the power. So McNemar retaining its
null at this scale is weak evidence of no change, and gating on it would let
real regressions through most of the time.

The reasoning behind the numbers: a rate regression is a regression whether or
not any scenario flipped its *majority*. Collapsing 3/5 to 2/5 into a single bit
discards most of the signal in a uniform shift, and a uniform shift is what a
slightly worse checkpoint produces.

**The failure mode of this choice, stated plainly.** The bootstrap can go red
when no scenario actually changed from failing to passing -- a rate moved,
nothing flipped. A user reading "regression" will picture scenarios that broke.
That is why the 2x2 is printed next to the verdict and never omitted: the gate
is the bootstrap, but the report always shows what did and did not flip, so the
red can be interpreted rather than merely obeyed.
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
    contingency,
    mcnemar_exact,
)

#: Below this many scenarios the percentile interval is optimistic. Measured at
#: a true difference of zero: 120 scenarios rejects ~7% of the time against a
#: nominal 5%, 300 rejects 4.8%, 500 rejects 4.0%. The basic (reverse
#: percentile) interval tracks it almost exactly at every size, which rules out
#: the bias that percentile intervals are known for -- so this is small-cluster
#: behaviour and BCa would not have fixed it. The honest response is to say so
#: rather than to add machinery.
MIN_UNITS_FOR_CALIBRATED_CI = 200


@dataclass
class TaskVerdict:
    task_id: str
    scene_id: str
    units: int
    rate_a: float
    rate_b: float
    cells: Discordance
    mcnemar: TestResult
    bootstrap: BootstrapResult
    gate: str = "ok"  # ok | regressed | improved
    notes: list[str] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        return self.gate == "regressed"


@dataclass
class Verdict:
    eligibility: Eligibility
    a: str
    b: str
    tasks: list[TaskVerdict] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        return any(t.regressed for t in self.tasks)

    @property
    def exit_code(self) -> int:
        """0 clean, 1 regression, 2 cannot be answered.

        A blocking condition is not a regression and must not be reported as
        one: "the two runs used different geometry" is a different sentence from
        "the policy got worse", and conflating them teaches people to ignore the
        gate.
        """
        if self.blocking:
            return 2
        return 1 if self.regressed else 0


def evaluate(
    eligibility: Eligibility,
    a: str,
    b: str,
    *,
    rule: Dichotomy = "majority",
    alpha: float = 0.05,
    resamples: int = 5000,
    seed: int = 0,
) -> Verdict:
    """Apply the gate, per task, and collect everything the report needs."""
    verdict = Verdict(eligibility=eligibility, a=a, b=b)

    for scene_id, hashes in sorted(eligibility.scene_hash_conflicts.items()):
        # Not a warning. The scenarios carry the same parameters but a different
        # world, so the comparison is between two things that were never the
        # same experiment.
        verdict.blocking.append(
            f"scene {scene_id!r} has {len(hashes)} different scene_hash values across the "
            "checkpoints being compared: the geometry changed between runs, so these "
            "results are not comparable. Re-run, or compare within one geometry."
        )

    if verdict.blocking:
        # Return before computing anything per task. Data that has just been
        # declared incomparable must not also yield a per-task verdict --
        # `regressed` would then be True on a comparison we refused to make, and
        # anyone reading the object rather than the exit code gets the wrong
        # answer.
        return verdict

    if len(eligibility.sessions) > 1:
        verdict.notes.append(
            f"this comparison spans {len(eligibility.sessions)} sessions, so warmup and "
            "thermal conditions differ across episodes. Success rates are unaffected; "
            "latency comparisons from it are not trustworthy."
        )
    if len(eligibility.harness_versions) > 1:
        verdict.notes.append(
            f"episodes were produced by {len(eligibility.harness_versions)} different harness "
            f"versions ({', '.join(sorted(eligibility.harness_versions))}). Harness changes "
            "can alter which observation parameters reach the benchmark."
        )

    by_task: dict[tuple[str, str], list[Unit]] = {}
    for unit in eligibility.units:
        by_task.setdefault((unit.key.scene_id, unit.key.task_id), []).append(unit)

    for (scene_id, task_id), units in sorted(by_task.items()):
        cells = contingency(units, a, b, rule)
        bootstrap = clustered_bootstrap(
            units, a, b, resamples=resamples, seed=seed, alpha=alpha
        )
        task_verdict = TaskVerdict(
            task_id=task_id,
            scene_id=scene_id,
            units=len(units),
            rate_a=sum(u.rate(a) for u in units) / len(units),
            rate_b=sum(u.rate(b) for u in units) / len(units),
            cells=cells,
            mcnemar=mcnemar_exact(cells),
            bootstrap=bootstrap,
        )

        if bootstrap.excludes_zero:
            task_verdict.gate = "regressed" if bootstrap.difference < 0 else "improved"

        if len(units) < MIN_UNITS_FOR_CALIBRATED_CI:
            task_verdict.notes.append(
                f"{len(units)} scenarios is below {MIN_UNITS_FOR_CALIBRATED_CI}, where the "
                "percentile interval measures ~7% false positives against a nominal 5%. "
                "Treat a marginal interval as marginal."
            )
        if bootstrap.excludes_zero and task_verdict.mcnemar.p_value >= alpha:
            task_verdict.notes.append(
                "the rate moved but few scenarios flipped their majority "
                f"(McNemar p={task_verdict.mcnemar.p_value:.3f}). A uniform shift, not a set "
                "of scenarios breaking -- see the 2x2 before acting."
            )
        verdict.tasks.append(task_verdict)

    return verdict


def render(verdict: Verdict) -> str:
    """The report. Overlap first, then the 2x2, then the tests, then the gate."""
    a, b = verdict.a, verdict.b
    lines: list[str] = []

    # Overlap above everything else, never in a footnote: silently comparing an
    # intersection while believing you compared everything is the failure this
    # exists to prevent.
    lines.extend(f"  {line}" for line in verdict.eligibility.summary_lines())
    lines.append("")

    for blocker in verdict.blocking:
        lines.append(f"  BLOCKED: {blocker}")
    if verdict.blocking:
        lines.append("")
        return "\n".join(lines)

    for task in verdict.tasks:
        lines.append(f"  {task.scene_id} / {task.task_id}   ({task.units} scenarios)")
        lines.append(f"    {a}: {task.rate_a:.1%}    {b}: {task.rate_b:.1%}")
        cells = task.cells
        lines.append(f"                        {b} fails   {b} succeeds")
        lines.append(
            f"      {a} fails        {cells.both_fail:>9}   {cells.only_b_passes:>11}"
        )
        lines.append(
            f"      {a} succeeds     {cells.only_a_passes:>9}   {cells.both_pass:>11}"
        )
        lines.append(
            f"    McNemar (reported):  p={task.mcnemar.p_value:.4f}  "
            f"n_discordant={task.mcnemar.n}"
        )
        lines.append(
            f"    bootstrap (gates):   {task.bootstrap.difference:+.3f} "
            f"[{task.bootstrap.low:+.3f}, {task.bootstrap.high:+.3f}]"
        )
        marker = {"regressed": "REGRESSED", "improved": "improved", "ok": "no change"}
        lines.append(f"    verdict: {marker[task.gate]}")
        for note in task.notes:
            lines.append(f"      note: {note}")
        lines.append("")

    for note in verdict.notes:
        lines.append(f"  note: {note}")

    return "\n".join(lines)


__all__ = [
    "MIN_UNITS_FOR_CALIBRATED_CI",
    "TaskVerdict",
    "Verdict",
    "evaluate",
    "render",
]
