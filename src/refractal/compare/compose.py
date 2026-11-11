"""Assemble a `ComparisonReport` from a verdict and the units behind it.

Everything the distributional layer produces was reachable only from tests:
`analysis_hash`, the grain on every test line, the per-arm scalars with their
parameters, the install provenance. The CLI emitted the binary shape, so none
of it was visible to anyone running the tool -- which is the same defect as a
check that does not print when it finds nothing, one level up.

Two readings, not two reports
-----------------------------

The verdict is the gate: one bit per scenario, Holm-corrected, with an exit
code. It stays, and it stays first, because it needs no convention defended.
This is the distributional reading of the same rows, and it is reported
alongside rather than instead -- the docs already say the McNemar path is the
convention-free headline and simultaneously the lossiest instrument here, and
printing only one of the two would make that sentence unreadable.

No new flags
------------

A budget, a horizon and a ghost penalty could all have been flags. They are
read from the data instead, and where the data cannot answer, the scalar is
reported as undefined WITH ITS REASON rather than omitted or defaulted.

The penalty is the case that matters: what an episode that can never succeed
contributes to a mean is a choice, and there is no measurement to fall back
on. A default would bury that choice in this function, so a run with TERMINAL
episodes and no declared penalty reports the restricted mean as undefined and
says why. That is the honest output, and it is also the one that prompts
somebody to declare a penalty.

Macro-average, never pooled
---------------------------

Every per-arm scalar is the mean of per-cell values, not a statistic over
pooled episodes. Pooling weights a scenario by how many episodes it happens to
have, so a scene with more seeds would quietly count for more -- and the
number would move when the seed budget changed without anything about the
policy changing.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .analysis import AnalysisSpec
from .distribution import DistributionError, estimate, median, rmst, success_at
from .pairing import Eligibility, Unit
from .phases import conditional_rates, orders_differ, traversal_of
from .report import (
    ArmReport,
    Authorship,
    ComparisonReport,
    Reproducible,
    Scalar,
    TestLine,
)
from .verdict import ArmContext, Verdict

#: What any run can currently say about its arms' server configuration.
#:
#: The harness's HELLO handshake does carry some of it -- the model server's
#: class name, and `observation_params`, which for the LeRobot server means
#: `send_state` and `send_wrist_image`. It does NOT carry `chunk_size`, so a
#: handshake check would narrow this and not close it. Until one exists, the
#: honest line is that nothing checked, with the one thing that does checked
#: named.
SERVER_ARGS_UNVERIFIED = (
    "declared, NOT verified: no server-config check runs, so plan_id records "
    "how the arms were meant to be driven rather than how they were. The "
    "declared chunk length is checked against the realized steps/decisions "
    "after the fact, and only where `decisions` was measured"
)

#: Phrases `evaluate` uses for a check that did not run. Matched rather than
#: flagged on the note itself, because the notes are strings by the time they
#: reach here -- and each phrase is produced in exactly one place, so a
#: rewording that broke this would be caught by the test below.
_DECLINE_MARKERS = ("did not run", "not compared", "not checked")


def _budget_of(units: Sequence[Unit], arms: Sequence[str]) -> int | None:
    """The one budget the inferential claim is made at, if there is one.

    None when the rows disagree or do not record it. Not reconstructed from
    the plan beside them: that plan is almost certainly the one that ran, and
    "almost certainly" is not a basis for the bound every functional is
    evaluated under.
    """
    seen = {
        outcome.step_budget
        for unit in units
        for arm in arms
        for outcome in unit.outcomes.get(arm, {}).values()
        if outcome.step_budget is not None
    }
    return seen.pop() if len(seen) == 1 else None


def _macro(values: Sequence[float | None]) -> tuple[float | None, int, int]:
    """Mean over the cells where the functional is defined.

    Returns the mean, how many cells contributed, and how many there were. The
    second and third are reported, because a median averaged over 3 of 8 cells
    is a different claim from one averaged over 8.
    """
    defined = [v for v in values if v is not None]
    if not defined:
        return None, 0, len(values)
    return sum(defined) / len(defined), len(defined), len(values)


def _arm_scalars(
    units: Sequence[Unit], arm: str, spec: AnalysisSpec, budget: int | None
) -> tuple[Scalar, ...]:
    if budget is None:
        reason = ("the rows do not record a single step_budget, so there is no "
                  "bound to evaluate a distribution against")
        return tuple(
            Scalar(name=n, value=None, undefined_reason=reason)
            for n in ("median time-to-success", "success@budget",
                      "restricted mean")
        )

    estimates = []
    for unit in units:
        seeds = unit.outcomes.get(arm)
        if not seeds:
            continue
        try:
            estimates.append(estimate(unit, arm, ghost_penalty=spec.ghost_penalty))
        except DistributionError:
            # A cell that cannot be read is a cell that does not contribute,
            # and the count in the parameters is what makes that visible.
            continue

    med, got, total = _macro([median(e) for e in estimates])
    cells = (("cells", f"{got}/{total}"),)
    out = [
        Scalar(name="median time-to-success", value=med, parameters=cells,
               undefined_reason=None if med is not None else
               "no cell had more than half its episodes succeed"),
        Scalar(name="success@budget",
               value=_macro([success_at(e, budget) for e in estimates])[0],
               parameters=(("budget", str(budget)),) + cells),
    ]

    tau = spec.tau if spec.tau is not None else budget
    try:
        value = _macro([rmst(e, tau, ghost_penalty=spec.ghost_penalty)
                        for e in estimates])[0]
        reason = None
    except DistributionError as exc:
        value, reason = None, str(exc)
    out.append(Scalar(name="restricted mean", value=value,
                      parameters=(("tau", str(tau)),
                                  ("ghost_penalty", str(spec.ghost_penalty)))
                      + cells,
                      undefined_reason=reason))
    return tuple(out)


def _provenance_of(
    episodes: Mapping[str, Sequence[Mapping[str, object]]] | None,
    checkpoints: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Which non-native fields these rows carry, and by which route.

    Summarised per field rather than per row, because a reader needs to know
    that a median rests on a derived `step_to_success` -- not which 1,685
    rows it rests on. The route is kept, not collapsed to "derived": a field
    measured from a different artefact and one reconstructed by arithmetic are
    not the same evidence even where the arithmetic is exact.
    """
    counts: dict[str, dict[str, int]] = {}
    for arm in checkpoints:
        for row in (episodes or {}).get(arm) or ():
            entries = row.get("field_provenance") or ()
            if isinstance(entries, Mapping):
                entries = entries.items()
            for field, route in entries:
                kind = str(route).split(":", 1)[0]
                counts.setdefault(str(field), {})
                counts[str(field)][kind] = counts[str(field)].get(kind, 0) + 1
    out = []
    for field in sorted(counts):
        shown = ", ".join(f"{kind} on {n} row(s)"
                          for kind, n in sorted(counts[field].items()))
        out.append((field, shown))
    return tuple(out)


def compose_report(
    verdict: Verdict,
    eligibility: Eligibility,
    *,
    plan_id: str,
    tier: str,
    spec: AnalysisSpec | None = None,
    arms: Mapping[str, Mapping[str, object]] | None = None,
    context: Mapping[str, ArmContext] | None = None,
    authorship: Authorship | None = None,
    #: Raw rows per arm, for the per-phase reading. Units carry one outcome
    #: per episode and not its phase records, so this cannot come from them.
    episodes: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> ComparisonReport:
    """The distributional reading of a verdict's rows."""
    checkpoints = list(verdict.checkpoints)
    budget = _budget_of(eligibility.units, checkpoints)
    spec = spec or AnalysisSpec(
        dichotomy="majority",
        budget=budget,
        tau=budget,
        family=tuple((verdict.baseline, c) for c in checkpoints
                     if c != verdict.baseline),
    )

    # Read before the arm loop, because the refusal decides whether the
    # per-phase scalars may be reported at all.
    traversals = [traversal_of(arm, (episodes or {}).get(arm) or ())
                  for arm in checkpoints]
    with_phases = [t for t in traversals if t.order]
    chained_refusal = orders_differ(with_phases) if len(with_phases) > 1 else None

    arm_reports = []
    for arm in checkpoints:
        apparatus, total = eligibility.apparatus_by_arm.get(arm, (0, 0))
        scalars = list(_arm_scalars(eligibility.units, arm, spec, budget))
        if not chained_refusal:
            # Chained rates: a normal run IS the chained condition. The
            # isolated one needs a run from canonical states, which no
            # results directory contains, so compounding stays out of reach
            # and is declined with that reason rather than approximated.
            for rate in conditional_rates((episodes or {}).get(arm) or (),
                                          credit=spec.phase_credit):
                scalars.append(Scalar(
                    name=f"phase {rate.phase_id!r} (chained)",
                    value=rate.rate,
                    parameters=(("attempted", str(rate.attempted)),
                                ("credit", spec.phase_credit)),
                    undefined_reason=None if rate.rate is not None else
                    "never attempted in any episode, so there is nothing to "
                    "condition on",
                ))
        arm_reports.append(ArmReport(
            arm=arm,
            descriptor=dict((arms or {}).get(arm) or {}),
            scalars=tuple(scalars),
            apparatus=apparatus,
            episodes=total,
        ))

    # Grouped by arm pair, with every contrast under it naming its own task
    # and scene. One entry per contrast printed eight identical-looking
    # "baseline vs candidate" blocks on the first fixture that had eight
    # scenes -- a reader could not tell which line belonged to which scene, or
    # that they were different lines at all.
    grouped: dict[tuple[str, str], list[TestLine]] = {}
    for contrast in (c for t in verdict.tasks for c in t.contrasts):
        where = (("task", contrast.task_id), ("scene", contrast.scene_id),
                 ("dichotomy", spec.dichotomy))
        grouped.setdefault((contrast.baseline, contrast.candidate), []).extend([
            # The gate, and the only line Holm applies to. Pairing
            # `adjusted_p` with McNemar's raw p printed `p=1 holm=0.2136` on a
            # real run -- an adjustment BELOW the value it adjusts, which is
            # arithmetically impossible and read as a tighter result. The
            # family is the bootstrap contrasts; McNemar is not in it.
            TestLine(
                name="clustered bootstrap (gates)",
                p_value=contrast.bootstrap.p_value,
                # One bit per scenario: the dichotomy collapses every seed
                # before the difference is taken.
                grain="one bit per scenario",
                adjusted_p=contrast.adjusted_p,
                m=verdict.family_size,
                detail=where + (("difference", f"{contrast.bootstrap.difference:+.3f}"),),
            ),
            TestLine(
                name="McNemar (exact)",
                p_value=contrast.mcnemar.p_value,
                grain="one bit per scenario",
                # Deliberately absent. McNemar is reported and never gates, so
                # it is not in the Holm family and has no adjusted value; a
                # number here would imply it was corrected with the others.
                adjusted_p=None,
                detail=where + (("gates", "no"),),
            ),
        ])
    pairwise = [(pair, tuple(lines)) for pair, lines in grouped.items()]

    declined = [n for n in verdict.notes
                if any(m in n for m in _DECLINE_MARKERS)]
    if not any(u.outcomes and any(
            o.step_to_success is not None
            for seeds in u.outcomes.values() for o in seeds.values())
            for u in eligibility.units):
        declined.append(
            "no distributional test ran: no episode records step_to_success, so "
            "every arm's time-to-success distribution is empty. Rows written "
            "before schema v2 carry no step data and cannot be upgraded into it."
        )
    if chained_refusal:
        declined.append(chained_refusal)
    elif not with_phases:
        declined.append(
            "per-phase rates were not read: no episode records phase_records, "
            "so the task declares no phases or the adapter reported none."
        )
    # Out of reach rather than not implemented, and the distinction matters to
    # a reader: compounding compares a chained run against an isolated one,
    # and a results directory holds only the chained condition.
    declined.append(
        "compounding was not computed: it is the isolated phase rates against "
        "the chained episode rate, and an isolated run starts every phase from "
        "a canonical state. These results are the chained condition only."
    )

    editable = sorted({name for arm in checkpoints
                       for name in ((context or {}).get(arm) or ArmContext()).editable})
    checked = [arm for arm in checkpoints
               if ((context or {}).get(arm) or ArmContext()).installs_checked]

    return ComparisonReport(
        plan_id=plan_id,
        analysis_hash=spec.analysis_hash(),
        tier=tier,
        m=verdict.family_size,
        authorship=authorship or Authorship(),
        reproducible=Reproducible(
            complete=not verdict.blocking,
            missing=tuple(verdict.blocking[:1]),
            editable=tuple(editable),
            installs_checked=bool(checkpoints) and len(checked) == len(checkpoints),
            provenance=_provenance_of(episodes, checkpoints),
            server_args_verified=SERVER_ARGS_UNVERIFIED,
        ),
        arms=tuple(arm_reports),
        pairwise=tuple(pairwise),
        traversals=tuple(t.render() for t in with_phases),
        declined=tuple(declined),
        blocking=tuple(verdict.blocking),
    )


__all__ = ["compose_report"]
