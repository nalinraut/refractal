"""Whether the episodes about to be pooled were measured alike.

The pool-scope half of a pair, and deliberately not the other half.

``execute.metrics.check_extractors_ran`` asks a SESSION question: did every
extractor this catalog declared produce anything while this run was going. It
needs the declaration, and it answers a configuration fault, at the moment the
operator is still watching.

This asks a POOL question: are the episodes about to be compared measured the
same way. It needs only rows, and it answers a comparability fault, about data
that may span a resumed run, several sessions, or a session predating an
extractor entirely.

**They take different inputs on purpose.** One cannot be implemented in terms
of the other, so neither can quietly cover for the other's absence. That is
not tidiness: two enforcement points of one rule make each copy untestable,
because breaking either leaves the other passing and the mutation registers as
nothing.

Why this matters
----------------

A verdict computed from a metric present on 400 rows and absent on 200 is not
the verdict it claims to be. ``Threshold.holds`` counts a missing metric as
failing, correctly -- absence is not evidence of success -- but that makes
those 200 failures indistinguishable from policy failures once the rate is
pooled. The comparison reports a number that means two different things.

This is the overlap-report problem in a new place. ``compare`` already refuses
to pool episodes whose scene geometry differs and reports which seeds were
dropped for pairing; measured-unlike is the same class of defect one column
over.

What it cannot see
------------------

Which metric a success rule thresholded. The rule lives in ``task_hash`` and in
the plan, never in the row, and ``compare`` reads rows. So this checks coverage
rather than relevance: a metric present unevenly is reported whether or not a
rule named it.

That is the conservative direction. An unevenly measured observational metric
is a finding a reader can dismiss in a second; an unevenly measured
*thresholded* one silently changes the verdict, and catching it required no
extra schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class UnevenMetric:
    """One metric, measured on some episodes of a task and not others."""

    task_hash: str
    metric: str
    present: int
    absent: int

    @property
    def episodes(self) -> int:
        return self.present + self.absent

    def __str__(self) -> str:
        return (
            f"{self.metric!r} on task {self.task_hash[:19]}...: "
            f"present on {self.present} of {self.episodes} episode(s), "
            f"absent on {self.absent}"
        )


def _names(row: Mapping[str, Any]) -> set[str]:
    """Metric names on one row, however pyarrow handed the map over.

    A map column reads back as a list of pairs rather than a dict, and a
    reader that accepted only one shape would report every row as unmeasured
    against the other. Both are accepted; neither is assumed.
    """
    metrics = row.get("metrics")
    if not metrics:
        return set()
    if isinstance(metrics, Mapping):
        return {k for k, v in metrics.items() if v is not None}
    return {k for k, v in metrics if v is not None}


def uneven_metric_coverage(rows: Iterable[Mapping[str, Any]]) -> list[UnevenMetric]:
    """Metrics present on some episodes of a task and missing from others.

    Grouped by ``task_hash`` rather than ``task_id``: the hash is what the
    comparison joins on, and two tasks sharing an id across catalogs are not
    the same task. Grouping by the label would merge them and report their
    difference as unevenness.

    Infra failures are excluded. A crashed worker measured nothing, and that
    is already its own column: counting it here would report every run with a
    crash as unevenly measured, which is true and useless.
    """
    by_task: dict[str, list[set[str]]] = {}
    for row in rows:
        if row.get("is_infra_failure"):
            continue
        by_task.setdefault(row.get("task_hash", ""), []).append(_names(row))

    findings: list[UnevenMetric] = []
    for task_hash, per_episode in sorted(by_task.items()):
        seen: set[str] = set()
        for names in per_episode:
            seen |= names
        for metric in sorted(seen):
            present = sum(1 for names in per_episode if metric in names)
            if present != len(per_episode):
                findings.append(UnevenMetric(
                    task_hash=task_hash,
                    metric=metric,
                    present=present,
                    absent=len(per_episode) - present,
                ))
    return findings


def describe(findings: Sequence[UnevenMetric]) -> str:
    """One line per finding, for the block `compare` prints above its verdict."""
    if not findings:
        return ""
    lines = [
        f"{len(findings)} metric(s) measured unevenly across the episodes being "
        "pooled. A verdict computed from a metric present on some episodes and "
        "absent on others is not the rate it appears to be: a missing metric "
        "fails its threshold, and those failures are indistinguishable from "
        "policy failures once pooled."
    ]
    lines += [f"  {f}" for f in findings]
    return "\n".join(lines)


__all__ = ["UnevenMetric", "describe", "uneven_metric_coverage"]
