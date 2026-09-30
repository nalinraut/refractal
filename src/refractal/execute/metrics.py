"""Running metric extractors, and turning what they produce into a verdict.

Where the two additions meet. An adapter hands over whatever state it has; the
extractors declared on the scene turn that into named numbers; the rule
declared on the task turns those numbers into ``success``.

**At write time, never at read time**, and the second half is the reason. The
rule is inside ``task_hash``, so editing a threshold makes a different
experiment. Computing the verdict when results are read would let that edit
silently re-score episodes already recorded, which is the one thing the
identity model exists to prevent. Deciding at write time makes it impossible
rather than discouraged.

It also leaves ``compare`` untouched. It reads ``success`` exactly as it always
has and never sees a metric, so McNemar, Holm, the clustered bootstrap and the
k-checkpoint outcome patterns all work with no change at all.

The extractor contract
----------------------

::

    def extractor(state, context) -> Mapping[str, float]

``state`` is whatever the adapter collected: it is the adapter's type and this
module never inspects it. ``context`` carries the declared ``args`` plus the
episode's own ``task_id``, ``instruction`` and ``provider_ref``, which is what
lets one extractor declared on a scene read a goal that differs per task.

Returning a name the extractor did not declare in ``produces`` is refused. The
declaration is what ``refractal plan`` checks a success rule against, and a
declaration that does not match what runs makes that check worthless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..schema.errors import RefractalError
from ..schema.importstr import resolve_import_string
from ..schema.models import MetricExtractor, SuccessRule


class MetricError(RefractalError):
    """An extractor could not run, or produced something it never declared."""


@dataclass(frozen=True)
class Measured:
    """What the extractors produced for one episode, and who produced it."""

    #: Metric name to value. Missing rather than null when an extractor did not
    #: produce one: a null would claim it ran and measured nothing.
    values: dict[str, float]
    #: Import strings of the extractors that returned at least one value. The
    #: request-and-fulfilment pair for `metrics_from`, so a declared extractor
    #: that silently produced nothing is visible per episode rather than
    #: inferred from an absence.
    ran: list[str]


def extract(
    extractors: Sequence[MetricExtractor],
    state: Any,
    *,
    task_id: str = "",
    instruction: str = "",
    provider_ref: Mapping[str, Any] | None = None,
) -> Measured:
    """Run every declared extractor over one episode's state."""
    values: dict[str, float] = {}
    ran: list[str] = []
    for declared in extractors:
        fn = resolve_import_string(declared.extractor)
        context = {
            "args": dict(declared.args),
            "task_id": task_id,
            "instruction": instruction,
            "provider_ref": dict(provider_ref or {}),
        }
        produced = fn(state, context)
        if produced is None:
            continue
        undeclared = sorted(set(produced) - set(declared.produces))
        if undeclared:
            # Refused rather than dropped. `refractal plan` checks a success
            # rule against `produces`, so an extractor writing names it never
            # declared makes that check meaningless in the direction that
            # matters: a rule could threshold a metric the catalog says
            # nothing produces, be refused at plan time, and be perfectly
            # satisfiable at run time.
            raise MetricError(
                f"extractor {declared.extractor!r} produced "
                f"{', '.join(repr(u) for u in undeclared)}, which it does not "
                f"declare. Declared: {', '.join(declared.produces)}. Add the "
                "name to 'produces' so a success rule can be checked against it."
            )
        kept = {k: float(v) for k, v in produced.items() if v is not None}
        if kept:
            ran.append(declared.extractor)
        values.update(kept)
    return Measured(values=values, ran=ran)


def verdict(rule: SuccessRule | None, measured: Measured, fallback: bool) -> bool:
    """The episode's ``success``.

    ``fallback`` is the provider's own boolean, and it stands when no rule is
    declared. That is not a default so much as the existing behaviour named:
    a wrapped suite owns its definition of success, and every episode recorded
    before this existed was decided that way.
    """
    if rule is None:
        return fallback
    return rule.holds(measured.values)


def check_extractors_ran(
    declared: Sequence[MetricExtractor], rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Declared extractors that produced nothing across every episode.

    A request and its fulfilment, compared. An extractor that never ran looks
    identical to one that ran and found nothing worth reporting, and the
    difference is the whole run: a metric absent everywhere makes a threshold
    on it fail everywhere, which reads as a policy that never succeeds.

    Returns the names rather than raising, so the caller decides whether a
    partial run is an error. Nothing is inferred from the metrics column
    itself: the check reads ``metrics_from``, which records what ran, because
    an absent value cannot distinguish "did not run" from "ran and returned
    nothing".
    """
    if not declared:
        return []
    seen: set[str] = set()
    for row in rows:
        for name in row.get("metrics_from") or ():
            seen.add(name)
    return [d.extractor for d in declared if d.extractor not in seen]


__all__ = [
    "Measured",
    "MetricError",
    "check_extractors_ran",
    "extract",
    "verdict",
]
