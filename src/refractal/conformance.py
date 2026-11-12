"""Mechanical properties a backend must have, checkable from outside it.

What this is for, and why it did not exist until now.

Refractal's planner says which scenario each episode is. A backend that drives
someone else's harness has to map those onto whatever the harness indexes by,
and the vla-eval harness indexes init states by its OWN episode counter:
``episode_idx = ep`` over ``range(episodes_per_task)``. Those two numbers
coincide only when a worker holds a task's whole zero-based range. Give it a
slice -- indices 5..9 -- and it runs 0..4 while every recorded row claims 5..9.
Rows describing episodes that did not happen, with nothing anywhere
disagreeing.

The vla-eval bridge refuses that, per group, before the group runs. The
planner refuses the plan that would cause it, which is the primary fix. But
the second layer exists because one adapter happens to have it, not because
anything requires it -- and the corrected wording of the planner's own refusal
says so: "a backend without that check would record the wrong rows silently."

That matters more later than now. A customer-supplied backend is recorded as
``authorship: customer``, which says the measurements came from code nobody
here wrote. It does not say the code checks its own inputs. This is the only
place that can.

Each requirement here takes a CALLABLE and exercises it, rather than reading a
backend's source or asking it to declare compliance. A declaration of
conformance is a label; refusing a bad input is a behaviour, and only the
second is worth requiring. Both directions are checked: a requirement that
only confirms refusals would be satisfied by a backend that refuses
everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


class ConformanceError(AssertionError):
    """A backend does not have a property Refractal's correctness rests on."""


@dataclass(frozen=True)
class Episode:
    """The minimum a conformance fixture needs to look like a planned episode."""

    scenario_hash: str
    seed: int = 0


def _scenarios(indices: Sequence[int]) -> dict[str, dict[str, Any]]:
    return {f"sha256:sc{i}": {"init_state_index": i} for i in indices}


def _episodes(indices: Sequence[int]) -> list[Episode]:
    return [Episode(scenario_hash=f"sha256:sc{i}") for i in indices]


def require_index_contract(
    admit: Callable[[Sequence[Any], Mapping[str, Any]], None],
) -> None:
    """``admit`` must refuse a group that does not start at zero.

    A backend indexing by its own counter must reject any group whose
    ``init_state_index`` values are not exactly ``range(len(group))``, because
    it cannot express that assignment and would silently run a different set of
    initial conditions than the rows claim.

    ``admit`` is the backend's group-admission function -- for the vla-eval
    bridge that is ``check_index_contract(episodes, scenarios)``. It must raise
    on each bad case and must NOT raise on the good one.

    The cases are chosen to be the ones a real planner can produce:

    * ``[5, 6, 7, 8, 9]`` -- a worker holding the second half of a task's
      range, which is exactly what a plan below the partition floor produces.
    * ``[0, 1, 2, 4]`` -- a gap, from a scenario set that filtered one out.
    * ``[1, 2, 3]`` -- off by one, the version that looks most nearly right.

    The contiguous-from-zero case is required to PASS, so a backend that
    refuses every group does not conform by accident.
    """
    good = [0, 1, 2, 3]
    try:
        admit(_episodes(good), _scenarios(good))
    except Exception as exc:  # noqa: BLE001
        raise ConformanceError(
            f"refused a group whose init_state_index values are exactly "
            f"{good}, which is the assignment every correct plan produces: "
            f"{exc!r}. A backend that refuses everything is not conforming, it "
            "is unusable."
        ) from exc

    for bad, why in (
        ([5, 6, 7, 8, 9], "a worker holding a slice of a task's range, which is "
                          "what a plan below the partition floor produces"),
        ([0, 1, 2, 4], "a gap, which a filtered scenario set produces"),
        ([1, 2, 3], "off by one -- the version that looks most nearly right"),
    ):
        try:
            admit(_episodes(bad), _scenarios(bad))
        except Exception:  # noqa: BLE001
            continue
        raise ConformanceError(
            f"accepted a group whose init_state_index values are {bad} ({why}). "
            "A backend that indexes episodes by its own counter would run "
            f"{list(range(len(bad)))} while every row claims {bad}, so the "
            "rows would describe episodes that never ran and nothing would "
            "disagree. Refuse the group instead."
        )


#: Every requirement in this module, so a backend can run the suite rather
#: than remember the list. Named rather than discovered by prefix: a
#: requirement that stops being collected passes by not running, which is the
#: failure this project writes down.
REQUIREMENTS: tuple[tuple[str, Callable[..., None]], ...] = (
    ("index contract", require_index_contract),
)


__all__ = [
    "ConformanceError",
    "Episode",
    "REQUIREMENTS",
    "require_index_contract",
]
