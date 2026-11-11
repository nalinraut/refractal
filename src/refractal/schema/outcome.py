"""How an episode ended, as four values every adapter maps into.

``success`` is a boolean, and a boolean cannot separate the two ways an
episode fails to finish. An episode that ran out its step budget with the
scene still recoverable is censored at T > budget; one that put the object on
the floor has T = infinity. Both record ``success: false`` today, and every
distributional statistic needs them apart -- the first contributes its budget
to a time-to-success distribution, the second contributes a penalty the caller
has to choose.

Four values, and the fourth is not an outcome at all:

``SUCCESS``
    The predicate was satisfied, at ``step_to_success``.

``BUDGET_EXHAUSTED``
    Ran to ``step_budget`` without satisfying it, scene state still
    recoverable. Censoring.

``TERMINAL``
    The scene reached a state where the predicate cannot be satisfied.
    T = infinity.

``APPARATUS``
    The simulator crashed, the policy server timed out, physics produced NaN.
    Not the policy's fault, so it enters no distribution and no test, and is
    reported as a rate instead. A climbing rate is a bug signal.

Why an adapter never writes these directly
------------------------------------------

An adapter declares a TERMINATION MAP -- its own vocabulary to these four --
and the map is a hashed input. If adapters produced the enum in code the
interpretation would stop being a declared artifact, and two runs that read
the same simulator differently could not be told apart mechanically. The map
exists so that difference is refusable rather than invisible.
"""

from __future__ import annotations

import enum
import math


class Outcome(str, enum.Enum):
    """The four values. ``str`` so it round-trips through Parquet as itself."""

    SUCCESS = "SUCCESS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TERMINAL = "TERMINAL"
    APPARATUS = "APPARATUS"


class TerminalBasis(str, enum.Enum):
    """On what authority a TERMINAL is terminal.

    ``TERMINAL`` asserts that success became impossible, which is a claim
    about the TASK. A simulator ending an episode is a claim about the
    SIMULATOR, and for some flags those diverge: a real vehicle comes off a
    kerb and keeps driving, so MetaDrive's ``crash_sidewalk`` is terminal
    because MetaDrive says so, not because the route became unachievable.

    This changes no statistic. Both kinds get T = infinity and the ghost
    penalty, so they are identical to every test here. What it changes is what
    a write-up can honestly claim, and it makes a sensitivity test expressible:
    does the ranking survive excluding SIMULATOR-basis terminals rather than
    penalising them?

    Part of the hashed termination map, for the same reason the mapping is.
    """

    TASK = "TASK"
    SIMULATOR = "SIMULATOR"


#: What `success` and `is_infra_failure` are on each outcome. They stay in the
#: schema and stay derived from here, so the three existing readers --
#: pairing.py, sweep.py and the vla-eval runner -- are untouched by this
#: change. Deprecating them is later work and a separate decision.
DERIVED: dict[Outcome, tuple[bool, bool]] = {
    Outcome.SUCCESS: (True, False),
    Outcome.BUDGET_EXHAUSTED: (False, False),
    Outcome.TERMINAL: (False, False),
    Outcome.APPARATUS: (False, True),
}


def derived_flags(outcome: Outcome) -> tuple[bool, bool]:
    """``(success, is_infra_failure)`` for an outcome.

    Total by construction: a new enum member with no entry raises here rather
    than silently reading as a failure that is not an apparatus failure, which
    is what a ``.get(outcome, (False, False))`` would have done.
    """
    try:
        return DERIVED[outcome]
    except KeyError:  # pragma: no cover - unreachable while DERIVED is total
        raise ValueError(
            f"{outcome!r} has no derived (success, is_infra_failure). Every "
            "Outcome must declare them: a missing entry would otherwise read "
            "as an ordinary policy failure."
        ) from None




class TerminationMapError(ValueError):
    """An adapter's map does not cover what the adapter can emit."""


class TerminationMap:
    """One adapter's vocabulary, declared as a total map into ``Outcome``.

    Total over the adapter's own flags. An unmapped flag RAISES: it is not
    defaulted to TERMINAL, to APPARATUS, or to anything else, because a silent
    default is how a simulator-specific flag becomes a fabricated statistic.
    The map is small and writing it out is the point.

    Out-of-vocabulary sentinels get explicit entries rather than falling
    through the totality rule. MetaDrive's adapter returns ``"unknown"`` when
    the episode ended and no flag was set, and raises when two are set at
    once; both are apparatus problems rather than policy outcomes, and both
    are declared here so they are auditable instead of implicit.

    Hashed. ``digest()`` goes beside the existing harness-surface check, and
    comparing two arms under different maps is refused with no override --
    unlike a harness surface, a different map changes the numbers directly
    rather than putting them in doubt.
    """

    def __init__(
        self,
        adapter: str,
        mapping: dict[str, tuple[Outcome, TerminalBasis | None]],
    ) -> None:
        self.adapter = adapter
        for flag, entry in mapping.items():
            outcome, basis = entry
            if outcome is Outcome.TERMINAL and basis is None:
                raise TerminationMapError(
                    f"{adapter}: {flag!r} maps to TERMINAL with no terminal_basis. "
                    "TERMINAL asserts success became impossible, which is a claim "
                    "about the task; a simulator merely ending an episode is a "
                    "claim about the simulator. Declare TASK or SIMULATOR."
                )
            if outcome is not Outcome.TERMINAL and basis is not None:
                raise TerminationMapError(
                    f"{adapter}: {flag!r} maps to {outcome.value} and also "
                    f"declares terminal_basis {basis.value}. Only TERMINAL has one."
                )
        self.mapping = dict(mapping)

    def resolve(self, flag: str) -> tuple[Outcome, TerminalBasis | None]:
        if flag not in self.mapping:
            raise TerminationMapError(
                f"{self.adapter}: {flag!r} is not in the termination map. Every "
                f"flag the adapter can emit must be declared; the map covers "
                f"{sorted(self.mapping)}. An unmapped flag is not defaulted, "
                "because a default would make a simulator-specific state into a "
                "statistic nobody chose."
            )
        return self.mapping[flag]

    def covers(self, vocabulary: "tuple[str, ...] | list[str]") -> None:
        """Raise unless every flag the adapter declares is mapped.

        Checked against the adapter's declared vocabulary rather than against
        what has been observed. A flag that has never fired is exactly the one
        whose mapping nobody has thought about.
        """
        missing = sorted(set(vocabulary) - set(self.mapping))
        if missing:
            raise TerminationMapError(
                f"{self.adapter}: termination map is not total. Unmapped: "
                f"{missing}. These can be emitted and have no outcome."
            )

    def document(self) -> dict:
        """The canonical form that gets hashed."""
        return {
            "adapter": self.adapter,
            "mapping": {
                flag: {
                    "outcome": outcome.value,
                    "terminal_basis": basis.value if basis else None,
                }
                for flag, (outcome, basis) in sorted(self.mapping.items())
            },
        }

    def digest(self) -> str:
        from .canonical import hash_obj

        return hash_obj(self.document())


class OutcomeInvariantError(ValueError):
    """A row's outcome and its step fields disagree."""


def validate_row(
    *,
    outcome: Outcome,
    step_to_success: int | None,
    step_budget: int | None,
    steps: int,
    decisions: int | None = None,
) -> None:
    """The invariants, checked where the row is written.

    Enforced on WRITE rather than by the Arrow schema, for one reason that is
    not a style choice: ``step_budget`` is NOT NULL for every row this version
    writes, but rows written before it exist and have no budget recorded
    anywhere. Making the column non-nullable in Arrow would make those rows
    unreadable, and reconstructing a budget for them would be inventing the
    number every later functional is bounded by. So the column is nullable in
    the schema and mandatory at the point of writing, and `schema_version`
    tells the two eras apart.
    """
    if step_budget is None:
        raise OutcomeInvariantError(
            "step_budget is required on every episode written by this version. "
            "It is recorded rather than read from the plan at analysis time, "
            "because every functional over the resulting distribution is "
            "bounded by the smallest budget in the cell."
        )

    if outcome is Outcome.SUCCESS:
        if step_to_success is None:
            raise OutcomeInvariantError(
                "SUCCESS with no step_to_success. The step the predicate was "
                "satisfied at is the observation; without it the episode "
                "contributes nothing to a time-to-success distribution."
            )
        if step_to_success > step_budget:
            raise OutcomeInvariantError(
                f"step_to_success {step_to_success} exceeds step_budget "
                f"{step_budget}: the predicate cannot have been satisfied after "
                "the episode was required to stop."
            )
    elif step_to_success is not None:
        raise OutcomeInvariantError(
            f"{outcome.value} with step_to_success={step_to_success}. Only "
            "SUCCESS has one; a non-null here would put a success time into a "
            "distribution for an episode that did not succeed."
        )

    if decisions is None:
        raise OutcomeInvariantError(
            "decisions is required on every episode written by this version. "
            "steps / decisions is the realized chunk length, and it is the only "
            "measured answer to how many environment steps one policy inference "
            "drove. A declared chunk length checked against nothing is an "
            "assertion."
        )
    if steps > 0 and decisions <= 0:
        raise OutcomeInvariantError(
            f"decisions={decisions} with {steps} step(s): an episode that "
            "stepped made at least one inference call."
        )
    # steps == 0 and decisions == 0 is legitimate and is what an APPARATUS
    # failure before the first step looks like -- a worker that crashed on
    # start-up inferred nothing and stepped nothing.

    if outcome is Outcome.BUDGET_EXHAUSTED:
        # The STRICT form, which holds whenever the policy decides once per
        # step. Every adapter in this repo does: MetaDrive calls act() then
        # env.step() once each iteration, and the vla-eval path runs one
        # conn.act() and one apply_action per loop with is_done() checked
        # between -- LIBERO's step() even asserts a 7-element action, so a
        # chunk raises rather than unrolling.
        #
        # The general form, for when a chunking adapter first appears:
        #
        #     step_budget <= steps  and  steps - step_budget < steps / decisions
        #
        # It is deliberately NOT implemented yet. Written against no example it
        # would be a guess, and the strict check failing loudly on the first
        # chunked episode is a better way to learn the real shape than a
        # permissive rule that silently accepts whatever arrives.
        if decisions == steps and steps != step_budget:
            raise OutcomeInvariantError(
                f"BUDGET_EXHAUSTED after {steps} steps with a budget of "
                f"{step_budget}, and one decision per step. The episode did not "
                "stop at its budget, so something other than the budget ended it."
            )
        if decisions != steps:
            raise OutcomeInvariantError(
                f"BUDGET_EXHAUSTED with {steps} steps over {decisions} decisions "
                f"(realized chunk length {steps / decisions:.2f}). Only the "
                "one-decision-per-step form is enforced; the chunked form is "
                "specified in this function's comments and should be implemented "
                "deliberately now that an adapter needs it, rather than inferred "
                "from this row."
            )


#: Chunk lengths an arm's rows are consistent with: ``(lo, hi)``, where ``hi``
#: of None means unbounded above. ``None`` in place of the pair means the rows
#: contradict each other -- no single chunk length explains them all.
ChunkRange = "tuple[int, int | None] | None"


def chunk_lengths_consistent_with(steps: int, decisions: int) -> ChunkRange:
    """Chunk lengths k for which ``ceil(steps / k) == decisions``, as a range.

    ``ceil(steps/k) == d`` rearranges to ``steps/d <= k < steps/(d-1)``, so one
    row admits a contiguous interval:

        104 steps / 11 calls -> [9.45, 10.4)  -> 10..10
         81 steps /  9 calls -> [9, 10.125)   ->  9..10
        104 steps / 21 calls -> [4.95, 5.2)   ->  5..5

    A range rather than a set, for two reasons. **One decision leaves the upper
    bound unbounded** -- ``ceil(steps/k) == 1`` for every k at least as long as
    the episode -- and an enumerated set silently truncated that to ``{steps}``,
    which would have falsely refused a correct declaration on any suite with
    episodes shorter than its chunk length. And it avoids walking ``range(1,
    steps + 1)`` per row, which is wasted work on a 220-step episode.
    """
    if decisions <= 0 or steps <= 0:
        return None
    lo = -(-steps // decisions)                      # ceil(steps / decisions)
    if decisions == 1:
        return (lo, None)
    hi = -(-steps // (decisions - 1)) - 1            # ceil(steps/(d-1)) - 1
    return (lo, hi) if lo <= hi else None


def intersect_chunk_ranges(a: ChunkRange, b: ChunkRange) -> ChunkRange:
    """Both rows' ranges, or None where they cannot both hold."""
    if a is None or b is None:
        return None
    lo = max(a[0], b[0])
    hi = a[1] if b[1] is None else (b[1] if a[1] is None else min(a[1], b[1]))
    if hi is not None and lo > hi:
        return None
    return (lo, hi)


def validate_phase_records(
    records, *, steps: int, phase_names=None
) -> None:
    """One record per phase, and the two step facts in a possible order.

    **One attempt per phase per episode**, enforced here rather than left to
    convention, because the whole of per-phase reporting rests on it:
    ``P(all) = prod P(k | prefix)`` is a product over phases, and a second
    attempt at one of them makes the left side a different quantity from the
    right. A repeated ``phase_id`` is refused rather than merged.

    Genuine repetition is expressed as distinct phases. "Pick item 3" is not a
    second attempt at "pick item 1" -- the scene state differs -- so distinct
    phases is the more honest encoding as well as the one that keeps the
    product well formed. It also defers loops, deliberately.

    `first_satisfied_step` is permitted on a phase whose `outcome` is a
    failure: that is exactly the case the second field exists for, a predicate
    that held and then stopped holding. What is refused is the pair that
    cannot have happened in time.
    """
    seen: set[str] = set()
    for record in records or ():
        phase_id = record.get("phase_id")
        if not phase_id:
            raise OutcomeInvariantError(
                "a phase record with no phase_id: phases are paired by id and "
                "never by position, so a record without one cannot be paired "
                "with anything."
            )
        if phase_id in seen:
            raise OutcomeInvariantError(
                f"phase {phase_id!r} has more than one record in this episode. "
                "One attempt per phase per episode: a retry inside a phase "
                "belongs to that phase, and a second record would break the "
                "conditional product per-phase rates are built on. Genuine "
                "repetition is a distinct phase."
            )
        seen.add(phase_id)

        if phase_names is not None and phase_id not in phase_names:
            raise OutcomeInvariantError(
                f"phase {phase_id!r} is recorded but the task does not declare "
                f"it. Declared: {sorted(phase_names)}."
            )

        first = record.get("first_satisfied_step")
        start, end = record.get("start_step"), record.get("end_step")
        if first is None:
            continue
        if not record.get("attempted"):
            raise OutcomeInvariantError(
                f"phase {phase_id!r} was not attempted and has "
                f"first_satisfied_step={first}. A predicate cannot have held "
                "during a phase that never ran."
            )
        if start is not None and first < start:
            raise OutcomeInvariantError(
                f"phase {phase_id!r} reports first_satisfied_step={first} "
                f"before its start_step={start}."
            )
        if first > steps:
            raise OutcomeInvariantError(
                f"phase {phase_id!r} reports first_satisfied_step={first} in an "
                f"episode of {steps} step(s)."
            )
        if end is not None and first > end:
            raise OutcomeInvariantError(
                f"phase {phase_id!r} reports first_satisfied_step={first} after "
                f"its end_step={end}. The predicate cannot first have held "
                "after the phase stopped."
            )


__all__ = [
    "Outcome",
    "TerminalBasis",
    "TerminationMap",
    "TerminationMapError",
    "DERIVED",
    "derived_flags",
    "ChunkRange",
    "chunk_lengths_consistent_with",
    "intersect_chunk_ranges",
    "validate_phase_records",
    "validate_row",
    "OutcomeInvariantError",
]
