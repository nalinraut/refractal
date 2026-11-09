"""A provider that runs no physics.

Step 2's whole point: validate the results schema, the writer, resume and the
partition layout with no simulator, no Docker and no policy in sight. If the
plumbing is wrong, it is wrong here, where a test round-trips in milliseconds
instead of after a GPU spins up.

It stays useful permanently. Every test of `compare` runs against data this
produced, because it is the only source where the ground truth is *known* --
you can ask for "checkpoint B is genuinely 10 points better on task X and
identical on task Y" and then assert the verdict recovers it.

Outcomes are a deterministic function of `episode_id`, never of a global RNG.
That means a resumed run produces exactly the results the interrupted one would
have, so resume can be tested for correctness rather than merely for not
crashing.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

from ..schema.outcome import Outcome, TerminalBasis, TerminationMap
from ..schema.plan import PlannedEpisode


def _unit(episode_id: str, salt: str) -> float:
    """A stable uniform in [0, 1) from an episode id and a label."""
    digest = hashlib.sha256(f"{episode_id}|{salt}".encode("utf-8")).digest()
    return struct.unpack(">Q", digest[:8])[0] / float(1 << 64)


#: The fake's whole vocabulary, declared as a total map.
#:
#: Two entries, because the fake has two ways to end and no irrecoverable
#: state. A two-entry total map is a valid map, and this is a faithful preview
#: of a LIBERO map, whose real vocabulary is also close to success-or-timeout.
#:
#: `worker_crashed` is here rather than falling through the totality rule: it
#: is an out-of-vocabulary sentinel for a simulated infrastructure failure, and
#: APPARATUS is what it means. Declaring it keeps it auditable instead of
#: implicit.
FAKE_TERMINATION_MAP = TerminationMap(
    "fake",
    {
        "predicate_satisfied": (Outcome.SUCCESS, None),
        "horizon": (Outcome.BUDGET_EXHAUSTED, None),
        "worker_crashed": (Outcome.APPARATUS, None),
        # Two terminals, one per basis, because the functional layer lets
        # ghost_penalty treat them differently and a fixture with only one
        # kind cannot exercise that.
        "object_unrecoverable": (Outcome.TERMINAL, TerminalBasis.TASK),
        "simulator_gave_up": (Outcome.TERMINAL, TerminalBasis.SIMULATOR),
    },
)


@dataclass
class FakeProvider:
    """Configurable synthetic outcomes.

    ``success_rate`` may be keyed per ``(checkpoint_id, task_id)`` so a test can
    construct a known difference. A flat rate is the boring default.
    """

    #: The provider declares its own map. `local` reads it off the
    #: instance rather than importing it, so a different provider in the
    #: same backend brings its own vocabulary with it.
    TERMINATION_MAP = FAKE_TERMINATION_MAP

    success_rate: float = 0.7
    per_task: dict[tuple[str, str], float] = field(default_factory=dict)
    #: Infra failures are injected on purpose. They must leave the denominator,
    #: so something has to be able to produce them.
    infra_failure_rate: float = 0.0
    #: Per-scenario difficulty, which is what makes this fixture able to test
    #: clustering at all.
    #:
    #: With this at zero every episode is an independent draw, so seeds within a
    #: scenario are uncorrelated and a clustered analysis gives almost exactly
    #: the same answer as a wrong unclustered one -- meaning the fixture cannot
    #: tell them apart. Real scenarios are nothing like that: a pose near the
    #: edge of the workspace fails on all five seeds, and an easy one succeeds
    #: on all five. That correlation is precisely why five seeds at one pose are
    #: one observation rather than five, and a fixture that omits it cannot
    #: exercise the mistake it exists to catch.
    #:
    #: The offset is a stable function of ``scenario_hash``, so the same
    #: scenario is equally hard for every checkpoint -- which is what keeps the
    #: injected per-task difference the *only* systematic difference between
    #: them.
    scenario_spread: float = 0.0
    #: Per-(scenario, checkpoint) difficulty: the checkpoint x scenario
    #: interaction, and the only thing that makes clustering matter here.
    #:
    #: ``scenario_spread`` alone is shared by every checkpoint, so a *paired*
    #: difference cancels it and an unclustered test stays calibrated -- which is
    #: measurable, and surprising if you expected the textbook warning to apply
    #: unconditionally. What does not cancel is a scenario being hard for one
    #: checkpoint and easy for another, correlated across that scenario's seeds.
    #: That is what a real regression looks like: it does not lower every
    #: scenario a little, it destroys a few and leaves the rest alone.
    #:
    #: Under the null this averages to zero, so any systematic difference is
    #: still only what ``per_task`` injects -- but the per-scenario differences
    #: now carry variance that an episode-level test cannot see.
    interaction_spread: float = 0.0
    #: Redraws every outcome without touching identity. Only useful for
    #: repeating an experiment many times to measure how often a statistical
    #: test is wrong -- which is the only way to choose between tests.
    salt: str = ""
    phases: tuple[str, ...] = ("reach", "grasp", "transport", "place")
    #: Fraction of episodes that reach an irrecoverable state BEFORE the
    #: budget. Zero by default, so every existing test keeps its distribution;
    #: the Phase 2 fixtures turn it on, because nothing else in this provider
    #: produces a TERMINAL and ghost_penalty has nothing to weigh without one.
    terminal_rate: float = 0.0
    #: Of those, the share whose basis is SIMULATOR rather than TASK.
    simulator_basis_share: float = 0.5
    #: Multiplies how long a SUCCESS takes, per checkpoint or flat. This is
    #: what lets two arms sit at the same success-at-budget while differing in
    #: median and RMST -- one finishing briskly, the other scraping in late --
    #: which is the case the whole phase exists to detect.
    speed: float | dict[str, float] = 1.0
    min_steps: int = 40
    max_steps: int = 400
    sec_per_step: float = 0.004

    def rate_for(self, episode: PlannedEpisode) -> float:
        """Success probability for this episode: task-level rate, shifted by scenario."""
        base = self.per_task.get((episode.checkpoint_id, episode.task_id), self.success_rate)
        offset = 0.0
        if self.scenario_spread:
            offset += self.scenario_spread * (
                2 * _unit(episode.scenario_hash, self.salt + "difficulty") - 1
            )
        if self.interaction_spread:
            offset += self.interaction_spread * (
                2
                * _unit(
                    f"{episode.scenario_hash}|{episode.checkpoint_id}",
                    self.salt + "interaction",
                )
                - 1
            )
        return min(1.0, max(0.0, base + offset))

    def run(self, episode: PlannedEpisode) -> dict:
        """Produce one episode outcome. Pure: same episode_id, same result."""
        if _unit(episode.episode_id, self.salt + "infra") < self.infra_failure_rate:
            return {
                "success": False,
                "phase_outcomes": [],
                # Always set: it is the input to the termination map, and a null
                # would be an episode whose outcome cannot be derived.
                "termination_reason": "worker_crashed",
                # Free-form detail BEYOND the flag, and there is none. It used
                # to repeat the flag; nothing read the copy.
                "failure_reason": None,
                "is_infra_failure": True,
                "steps": 0,
                "decisions": 0,
                "elapsed_sec": 0.0,
            }

        roll = _unit(episode.episode_id, self.salt + "outcome")
        success = roll < self.rate_for(episode)

        # A failure stops at some phase; a success clears them all. Phases are
        # sequential, so a later one can only be true if every earlier one is.
        if success:
            reached = len(self.phases)
        else:
            reached = int(_unit(episode.episode_id, self.salt + "phase") * len(self.phases))
        outcomes = [(name, i < reached) for i, name in enumerate(self.phases)]
        # NOT the phase reached. This wrote `self.phases[reached]` into a
        # column the schema documents as "the reason, where success is the
        # verdict" -- a provider putting the wrong semantic into a
        # documented column, which is a defect rather than a second
        # legitimate use. How far an episode got is `phase_outcomes`, which
        # already carries it: over all 360 affected rows on disk, every
        # phase-name value was already a key in its own phase_outcomes.
        #
        # The fake has no irrecoverable state, so its vocabulary is two
        # words and its map is two entries. A two-entry total map is a
        # valid map, and a faithful preview of what a LIBERO map looks
        # like: success-or-timeout.
        # The fake has no irrecoverable state unless `terminal_rate` asks for
        # one, and its vocabulary stays small: a two-entry map was a faithful
        # preview of LIBERO's, and the terminals exist so the functional layer
        # has something for ghost_penalty to weigh.
        #
        # A success is NOT the phase reached. This wrote `self.phases[reached]`
        # into a column documented as "the reason, where success is the
        # verdict" -- the wrong semantic in a documented column, fixed at the
        # source and migrated out of the 360 rows it had already written.
        terminal_roll = _unit(episode.episode_id, self.salt + "terminal")
        terminal = not success and terminal_roll < self.terminal_rate
        if success:
            termination = "predicate_satisfied"
        elif terminal:
            basis = _unit(episode.episode_id, self.salt + "basis")
            termination = ("simulator_gave_up"
                           if basis < self.simulator_basis_share
                           else "object_unrecoverable")
        else:
            termination = "horizon"

        # Duration. A MODE AND A TAIL, not uniform noise: three uniforms summed
        # concentrate near the middle and still reach the ends, so quantiles
        # and RMST read differently from each other. Uniform lengths make the
        # median and the mean agree and the whole phase untestable.
        u = (_unit(episode.episode_id, self.salt + "d1")
             + _unit(episode.episode_id, self.salt + "d2")
             + _unit(episode.episode_id, self.salt + "d3")) / 3.0
        span = self.max_steps - self.min_steps

        if success:
            # `speed` stretches the whole distribution without moving the
            # success RATE, which is the two-arm case this fixture exists for:
            # same success-at-budget, different median and RMST.
            speed = (self.speed.get(episode.checkpoint_id, 1.0)
                     if isinstance(self.speed, dict) else self.speed)
            steps = self.min_steps + int(u * span * 0.55 * speed)
            steps = max(1, min(steps, episode.max_steps))
        elif terminal:
            # Strictly before the budget: a terminal AT the budget is
            # indistinguishable from running out of it, and the two are the
            # distinction this whole change exists to record.
            steps = max(1, min(int(u * (episode.max_steps - 1)) + 1,
                               episode.max_steps - 1))
        else:
            steps = episode.max_steps

        return {
            "success": success,
            "phase_outcomes": outcomes,
            "termination_reason": termination,
            # Free-form detail beyond the flag, not a copy of it. The old
            # comment argued a null could not tell "succeeded" from "failed
            # for no recorded reason" -- right, and now termination_reason's
            # job, where it is never null.
            "failure_reason": None,
            "is_infra_failure": False,
            "steps": steps,
            # One inference per step: this provider has no chunking, so the
            # realized chunk length is exactly 1.
            "decisions": steps,
            "elapsed_sec": round(steps * self.sec_per_step, 4),
        }


__all__ = ["FakeProvider"]
