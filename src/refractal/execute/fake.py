"""A benchmark that runs no physics.

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

from ..schema.plan import PlannedEpisode


def _unit(episode_id: str, salt: str) -> float:
    """A stable uniform in [0, 1) from an episode id and a label."""
    digest = hashlib.sha256(f"{episode_id}|{salt}".encode("utf-8")).digest()
    return struct.unpack(">Q", digest[:8])[0] / float(1 << 64)


@dataclass
class FakeBenchmark:
    """Configurable synthetic outcomes.

    ``success_rate`` may be keyed per ``(checkpoint_id, task_id)`` so a test can
    construct a known difference. A flat rate is the boring default.
    """

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
                "terminal_phase": None,
                "failure_reason": "worker_crashed",
                "is_infra_failure": True,
                "steps": 0,
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
        terminal = self.phases[min(reached, len(self.phases) - 1)]

        span = self.max_steps - self.min_steps
        steps = self.min_steps + int(_unit(episode.episode_id, self.salt + "steps") * span)
        if success:
            steps = self.min_steps + int(steps * 0.6)  # successes finish sooner

        return {
            "success": success,
            "phase_outcomes": outcomes,
            "terminal_phase": terminal,
            # Never null on a failure: null is what a success writes, so a
            # nullable reason cannot distinguish "succeeded" from "the policy
            # failed for no recorded reason".
            "failure_reason": None if success else "policy_failure",
            "is_infra_failure": False,
            "steps": steps,
            "elapsed_sec": round(steps * self.sec_per_step, 4),
        }


__all__ = ["FakeBenchmark"]
