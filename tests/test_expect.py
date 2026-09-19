"""The expected-duration estimate: the second number, honest about being second.

`plan.estimated_seconds` is a bound. This is what it will probably take. Two
numbers, each labelled with what it is, rather than one a reader has to guess
the meaning of.
"""

from __future__ import annotations

import unittest

from refractal.expect import expected_seconds
from test_vla_eval_loop import make_plan


def prior(steps_by_task, checkpoints=("pi0", "pi05"), task_prefix=""):
    rows = []
    for task, steps in steps_by_task.items():
        for checkpoint in checkpoints:
            for n in steps:
                rows.append({
                    "task_id": f"{task_prefix}{task}", "checkpoint_id": checkpoint,
                    "steps": n, "is_infra_failure": False,
                })
    return rows


class TestItIsCappedPerEpisodeNotOnTheMean(unittest.TestCase):
    """The bug this was written with, kept as a regression.

    A prior run under a looser budget has episodes longer than this plan allows.
    Capping the MEAN lets those episodes pull the average up before the cap that
    would have stopped them applies. Measured on the real LIBERO data: 100 steps
    against a correct 95, which made the expected duration exactly equal the
    bound and therefore useless.
    """

    def test_long_prior_episodes_are_truncated_individually(self):
        plan = make_plan(scenarios=2, seeds=(0,), checkpoints=("pi0",))
        # max_steps is 300 in the fixture; prior episodes straddle it.
        rows = prior({"0": [100, 100, 900, 900]}, checkpoints=("pi0",))
        got = expected_seconds(plan, rows)
        # Per-episode: mean(100,100,300,300) = 200. On the mean: min(500,300)=300.
        shape = plan.scenes[0].resource_shape
        per_episode = 200 / 1000 * shape.sec_per_1k_steps * len(
            [e for s in plan.scenes for w in s.workers for e in w.episodes])
        self.assertAlmostEqual(got.seconds, int(per_episode), delta=1)

    def test_it_never_exceeds_the_bound(self):
        """An expectation above the bound would be incoherent: the bound costs
        every episode at its cap, and nothing can run longer than its cap."""
        plan = make_plan(scenarios=3, seeds=(0, 1), checkpoints=("pi0",))
        rows = prior({"0": [10_000] * 5}, checkpoints=("pi0",))
        got = expected_seconds(plan, rows)
        self.assertLessEqual(got.seconds, plan.estimated_seconds + 1)


class TestItSaysWhatItLearnedFrom(unittest.TestCase):
    def test_different_checkpoints_are_flagged(self):
        """Episode length is a property of the POLICY. A worse checkpoint times
        out more and runs longer, so borrowing a better one's distribution
        under-estimates exactly when somebody is waiting."""
        plan = make_plan(scenarios=2, checkpoints=("pi0", "pi05"))
        rows = prior({"0": [50, 60]}, checkpoints=("something-else",))
        got = expected_seconds(plan, rows)
        self.assertTrue(got.checkpoints_differ)
        joined = " ".join(got.notes)
        self.assertIn("property of the policy", joined)
        self.assertIn("under-estimate", joined)

    def test_same_checkpoints_are_not_flagged(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        rows = prior({"0": [50, 60]}, checkpoints=("pi0",))
        got = expected_seconds(plan, rows)
        self.assertFalse(got.checkpoints_differ)
        self.assertEqual(got.notes, [])

    def test_unmatched_tasks_are_flagged_and_fall_back(self):
        """task_id is a LABEL. task_hash would be the honest key and moves when
        max_steps moves -- so it never matches across the change this estimate is
        most useful for. The fallback is pooled, and it has to say so."""
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        rows = prior({"0": [50, 60]}, checkpoints=("pi0",), task_prefix="renamed-")
        got = expected_seconds(plan, rows)
        self.assertEqual(got.matched_tasks, 0)
        self.assertIn("labels", " ".join(got.notes))
        self.assertGreater(got.seconds, 0, "it must still produce a number")

    def test_infra_failures_are_excluded(self):
        """A crashed container is not evidence about how long an episode takes,
        the same reason it is excluded from a success denominator."""
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        rows = prior({"0": [100, 100]}, checkpoints=("pi0",))
        rows.append({"task_id": "0", "checkpoint_id": "pi0", "steps": 1,
                     "is_infra_failure": True})
        got = expected_seconds(plan, rows)
        self.assertEqual(got.sample, 2)

    def test_no_usable_prior_returns_none(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        self.assertIsNone(expected_seconds(plan, []))
        self.assertIsNone(expected_seconds(plan, [
            {"task_id": "0", "checkpoint_id": "pi0", "steps": 0, "is_infra_failure": False}
        ]))


class TestItStaysOutOfThePlan(unittest.TestCase):
    def test_the_estimate_is_not_written_into_plan_json(self):
        """It is fitted to particular hardware and particular checkpoints, which
        makes it render-time by the project's own rule. A plan is portable."""
        import json

        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        before = plan.plan_id
        rows = prior({"0": [50]}, checkpoints=("pi0",))
        expected_seconds(plan, rows)
        document = json.loads(plan.to_json())
        self.assertEqual(plan.plan_id, before)
        self.assertNotIn("expected_seconds", document)
        self.assertNotIn("expectation", document)


if __name__ == "__main__":
    unittest.main()
