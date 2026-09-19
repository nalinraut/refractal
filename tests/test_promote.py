"""Tier promotion: reusing a smaller run inside a larger one, explicitly.

`tier` stays in `plan_id`, so a smoke run and a full run are different
experiments. This does not merge them. It pools their rows for one comparison,
on request, and refuses when they do not nest.
"""

from __future__ import annotations

import unittest

from refractal.promote import PromotionError, check_nesting, promote
from tests.test_vla_eval_loop import make_plan


def rows_for(plan, ids=None):
    out = []
    for scene in plan.scenes:
        for worker in scene.workers:
            for episode in worker.episodes:
                if ids is not None and episode.episode_id not in ids:
                    continue
                out.append({
                    "episode_id": episode.episode_id,
                    "checkpoint_id": episode.checkpoint_id,
                    "scenario_hash": episode.scenario_hash,
                    "success": True,
                })
    return out


class TestNestingIsExact(unittest.TestCase):
    """The condition is exact, not approximate, and that is the design.

    `episode_id` covers scene_hash, task_hash, scenario_hash, seed and
    checkpoint -- and NOT tier. So a genuine subset matches id-for-id, and
    anything else is a different experiment rather than a near-miss.
    """

    def test_a_strict_subset_promotes(self):
        full = make_plan(scenarios=4, seeds=(0, 1), checkpoints=("pi0", "pi05"))
        every = [e.episode_id for s in full.scenes for w in s.workers for e in w.episodes]
        smoke = rows_for(full, set(every[:6]))
        self.assertEqual(check_nesting(full, smoke), set())
        pooled, promotion = promote(full, [], smoke, "sha256:smoke")
        self.assertEqual(len(pooled), 6)
        self.assertEqual(promotion.episodes, 6)
        self.assertEqual(promotion.remaining, len(every) - 6)

    def test_a_foreign_episode_is_refused(self):
        full = make_plan(scenarios=4, checkpoints=("pi0",))
        smoke = rows_for(full)
        smoke.append({"episode_id": "sha256:not-in-this-plan", "checkpoint_id": "pi0",
                      "scenario_hash": "x", "success": True})
        with self.assertRaises(PromotionError) as ctx:
            promote(full, [], smoke, "sha256:other")
        message = str(ctx.exception)
        self.assertIn("do not nest", message)
        self.assertIn("NOT tier", message)

    def test_a_changed_catalog_is_refused_by_construction(self):
        """The case that makes the exactness worth having: the tier is the same
        and something else moved, so the ids do not match and nothing pools."""
        original = make_plan(scenarios=3, checkpoints=("pi0",))
        # Same shape, different scene -> different scene_hash -> different ids.
        changed = make_plan(scenarios=3, checkpoints=("pi0",), scenes=("other",))
        with self.assertRaises(PromotionError) as ctx:
            promote(original, [], rows_for(changed), "sha256:changed")
        self.assertIn("do not nest", str(ctx.exception))

    def test_an_empty_promotion_is_refused_rather_than_silently_nothing(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        with self.assertRaises(PromotionError) as ctx:
            promote(plan, [], [], "sha256:nothing")
        self.assertIn("actually happened", str(ctx.exception))


class TestDuplicatesAreRefusedRatherThanPooled(unittest.TestCase):
    """The question a set cannot answer, asked separately.

    `check_nesting` does a set difference: it answers "is every promoted id a
    planned id". That is necessary and not sufficient -- it cannot see the same
    id twice, and a doubled row pooled into a comparison weights one scenario
    double.

    Caught by an audit, in code written the same day as the pattern entry saying
    to match the container to the question. Which is why the entry now also says
    WHEN to check: at write time, for any collection keyed by an identity.
    """

    def _rows(self, plan, ids=None):
        return rows_for(plan, ids)

    def test_a_doubled_episode_is_refused(self):
        plan = make_plan(scenarios=3, checkpoints=("pi0",))
        rows = self._rows(plan)
        with self.assertRaises(PromotionError) as ctx:
            promote(plan, [], rows + [rows[0]], "sha256:dup")
        message = str(ctx.exception)
        self.assertIn("duplicated episode id", message)
        self.assertIn("weight it double", message)

    def test_the_message_names_the_run_not_the_symptom(self):
        """`compare` would block downstream saying "duplicate rows", which sends
        the reader to this comparison rather than to the run that supplied them."""
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        rows = self._rows(plan)
        with self.assertRaises(PromotionError) as ctx:
            promote(plan, [], rows + [rows[0]], "sha256:dup")
        self.assertIn("defect in that run", str(ctx.exception))

    def test_rows_and_episodes_are_counted_separately(self):
        """The count used to be called `episodes` and count rows. The name is how
        the bug survived review, so the fix is both."""
        plan = make_plan(scenarios=3, checkpoints=("pi0",))
        _, promotion = promote(plan, [], self._rows(plan), "sha256:ok")
        self.assertEqual(promotion.rows, promotion.episodes)
        self.assertEqual(promotion.episodes, 3)


class TestPromotionFillsGapsRatherThanOverwriting(unittest.TestCase):
    def test_the_targets_own_rows_win(self):
        """If an episode ran under both plans, the one belonging to THIS
        experiment is the one to keep."""
        plan = make_plan(scenarios=4, checkpoints=("pi0",))
        every = [e.episode_id for s in plan.scenes for w in s.workers for e in w.episodes]
        target = [{**r, "success": True} for r in rows_for(plan, set(every[:2]))]
        promoted = [{**r, "success": False} for r in rows_for(plan)]
        pooled, promotion = promote(plan, target, promoted, "sha256:smoke")

        self.assertEqual(len(pooled), len(every), "every episode exactly once")
        self.assertEqual(len({r["episode_id"] for r in pooled}), len(every))
        kept = {r["episode_id"]: r["success"] for r in pooled}
        for episode_id in every[:2]:
            self.assertTrue(kept[episode_id], "the target's own row must win")
        self.assertEqual(promotion.episodes, len(every) - 2)
        self.assertEqual(promotion.rows, promotion.episodes)

    def test_remaining_counts_what_still_has_to_run(self):
        plan = make_plan(scenarios=5, checkpoints=("pi0",))
        every = [e.episode_id for s in plan.scenes for w in s.workers for e in w.episodes]
        pooled, promotion = promote(plan, [], rows_for(plan, set(every[:2])), "sha256:s")
        self.assertEqual(promotion.remaining, len(every) - 2)


class TestAgainstTheRealTierMechanism(unittest.TestCase):
    """The fixtures above construct subsets by hand. This uses `tier` itself.

    Which matters, because the whole claim is that the tiers NEST -- that
    `TIER_FRACTION` subsamples by a deterministic hash fraction so a smoke
    scenario is a regression scenario is a full scenario. If that were false,
    promotion would refuse on every real catalog and the hand-built fixtures
    would never notice.
    """

    def _plan(self, tier):
        import shutil
        import tempfile
        from pathlib import Path

        import yaml

        from refractal.resolve import resolve

        root = Path(tempfile.mkdtemp())
        shutil.copytree(
            Path(__file__).resolve().parents[1] / "examples" / "catalog",
            root, dirs_exist_ok=True,
        )
        run = yaml.safe_load((root / "run.yaml").read_text())
        run["run"]["tier"] = tier
        (root / "run.yaml").write_text(yaml.safe_dump(run))
        return resolve(root, hardware_profile="rtx5090")

    def test_the_tiers_actually_nest(self):
        smoke, full = self._plan("smoke"), self._plan("full")
        self.assertNotEqual(smoke.plan_id, full.plan_id, "tier is in plan_id")
        self.assertLess(smoke.total_episodes, full.total_episodes)
        self.assertEqual(
            check_nesting(full, rows_for(smoke)), set(),
            "smoke must be a strict subset of full, or promotion is impossible "
            "on every real catalog",
        )

    def test_a_smoke_run_promotes_into_a_full_run(self):
        smoke, full = self._plan("smoke"), self._plan("full")
        pooled, promotion = promote(full, [], rows_for(smoke), smoke.plan_id)
        self.assertEqual(promotion.episodes, smoke.total_episodes)
        self.assertEqual(
            promotion.remaining, full.total_episodes - smoke.total_episodes
        )
        self.assertEqual(len(pooled), smoke.total_episodes)

    def test_promotion_does_not_merge_the_identities(self):
        """It pools rows for one comparison. The two runs remain two
        experiments, and their plan_ids are untouched."""
        smoke, full = self._plan("smoke"), self._plan("full")
        before = (smoke.plan_id, full.plan_id)
        promote(full, [], rows_for(smoke), smoke.plan_id)
        self.assertEqual((smoke.plan_id, full.plan_id), before)

    def test_full_does_not_promote_into_smoke(self):
        """The direction matters: a superset does not nest inside a subset."""
        smoke, full = self._plan("smoke"), self._plan("full")
        with self.assertRaises(PromotionError):
            promote(smoke, [], rows_for(full), full.plan_id)


if __name__ == "__main__":
    unittest.main()
