"""Metrics to a verdict: the rule, its identity, and the refusal.

Every check here is written so it can be seen to fail. A guard nobody has
watched reject something is a guard nobody knows works, which this project has
paid for once already.
"""

from __future__ import annotations

import unittest

from refractal.schema.errors import CatalogError
from refractal.schema.identity import metric_sources_for, task_hash
from refractal.schema.models import (
    MetricExtractor,
    Scene,
    SuccessRule,
    Task,
    Threshold,
)
from refractal.resolve.expand import refuse_unproduced_metrics


ROUTE = "refractal_metadrive.metrics:route"
SAFETY = "refractal_metadrive.metrics:safety"


def scene(metrics=()) -> Scene:
    return Scene(id="town-01", engine="mujoco", model="a.xml", metrics=list(metrics))


def task(success=None, **kw) -> Task:
    return Task(
        id="drive-north", scene="town-01", instruction="drive north",
        predicate="refractal.predicates:from_provider", success=success, **kw
    )


def rule(*thresholds) -> SuccessRule:
    return SuccessRule(all_of=list(thresholds))


class Catalog:
    """The two attributes `refuse_unproduced_metrics` reads. Nothing more.

    A real catalog needs files on disk; the refusal is a pure function of two
    lists, and a fixture that needed a temp directory would be testing the
    loader.
    """

    def __init__(self, scenes, tasks):
        self._scenes = {s.id: s for s in scenes}
        self.tasks = tasks

    def scene(self, scene_id):
        return self._scenes[scene_id]


class TestThreshold(unittest.TestCase):
    def test_exactly_one_bound(self):
        """Neither bound, or both, is a mistake with no sensible reading."""
        with self.assertRaises(ValueError):
            Threshold(metric="route_completion")
        with self.assertRaises(ValueError):
            Threshold(metric="route_completion", at_least=0.9, at_most=0.8)

    def test_the_fixture_crosses_the_threshold_in_both_directions(self):
        """A test whose metric never crosses proves only that nothing errored.

        The invariant is "at_least passes above and fails below", and a fixture
        that only ever sits above cannot distinguish that from "always passes".
        """
        t = Threshold(metric="route_completion", at_least=0.95)
        self.assertTrue(t.holds(0.96), "above the bound must pass")
        self.assertTrue(t.holds(0.95), "at the bound must pass: at_least is inclusive")
        self.assertFalse(t.holds(0.94), "below the bound must fail")

        u = Threshold(metric="collisions", at_most=0)
        self.assertTrue(u.holds(0), "at the bound must pass: at_most is inclusive")
        self.assertFalse(u.holds(1), "above the bound must fail")

    def test_a_missing_metric_fails(self):
        """Absence is not evidence of success.

        The alternative -- treating an absent metric as satisfied -- makes a
        broken extractor look like a perfect policy, which is the failure mode
        that is hardest to notice because it points the flattering way.
        """
        self.assertFalse(Threshold(metric="gone", at_least=0.0).holds(None))
        self.assertFalse(rule(Threshold(metric="gone", at_most=99.0)).holds({}))


class TestSuccessRule(unittest.TestCase):
    def test_every_threshold_must_hold(self):
        r = rule(
            Threshold(metric="route_completion", at_least=0.95),
            Threshold(metric="collisions", at_most=0),
        )
        self.assertTrue(r.holds({"route_completion": 0.99, "collisions": 0}))
        self.assertFalse(r.holds({"route_completion": 0.99, "collisions": 1}),
                         "one collision fails the run however complete the route")
        self.assertFalse(r.holds({"route_completion": 0.50, "collisions": 0}),
                         "a clean half-route is still a failure")

    def test_metrics_named_is_deduplicated_and_sorted(self):
        r = rule(
            Threshold(metric="ttc", at_least=1.0),
            Threshold(metric="ttc", at_most=99.0),
            Threshold(metric="collisions", at_most=0),
        )
        self.assertEqual(r.metrics_named(), ["collisions", "ttc"])


class TestIdentity(unittest.TestCase):
    """The rule is hashed as a unit: thresholds AND the extractors it names."""

    def setUp(self):
        self.scene = scene([
            MetricExtractor(extractor=ROUTE, produces=["route_completion"]),
            MetricExtractor(extractor=SAFETY, produces=["collisions", "min_ttc"]),
        ])
        self.rule = rule(
            Threshold(metric="route_completion", at_least=0.95),
            Threshold(metric="collisions", at_most=0),
        )

    def hash_of(self, t: Task, s: Scene | None = None) -> str:
        s = s or self.scene
        return task_hash(t, metric_sources=metric_sources_for(s, t))

    def test_no_rule_hashes_as_it_always_did(self):
        """Every task_hash recorded before this existed stays valid."""
        plain = task()
        self.assertEqual(self.hash_of(plain), task_hash(plain))

    def test_moving_a_threshold_moves_the_hash(self):
        """0.95 to 0.90 is a different experiment and must not join."""
        strict = self.hash_of(task(success=self.rule))
        loose = self.hash_of(task(success=rule(
            Threshold(metric="route_completion", at_least=0.90),
            Threshold(metric="collisions", at_most=0),
        )))
        self.assertNotEqual(strict, loose)

    def test_swapping_a_named_extractor_moves_the_hash(self):
        """The rule is not defined without what computes its inputs.

        Same thresholds, different code producing `route_completion`: what
        passes changes, so the experiment changes.
        """
        before = self.hash_of(task(success=self.rule))
        other = scene([
            MetricExtractor(extractor="other.pkg:route", produces=["route_completion"]),
            MetricExtractor(extractor=SAFETY, produces=["collisions", "min_ttc"]),
        ])
        self.assertNotEqual(before, self.hash_of(task(success=self.rule), other))

    def test_adding_an_observational_metric_moves_nothing(self):
        """The property the metrics column exists to express.

        `peak_jerk` is measured and not thresholded. Measuring more about a run
        does not make it a different run, so results already recorded stay
        joinable.
        """
        before = self.hash_of(task(success=self.rule))
        richer = scene([
            MetricExtractor(extractor=ROUTE, produces=["route_completion"]),
            MetricExtractor(extractor=SAFETY, produces=["collisions", "min_ttc"]),
            MetricExtractor(extractor="new.pkg:comfort", produces=["peak_jerk"]),
        ])
        self.assertEqual(before, self.hash_of(task(success=self.rule), richer))

    def test_swapping_an_unnamed_extractor_moves_nothing(self):
        """`min_ttc` is produced and not thresholded, so its source is not identity."""
        before = self.hash_of(task(success=self.rule))
        moved = scene([
            MetricExtractor(extractor=ROUTE, produces=["route_completion"]),
            MetricExtractor(extractor=SAFETY, produces=["collisions"]),
            MetricExtractor(extractor="elsewhere:ttc", produces=["min_ttc"]),
        ])
        self.assertEqual(before, self.hash_of(task(success=self.rule), moved))


class TestPlanTimeRefusal(unittest.TestCase):
    def test_a_metric_nothing_produces_is_refused(self):
        """The failure being guarded against, demonstrated rather than described.

        Without this, `comfort` reads as None on every episode, `holds` counts
        a missing metric as failing, and the whole task scores zero looking
        exactly like a bad policy. Found after the run, at the cost of the run.
        """
        cat = Catalog(
            [scene([MetricExtractor(extractor=ROUTE, produces=["route_completion"])])],
            [task(success=rule(Threshold(metric="comfort", at_least=0.5)))],
        )
        with self.assertRaises(CatalogError) as ctx:
            refuse_unproduced_metrics(cat)
        message = str(ctx.exception)
        self.assertIn("comfort", message, "names the metric that is missing")
        self.assertIn("route_completion", message, "and what IS available")

    def test_a_produced_metric_passes(self):
        cat = Catalog(
            [scene([MetricExtractor(extractor=ROUTE, produces=["route_completion"])])],
            [task(success=rule(Threshold(metric="route_completion", at_least=0.95)))],
        )
        refuse_unproduced_metrics(cat)

    def test_a_task_with_no_rule_is_not_examined(self):
        """A catalog declaring no metrics at all must still plan."""
        refuse_unproduced_metrics(Catalog([scene()], [task()]))


class TestExtractorDeclaration(unittest.TestCase):
    def test_a_repeated_metric_name_is_refused(self):
        with self.assertRaises(ValueError):
            MetricExtractor(extractor=ROUTE, produces=["a", "a"])

    def test_produces_may_not_be_empty(self):
        """An extractor writing nothing is a declaration with no content."""
        with self.assertRaises(ValueError):
            MetricExtractor(extractor=ROUTE, produces=[])


if __name__ == "__main__":
    unittest.main()
