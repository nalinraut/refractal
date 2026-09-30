"""Extractors run, the rule decides, and both are recorded.

Every check is written so it can be watched to fail.
"""

from __future__ import annotations

import unittest

from refractal.execute.metrics import (
    MetricError,
    Measured,
    check_extractors_ran,
    extract,
    verdict,
)
from refractal.schema.models import MetricExtractor, SuccessRule, Threshold


# --- extractors used as fixtures -------------------------------------------
# Module-level so an import string resolves to them.

def route(state, context):
    """Reads a per-task goal out of `provider_ref`, which is the whole point."""
    goal = context["provider_ref"].get("goal_m", 100.0)
    return {"route_completion": min(1.0, state["travelled_m"] / goal)}


def safety(state, context):
    return {"collisions": float(state["hits"]), "min_ttc": state["ttc"]}


def silent(state, context):
    """Declares a metric and never produces it: the failure worth catching."""
    return {}


def liar(state, context):
    """Produces a name it never declared."""
    return {"undeclared_thing": 1.0}


def nothing(state, context):
    return None


ROUTE = "tests.test_metrics_runner:route"
SAFETY = "tests.test_metrics_runner:safety"
SILENT = "tests.test_metrics_runner:silent"
LIAR = "tests.test_metrics_runner:liar"
NOTHING = "tests.test_metrics_runner:nothing"


def ex(import_string, produces, **args):
    return MetricExtractor(extractor=import_string, produces=produces, args=args)


STATE = {"travelled_m": 95.0, "hits": 0, "ttc": 2.4}


class TestExtract(unittest.TestCase):
    def test_values_and_who_produced_them(self):
        m = extract([ex(ROUTE, ["route_completion"]), ex(SAFETY, ["collisions", "min_ttc"])], STATE)
        self.assertEqual(sorted(m.values), ["collisions", "min_ttc", "route_completion"])
        self.assertEqual(sorted(m.ran), sorted([ROUTE, SAFETY]))

    def test_a_scene_extractor_reads_its_task(self):
        """One extractor on the scene, parameterised by the task it is running.

        This is what keeps extractors scene-scoped when two tasks on one map
        have different goals. Without `provider_ref` on the episode there is no
        sound channel for it: the alternatives were parsing the instruction or
        branching on task_id.
        """
        near = extract([ex(ROUTE, ["route_completion"])], STATE,
                       provider_ref={"goal_m": 100.0})
        far = extract([ex(ROUTE, ["route_completion"])], STATE,
                      provider_ref={"goal_m": 200.0})
        self.assertAlmostEqual(near.values["route_completion"], 0.95)
        self.assertAlmostEqual(far.values["route_completion"], 0.475)
        self.assertNotEqual(near.values, far.values,
                            "the same extractor must answer differently per task")

    def test_an_extractor_that_produces_nothing_is_not_recorded_as_having_run(self):
        m = extract([ex(SILENT, ["comfort"])], STATE)
        self.assertEqual(m.values, {})
        self.assertEqual(m.ran, [], "producing nothing is not running")

    def test_returning_none_is_tolerated(self):
        self.assertEqual(extract([ex(NOTHING, ["x"])], STATE).ran, [])

    def test_an_undeclared_metric_is_refused(self):
        """The declaration is what plan-time checking rests on.

        If an extractor may write names it never declared, a rule could
        threshold a metric the catalog says nothing produces, be refused at
        plan time, and be perfectly satisfiable at run time. The refusal keeps
        the declaration worth checking.
        """
        with self.assertRaises(MetricError) as ctx:
            extract([ex(LIAR, ["declared_thing"])], STATE)
        self.assertIn("undeclared_thing", str(ctx.exception))
        self.assertIn("declared_thing", str(ctx.exception), "says what IS declared")


class TestVerdict(unittest.TestCase):
    RULE = SuccessRule(all_of=[
        Threshold(metric="route_completion", at_least=0.95),
        Threshold(metric="collisions", at_most=0),
    ])

    def test_no_rule_defers_to_the_provider(self):
        """Every episode recorded before this existed was decided this way."""
        empty = Measured(values={}, ran=[])
        self.assertTrue(verdict(None, empty, fallback=True))
        self.assertFalse(verdict(None, empty, fallback=False))

    def test_the_rule_overrides_the_provider_in_both_directions(self):
        """A fixture that only ever agrees with the fallback proves nothing."""
        passing = Measured(values={"route_completion": 0.99, "collisions": 0.0}, ran=[])
        failing = Measured(values={"route_completion": 0.99, "collisions": 1.0}, ran=[])
        self.assertTrue(verdict(self.RULE, passing, fallback=False),
                        "the rule can pass an episode the provider failed")
        self.assertFalse(verdict(self.RULE, failing, fallback=True),
                         "and fail one the provider passed")

    def test_end_to_end_from_state_to_verdict(self):
        declared = [ex(ROUTE, ["route_completion"]), ex(SAFETY, ["collisions", "min_ttc"])]
        clean = extract(declared, {"travelled_m": 99.0, "hits": 0, "ttc": 3.0})
        crashed = extract(declared, {"travelled_m": 99.0, "hits": 1, "ttc": 0.2})
        self.assertTrue(verdict(self.RULE, clean, fallback=False))
        self.assertFalse(verdict(self.RULE, crashed, fallback=True),
                         "one collision fails however complete the route")


class TestExtractorReceipt(unittest.TestCase):
    def test_an_extractor_that_never_ran_is_named(self):
        """A declared extractor producing nothing everywhere is the silent failure.

        A threshold on its metric fails every episode, and the run reads as a
        policy that never succeeds.
        """
        declared = [ex(ROUTE, ["route_completion"]), ex(SILENT, ["comfort"])]
        rows = [{"metrics_from": [ROUTE]} for _ in range(50)]
        self.assertEqual(check_extractors_ran(declared, rows), [SILENT])

    def test_nothing_is_reported_when_all_ran(self):
        declared = [ex(ROUTE, ["route_completion"])]
        rows = [{"metrics_from": [ROUTE]}]
        self.assertEqual(check_extractors_ran(declared, rows), [])

    def test_it_reads_the_receipt_and_not_the_values(self):
        """An absent value cannot distinguish 'did not run' from 'ran, found nothing'.

        A row whose `metrics` is empty but whose `metrics_from` names the
        extractor is an extractor that ran. Reading the values would call that
        a failure.
        """
        declared = [ex(ROUTE, ["route_completion"])]
        rows = [{"metrics": {}, "metrics_from": [ROUTE]}]
        self.assertEqual(check_extractors_ran(declared, rows), [])

    def test_no_declared_extractors_is_not_a_finding(self):
        self.assertEqual(check_extractors_ran([], [{"metrics_from": []}]), [])


if __name__ == "__main__":
    unittest.main()
