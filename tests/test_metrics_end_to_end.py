"""Metrics survive a real write and a real read.

Read the artifact, not a counter. Everything up to here tested functions
against dicts; this runs a plan, writes Parquet, and reads it back, because a
map column that a worker builds correctly and the writer drops is a failure
none of the unit tests can see.
"""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from refractal.execute import read_episodes, run_local
from refractal.resolve import resolve


# Extractors resolved by import string from the catalog written below. They
# read the fake provider's outcome dict, which is that backend's whole state.

def margin(state, context):
    """A metric that varies, so a threshold on it can go either way."""
    return {"reach_margin": 0.9 if state["success"] else 0.1}


def silent(state, context):
    """Declared and never produces. The receipt has to notice."""
    return {}


CATALOG = "tests.test_metrics_end_to_end:margin"
SILENT = "tests.test_metrics_end_to_end:silent"


def write_catalog(root: Path, *, rule: bool, with_silent: bool = False) -> Path:
    from refractal.init import init

    init(str(root))
    catalog = root / "catalog"

    scenes = (catalog / "scenes.yaml").read_text()
    extra = f"      - extractor: {SILENT}\n        produces: [never_written]\n" if with_silent else ""
    scenes = scenes.replace(
        "    resource_shape:",
        f"    metrics:\n      - extractor: {CATALOG}\n"
        f"        produces: [reach_margin]\n{extra}"
        "    resource_shape:", 1)
    (catalog / "scenes.yaml").write_text(scenes)

    if rule:
        tasks = (catalog / "tasks.yaml").read_text()
        marker = "\n  - id:"
        first_end = tasks.index(marker, tasks.index("  - id:") + 1)
        tasks = (tasks[:first_end] +
                 "\n    success:\n      all_of:\n"
                 "        - {metric: reach_margin, at_least: 0.5}\n" +
                 tasks[first_end:])
        (catalog / "tasks.yaml").write_text(tasks)
    return catalog


class TestMetricsLandInParquet(unittest.TestCase):
    def run_plan(self, root: Path, catalog: Path):
        plan = resolve(str(catalog), hardware_profile="laptop")
        run_local(
            plan, str(root / "results"),
            session_id="testsession", resume=False,
            started_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        )
        return plan, read_episodes(str(root / "results"), plan.plan_id).to_pylist()

    def test_metrics_are_readable_from_the_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows = self.run_plan(root, write_catalog(root, rule=False))
            self.assertTrue(rows)

            # pyarrow reads a map column as a list of pairs, and a test that
            # accepted either shape would not notice the column becoming a
            # struct.
            with_metrics = [r for r in rows if r["metrics"]]
            self.assertTrue(with_metrics, "no row carries a metric")
            values = dict(with_metrics[0]["metrics"])
            self.assertIn("reach_margin", values)
            self.assertIn(values["reach_margin"], (0.9, 0.1))

            self.assertEqual(
                list(with_metrics[0]["metrics_from"]), [CATALOG],
                "the receipt must name the extractor that ran",
            )

    def test_the_rule_decides_the_verdict_that_was_written(self):
        """Not that success exists: that it follows the metric, both ways.

        `reach_margin` is 0.9 on a provider-success and 0.1 otherwise, and the
        threshold is 0.5. So the written verdict must equal the metric test on
        every row, and both outcomes must be present or the check proves
        nothing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, rows = self.run_plan(root, write_catalog(root, rule=True))
            decided = [r for r in rows if r["metrics"]]
            self.assertTrue(decided)

            outcomes = set()
            for row in decided:
                margin_value = dict(row["metrics"])["reach_margin"]
                self.assertEqual(
                    row["success"], margin_value >= 0.5,
                    f"verdict {row['success']} does not follow reach_margin={margin_value}",
                )
                outcomes.add(row["success"])
            self.assertEqual(
                outcomes, {True, False},
                "the fixture must produce both verdicts or it cannot tell a "
                "working rule from one that always returns the same answer",
            )

    def test_an_extractor_that_never_produced_is_named_by_the_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = write_catalog(root, rule=False, with_silent=True)
            plan, rows = self.run_plan(root, catalog)

            from refractal.execute.metrics import check_extractors_ran

            declared = plan.scenes[0].metrics
            self.assertEqual(len(declared), 2)
            missing = check_extractors_ran(
                declared, [{"metrics_from": list(r["metrics_from"] or ())} for r in rows]
            )
            self.assertEqual(
                missing, [SILENT],
                "a declared extractor that produced nothing on every episode "
                "must be named, because a threshold on its metric would fail "
                "everywhere and read as a bad policy",
            )


if __name__ == "__main__":
    unittest.main()
