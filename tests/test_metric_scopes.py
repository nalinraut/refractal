"""Two checks, two scopes, and neither is the other.

The pair exists because one question cannot answer both. `run` asks whether
every DECLARED extractor produced anything this session: a configuration
fault, caught while the operator is watching. `compare` asks whether the
episodes being POOLED were measured alike: a comparability fault, over data
that may span sessions.

They take different inputs, which is what makes them independently testable.
If either were implemented in terms of the other, breaking one would leave the
other passing and the mutation would register as nothing.
"""

from __future__ import annotations

import unittest

from refractal.compare.coverage import describe, uneven_metric_coverage
from refractal.execute.metrics import check_extractors_ran
from refractal.schema.models import MetricExtractor


def ex(name, produces):
    return MetricExtractor(extractor=name, produces=produces)


def row(task_hash="t1", metrics=None, ran=(), infra=False):
    return {
        "task_hash": task_hash,
        "metrics": metrics,
        "metrics_from": list(ran),
        "is_infra_failure": infra,
    }


class TestTheScopesAreIndependent(unittest.TestCase):
    """Each check must see a fault the other is blind to.

    This is the test that would fail if they ever collapsed into one
    implementation, which is the thing worth preventing.
    """

    def test_run_sees_a_fault_compare_cannot(self):
        """An extractor silent on EVERY episode is even coverage.

        Nothing is missing relative to anything else, so the pool check is
        correctly silent. Only the declaration reveals that something was
        asked for and never arrived.
        """
        rows = [row(metrics={"a": 1.0}, ran=["pkg:a"]) for _ in range(10)]
        self.assertEqual(uneven_metric_coverage(rows), [],
                         "uniformly absent is not uneven")
        self.assertEqual(
            check_extractors_ran([ex("pkg:a", ["a"]), ex("pkg:b", ["b"])], rows),
            ["pkg:b"],
            "only the declaration can reveal a metric nobody ever produced",
        )

    def test_compare_sees_a_fault_run_cannot(self):
        """A metric on some episodes and not others, with nothing declared.

        `compare` has no catalog: a resumed run, or rows from a session that
        predates an extractor, arrive as rows alone. The session check has no
        declaration to compare against and reports nothing.
        """
        rows = ([row(metrics={"a": 1.0}, ran=["pkg:a"]) for _ in range(6)]
                + [row(metrics=None, ran=[]) for _ in range(4)])
        self.assertEqual(check_extractors_ran([], rows), [],
                         "with nothing declared there is nothing to fulfil")
        uneven = uneven_metric_coverage(rows)
        self.assertEqual(len(uneven), 1)
        self.assertEqual(uneven[0].metric, "a")
        self.assertEqual((uneven[0].present, uneven[0].absent), (6, 4))


class TestPoolCoverage(unittest.TestCase):
    def test_even_coverage_is_not_a_finding(self):
        rows = [row(metrics={"a": 1.0, "b": 2.0}) for _ in range(5)]
        self.assertEqual(uneven_metric_coverage(rows), [])

    def test_grouped_by_task_hash_not_task_id(self):
        """Two tasks measured differently is not unevenness within either."""
        rows = ([row(task_hash="t1", metrics={"a": 1.0}) for _ in range(3)]
                + [row(task_hash="t2", metrics={"b": 1.0}) for _ in range(3)])
        self.assertEqual(uneven_metric_coverage(rows), [],
                         "each task is internally consistent")

    def test_infra_failures_are_excluded(self):
        """A crashed worker measured nothing, and that is its own column.

        Counting it would report every run containing a crash as unevenly
        measured: true, and useless.
        """
        rows = ([row(metrics={"a": 1.0}) for _ in range(5)]
                + [row(metrics=None, infra=True)])
        self.assertEqual(uneven_metric_coverage(rows), [])

    def test_a_map_read_back_as_pairs_is_understood(self):
        """pyarrow hands a map column over as pairs, not a dict.

        A reader accepting only one shape would call every row unmeasured
        against the other, and report perfect coverage as total absence.
        """
        as_pairs = [row(metrics=[("a", 1.0)]) for _ in range(4)]
        self.assertEqual(uneven_metric_coverage(as_pairs), [])
        mixed = as_pairs + [row(metrics=[])]
        self.assertEqual(len(uneven_metric_coverage(mixed)), 1)

    def test_a_null_value_counts_as_absent(self):
        """Present-but-null is not measured, and a threshold on it fails."""
        rows = ([row(metrics={"a": 1.0}) for _ in range(3)]
                + [row(metrics={"a": None})])
        uneven = uneven_metric_coverage(rows)
        self.assertEqual(len(uneven), 1)
        self.assertEqual((uneven[0].present, uneven[0].absent), (3, 1))

    def test_the_description_says_what_the_number_is_of(self):
        rows = ([row(metrics={"a": 1.0}) for _ in range(6)]
                + [row(metrics=None) for _ in range(4)])
        text = describe(uneven_metric_coverage(rows))
        self.assertIn("6 of 10", text, "present out of episodes, both stated")
        self.assertIn("absent on 4", text)
        self.assertEqual(describe([]), "", "no findings, nothing printed")


if __name__ == "__main__":
    unittest.main()
