"""Recovering the v2 columns on rows written without them, and refusing to
guess where the partition that makes it exact does not hold."""

import importlib.util
import pathlib
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

_spec = importlib.util.spec_from_file_location(
    "backfill", pathlib.Path("scripts/backfill_step_fields.py"))
backfill = importlib.util.module_from_spec(_spec)
sys.modules["backfill"] = backfill
_spec.loader.exec_module(backfill)


def table(rows, *, thin=False):
    """A pre-v2 row set: steps and success observed, the v2 columns absent.

    Carries the schema's non-nullable identity columns, because every real row
    does -- a fixture without them tests a file that cannot exist, and the
    first version of this one did, which is how the missing-column refusal
    below came to be written.
    """
    columns = {
        "episode_id": [r[0] for r in rows],
        "steps": pa.array([r[1] for r in rows], pa.int32()),
        "success": [r[2] for r in rows],
    }
    if not thin:
        n = len(rows)
        columns.update({
            "scenario_hash": [f"sha256:sc{i}" for i in range(n)],
            "base_scenario_hash": [f"sha256:sc{i}" for i in range(n)],
            "scene_id": ["scene"] * n, "scene_hash": ["sha256:scene"] * n,
            "task_id": ["task"] * n, "task_hash": ["sha256:task"] * n,
            "checkpoint_id": ["pi0"] * n, "seed": pa.array(range(n), pa.int32()),
            "session_id": ["s"] * n, "worker_id": ["w"] * n,
            "execution_mode": ["serial"] * n,
            "is_infra_failure": [False] * n,
            "elapsed_sec": pa.array([1.0] * n, pa.float64()),
        })
    return pa.table(columns)


class TheDerivationIsExactWhereThePartitionHolds(unittest.TestCase):
    def test_successes_get_their_time_and_failures_get_the_budget(self):
        got, tally = backfill.derive(
            table([("a", 104, True), ("b", 220, False), ("c", 63, True)]), {})
        self.assertEqual(got.column("outcome").to_pylist(),
                         ["SUCCESS", "BUDGET_EXHAUSTED", "SUCCESS"])
        self.assertEqual(got.column("step_to_success").to_pylist(), [104, None, 63])
        self.assertEqual(got.column("step_budget").to_pylist(), [220, 220, 220])
        self.assertEqual(tally["success"], 2)

    def test_the_budget_comes_from_the_failures_not_from_max_steps(self):
        """The longest success is a lower bound on the budget, not the budget.
        A file with no failure cannot say what the limit was."""
        self.assertIsNone(backfill.budget_of([104, 63], [True, True]))
        self.assertEqual(backfill.budget_of([104, 220, 220], [True, False, False]), 220)
        # Failures disagreeing means two budgets in one file; unanswerable.
        self.assertIsNone(backfill.budget_of([220, 400], [False, False]))

    def test_with_no_failure_the_outcome_is_null_with_a_reason(self):
        got, tally = backfill.derive(table([("a", 104, True)]), {})
        self.assertEqual(got.column("outcome").to_pylist(), ["SUCCESS"])
        self.assertEqual(got.column("step_budget").to_pylist(), [None])
        route = dict(got.column("field_provenance").to_pylist()[0])
        self.assertNotIn("step_budget", route)


class TheGuardRefusesRatherThanReclassifying(unittest.TestCase):
    def test_a_success_at_the_budget_refuses_the_file(self):
        """The failure mode the whole script is shaped around.

        The partition reads "steps below the budget" as success and "steps at
        the budget" as exhaustion. A success landing exactly on the budget
        would be reclassified as BUDGET_EXHAUSTED and its success time
        discarded -- silently, and in the direction that flatters the
        apparatus. It holds on the recorded data (successes 63..210, every
        failure at 220) and nothing stops a future run from breaking it.
        """
        with self.assertRaises(backfill.BackfillRefused) as caught:
            backfill.derive(
                table([("a", 220, True), ("b", 220, False)]), {})
        message = str(caught.exception)
        self.assertIn("220", message)
        self.assertIn("Refusing rather than deriving", message)

    def test_it_names_the_offending_episode(self):
        with self.assertRaises(backfill.BackfillRefused) as caught:
            backfill.derive(
                table([("ok", 100, True), ("edge", 220, True), ("f", 220, False)]), {})
        self.assertIn("edge", str(caught.exception))


class DecisionsAreMeasuredNotDerived(unittest.TestCase):
    def test_they_come_from_the_hash_log(self):
        got, tally = backfill.derive(
            table([("a", 104, True), ("b", 220, False)]), {"a": 11, "b": 22})
        self.assertEqual(got.column("decisions").to_pylist(), [11, 22])
        route = dict(got.column("field_provenance").to_pylist()[0])
        self.assertIn("measured:", route["decisions"])
        self.assertEqual(tally["decisions measured"], 2)

    def test_without_a_log_they_are_null_with_a_reason(self):
        """Never derived from the declared chunk size: `steps / decisions`
        would then be a restatement of the declaration, which is the assertion
        `validate_row` refuses to accept as a measurement."""
        got, _ = backfill.derive(table([("a", 104, True)]), {})
        self.assertEqual(got.column("decisions").to_pylist(), [None])
        route = dict(got.column("field_provenance").to_pylist()[0])
        self.assertIn("unmeasured:", route["decisions"])
        self.assertIn("per environment step", route["decisions"])


class ProvenanceIsPerFieldAndTheRowStaysDistinguishable(unittest.TestCase):
    def test_each_field_names_its_own_route(self):
        """One row, three provenance classes: steps observed, step_to_success
        derived from it, decisions measured from a different artefact."""
        got, _ = backfill.derive(table([("a", 104, True)]), {"a": 11})
        route = dict(got.column("field_provenance").to_pylist()[0])
        self.assertTrue(route["step_to_success"].startswith("derived:"))
        self.assertTrue(route["decisions"].startswith("measured:"))
        self.assertTrue(route["outcome"].startswith("derived:"))
        self.assertNotIn("steps", route)  # observed, original, nothing to say

    def test_a_backfilled_row_is_not_a_native_v2_row(self):
        """schema_version stays null. A native v2 row was written by a writer
        that observed these fields; these were reconstructed afterwards, and a
        reader asking which must not get the same answer from both."""
        got, _ = backfill.derive(table([("a", 104, True)]), {})
        self.assertEqual(got.column("schema_version").to_pylist(), [None])

    def test_the_result_matches_the_episode_schema(self):
        from refractal.execute.results import EPISODES_SCHEMA

        got, _ = backfill.derive(table([("a", 104, True)]), {})
        self.assertEqual(got.schema.names, EPISODES_SCHEMA.names)


class AShortInputIsRefusedClearly(unittest.TestCase):
    def test_missing_identity_columns_refuse_with_their_names(self):
        """Rather than an ArrowInvalid from inside the writer, raised after
        every other file was processed and blaming the derivation."""
        with self.assertRaises(backfill.BackfillRefused) as caught:
            backfill.derive(table([("a", 104, True)], thin=True), {})
        self.assertIn("scenario_hash", str(caught.exception))
        self.assertIn("cannot invent the identity", str(caught.exception))


class NothingIsWrittenInPlace(unittest.TestCase):
    def test_the_original_file_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "part-0.parquet"
            pq.write_table(table([("a", 104, True), ("b", 220, False)]), path)
            before = path.read_bytes()
            code = backfill.main(["backfill", tmp, "--write"])
            self.assertEqual(code, 0)
            self.assertEqual(path.read_bytes(), before)
            siblings = list(pathlib.Path(tmp).glob("*" + backfill.SUFFIX))
            self.assertEqual(len(siblings), 1)

    def test_a_refusal_exits_nonzero_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "part-0.parquet"
            pq.write_table(table([("edge", 220, True), ("f", 220, False)]), path)
            code = backfill.main(["backfill", tmp, "--write"])
            self.assertEqual(code, 1)
            self.assertEqual(list(pathlib.Path(tmp).glob("*" + backfill.SUFFIX)), [])


class AnAbsenceIsNotTheSameAsLookingInTheWrongPlace(unittest.TestCase):
    """The class behind a bug this script already had.

    Reading `scenes[].episodes[]` -- which does not exist, because the planner
    groups episodes under the worker that runs them -- reported "no plan.json
    in reach" against a tree containing one. A lookup that says "absent" when
    it means "I looked somewhere that does not exist" cannot be told apart
    from a real absence, and the next change to the planner's shape
    reproduces it.
    """

    def _in(self, contents=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        if contents is not None:
            (root / "plan.json").write_text(contents)
        return root

    def test_no_file_says_so(self):
        _, why = backfill.declared_budget(self._in())
        self.assertIn("no plan.json anywhere", why)

    def test_a_file_that_will_not_parse_is_distinguished(self):
        _, why = backfill.declared_budget(self._in("{not json"))
        self.assertIn("could be read", why)
        self.assertIn("JSONDecodeError", why)

    def test_a_file_without_the_field_names_the_path_searched(self):
        """The one that matters: the file IS there, and the message has to
        make a wrong path visible the first time anyone reads it."""
        import json as _json

        root = self._in(_json.dumps({"scenes": [{"episodes": [{"max_steps": 220}]}]}))
        _, why = backfill.declared_budget(root)
        self.assertIn("parsed but carries no", why)
        self.assertIn("scenes[].workers[].episodes[].max_steps", why)
        self.assertNotIn("no plan.json anywhere", why)

    def test_the_real_shape_is_found_and_says_where(self):
        import json as _json

        root = self._in(_json.dumps(
            {"scenes": [{"workers": [{"episodes": [{"max_steps": 220}]}]}]}))
        budget, why = backfill.declared_budget(root)
        self.assertEqual(budget, 220)
        self.assertTrue(why.startswith("declared:"))
        self.assertIn("workers", why)


if __name__ == "__main__":
    unittest.main()
