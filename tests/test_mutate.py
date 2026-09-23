"""The mutation runner, which is a checker, and therefore needs one.

This project has now found several checkers with nothing checking them: a
recorder gate that silently did nothing, a worker whose written rows were
believed because a counter said so, and this -- a mutation runner that reported
"0 failures" from a run that executed 0 tests.

The suite is never actually run here. ``run_suite`` is substituted, because what
is under test is the judgement the runner makes about a result, not the result.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mutate  # noqa: E402
from mutate import MutationError, check  # noqa: E402


class Case(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.target = self.root / "src.py"
        self.target.write_text("value = 1\n", encoding="utf-8")
        self._root, mutate.ROOT = mutate.ROOT, self.root
        self._sentinel, mutate.SENTINEL = mutate.SENTINEL, self.root / ".sentinel"
        self._run = mutate.run_suite

    def tearDown(self):
        mutate.ROOT, mutate.SENTINEL, mutate.run_suite = (
            self._root, self._sentinel, self._run)
        self._dir.cleanup()

    def given(self, tests, failures):
        mutate.run_suite = lambda python=None: (tests, failures, "")

    def mutate_it(self, **kw):
        base = kw.pop("baseline", mutate.Baseline(10))
        return check("m", "src.py", [("value = 1", "value = 2")],
                     baseline=base, **kw)


class TestItRefusesAnUntrustworthyRun(Case):
    def test_zero_failures_from_zero_tests_is_an_error(self):
        """The incident. `0 failures` from a run that executed nothing looked
        exactly like a mutation that did not bite."""
        self.given(tests=0, failures=0)
        with self.assertRaises(MutationError) as ctx:
            self.mutate_it()
        self.assertIn("0 tests", str(ctx.exception))

    def test_a_subset_is_refused_as_loudly_as_nothing(self):
        """Running fewer tests than the baseline is not evidence about this
        mutation either, and it is the case a failure count cannot show."""
        self.given(tests=4, failures=1)
        with self.assertRaises(MutationError) as ctx:
            self.mutate_it()
        self.assertIn("4 tests", str(ctx.exception))

    def test_more_tests_than_the_baseline_is_also_refused(self):
        """Whichever direction it moved. A run that gained tests is a different
        suite, and the number would be about something else."""
        self.given(tests=99, failures=1)
        with self.assertRaises(MutationError):
            self.mutate_it()

    def test_a_mutation_that_breaks_nothing_is_UNDETERMINED_without_a_witness(self):
        """Not a coverage gap. Not a pass either.

        A mutation nothing notices means the check it breaks is untested, OR
        that the mutation changes no behaviour at all. Those are opposite
        findings and a failure count cannot tell them apart, so the result
        stays undetermined until something shows the mutation did anything.
        """
        self.given(tests=10, failures=0)
        with self.assertRaises(MutationError) as ctx:
            self.mutate_it()
        self.assertIn("UNDETERMINED", str(ctx.exception))

    def test_a_real_bite_returns_the_count(self):
        self.given(tests=10, failures=3)
        self.assertEqual(self.mutate_it(), 3)


class TestItAlwaysRestores(Case):
    def test_the_file_comes_back_after_a_normal_run(self):
        self.given(tests=10, failures=2)
        self.mutate_it()
        self.assertEqual(self.target.read_text(encoding="utf-8"), "value = 1\n")

    def test_the_file_comes_back_when_the_suite_explodes(self):
        def boom(python=None):
            raise RuntimeError("the runner died")

        mutate.run_suite = boom
        with self.assertRaises(RuntimeError):
            self.mutate_it()
        self.assertEqual(self.target.read_text(encoding="utf-8"), "value = 1\n")

    def test_a_mutation_that_does_not_apply_leaves_the_file_alone(self):
        self.given(tests=10, failures=1)
        with self.assertRaises(MutationError) as ctx:
            check("m", "src.py", [("absent", "x")], baseline=mutate.Baseline(10))
        self.assertIn("did not apply", str(ctx.exception))
        self.assertEqual(self.target.read_text(encoding="utf-8"), "value = 1\n")


class TestAnInterruptedSweep(Case):
    """SIGKILL does not run a `finally`. A sweep killed mid-mutation leaves the
    tree changed, and the next run's baseline fails for a reason that reads
    exactly like a real regression -- which cost a confusing minute."""

    def test_the_sentinel_names_the_file_while_mutated(self):
        seen = {}

        def peek(python=None):
            seen["sentinel"] = mutate.SENTINEL.read_text(encoding="utf-8")
            seen["content"] = self.target.read_text(encoding="utf-8")
            return (10, 1, "")

        mutate.run_suite = peek
        self.mutate_it()
        self.assertIn("src.py", seen["sentinel"])
        self.assertEqual(seen["content"], "value = 2\n", "the file was mutated")

    def test_it_is_gone_once_restored(self):
        self.given(tests=10, failures=1)
        self.mutate_it()
        self.assertFalse(mutate.SENTINEL.exists())

    def test_starting_on_top_of_one_is_refused(self):
        mutate.SENTINEL.write_text("src.py\nsome mutation\n", encoding="utf-8")
        with self.assertRaises(MutationError) as ctx:
            mutate.refuse_if_interrupted()
        self.assertIn("did not restore", str(ctx.exception))
        self.assertIn("src.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestTheBaselineIsATypeNotANumber(Case):
    """A verdict from a red tree is not a verdict.

    `main` refuses to start on a failing tree. Every caller reaching for
    `check` directly was trusting itself to have done the same, and that failed
    exactly once in the way it always does: the caller took the test count from
    `run_suite` and threw the failure count away. Two sweeps of verdicts were
    computed against a tree already broken by an edit to the mutation runner
    itself, so a kill might have been the pre-existing failure and a survival
    might have been masked by it.

    So the check moved to where the evidence is used rather than where it was
    convenient to put it -- which is the same move that put the test-count
    check here rather than in the caller.
    """

    def test_a_bare_count_is_refused(self):
        """The shape of the misuse: a caller that has not proven the tree was
        green cannot produce the thing `check` requires."""
        self.given(tests=10, failures=3)
        with self.assertRaises(MutationError) as ctx:
            self.mutate_it(baseline=10)
        self.assertIn("green_baseline", str(ctx.exception))

    def test_and_the_file_is_never_touched(self):
        """It refuses BEFORE applying, so a bad call cannot also leave the
        tree mutated."""
        self.given(tests=10, failures=3)
        with self.assertRaises(MutationError):
            self.mutate_it(baseline=10)
        self.assertEqual(self.target.read_text(encoding="utf-8"), "value = 1\n")

    def test_a_red_tree_yields_no_baseline_at_all(self):
        self.given(tests=10, failures=2)
        with self.assertRaises(MutationError) as ctx:
            mutate.green_baseline()
        self.assertIn("already failing", str(ctx.exception))

    def test_a_green_tree_yields_one(self):
        self.given(tests=10, failures=0)
        self.assertEqual(mutate.green_baseline(), 10)
