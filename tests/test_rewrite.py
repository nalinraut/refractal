"""The edit helper, tested against the failures it exists to prevent.

Each test below is a real incident, reduced. They are not hypothetical: every
one of them shipped a change that was reported as done and was not done.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rewrite import RewriteError, insert_before, sub, subs  # noqa: E402


class Temp:
    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "f.py"
        self.path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        return self.path

    def __exit__(self, *exc):
        self._dir.cleanup()


class TestItRefusesToDoNothing(unittest.TestCase):
    def test_a_missing_pattern_raises_rather_than_returning_the_text(self):
        """The whole incident, in one line. `str.replace` returns the string
        unchanged and the write succeeds."""
        with Temp() as path:
            before = path.read_text(encoding="utf-8")
            with self.assertRaises(RewriteError) as ctx:
                sub(path, "delta", "epsilon")
            self.assertIn("no match", str(ctx.exception))
            self.assertEqual(path.read_text(encoding="utf-8"), before,
                             "a failed edit must not write")

    def test_the_message_shows_what_was_looked_for(self):
        with Temp() as path:
            with self.assertRaises(RewriteError) as ctx:
                sub(path, "not here", "x")
            self.assertIn("not here", str(ctx.exception))

    def test_a_replacement_identical_to_the_text_is_refused(self):
        """Matching and changing nothing is the same failure wearing a match:
        the edit reports success and the file is as it was."""
        with Temp() as path:
            with self.assertRaises(RewriteError) as ctx:
                sub(path, "beta", "beta")
            self.assertIn("unchanged", str(ctx.exception))


class TestItRefusesToGuess(unittest.TestCase):
    def test_more_matches_than_expected_raises(self):
        with Temp() as path:
            path.write_text("x\nx\n", encoding="utf-8")
            with self.assertRaises(RewriteError) as ctx:
                sub(path, "x", "y")
            self.assertIn("found 2", str(ctx.exception))

    def test_a_count_can_be_declared(self):
        with Temp() as path:
            path.write_text("x\nx\n", encoding="utf-8")
            self.assertEqual(sub(path, "x", "y", count=2), 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "y\ny\n")

    def test_count_none_accepts_any_number_but_not_zero(self):
        with Temp() as path:
            path.write_text("x\nx\n", encoding="utf-8")
            self.assertEqual(sub(path, "x", "y", count=None), 2)
            with self.assertRaises(RewriteError):
                sub(path, "zzz", "y", count=None)


class TestABatchIsAllOrNothing(unittest.TestCase):
    """The `plan_id` incident exactly: four substitutions, two asserted, one
    silently absent. The file was written with the edit half-applied and the
    failure surfaced somewhere else as a different bug."""

    def test_one_bad_pair_writes_none_of_them(self):
        with Temp() as path:
            before = path.read_text(encoding="utf-8")
            with self.assertRaises(RewriteError):
                subs(path, [("alpha", "ALPHA"), ("nope", "x"), ("beta", "BETA")])
            self.assertEqual(path.read_text(encoding="utf-8"), before,
                             "a partially-matched batch must write nothing")

    def test_a_good_batch_applies_in_order(self):
        with Temp() as path:
            self.assertEqual(subs(path, [("alpha", "one"), ("beta", "two")]), 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\ngamma\n")

    def test_later_pairs_see_earlier_edits(self):
        """So a batch reads the way it is written, top to bottom."""
        with Temp() as path:
            subs(path, [("alpha", "beta"), ("beta\nbeta", "merged")])
            self.assertEqual(path.read_text(encoding="utf-8"), "merged\ngamma\n")


class TestInsertBefore(unittest.TestCase):
    def test_it_anchors_or_raises(self):
        with Temp() as path:
            insert_before(path, "beta", "inserted\n")
            self.assertEqual(path.read_text(encoding="utf-8"),
                             "alpha\ninserted\nbeta\ngamma\n")
            with self.assertRaises(RewriteError):
                insert_before(path, "absent", "x")


class TestThereIsNoUncheckedPath(unittest.TestCase):
    def test_every_public_function_goes_through_the_counter(self):
        """The property that makes this a mechanism rather than a habit.

        `assert after every replacement` failed because it was applied to some
        calls and not others in the same script. This holds only while no
        function here can write without counting first, so it is checked rather
        than intended.
        """
        import ast

        source = Path(__file__).resolve().parents[1] / "scripts" / "rewrite.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        writers, checked = set(), set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            calls = {
                n.func.attr for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            }
            names = {
                n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            if "write_text" in calls:
                writers.add(node.name)
                if "_check" in names:
                    checked.add(node.name)
        self.assertTrue(writers, "no writer found; the test has drifted")
        self.assertEqual(
            writers - checked, set(),
            "a function writes without counting first, which is the whole bug",
        )


if __name__ == "__main__":
    unittest.main()
