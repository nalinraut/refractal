"""The seconds test, as a test.

`pip install refractal`, `refractal init`, `refractal plan`, `refractal run`,
`refractal compare`, on a machine with no simulator. The claim is only true if
what `init` writes actually runs, so this runs it.
"""

import tempfile
import unittest
from pathlib import Path

from refractal.cli import main
from refractal.execute import read_episodes
from refractal.init import InitError, init
from refractal.resolve import resolve
from refractal.schema import load_catalog


class TestInitExample(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="refractal-init-")
        self.root = Path(self._tmp.name)
        init(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_it_plans_immediately(self):
        plan = resolve(self.root / "catalog", hardware_profile="laptop")
        self.assertGreater(plan.total_episodes, 0)
        self.assertEqual(len(plan.checkpoints), 2)

    def test_the_example_declares_no_filter(self):
        """A filtered example could not be planned until you had an engine.

        `refractal plan` refuses a declared filter with no build.lock, and
        `refractal build` needs the simulator -- so a filter here would make the
        first command of the seconds test fail.
        """
        catalog = load_catalog(self.root / "catalog")
        self.assertTrue(all(s.filter is None for s in catalog.scenario_sets))

    def test_it_ships_no_build_lock(self):
        """A committed lock is one nobody can regenerate.

        And the first thing anyone does after init is widen the grid, which
        would invalidate it and produce a staleness error before they had run
        anything.
        """
        self.assertFalse((self.root / "catalog" / "build.lock").exists())

    def test_widening_the_grid_still_plans(self):
        path = self.root / "catalog" / "scenarios.yaml"
        path.write_text(
            path.read_text(encoding="utf-8").replace("steps: 6", "steps: 9"), encoding="utf-8"
        )
        plan = resolve(self.root / "catalog", hardware_profile="laptop")
        self.assertEqual(len(plan.scenes[0].scenarios), 9 * 5)

    def test_import_strings_in_the_example_actually_resolve(self):
        from refractal.schema.importstr import resolve_import_string

        catalog = load_catalog(self.root / "catalog")
        for task in catalog.tasks:
            self.assertTrue(callable(resolve_import_string(task.predicate)))
        for checkpoint in catalog.run.checkpoints:
            resolve_import_string(checkpoint.server)

    def test_refuses_to_overwrite_without_force(self):
        with self.assertRaises(InitError) as ctx:
            init(self.root)
        self.assertIn("--force", str(ctx.exception))
        init(self.root, force=True)


class TestSecondsTestEndToEnd(unittest.TestCase):
    def test_init_plan_run_compare(self):
        with tempfile.TemporaryDirectory(prefix="refractal-seconds-") as tmp:
            root = Path(tmp)
            catalog = root / "catalog"
            plan_path = root / "plan.json"
            results = root / "results"

            self.assertEqual(main(["init", str(root)]), 0)
            self.assertEqual(
                main(["plan", str(catalog), "--hardware", "laptop", "-o", str(plan_path)]), 0
            )
            self.assertEqual(
                main([
                    "run", str(plan_path), "-o", str(results),
                    "--catalog", str(catalog), "--session-id", "0" * 32,
                ]),
                0,
            )
            plan = resolve(catalog, hardware_profile="laptop")
            self.assertEqual(
                read_episodes(str(results), plan.plan_id).num_rows, plan.total_episodes
            )
            # 0 or 1 are both legitimate verdicts on synthetic data; 2 would mean
            # the comparison could not be answered at all.
            self.assertIn(main(["compare", str(results), plan.plan_id]), (0, 1))


if __name__ == "__main__":
    unittest.main()
