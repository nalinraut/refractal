"""A plan runs with no catalog in reach.

`refractal plan` compiles a catalog away. Everything a backend needs to ACT has
to be inside plan.json, because a worker in a container has the plan mounted
and nothing else.

That property is easy to break by ADDITION, and silently. Add a field to the
catalog, use it in a runner, forget to carry it into the plan, and nothing
fails until a consumer needs it -- at which point the run is already going.
It broke three times in one afternoon while metrics were added:

* ``provider_ref`` was not on the episode, so a scene-scoped extractor had no
  sound way to read a per-task goal
* ``PlannedScene.metrics`` did not exist, so the extractors could not be found
* ``PlannedEpisode.success`` did not exist, so the rule could not be applied

Each was caught by reading, not by a test. This is the test.

A SUBPROCESS, not an in-process call. The catalog is deleted before the run,
and an in-process test would keep working off objects already loaded: the
loader's caches, an import that resolved earlier, a Path that still points at a
directory the OS has not reclaimed. A fresh interpreter with the directory gone
is the only arrangement where "needs nothing but the plan" is actually being
asserted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def refractal(*args, cwd, env=None):
    """Run the CLI in a fresh interpreter with src/ on the path."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "src")
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "refractal.cli", *args],
        cwd=cwd, env=environment, capture_output=True, text=True, timeout=600,
    )


class TestPlanIsSelfContained(unittest.TestCase):
    def test_a_plan_runs_after_its_catalog_is_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init = refractal("init", ".", cwd=root)
            self.assertEqual(init.returncode, 0, init.stderr)

            plan_dir = root / "elsewhere"
            plan_dir.mkdir()
            planned = refractal(
                "plan", "catalog", "--hardware", "laptop",
                "-o", str(plan_dir / "plan.json"), cwd=root,
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)

            # The whole point. Everything the catalog said must now be inside
            # the plan or it is gone.
            shutil.rmtree(root / "catalog")
            self.assertFalse((root / "catalog").exists())

            # No --catalog, and nothing to point it at if there were.
            run = refractal(
                "run", "plan.json", "-o", "results", cwd=plan_dir,
            )
            self.assertEqual(
                run.returncode, 0,
                "a plan must run with no catalog in reach. If this fails, "
                "something a runner needs was left in the catalog and never "
                "carried into plan.json.\n"
                f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}",
            )

            parts = list((plan_dir / "results").rglob("part-*.parquet"))
            self.assertTrue(parts, "the run wrote no episodes")

    def test_declared_values_reach_the_plan(self):
        """Values, not keys. The first version of this asserted key presence.

        Every one of these fields has a default, so the key is in the JSON as
        ``{}``, ``null`` or ``[]`` whether the catalog's value propagates or
        not. Deleting the three lines that carry them left all three keys
        present and the test passing: a check that fired and did not matter.

        So the fixture has to be in the configuration the invariant needs. The
        catalog below DECLARES an extractor and a success rule, and the
        assertions are that those exact values arrive. With an example catalog
        that declares neither, there is nothing for the propagation to fail to
        carry.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(refractal("init", ".", cwd=root).returncode, 0)
            catalog = root / "catalog"

            scenes = (catalog / "scenes.yaml").read_text()
            scenes = scenes.replace(
                "    resource_shape:",
                "    metrics:\n"
                "      - extractor: refractal.example:ee_near\n"
                "        produces: [reach_margin]\n"
                "    resource_shape:", 1)
            (catalog / "scenes.yaml").write_text(scenes)

            tasks = (catalog / "tasks.yaml").read_text()
            first = tasks.index("  - id:")
            end = tasks.index("\n  - id:", first + 1) if "\n  - id:" in tasks[first + 1:] else len(tasks)
            tasks = (tasks[:end] +
                     "\n    provider_ref: {slot: 7}\n"
                     "    success:\n"
                     "      all_of:\n"
                     "        - {metric: reach_margin, at_least: 0.25}\n" +
                     tasks[end:])
            (catalog / "tasks.yaml").write_text(tasks)

            out = root / "plan.json"
            planned = refractal("plan", "catalog", "--hardware", "laptop",
                                "-o", str(out), cwd=root)
            self.assertEqual(planned.returncode, 0, planned.stderr)
            plan = json.loads(out.read_text())

            scene = plan["scenes"][0]
            self.assertEqual(
                [m["extractor"] for m in scene["metrics"]],
                ["refractal.example:ee_near"],
                "the scene's extractors did not reach the plan",
            )
            self.assertEqual(scene["metrics"][0]["produces"], ["reach_margin"])

            declared = [
                e for w in scene["workers"] for e in w["episodes"]
                if e["provider_ref"]
            ]
            self.assertTrue(declared, "provider_ref did not reach any episode")
            self.assertEqual(declared[0]["provider_ref"], {"slot": 7})
            self.assertIsNotNone(
                declared[0]["success"], "the success rule did not reach the episode")
            self.assertEqual(
                declared[0]["success"]["all_of"][0]["metric"], "reach_margin")
            self.assertEqual(
                declared[0]["success"]["all_of"][0]["at_least"], 0.25)


if __name__ == "__main__":
    unittest.main()
