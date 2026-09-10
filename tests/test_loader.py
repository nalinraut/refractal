import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from refractal.schema import CatalogError, load_catalog

CATALOG = Path(__file__).resolve().parents[1] / "examples" / "catalog"


class TempCatalog:
    """A writable copy of the example catalog, for mutation tests."""

    def __enter__(self) -> Path:
        self._dir = tempfile.mkdtemp(prefix="refractal-catalog-")
        self.root = Path(self._dir) / "catalog"
        shutil.copytree(CATALOG, self.root)
        return self.root

    def __exit__(self, *exc):
        shutil.rmtree(self._dir, ignore_errors=True)

    def edit(self, filename: str, mutate):
        path = self.root / filename
        doc = yaml.safe_load(path.read_text())
        mutate(doc)
        path.write_text(yaml.safe_dump(doc, sort_keys=False))


class TestLoad(unittest.TestCase):
    def test_example_catalog_loads(self):
        catalog = load_catalog(CATALOG)
        self.assertEqual([s.id for s in catalog.scenes], ["vial-rack-v1", "cube-bowl-v1"])
        self.assertEqual(len(catalog.tasks), 3)
        self.assertEqual(catalog.run.seed_values(), [0, 1, 2])

    def test_tasks_for_defaults_to_every_task_on_the_scene(self):
        catalog = load_catalog(CATALOG)
        vial_set = next(ss for ss in catalog.scenario_sets if ss.id == "vial-grid")
        # Tasks are not in the physics model, so adding one costs episodes but
        # never splits a batch.
        self.assertEqual(
            [t.id for t in catalog.tasks_for(vial_set)], ["vial-slot-4", "vial-slot-7"]
        )

    def test_missing_file(self):
        with TempCatalog() as root:
            (root / "run.yaml").unlink()
            with self.assertRaises(CatalogError) as ctx:
                load_catalog(root)
            self.assertIn("run.yaml", str(ctx.exception))


class TestCrossFileValidation(unittest.TestCase):
    def test_task_references_unknown_scene(self):
        with TempCatalog() as root:
            tc = TempCatalog.__new__(TempCatalog)
            tc.root = root
            tc.edit("tasks.yaml", lambda d: d["tasks"][0].__setitem__("scene", "nope"))
            with self.assertRaises(CatalogError) as ctx:
                load_catalog(root)
            self.assertIn("nope", str(ctx.exception))

    def test_scenario_set_naming_a_task_from_another_scene(self):
        # Not a nicety: the scene is the affinity key, so such an episode would
        # have no batch it could legally run in.
        with TempCatalog() as root:
            tc = TempCatalog.__new__(TempCatalog)
            tc.root = root
            tc.edit(
                "scenarios.yaml",
                lambda d: d["scenario_sets"][0].__setitem__("tasks", ["cube-in-bowl"]),
            )
            with self.assertRaises(CatalogError) as ctx:
                load_catalog(root)
            self.assertIn("cannot cross scenes", str(ctx.exception))

    def test_missing_model_file(self):
        with TempCatalog() as root:
            (root / "assets" / "vial_rack.xml").unlink()
            with self.assertRaises(CatalogError) as ctx:
                load_catalog(root)
            self.assertIn("vial_rack.xml", str(ctx.exception))

    def test_duplicate_ids(self):
        with TempCatalog() as root:
            tc = TempCatalog.__new__(TempCatalog)
            tc.root = root
            tc.edit("tasks.yaml", lambda d: d["tasks"].append(dict(d["tasks"][0])))
            with self.assertRaises(CatalogError) as ctx:
                load_catalog(root)
            self.assertIn("duplicate", str(ctx.exception))


class TestWarnings(unittest.TestCase):
    def test_missing_model_hash_warns_rather_than_fails(self):
        catalog = load_catalog(CATALOG)
        self.assertTrue(any("model_hash" in w for w in catalog.warnings))

    def test_missing_resource_shape_warns(self):
        with TempCatalog() as root:
            tc = TempCatalog.__new__(TempCatalog)
            tc.root = root
            tc.edit("scenes.yaml", lambda d: d["scenes"][0].pop("resource_shape"))
            catalog = load_catalog(root)
            self.assertTrue(any("resource_shape" in w for w in catalog.warnings))


class TestDeterminismAcrossProcesses(unittest.TestCase):
    """The step-1 gate: the same catalog produces the same identity, twice.

    Run in separate interpreters, because the failure this guards against is
    hash randomization and dict-ordering leakage, and both are invisible inside
    one process.
    """

    SNIPPET = (
        "import json,sys;"
        "from refractal.schema import load_catalog;"
        "from refractal.generators import linspace_grid;"
        "from refractal.schema import scenario_hash;"
        "c=load_catalog(sys.argv[1]);"
        "ss=c.scenario_sets[0];"
        "rows=linspace_grid(ss.params, ss.generator_seed);"
        "print(json.dumps({'plan_id': c.plan_id(),"
        " 'scenarios': [scenario_hash(r) for r in rows]}))"
    )

    def _run(self) -> str:
        env_root = str(Path(__file__).resolve().parents[1] / "src")
        out = subprocess.run(
            [sys.executable, "-c", self.SNIPPET, str(CATALOG)],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": env_root, "PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"},
            check=True,
        )
        return out.stdout.strip()

    def test_identical_across_processes(self):
        self.assertEqual(self._run(), self._run())


if __name__ == "__main__":
    unittest.main()
