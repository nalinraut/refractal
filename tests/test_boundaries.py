"""The dependency rules, as a test rather than as a comment.

``schema`` and ``resolve`` must be importable and fully testable on a bare
laptop -- no Docker, no GPU, no simulator, no harness. That is not a style
preference: it is the property that lets you inspect a plan and its cost before
spending anything, and it is what makes the compiler architecture worth having.

It is also the rule most likely to be broken by accident, because breaking it
looks like a convenience. Hence the test.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SRC = str(Path(__file__).resolve().parents[1] / "src")

FORBIDDEN = [
    "docker",
    "torch",
    "mujoco",
    "jax",
    "isaacsim",
    "vla_eval",
    "pyarrow",
    "duckdb",
    "scipy",
    "fsspec",
    "numpy",
]


class TestSchemaImportsNothingHeavy(unittest.TestCase):
    def test_no_forbidden_module_is_imported(self):
        snippet = (
            "import sys, json;"
            "import refractal.schema;"
            "import refractal.generators;"
            f"print(json.dumps(sorted(m for m in {FORBIDDEN!r} if m in sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        self.assertEqual(
            out.stdout.strip(),
            "[]",
            "refractal.schema pulled in a module it is not allowed to depend on",
        )

    def test_import_strings_are_not_resolved_at_load_time(self):
        """Loading a catalog must not import what its strings name.

        The catalog is BUILT HERE with a deliberately absent module, rather than
        relying on the shipped example being broken. It used to rely on that --
        `examples/catalog` named `so101_eval`, a package descoped weeks ago, and
        this test's comment recorded the fact instead of anyone fixing it.

        A guard that depends on a defect is worse than it looks in both
        directions: it silently weakens the day the defect is fixed, and while
        the defect stands it reads as sanctioned. Now the absent module is the
        test's own, and `examples/catalog` resolves -- asserted separately by
        TestShippedExamplesResolve.
        """
        catalog = tempfile.mkdtemp()
        source = Path(__file__).resolve().parents[1] / "examples" / "catalog"
        shutil.copytree(source, catalog, dirs_exist_ok=True)
        tasks = yaml.safe_load((Path(catalog) / "tasks.yaml").read_text())
        for task in tasks["tasks"]:
            task["predicate"] = "definitely_not_installed.predicates:whatever"
            task.pop("phases", None)
        (Path(catalog) / "tasks.yaml").write_text(yaml.safe_dump(tasks))

        snippet = (
            "import sys;"
            "from refractal.schema import load_catalog;"
            "load_catalog(sys.argv[1]);"
            "assert 'definitely_not_installed' not in sys.modules;"
            "print('ok')"
        )
        out = subprocess.run(
            [sys.executable, "-c", snippet, catalog],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        self.assertEqual(out.stdout.strip(), "ok")


class TestShippedExamplesResolve(unittest.TestCase):
    """Every import string in a shipped example catalog must import.

    The guard that should have existed. `examples/catalog` named `so101_eval` --
    a package descoped weeks ago -- in three places, and the only thing that knew
    was a comment in another test explaining why it was fine there.

    It is the catalog a reader opens to learn the format, and `refractal init`
    generates a DIFFERENT one whose strings do resolve. Two examples shipped and
    one of them was broken.

    Deliberately not run against user catalogs: an import string naming something
    the author has installed and we do not is correct, which is the whole reason
    `plan` does not resolve them. This asserts only what this repository ships.
    """

    def test_every_example_catalog_resolves(self):
        from refractal.schema import load_catalog
        from refractal.schema.importstr import resolve_import_string

        root = Path(__file__).resolve().parents[1] / "examples"
        checked = 0
        for catalog in sorted(p for p in root.iterdir() if (p / "scenes.yaml").is_file()):
            loaded = load_catalog(catalog)
            strings = (
                {t.predicate for t in loaded.tasks}
                | {p.predicate for t in loaded.tasks for p in t.phases}
                | {s.generator for s in loaded.scenario_sets}
                | {s.filter for s in loaded.scenario_sets if s.filter}
                | {c.server for c in loaded.run.checkpoints}
            )
            for string in sorted(strings):
                checked += 1
                try:
                    resolve_import_string(string)
                except Exception as exc:
                    self.fail(
                        f"{catalog.name} names {string!r}, which does not import "
                        f"({type(exc).__name__}). A shipped example teaches the "
                        "format; one whose strings dangle teaches it wrongly, and a "
                        "reader cannot tell which dangling name is deliberate."
                    )
        self.assertGreater(checked, 0, "no example catalogs found to check")


if __name__ == "__main__":
    unittest.main()
