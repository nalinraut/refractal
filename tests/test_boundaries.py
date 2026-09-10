"""The dependency rules, as a test rather than as a comment.

``schema`` and ``resolve`` must be importable and fully testable on a bare
laptop -- no Docker, no GPU, no simulator, no harness. That is not a style
preference: it is the property that lets you inspect a plan and its cost before
spending anything, and it is what makes the compiler architecture worth having.

It is also the rule most likely to be broken by accident, because breaking it
looks like a convenience. Hence the test.
"""

import subprocess
import sys
import unittest
from pathlib import Path

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
        # The example catalog references so101_eval, which does not exist here.
        # Loading it must still succeed: predicates and servers are resolved by
        # execute, generators and filters by resolve, and neither is schema's
        # business.
        snippet = (
            "import sys;"
            "from refractal.schema import load_catalog;"
            "load_catalog(sys.argv[1]);"
            "assert 'so101_eval' not in sys.modules;"
            "print('ok')"
        )
        catalog = str(Path(__file__).resolve().parents[1] / "examples" / "catalog")
        out = subprocess.run(
            [sys.executable, "-c", snippet, catalog],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        self.assertEqual(out.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
