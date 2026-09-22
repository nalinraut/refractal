"""The dependency rules, as a test rather than as a comment.

``schema`` and ``resolve`` must be importable and fully testable on a bare
laptop -- no Docker, no GPU, no simulator, no harness. That is not a style
preference: it is the property that lets you inspect a plan and its cost before
spending anything, and it is what makes the compiler architecture worth having.

It is also the rule most likely to be broken by accident, because breaking it
looks like a convenience. Hence the test.
"""

import re
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


#: Crossings from a planning layer into an execution layer that are allowed,
#: each with the reason it is allowed. A crossing not named here fails; a
#: crossing named here that no longer exists also fails, because permission that
#: outlives its use is permission nobody is checking.
DECLARED_CROSSINGS = {
    ("compare/verdict.py", "refractal.execute.harness"): (
        "`harness_surface` is a precondition `compare` enforces, and the value "
        "is already in the rows it reads. The import exists only to turn "
        "'something moved' into a list of files and the assumptions that rest "
        "on them. Lazy, inside the function, so it creates no load-time "
        "dependency on anything `execute` needs."
    ),
}

#: The layers that must not depend on how a run happens.
PLANNING_LAYERS = ("resolve", "compare")
EXECUTION_LAYERS = ("execute", "render")


def _imports(path: Path, package_root: Path) -> set[str]:
    """Every module this file imports, absolute, relative ones resolved.

    Parsed rather than grepped: a lazy import inside a function is still a
    dependency, and a string search cannot tell `from ..execute import x` from
    the word execute in a docstring.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    here = path.relative_to(package_root).parent.parts
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ("refractal", *here[: len(here) - node.level + 1])
                found.add(".".join((*base, node.module) if node.module else base))
            elif node.module:
                found.add(node.module)
    return found


class TestPlanningDoesNotDependOnExecution(unittest.TestCase):
    """`resolve` and `compare` must not know how a run happens.

    They were clean when this was written, and nothing was stopping the next
    change from making them otherwise. That is the shape this project keeps
    finding: a property true by inspection with no guard on it.

    Video is the example that prompted it. Recording frames is render-time, it
    is deliberately absent from `plan_id`, and it reaches only `execute` and
    `render`. None of that was enforced.
    """

    def setUp(self):
        self.root = Path(SRC) / "refractal"

    def _crossings(self):
        found = []
        for layer in PLANNING_LAYERS:
            for path in sorted((self.root / layer).rglob("*.py")):
                rel = str(path.relative_to(self.root))
                for module in _imports(path, self.root):
                    tail = module.split(".")
                    if "refractal" in tail:
                        after = tail[tail.index("refractal") + 1:]
                        if after and after[0] in EXECUTION_LAYERS:
                            found.append((rel, module))
        return found

    def test_every_crossing_is_declared(self):
        undeclared = [c for c in self._crossings() if c not in DECLARED_CROSSINGS]
        self.assertEqual(
            undeclared, [],
            "planning layers reached into execution without a declared reason. "
            "Either move the dependency, or add it to DECLARED_CROSSINGS with "
            "why it is allowed.",
        )

    def test_no_declared_crossing_is_stale(self):
        """An allowlist entry that no longer matches anything is dead permission."""
        actual = set(self._crossings())
        for entry in DECLARED_CROSSINGS:
            self.assertIn(
                entry, actual,
                f"{entry} is declared as an allowed crossing and no longer "
                "exists. Remove it, so the list keeps meaning something.",
            )

    def test_recording_concepts_stay_out_of_planning(self):
        """The specific case that prompted the rule, named so it cannot drift."""
        for layer in PLANNING_LAYERS:
            for path in sorted((self.root / layer).rglob("*.py")):
                text = path.read_text(encoding="utf-8")
                for word in ("record_video", "FrameBuffer", "frame_every",
                             "write_strips", "artifact_uri"):
                    self.assertNotIn(
                        word, text,
                        f"{path.name} mentions {word}: whether a run recorded "
                        "frames is render-time and must not reach planning.",
                    )


class TestTheDocsNameFlagsThatExist(unittest.TestCase):
    """Every ``--flag`` the docs mention is one the CLI offers.

    The docs drifted exactly here: the Compose flags table listed two flags that
    live on ``run`` as though they were ``render``'s, and omitted six that had
    been added since it was written. Nothing failed, because nothing looked.

    A doc that names a flag which does not exist is worse than no doc -- the
    reader types it, argparse exits 2, and the tool looks broken rather than the
    page looking stale.
    """

    #: Flags belonging to other tools, appearing in console transcripts the docs
    #: show for context: uv, pip, twine, and a model server's own command line.
    FOREIGN = {
        "--python", "--no-cache-dir", "--repository", "--address", "--short",
    }

    @staticmethod
    def _cli_flags() -> set[str]:
        import argparse

        from refractal.cli import build_parser

        found: set[str] = set()

        def walk(parser: argparse.ArgumentParser) -> None:
            for action in parser._actions:
                found.update(o for o in action.option_strings if o.startswith("--"))
                if isinstance(action, argparse._SubParsersAction):
                    for sub in action.choices.values():
                        walk(sub)

        walk(build_parser())
        return found

    def test_no_doc_names_a_flag_the_cli_does_not_have(self):
        docs = Path(__file__).resolve().parents[1] / "docs"
        self.assertTrue(docs.is_dir(), "docs/ not found")
        offered = self._cli_flags()
        mentioned: dict[str, set[str]] = {}
        for page in sorted(docs.glob("*.md")):
            for flag in set(re.findall(r"--[a-z][a-z0-9-]+", page.read_text("utf-8"))):
                mentioned.setdefault(flag, set()).add(page.name)

        unknown = {
            flag: sorted(pages)
            for flag, pages in mentioned.items()
            if flag not in offered and flag not in self.FOREIGN
        }
        self.assertEqual(
            unknown,
            {},
            "docs name flags the CLI does not offer (add to FOREIGN if the flag "
            f"belongs to another tool): {unknown}",
        )
