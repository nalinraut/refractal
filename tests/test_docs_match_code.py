"""Claims the docs make about the code, checked against the code.

Documentation rots in one direction: the code changes and the prose does not.
This file is the same ratchet ``scripts/lint_guards.py`` applies to guards --
a classification written once rots, so check it instead of asserting it.

Three claims, each chosen because it had already drifted when this was written:

``plan_schema``
    ``getting-started.md`` showed ``plan_schema 1`` and ``README.md`` showed
    ``plan_schema 3`` while the code emitted ``4``. Two documents, two different
    wrong values, in the field that tells a reader whether their plan file is
    still readable. It is the first number a new user sees.

import strings
    The only worked example for ``metrics[].extractor`` named
    ``refractal_metadrive.metrics:route``, a package that is not published and
    exists on one disk. Someone copying it gets ``ModuleNotFoundError`` from the
    documentation rather than from their own mistake.

CLI flags
    ``--benchmark`` became ``--provider`` and the docs kept neither. A flag the
    docs never name is an undiscoverable feature.

Illustrative placeholders are allowed and must look like placeholders: a module
beginning with ``your_`` is the reader's own package and is not imported. That
convention is the point -- it makes "this is yours to write" visible in the
example itself rather than in a sentence beside it.
"""

import argparse
import pathlib
import re
import unittest

from refractal.cli import build_parser
from refractal.schema.plan import PLAN_SCHEMA

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
#: A module spelled like this is the reader's, not ours, and is never imported.
PLACEHOLDER = ("your_", "some_package", "my_")


def _where(path):
    """Repo-relative when it is a file, verbatim when it is ``--help``."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def _texts():
    """Every surface that documents an import string -- including ``--help``.

    ``--help`` is read more often than the docs directory and was not covered
    by the first version of this test, which is how the same unpublished
    package survived in the help text of the very flag being documented while
    the markdown copy was being fixed.
    """
    texts = [(p, p.read_text()) for p in DOCS]
    parser = build_parser()
    chunks = []
    for action in parser._actions:
        for name in getattr(action, "choices", None) or {}:
            for act in action.choices[name]._actions:
                if act.help and act.help is not argparse.SUPPRESS:
                    chunks.append(act.help)
    texts.append((pathlib.Path("cli --help"), "\n".join(chunks)))
    return texts


class DocsMatchCode(unittest.TestCase):
    def test_plan_schema_claims_match_the_emitted_value(self):
        wrong = []
        for path, text in _texts():
            for m in re.finditer(r"plan_schema[` ]+(\d+)", text):
                if int(m.group(1)) != PLAN_SCHEMA:
                    line = text[: m.start()].count("\n") + 1
                    wrong.append(f"{_where(path)}:{line} says "
                                 f"plan_schema {m.group(1)}, code emits {PLAN_SCHEMA}")
        self.assertEqual(wrong, [], "\n".join(wrong))

    def test_import_strings_resolve_or_are_marked_as_the_readers_own(self):
        import importlib

        broken = []
        for path, text in _texts():
            # NOT only inline-backticked. The first version of this test matched
            # ``\`mod:Name\``` only, and the very next commit put an unpublished
            # package inside a fenced command block where nothing looked at it.
            # A reader copies from code blocks more readily than from prose.
            for m in re.finditer(r"\b([a-z_][a-z0-9_.]*):([A-Za-z_][A-Za-z0-9_]*)\b", text):
                mod, name = m.group(1), m.group(2)
                if mod.startswith(PLACEHOLDER):
                    continue
                if not mod.startswith("refractal"):
                    continue          # third-party names are the reader's to install
                line = text[: m.start()].count("\n") + 1
                try:
                    obj = importlib.import_module(mod)
                except Exception as exc:
                    broken.append(f"{_where(path)}:{line} {mod}:{name} "
                                  f"-> {type(exc).__name__}")
                    continue
                if not hasattr(obj, name):
                    broken.append(f"{_where(path)}:{line} {mod} has no {name}")
        self.assertEqual(broken, [], "\n".join(broken))

    def test_every_documented_flag_exists(self):
        parser = build_parser()
        real = set()
        for action in parser._actions:
            for sub in getattr(action, "choices", None) or {}:
                for act in action.choices[sub]._actions:
                    real.update(act.option_strings)
        for action in parser._actions:
            real.update(action.option_strings)

        unknown = []
        for path, text in _texts():
            for i, line in enumerate(text.splitlines(), 1):
                if "refractal " not in line:
                    continue
                for flag in re.findall(r"(--[a-z][a-z0-9\-]+)", line):
                    if flag not in real:
                        unknown.append(f"{_where(path)}:{i} {flag}")
        self.assertEqual(unknown, [], "\n".join(unknown))


if __name__ == "__main__":
    unittest.main()
