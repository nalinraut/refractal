"""Catalog identity must not depend on the host locale.

Prompted by upstream #135 (`config_loader.py` opens YAML with no explicit
encoding, so a CP936 Windows host raises `UnicodeDecodeError` on a packaged
config containing "pi-0.5" written with real Greek). Refractal had the identical
bug in `loader.py`.

The crash is the *benign* half. A strict codec fails loudly and someone fixes
it. A lenient one -- latin-1, cp1252 -- decodes every possible byte sequence
without error into different text, so a non-ASCII instruction yields a different
`task_hash` and a different `plan_id` with nothing raised anywhere. A host
environment variable silently forking content-addressed identity is the exact
failure this project exists to prevent, arriving through the back door.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from refractal.schema import CatalogError, hash_obj, load_catalog

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "examples" / "catalog"

# Greek pi + subscript zero: the same class of character that broke upstream.
NON_ASCII_INSTRUCTION = "place the π₀.5 vial"

PLAN_ID_SNIPPET = (
    "import json,sys;"
    "from refractal.schema import load_catalog;"
    "print(json.dumps({'plan_id': load_catalog(sys.argv[1]).plan_id()}))"
)


def catalog_with_non_ascii_instruction() -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory(prefix="refractal-encoding-")
    root = Path(tmp.name) / "catalog"
    shutil.copytree(CATALOG, root)
    tasks = root / "tasks.yaml"
    text = tasks.read_text(encoding="utf-8").replace(
        '"put the vial in slot 4"', f'"{NON_ASCII_INSTRUCTION}"'
    )
    # Write bytes explicitly: a test for encoding handling must not itself
    # depend on the ambient encoding.
    tasks.write_bytes(text.encode("utf-8"))
    return tmp


class TestLenientCodecCannotForkIdentity(unittest.TestCase):
    def test_mojibake_would_change_the_hash(self):
        """The failure the explicit encoding prevents, demonstrated directly.

        latin-1 never raises. It just produces different text, and different
        text is a different identity.
        """
        raw = NON_ASCII_INSTRUCTION.encode("utf-8")
        as_utf8 = raw.decode("utf-8")
        as_latin1 = raw.decode("latin-1")
        self.assertNotEqual(as_utf8, as_latin1)
        self.assertNotEqual(
            hash_obj({"instruction": as_utf8}), hash_obj({"instruction": as_latin1})
        )


class TestLoadIsLocaleIndependent(unittest.TestCase):
    """Load the same catalog under a C locale and under UTF-8; require agreement.

    `LC_ALL=C` with `PYTHONUTF8=0` reproduces #135 on Linux: the preferred
    encoding becomes ANSI_X3.4-1968 and an unqualified `read_text()` raises on
    the first non-ASCII byte.
    """

    def setUp(self):
        self._tmp = catalog_with_non_ascii_instruction()
        self.root = Path(self._tmp.name) / "catalog"

    def tearDown(self):
        self._tmp.cleanup()

    def _plan_id(self, **env_overrides) -> str:
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(REPO / "src"),
            **env_overrides,
        }
        out = subprocess.run(
            [sys.executable, "-c", PLAN_ID_SNIPPET, str(self.root)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return json.loads(out.stdout)["plan_id"]

    def test_c_locale_matches_utf8_locale(self):
        c_locale = self._plan_id(LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
        utf8_locale = self._plan_id(LC_ALL="C.UTF-8", LANG="C.UTF-8")
        self.assertEqual(c_locale, utf8_locale)

    def test_in_process_load_reads_utf8(self):
        catalog = load_catalog(self.root)
        self.assertEqual(catalog.task("vial-slot-4").instruction, NON_ASCII_INSTRUCTION)


class TestInvalidUtf8IsALoudError(unittest.TestCase):
    def test_bad_bytes_name_the_file(self):
        with tempfile.TemporaryDirectory(prefix="refractal-encoding-") as tmp:
            root = Path(tmp) / "catalog"
            shutil.copytree(CATALOG, root)
            # 0xff is not valid UTF-8 in any position.
            (root / "tasks.yaml").write_bytes(b"apiVersion: x\ntasks: [\xff]\n")
            with self.assertRaises(CatalogError) as ctx:
                load_catalog(root)
            self.assertIn("tasks.yaml", str(ctx.exception))
            self.assertIn("UTF-8", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
