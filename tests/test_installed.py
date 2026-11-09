"""Whether an installed distribution is one anybody else could install."""

import json
import pathlib
import tempfile
import unittest

from refractal.build.installed import (
    InstalledDist,
    editable_among,
    installed_dists,
)
from refractal.compare.report import Reproducible


def _dist(root, name, version, *, direct_url=None, record=None):
    """A dist-info directory shaped the way pip writes one."""
    d = pathlib.Path(root) / f"{name.replace('-', '_')}-{version}.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    if direct_url is not None:
        (d / "direct_url.json").write_text(json.dumps(direct_url))
    if record is not None:
        (d / "RECORD").write_text(record)
    return d


# What pip writes for a normally installed package: a sha256 per file.
WHEEL_RECORD = (
    "pkg/__init__.py,sha256=AAAA,120\n"
    "pkg/core.py,sha256=BBBB,4096\n"
    "pkg-1.0.dist-info/METADATA,sha256=CCCC,512\n"
    "pkg-1.0.dist-info/RECORD,,\n"
)

# What pip writes for an editable install: a .pth, and nothing about the
# sources the interpreter will actually import.
EDITABLE_RECORD = (
    "__editable__.pkg-1.0.pth,sha256=DDDD,41\n"
    "pkg-1.0.dist-info/METADATA,sha256=CCCC,512\n"
    "pkg-1.0.dist-info/RECORD,,\n"
)


class EditableIsDetected(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_an_editable_install_is_flagged_and_claims_no_digest(self):
        """The two halves are one decision.

        An editable distribution's RECORD describes a .pth file, so a digest
        over it would move when the packaging moves and hold still when the
        code changes. Claiming one would be worse than claiming nothing,
        because it would read as verification.
        """
        _dist(self.root, "libero", "0.1.0",
              direct_url={"url": "file:///home/x/libero",
                          "dir_info": {"editable": True}},
              record=EDITABLE_RECORD)
        got = installed_dists(["libero"], search_path=[self.root])["libero"]
        self.assertTrue(got.editable)
        self.assertEqual(got.source, "file:///home/x/libero")
        self.assertIsNone(got.record_digest)
        self.assertFalse(got.verifiable)
        self.assertIn("EDITABLE", got.render())

    def test_a_published_install_gets_a_digest_over_every_file(self):
        _dist(self.root, "mujoco", "3.2.0", record=WHEEL_RECORD)
        got = installed_dists(["mujoco"], search_path=[self.root])["mujoco"]
        self.assertFalse(got.editable)
        self.assertIsNone(got.source)
        self.assertTrue(got.record_digest.startswith("sha256:"))
        self.assertTrue(got.verifiable)

    def test_the_digest_moves_when_any_installed_file_changes(self):
        """One hash per package, covering every file pip wrote."""
        _dist(self.root, "mujoco", "3.2.0", record=WHEEL_RECORD)
        first = installed_dists(["mujoco"], search_path=[self.root])["mujoco"]
        edited = WHEEL_RECORD.replace("sha256=BBBB", "sha256=ZZZZ")
        (pathlib.Path(self.root) / "mujoco-3.2.0.dist-info" / "RECORD").write_text(edited)
        second = installed_dists(["mujoco"], search_path=[self.root])["mujoco"]
        self.assertNotEqual(first.record_digest, second.record_digest)

    def test_installed_from_a_local_path_but_not_editable(self):
        """Still a direct install, and its sources ARE pinned.

        pip wrote a real RECORD because it copied the files, so this is
        verifiable even though it did not come from an index. Recording the
        source without flagging it editable keeps those two facts apart.
        """
        _dist(self.root, "robosuite", "1.4.1",
              direct_url={"url": "file:///tmp/robosuite-1.4.1.tar.gz",
                          "archive_info": {}},
              record=WHEEL_RECORD)
        got = installed_dists(["robosuite"], search_path=[self.root])["robosuite"]
        self.assertFalse(got.editable)
        self.assertEqual(got.source, "file:///tmp/robosuite-1.4.1.tar.gz")
        self.assertTrue(got.verifiable)

    def test_the_default_search_path_is_the_environment(self):
        """Regression: `path=None` is not importlib.metadata's default.

        It iterates whatever it is given, so passing None explicitly raises
        TypeError instead of falling back to sys.path. Every other test here
        supplies a search_path, so none of them covered the call `build`
        actually makes.
        """
        got = installed_dists(["refractal"])
        self.assertIn("refractal", got)

    def test_a_missing_distribution_is_recorded_not_skipped(self):
        """"libero is not installed here" is a fact about the environment that
        wrote the lock. Dropping it makes a lock written without the provider
        look like one written with it."""
        got = installed_dists(["libero"], search_path=[self.root])["libero"]
        self.assertTrue(got.missing)
        self.assertEqual(got.document(), {"missing": True})

    def test_the_blank_hash_line_count_is_not_a_discriminator(self):
        """Documented because it reads exactly like one.

        Both RECORDs here have exactly one line with an empty hash field --
        RECORD's own, which carries no hash by construction. Measured the same
        on real installs: 1 for an editable checkout, 1 for numpy, 1 for
        pytest. A check built on it would fire on everything.
        """
        self.assertEqual(WHEEL_RECORD.count(",,"), 1)
        self.assertEqual(EDITABLE_RECORD.count(",,"), 1)

    def test_the_document_emits_only_what_is_set(self):
        """So a lock written before this field, and a normally installed
        package, both keep their shape."""
        self.assertEqual(
            InstalledDist(name="x", version="1.0").document(), {"version": "1.0"}
        )

    def test_editable_among_names_them_for_the_verdict(self):
        _dist(self.root, "libero", "0.1.0",
              direct_url={"url": "file:///a", "dir_info": {"editable": True}},
              record=EDITABLE_RECORD)
        _dist(self.root, "mujoco", "3.2.0", record=WHEEL_RECORD)
        dists = installed_dists(["libero", "mujoco"], search_path=[self.root])
        self.assertEqual(editable_among(dists), ("libero",))


class TheVerdictSaysSo(unittest.TestCase):
    def test_unchecked_is_distinguished_from_none(self):
        """An empty `editable` with no check behind it is not evidence of
        nothing, and the line must not read as though it were -- the same
        reason `declined` prints `none`."""
        self.assertIn("not checked", Reproducible().render())
        self.assertIn("published release",
                      Reproducible(installs_checked=True).render())

    def test_editable_is_stated_even_when_provenance_is_complete(self):
        """The orthogonal case, and the one that motivated this.

        Every provenance field can be present -- engine, version, seeds, the
        lot -- while the run is still unreproducible from any release, because
        the bytes that ran came from a working tree.
        """
        line = Reproducible(complete=True, installs_checked=True,
                            editable=("libero",)).render()
        self.assertTrue(line.startswith("yes"))
        self.assertIn("EDITABLE", line)
        self.assertIn("move no identity", line)


if __name__ == "__main__":
    unittest.main()
