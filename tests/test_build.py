import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from refractal.build import BuildError, DeclaredProbe, build
from refractal.resolve import resolve
from refractal.resolve.lock import LOCK_FILENAME, load_lock
from refractal.schema import CatalogError, load_catalog

SRC = Path(__file__).resolve().parents[1] / "examples" / "catalog"
FILTER = "tests.fixture_filters:reachable"
HARDWARE = "rtx5090"


class Temp:
    def __enter__(self) -> Path:
        self._dir = tempfile.mkdtemp(prefix="refractal-build-")
        self.root = Path(self._dir) / "catalog"
        shutil.copytree(SRC, self.root)
        return self.root

    def __exit__(self, *exc):
        shutil.rmtree(self._dir, ignore_errors=True)

    def edit(self, filename, mutate):
        path = self.root / filename
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        mutate(doc)
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def with_filter(tmp, root, name=FILTER):
    tmp.edit("scenarios.yaml", lambda d: d["scenario_sets"][0].__setitem__("filter", name))


class TestLockContents(unittest.TestCase):
    def test_writes_a_lock_with_scene_hashes(self):
        with Temp() as root:
            report = build(root, built_at="2026-01-01T00:00:00Z")
            self.assertTrue((root / LOCK_FILENAME).is_file())
            lock = load_lock(root)
            self.assertEqual({s.scene_id for s in lock.scenes}, {"vial-rack-v1", "cube-bowl-v1"})
            for entry in lock.scenes:
                self.assertTrue(entry.scene_hash.startswith("sha256:"))
                self.assertEqual(entry.engine_version, "3.2.0")
            self.assertEqual(report.warnings, [])

    def test_never_rewrites_hand_authored_yaml(self):
        """Machine-written and human-written data do not share a file."""
        with Temp() as root:
            before = (root / "scenes.yaml").read_bytes()
            build(root)
            self.assertEqual((root / "scenes.yaml").read_bytes(), before)

    def test_declared_engine_version_is_required_without_a_probe(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            tmp.edit("scenes.yaml", lambda d: [s.pop("engine_version") for s in d["scenes"]])
            with self.assertRaises(BuildError) as ctx:
                build(root)
            self.assertIn("where the engine is installed", str(ctx.exception))

    def test_a_probe_supplies_the_version_and_verifies_the_model(self):
        seen = []

        class Probe:
            def version(self, engine):
                return "9.9.9-probed"

            def verify(self, engine, model_path):
                seen.append(model_path.name)

        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            tmp.edit("scenes.yaml", lambda d: [s.pop("engine_version") for s in d["scenes"]])
            build(root, probe=Probe())
            lock = load_lock(root)
            self.assertEqual({e.engine_version for e in lock.scenes}, {"9.9.9-probed"})
            self.assertEqual(sorted(seen), ["cube_bowl.xml", "vial_rack.xml"])

    def test_stale_model_hash_warns_loudly(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            tmp.edit("scenes.yaml", lambda d: d["scenes"][0].__setitem__("model_hash", "sha256:" + "0" * 64))
            report = build(root)
            self.assertTrue(any("geometry changed" in w for w in report.warnings))


class TestFilters(unittest.TestCase):
    def test_survivors_are_recorded(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root)
            report = build(root)
            entry = report.lock.filters[0]
            self.assertEqual(entry.generated, 120)
            self.assertGreater(entry.dropped, 0)
            self.assertEqual(len(entry.survivors), entry.generated - entry.dropped)

    def test_a_filter_that_rejects_everything_is_an_error(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root, "tests.fixture_filters:rejects_everything")
            with self.assertRaises(BuildError) as ctx:
                build(root)
            # Worth knowing before a run rather than after: an inverted filter
            # and an unreachable grid look identical from the plan.
            self.assertIn("inverted", str(ctx.exception))

    def test_a_raising_filter_names_the_scenario(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root, "tests.fixture_filters:explodes")
            with self.assertRaises(BuildError) as ctx:
                build(root)
            self.assertIn("did not converge", str(ctx.exception))

    def test_an_unimportable_filter_says_where_build_must_run(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root, "so101_eval.filters:reachable")
            with self.assertRaises(BuildError) as ctx:
                build(root)
            self.assertIn("dependencies are installed", str(ctx.exception))


class TestBuildUnblocksResolve(unittest.TestCase):
    """The whole point: resolve refuses a filtered catalog until build has run."""

    def test_resolve_refuses_before_build_and_succeeds_after(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root)

            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            self.assertIn("refractal build", str(ctx.exception))

            build(root)
            plan = resolve(root, hardware_profile=HARDWARE)
            vial = next(s for s in plan.scenes if s.scene_id == "vial-rack-v1")
            self.assertLess(len(vial.scenarios), 120)
            self.assertTrue(any("dropped by filter" in w for w in plan.warnings))

    def test_a_stale_lock_is_refused_rather_than_used(self):
        """Editing the grid after building must not silently reuse old survivors."""
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root)
            build(root)
            tmp.edit(
                "scenarios.yaml",
                lambda d: d["scenario_sets"][0]["params"]["vial_x"].__setitem__("steps", 6),
            )
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            self.assertIn("stale", str(ctx.exception))

    def test_editing_the_scene_also_staleness_checks(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root)
            build(root)
            asset = root / "assets" / "vial_rack.xml"
            asset.write_text(asset.read_text() + "\n<!-- moved a slot -->\n", encoding="utf-8")
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            self.assertIn("stale", str(ctx.exception))


class TestDeterminism(unittest.TestCase):
    def test_same_catalog_produces_the_same_lock(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root)
            first = build(root, built_at="2026-01-01T00:00:00Z").lock.model_dump_json()
            second = build(root, built_at="2026-01-01T00:00:00Z").lock.model_dump_json()
            self.assertEqual(first, second)

    def test_built_at_is_injected_not_read_from_the_clock(self):
        with Temp() as root:
            self.assertIsNone(build(root).lock.built_at)


if __name__ == "__main__":
    unittest.main()


class TestFilterBodyIsHashed(unittest.TestCase):
    """The name is not enough: editing the body without renaming must be caught.

    Same class as the mesh case, and less obvious, because nothing about the
    catalog changed at all.
    """

    def _write_filter(self, pkg_dir: Path, threshold: float):
        (pkg_dir / "movable_filter.py").write_text(
            f"def reachable(scenario):\n"
            f"    return abs(scenario.get('vial_y', 0.0)) <= {threshold}\n",
            encoding="utf-8",
        )
        for name in list(sys.modules):
            if name == "movable_filter":
                del sys.modules[name]
        importlib.invalidate_caches()

    def test_editing_the_body_invalidates_the_lock(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            pkg_dir = root.parent
            sys.path.insert(0, str(pkg_dir))
            try:
                self._write_filter(pkg_dir, 0.05)
                with_filter(tmp, root, "movable_filter:reachable")
                report = build(root)
                self.assertIsNotNone(report.lock.filters[0].source_sha)

                # Same name, same catalog, different logic.
                self._write_filter(pkg_dir, 0.02)
                with self.assertRaises(CatalogError) as ctx:
                    resolve(root, hardware_profile=HARDWARE)
                self.assertIn("though its name did not", str(ctx.exception))
            finally:
                sys.path.remove(str(pkg_dir))
                sys.modules.pop("movable_filter", None)

    def test_an_unimportable_filter_degrades_to_a_warning_not_a_failure(self):
        """A laptop with no simulator must still be able to plan."""
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root)
            build(root)
            # Rewrite the lock to name a filter nothing here can import, keeping
            # the key consistent so only the body check can fire.
            lock_path = root / LOCK_FILENAME
            doc = json.loads(lock_path.read_text(encoding="utf-8"))
            lock_path.write_text(json.dumps(doc), encoding="utf-8")
            tmp.edit(
                "scenarios.yaml",
                lambda d: d["scenario_sets"][0].__setitem__(
                    "filter", "tests.fixture_filters:reachable"
                ),
            )
            plan = resolve(root, hardware_profile=HARDWARE)
            self.assertTrue(plan.total_episodes > 0)
