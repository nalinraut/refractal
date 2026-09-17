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
from refractal.resolve.lock import LOCK_FILENAME, LOCK_SCHEMA, load_lock
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

    def test_editing_the_scene_is_caught_by_the_scene_check(self):
        """Now reported directly rather than via the filter key.

        Editing a mesh invalidates both the scene hash and the filter key (which
        contains it). The scene check runs first and names the actual cause --
        the geometry moved -- instead of reporting a downstream consequence.
        """
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root)
            build(root)
            asset = root / "assets" / "vial_rack.xml"
            asset.write_text(asset.read_text() + "\n<!-- moved a slot -->\n", encoding="utf-8")
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            message = str(ctx.exception)
            self.assertIn("The geometry changed", message)
            self.assertIn("the lock is what planning trusts", message)


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


class TestCallableShapesAreChecked(unittest.TestCase):
    """The predicate/filter swap, caught mechanically rather than by a comment."""

    def test_a_predicate_in_a_filter_field_is_refused(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            # cube_in_bowl is (state, args) -- a predicate, not a filter.
            with_filter(tmp, root, "refractal.example:cube_in_bowl")
            with self.assertRaises(Exception) as ctx:
                build(root)
            message = str(ctx.exception)
            self.assertIn("must be (scenario) -> bool", message)
            self.assertIn("That is the shape of a predicate", message)

    def test_a_real_filter_passes(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root, FILTER)   # (scenario) -> bool
            self.assertEqual(len(build(root).lock.filters), 1)

    def test_a_filter_in_a_predicate_field_is_refused(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            tmp.edit(
                "tasks.yaml",
                lambda d: d["tasks"][0].__setitem__(
                    "predicate", "refractal.example:within_reach"
                ),
            )
            with self.assertRaises(Exception) as ctx:
                build(root)
            self.assertIn("must be (state, args) -> bool", str(ctx.exception))

    def test_an_unimportable_predicate_is_only_a_note(self):
        """A catalog may legitimately be built before its adapter is installed."""
        with Temp() as root:
            report = build(root)
            self.assertTrue(any("could not import" in n for n in report.notes))


class TestFilterDiagnosis(unittest.TestCase):
    """Three distinct ways a filter can be wrong, each named separately."""

    def test_wrong_shape_reports_the_shape_not_the_workspace(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root, "refractal.example:cube_in_bowl")
            self.assertIn("shape of a predicate", str(self._error(root)))

    def test_wrong_parameter_names_are_named(self):
        """Right arity, wrong grid: the error must show what the scenarios carry.

        `within_reach` reads cube_x/cube_y; the vial grid carries vial_x/vial_y.
        `.get()` returns the default and every scenario is rejected, which looks
        exactly like an unreachable workspace unless the parameters are printed.
        """
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root, "refractal.example:within_reach")
            message = str(self._error(root))
            self.assertIn("'friction', 'vial_x', 'vial_y'", message)
            self.assertIn("parameters this grid does not define", message)

    def test_genuinely_inverted_filter_still_reports(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            with_filter(tmp, root, "tests.fixture_filters:rejects_everything")
            self.assertIn("inverted", str(self._error(root)))

    def _error(self, root):
        with self.assertRaises(Exception) as ctx:
            build(root)
        return ctx.exception


class TestExternallyDefinedScenes(unittest.TestCase):
    """Scenes whose geometry lives in a wrapped benchmark, not in this catalog.

    A LIBERO scene is a BDDL file inside the installed `libero` package. There is
    nothing catalog-local to hash, and a placeholder `model:` path would put a lie
    in the artifact whose job is to not lie -- the catalog copied into every
    results directory as provenance.
    """

    EXTERNAL = {
        "id": "libero-spatial-3",
        "engine": "mujoco",
        "engine_version": "3.2.0",
        "external": {
            "provider": "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark",
            "ref": {"suite": "libero_spatial", "task_id": 3},
        },
        "resource_shape": [
            {"hardware_profile": "rtx5090", "envs_per_process": 1, "vram_per_env_mb": 0,
             "cpu_cores": 1, "sec_per_1k_steps": 72, "startup_sec": 3},
        ],
    }

    class Probe:
        """Stands in for a probe that can import the provider."""

        def __init__(self, digest="sha256:" + "ab" * 32):
            self.digest = digest

        def version(self, engine):
            return "3.2.0"

        def verify(self, engine, model_path):
            return None

        def external_scene_facts(self, scene):
            # A document, not a digest: Refractal hashes it, so there is exactly
            # one implementation of the canonicalisation.
            return {"provider": "stand-in", "digest": self.digest}

    def _catalog(self, tmp, root):
        tmp.edit("scenes.yaml", lambda d: d.__setitem__("scenes", [self.EXTERNAL]))
        tmp.edit("tasks.yaml", lambda d: d.__setitem__("tasks", [{
            **d["tasks"][0], "scene": "libero-spatial-3",
            "predicate": "refractal.example:cube_in_bowl"}]))
        tmp.edit("scenarios.yaml", lambda d: d.__setitem__("scenario_sets", [{
            **d["scenario_sets"][0], "scene": "libero-spatial-3",
            "params": {"init_state_index": {"range": [0, 9], "steps": 10}}}]))
        tmp.edit("run.yaml", lambda d: d["run"].__setitem__("tier", "full"))

    def test_plan_refuses_before_build(self):
        """Same failure shape as a filtered catalog: consistency beats avoidance."""
        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            self._catalog(tmp, root)
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            self.assertIn("refractal build", str(ctx.exception))

    def test_build_records_the_hash_and_plan_then_works(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            self._catalog(tmp, root)
            report = build(root, hardware_profile=HARDWARE, probe=self.Probe())
            entry = report.lock.scene_entry("libero-spatial-3")
            self.assertTrue(entry.external)
            self.assertIsNotNone(entry.ref_key)
            plan = resolve(root, hardware_profile=HARDWARE)
            from refractal.schema import hash_obj

            self.assertEqual(
                plan.scenes[0].scene_hash,
                hash_obj({"provider": "stand-in", "digest": self.Probe().digest}),
            )
            self.assertEqual(len(plan.scenes[0].scenarios), 10)

    def test_build_without_a_capable_probe_refuses(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            self._catalog(tmp, root)
            with self.assertRaises(BuildError) as ctx:
                build(root, hardware_profile=HARDWARE)
            self.assertIn("where the benchmark is installed", str(ctx.exception))

    def test_a_probe_that_returns_a_digest_instead_of_facts_is_refused(self):
        """Hashing happens in one place; a probe that hashes is a second one."""

        class HashingProbe(self.Probe):
            def external_scene_facts(self, scene):
                return "sha256:" + "cd" * 32  # a digest, not a document

        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            self._catalog(tmp, root)
            with self.assertRaises(BuildError) as ctx:
                build(root, hardware_profile=HARDWARE, probe=HashingProbe())
            self.assertIn("second implementation of the canonicalisation", str(ctx.exception))

    def test_the_facts_are_recorded_so_a_changed_hash_can_be_explained(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            self._catalog(tmp, root)
            report = build(root, hardware_profile=HARDWARE, probe=self.Probe())
            entry = report.lock.scene_entry("libero-spatial-3")
            self.assertEqual(entry.facts["provider"], "stand-in")

    def test_editing_the_ref_makes_the_lock_stale(self):
        """task_id 3 and task_id 4 are different scenes, and the hash must stop being trusted."""
        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            self._catalog(tmp, root)
            build(root, hardware_profile=HARDWARE, probe=self.Probe())
            tmp.edit("scenes.yaml", lambda d: d["scenes"][0]["external"]["ref"].__setitem__("task_id", 4))
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            self.assertIn("different provider or ref", str(ctx.exception))

    def test_scene_hash_reaches_the_plan_and_would_gate_compare(self):
        """The whole point of folding init states into scene_hash."""
        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            self._catalog(tmp, root)
            build(root, hardware_profile=HARDWARE, probe=self.Probe("sha256:" + "11" * 32))
            first = resolve(root, hardware_profile=HARDWARE)
            # LIBERO ships a different init-state file: the digest moves, so the
            # scene_hash moves, so compare refuses instead of partially joining.
            build(root, hardware_profile=HARDWARE, probe=self.Probe("sha256:" + "22" * 32))
            second = resolve(root, hardware_profile=HARDWARE)
            self.assertNotEqual(first.scenes[0].scene_hash, second.scenes[0].scene_hash)
            self.assertNotEqual(first.plan_id, second.plan_id)


class TestLockSchemaRefusal(unittest.TestCase):
    """The version gate on build.lock, exercised.

    Closed because it was a gap rather than a deferral. The refusal was written
    and correct; nobody had bumped `lock_schema`, so nothing had ever handed the
    reader a number it did not recognise. A fixture carrying a future version
    costs two lines and needs nothing that does not exist.

    That matters more than it looks: the LIBERO bridge lives in a separate repo,
    so two distributions have to agree on this file across a version boundary.
    The first time that gate fires for real, it should not be the first time it
    fires at all.
    """

    def _write_lock(self, root, mutate):
        path = root / LOCK_FILENAME
        doc = json.loads(path.read_text(encoding="utf-8"))
        mutate(doc)
        path.write_text(json.dumps(doc), encoding="utf-8")

    def test_a_future_lock_is_refused_not_read(self):
        with Temp() as root:
            build(root)
            self._write_lock(root, lambda d: d.__setitem__("lock_schema", LOCK_SCHEMA + 1))
            with self.assertRaises(CatalogError) as ctx:
                load_lock(root)
            message = str(ctx.exception)
            self.assertIn(f"lock_schema={LOCK_SCHEMA + 1}", message)
            self.assertIn("refractal build", message)

    def test_a_past_lock_is_also_refused(self):
        """A lock is regenerable, so there is no reason to attempt an old one."""
        with Temp() as root:
            build(root)
            self._write_lock(root, lambda d: d.__setitem__("lock_schema", LOCK_SCHEMA - 1))
            with self.assertRaises(CatalogError):
                load_lock(root)

    def test_a_missing_lock_schema_is_refused(self):
        with Temp() as root:
            build(root)
            self._write_lock(root, lambda d: d.pop("lock_schema"))
            with self.assertRaises(CatalogError):
                load_lock(root)

    def test_resolve_surfaces_the_refusal_rather_than_ignoring_the_lock(self):
        """The failure must not degrade into "no lock, plan anyway"."""
        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            with_filter(tmp, root)
            build(root)
            self._write_lock(root, lambda d: d.__setitem__("lock_schema", 99))
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            self.assertIn("lock_schema=99", str(ctx.exception))

    def test_the_current_schema_round_trips(self):
        with Temp() as root:
            build(root)
            self.assertEqual(load_lock(root).lock_schema, LOCK_SCHEMA)
