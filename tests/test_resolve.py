import ast
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from refractal.resolve import CapacityError, resolve
from refractal.resolve.expand import TIER_FRACTION, subsample
from refractal.schema import CatalogError, load_catalog
from refractal.schema.plan import PLAN_SCHEMA, PlanSchemaError, read_plan

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "examples" / "catalog"
HARDWARE = "rtx5090"


class Temp:
    """A writable copy of the example catalog."""

    def __enter__(self) -> Path:
        self._dir = tempfile.mkdtemp(prefix="refractal-resolve-")
        self.root = Path(self._dir) / "catalog"
        shutil.copytree(CATALOG, self.root)
        return self.root

    def __exit__(self, *exc):
        shutil.rmtree(self._dir, ignore_errors=True)

    def edit(self, filename, mutate):
        path = self.root / filename
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        mutate(doc)
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


class TestPlanShape(unittest.TestCase):
    def setUp(self):
        self.plan = resolve(CATALOG, hardware_profile=HARDWARE)

    def test_every_episode_assigned_exactly_once(self):
        # A dropped episode is a silently smaller denominator; a duplicated one
        # is a scenario counted twice in the comparison.
        for scene in self.plan.scenes:
            ids = [e.episode_id for w in scene.workers for e in w.episodes]
            self.assertEqual(len(ids), scene.episode_count)
            self.assertEqual(len(set(ids)), scene.episode_count)

    def test_episodes_are_self_describing(self):
        """A worker must be runnable from its assignment alone.

        The API reference gives workers bare episode-id strings, which are
        hashes -- nothing can recover (task, scenario, seed, checkpoint) from
        them, so `execute` would have nothing to run.
        """
        episode = self.plan.scenes[0].workers[0].episodes[0]
        self.assertTrue(episode.task_id)
        self.assertTrue(episode.scenario_hash)
        self.assertTrue(episode.checkpoint_id)
        self.assertIn(episode.seed, self.plan.seeds)

    def test_worker_ids_are_never_bare_indices(self):
        for scene in self.plan.scenes:
            for worker in scene.workers:
                self.assertRegex(worker.worker_id, r"^[a-z0-9/-]+/[a-z0-9-]+$")
                self.assertIn(scene.scene_id, worker.worker_id)

    def test_both_checkpoints_face_every_scenario(self):
        # Pairing is a property of the plan, not an analysis choice.
        for scene in self.plan.scenes:
            episodes = [e for w in scene.workers for e in w.episodes]
            by_ckpt = {}
            for episode in episodes:
                by_ckpt.setdefault(episode.checkpoint_id, set()).add(
                    (episode.task_hash, episode.scenario_hash, episode.seed)
                )
            self.assertEqual(len(by_ckpt), 2)
            a, b = by_ckpt.values()
            self.assertEqual(a, b)


class TestWorkerCountIsCapacityBound(unittest.TestCase):
    """The correction to `workers = ceil(episodes / envs_per_process)`."""

    def test_not_one_worker_per_episode(self):
        plan = resolve(CATALOG, hardware_profile=HARDWARE)
        workers = sum(len(s.workers) for s in plan.scenes)
        # envs_per_process is 1 for mujoco; the broken formula would give one
        # worker per episode.
        self.assertLess(workers, plan.total_episodes)
        self.assertLessEqual(workers, load_catalog(CATALOG).hardware(HARDWARE).max_workers)

    def test_ceiling_applies_when_capacity_is_ample(self):
        """Past ceil(episodes / envs_per_process) a worker cannot fill its batch."""
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            # One tiny scenario set, one task, one seed, one checkpoint: 2 episodes.
            tmp.edit(
                "scenarios.yaml",
                lambda d: d.__setitem__(
                    "scenario_sets",
                    [
                        {
                            **d["scenario_sets"][0],
                            "params": {"vial_x": {"choices": [0.1, 0.12]}},
                        }
                    ],
                ),
            )
            tmp.edit("tasks.yaml", lambda d: d.__setitem__("tasks", [d["tasks"][0]]))
            tmp.edit(
                "run.yaml",
                lambda d: (
                    d["run"].__setitem__("seeds", 1),
                    d["run"].__setitem__("checkpoints", d["run"]["checkpoints"][:1]),
                ),
            )
            tmp.edit("hardware.yaml", lambda d: d["hardware_profiles"][0].pop("max_workers"))
            tmp.edit("run.yaml", lambda d: d["run"].__setitem__("tier", "full"))
            plan = resolve(root, hardware_profile=HARDWARE)
            self.assertEqual(plan.total_episodes, 2)
            # 16 cores available, but only 2 useful workers.
            self.assertEqual(sum(len(s.workers) for s in plan.scenes), 2)


class TestRefusals(unittest.TestCase):
    def test_vram_oversubscription_refused(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root

            def add_checkpoints(doc):
                base = doc["run"]["checkpoints"][0]
                doc["run"]["checkpoints"] = [
                    {**base, "id": f"ckpt-{i}", "vram_mb": 8192} for i in range(6)
                ]

            tmp.edit("run.yaml", add_checkpoints)
            with self.assertRaises(CapacityError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            message = str(ctx.exception)
            # The error must name the device, the budget and the shortfall --
            # a refusal you cannot act on is barely better than a crash.
            self.assertIn("cuda:0", message)
            self.assertIn("32768", message)

    def test_missing_resource_shape_refused(self):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            tmp.edit("scenes.yaml", lambda d: d["scenes"][0].pop("resource_shape"))
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            self.assertIn("resource_shape", str(ctx.exception))

    def test_unknown_hardware_profile_refused(self):
        with self.assertRaises(CatalogError) as ctx:
            resolve(CATALOG, hardware_profile="h100-nonexistent")
        self.assertIn("h100-nonexistent", str(ctx.exception))

    def test_filter_without_lock_refused(self):
        """A filter needs the engine, so `resolve` must not guess at its result."""
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            tmp.edit(
                "scenarios.yaml",
                lambda d: d["scenario_sets"][0].__setitem__(
                    "filter", "so101_eval.filters:reachable"
                ),
            )
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile=HARDWARE)
            self.assertIn("refractal build", str(ctx.exception))


class TestTiers(unittest.TestCase):
    def _scenarios(self, tier):
        with Temp() as root:
            tmp = Temp.__new__(Temp)
            tmp.root = root
            tmp.edit("run.yaml", lambda d: d["run"].__setitem__("tier", tier))
            plan = resolve(root, hardware_profile=HARDWARE)
            return {
                scene.scene_id: {s.scenario_hash for s in scene.scenarios}
                for scene in plan.scenes
            }

    def test_ladder_is_strictly_nested(self):
        """Promotion must add work, not replace it.

        Two independent samples would give a smoke set that is not inside the
        regression set, so "smoke passed then regression failed" would carry no
        information about what changed.
        """
        smoke, regression, full = (
            self._scenarios("smoke"),
            self._scenarios("regression"),
            self._scenarios("full"),
        )
        for scene_id in full:
            self.assertTrue(smoke[scene_id] <= regression[scene_id], scene_id)
            self.assertTrue(regression[scene_id] <= full[scene_id], scene_id)

    def test_small_set_still_contributes_to_smoke(self):
        """A three-scenario set must not round to zero at 10%."""
        tiny = [
            type("S", (), {"scenario_hash": f"sha256:{i:064x}", "scenario_set_id": "tiny"})()
            for i in range(3)
        ]
        kept = subsample(tiny, "smoke")
        self.assertGreaterEqual(len(kept), 1)

    def test_full_keeps_everything(self):
        self.assertEqual(TIER_FRACTION["full"], 1.0)


class TestPlanFile(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan = resolve(CATALOG, hardware_profile=HARDWARE)
            plan.write(path)
            self.assertEqual(read_plan(path).plan_id, plan.plan_id)

    def test_unrecognised_plan_schema_is_refused_not_attempted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            resolve(CATALOG, hardware_profile=HARDWARE).write(path)
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["plan_schema"] = PLAN_SCHEMA + 1
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(PlanSchemaError) as ctx:
                read_plan(path)
            self.assertIn("upgrade", str(ctx.exception))

    def test_created_at_is_injected_not_read_from_the_clock(self):
        # A planner that reads the wall clock cannot be tested for determinism.
        self.assertIsNone(resolve(CATALOG, hardware_profile=HARDWARE).created_at)
        stamped = resolve(CATALOG, hardware_profile=HARDWARE, created_at="2026-01-01T00:00:00Z")
        self.assertEqual(stamped.created_at, "2026-01-01T00:00:00Z")


class TestDeterminism(unittest.TestCase):
    SNIPPET = (
        "import sys;"
        "from refractal.resolve import resolve;"
        "print(resolve(sys.argv[1], hardware_profile=sys.argv[2]).to_json())"
    )

    def _run(self):
        out = subprocess.run(
            [sys.executable, "-c", self.SNIPPET, str(CATALOG), HARDWARE],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(REPO / "src"),
                "PYTHONHASHSEED": "random",
            },
            check=True,
        )
        return out.stdout

    def test_byte_identical_across_processes(self):
        self.assertEqual(self._run(), self._run())


class TestEngineAbstraction(unittest.TestCase):
    def test_resolve_never_branches_on_engine_name(self):
        """Invariant 2, as a test rather than a comment.

        `engine` is used in exactly two places: image selection in `execute`,
        and as part of the resource-shape lookup key. A branch here means the
        abstraction broke, and the fix is to change what the shape reports.
        """
        engines = {"mujoco", "mjx", "isaac"}
        offenders = []
        for path in (REPO / "src" / "refractal" / "resolve").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            # Docstrings and comments are where the rule is *stated*, so they
            # have to be excluded or the invariant flags its own explanation.
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in engines
                    and id(node) not in docstrings
                ):
                    offenders.append(f"{path.name}:{node.lineno}: literal {node.value!r}")
                if isinstance(node, ast.Compare):
                    src = ast.dump(node)
                    if "'engine'" in src or "attr='engine'" in src:
                        offenders.append(f"{path.name}:{node.lineno}: comparison on engine")
        self.assertEqual(offenders, [], "resolve branches on engine name")


if __name__ == "__main__":
    unittest.main()


class TestTypeOnlyCollisions(unittest.TestCase):
    """The residual hazard of preserving the authored numeric type.

    Worth being exact about where it can fire, because the obvious guess is
    wrong. A type *edit* to a catalog changes `plan_id`, so before and after land
    in different comparison directories and `compare` never sees both -- it
    cannot report them as non-overlapping. The confusion is only reachable
    *within one plan*.
    """

    def _two_sets(self, root, first, second):
        tmp = Temp.__new__(Temp)
        tmp.root = root

        def mutate(doc):
            base = doc["scenario_sets"][0]
            doc["scenario_sets"] = [
                {**base, "id": "coarse", "params": {"friction": {"choices": [first]}}},
                {**base, "id": "fine", "params": {"friction": {"choices": [second]}}},
            ]

        tmp.edit("scenarios.yaml", mutate)
        tmp.edit("run.yaml", lambda d: d["run"].__setitem__("tier", "full"))
        return resolve(root, hardware_profile=HARDWARE)

    def test_a_type_edit_cannot_reach_the_overlap_report(self):
        """Verified rather than assumed: the two land in different comparisons."""
        with Temp() as a, Temp() as b:
            ta, tb = Temp.__new__(Temp), Temp.__new__(Temp)
            ta.root, tb.root = a, b
            for tmp, value in ((ta, 1), (tb, 1.0)):
                tmp.edit("scenarios.yaml", lambda d, v=value: d.__setitem__(
                    "scenario_sets",
                    [{**d["scenario_sets"][0], "params": {"friction": {"choices": [v]}}}]))
                tmp.edit("run.yaml", lambda d: d["run"].__setitem__("tier", "full"))
            self.assertNotEqual(
                resolve(a, hardware_profile=HARDWARE).plan_id,
                resolve(b, hardware_profile=HARDWARE).plan_id,
            )

    def test_within_one_plan_it_warns(self):
        with Temp() as root:
            plan = self._two_sets(root, 1, 1.0)
            self.assertEqual(len(plan.scenes[0].scenarios), 2)
            note = [w for w in plan.warnings if "numerically equal" in w]
            self.assertEqual(len(note), 1)
            self.assertIn("coarse", note[0])
            self.assertIn("fine", note[0])
            self.assertIn("two scenarios rather than one", note[0])

    def test_the_same_type_twice_is_deduped_not_warned(self):
        with Temp() as root:
            plan = self._two_sets(root, 1.0, 1.0)
            self.assertFalse([w for w in plan.warnings if "numerically equal" in w])

    def test_genuinely_different_values_do_not_warn(self):
        with Temp() as root:
            plan = self._two_sets(root, 1, 2)
            self.assertEqual(len(plan.scenes[0].scenarios), 2)
            self.assertFalse([w for w in plan.warnings if "numerically equal" in w])

    def test_identical_scenarios_across_sets_are_deduped_not_duplicated(self):
        """Two sets, one scenario. The bug this found, as a regression test.

        Deduplication used to happen only within a single scenario set, so two
        sets on one scene producing the same scenario yielded the same
        scenario_hash twice, the same episode_id twice, and a plan that ran and
        recorded every affected episode twice over.
        """
        with Temp() as root:
            plan = self._two_sets(root, 1.0, 1.0)
            scenarios = plan.scenes[0].scenarios
            self.assertEqual(len(scenarios), 1)
            episodes = [e for w in plan.scenes[0].workers for e in w.episodes]
            self.assertEqual(len(episodes), len({e.episode_id for e in episodes}))
            self.assertTrue(any("kept once" in w for w in plan.warnings))

    def test_dedup_is_reported_rather_than_silent(self):
        with Temp() as root:
            note = next(w for w in self._two_sets(root, 1.0, 1.0).warnings if "kept once" in w)
            self.assertIn("coarse", note)
            self.assertIn("fine", note)


class TestTheDefaultFixtureExercisesItsInvariants(unittest.TestCase):
    """Does any fixture put each invariant in a position to fire?

    A different question from "is the invariant correct", and the one that found
    three bugs. These assert the *fixture*, not the code: if someone tidies the
    deliberately-overlapping scenario sets out of examples/catalog, the invariant
    goes back to being untestable by the default path and this fails.
    """

    def setUp(self):
        self.plan = resolve(CATALOG, hardware_profile=HARDWARE)

    def test_the_fixture_contains_overlapping_scenario_sets(self):
        self.assertTrue(
            [w for w in self.plan.warnings if "kept once" in w],
            "examples/catalog no longer has overlapping scenario sets, so the "
            "per-scene dedup invariant is unexercised by the default fixture",
        )

    def test_and_the_overlap_is_deduped_rather_than_doubled(self):
        for scene in self.plan.scenes:
            episodes = [e for w in scene.workers for e in w.episodes]
            self.assertEqual(
                len(episodes), len({e.episode_id for e in episodes}), scene.scene_id
            )
            hashes = [s.scenario_hash for s in scene.scenarios]
            self.assertEqual(len(hashes), len(set(hashes)), scene.scene_id)

    def test_the_fixture_contains_a_scene_with_two_tasks(self):
        """Otherwise the map-not-struct phase_outcomes fix is unexercised."""
        by_scene = {}
        for task in load_catalog(CATALOG).tasks:
            by_scene.setdefault(task.scene, []).append(task.id)
        self.assertTrue(
            any(len(v) > 1 for v in by_scene.values()),
            "no scene has two tasks, so nothing writes differing phase sets to one file",
        )

    def test_the_fixture_contains_a_scene_with_no_declared_phases(self):
        """The other half of that: one task with phases, one without."""
        tasks = load_catalog(CATALOG).tasks
        self.assertTrue(any(t.phases for t in tasks))
        self.assertTrue(any(not t.phases for t in tasks))


class TestExplainingWhyTwoPlansDiffer(unittest.TestCase):
    """A digest says something moved and nothing about what.

    Third instance of the same gap: the harness surface records a per-file
    manifest beside its digest, an external scene records its facts beside its
    hash, and a plan records the document its id was computed from. Each one
    turns "these did not join" from a bisect into a sentence.
    """

    def _plan(self, mutate=None):
        with Temp() as root:
            if mutate is not None:
                tmp = Temp.__new__(Temp)
                tmp.root = root
                tmp.edit("run.yaml", mutate)
            return resolve(root, hardware_profile=HARDWARE)

    def test_identical_plans_have_nothing_to_explain(self):
        from refractal.schema.plan import explain_identity_difference

        first, second = self._plan(), self._plan()
        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(explain_identity_difference(first, second), [])

    def test_it_names_the_field_that_moved(self):
        from refractal.schema.plan import explain_identity_difference

        before = self._plan()
        after = self._plan(lambda d: d["run"].__setitem__("seeds", 5))
        lines = explain_identity_difference(before, after)
        self.assertEqual(lines, ["run.seeds: 3 -> 5"])

    def test_it_reaches_into_server_args(self):
        """The 55-point class: a parameter buried in a nested dict."""
        from refractal.schema.plan import explain_identity_difference

        def mutate(doc):
            doc["run"]["checkpoints"][1]["server_args"]["max_batch_size"] = 32

        lines = explain_identity_difference(self._plan(), self._plan(mutate))
        self.assertEqual(
            lines, ["run.checkpoints[1].server_args.max_batch_size: 8 -> 32"]
        )

    def test_it_reports_several(self):
        from refractal.schema.plan import explain_identity_difference

        def mutate(doc):
            doc["run"]["seeds"] = 5
            doc["run"]["tier"] = "smoke"

        lines = explain_identity_difference(self._plan(), self._plan(mutate))
        self.assertEqual(len(lines), 2)
        self.assertTrue(any("run.seeds" in line for line in lines))
        self.assertTrue(any("run.tier" in line for line in lines))

    def test_a_plan_without_an_identity_document_says_so(self):
        """Honest about the limit rather than silently explaining nothing."""
        from refractal.schema.plan import explain_identity_difference

        before = self._plan()
        after = self._plan(lambda d: d["run"].__setitem__("seeds", 5))
        stripped = before.model_copy(update={"identity": {}})
        lines = explain_identity_difference(stripped, after)
        self.assertIn("predates identity recording", lines[0])

    def test_the_identity_document_is_what_the_id_was_computed_from(self):
        from refractal.schema.plan import Plan
        from refractal.schema import hash_obj

        plan = self._plan()
        self.assertEqual(plan.plan_id, hash_obj(plan.identity))

    def test_explain_accepts_a_results_directory(self):
        """The plan is copied in as provenance; using it should not need the layout."""
        import tempfile

        from refractal.execute import run_local
        from refractal.schema.plan import read_plan

        with Temp() as root, tempfile.TemporaryDirectory() as results:
            plan = resolve(root, hardware_profile=HARDWARE)
            run_local(plan, results, session_id="0" * 32, catalog_root=str(root))
            # a results root with one comparison in it
            self.assertEqual(read_plan(results).plan_id, plan.plan_id)
            # and the comparison directory itself
            comparison = next(Path(results).glob("comparison_id=*"))
            self.assertEqual(read_plan(comparison).plan_id, plan.plan_id)

    def test_an_ambiguous_results_root_lists_the_comparisons(self):
        import tempfile

        from refractal.execute import run_local
        from refractal.schema.plan import read_plan

        with Temp() as root, tempfile.TemporaryDirectory() as results:
            run_local(resolve(root, hardware_profile=HARDWARE), results,
                      session_id="0" * 32)
            tmp = Temp.__new__(Temp); tmp.root = root
            tmp.edit("run.yaml", lambda d: d["run"].__setitem__("seeds", 5))
            run_local(resolve(root, hardware_profile=HARDWARE), results,
                      session_id="1" * 32)
            with self.assertRaises(CatalogError) as ctx:
                read_plan(results)
            self.assertIn("holds 2 comparisons", str(ctx.exception))
