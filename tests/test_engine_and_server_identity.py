"""What ran, read from what ran: the engine version from the installed engine, and
the model server's environment from the server, on every row (audit finding 9)."""

import unittest

from tests.test_build import Temp
from tests.test_vla_eval_loop import FakeHarness, LoopCase, make_plan

from refractal.build import build
from refractal.execute.vla_eval import server_identity
from refractal.resolve.lock import load_lock


class Probe:
    def version(self, engine):
        return "3.2.3"

    def verify(self, engine, model_path):
        pass


class TestTheEngineVersionIsMeasured(unittest.TestCase):
    def test_the_installed_version_wins_over_the_declared_one(self):
        """The catalog declares 3.2.0; the engine present is 3.2.3."""
        with Temp() as root:
            report = build(root, probe=Probe())
            self.assertEqual({e.engine_version for e in load_lock(root).scenes}, {"3.2.3"})
            self.assertTrue(any("declares engine_version '3.2.0'" in w for w in report.warnings))

    def test_a_catalog_with_a_wrong_declaration_still_plans_after_build(self):
        """The lock hashes the measured version; freshness must not compare it
        with a scene_hash recomputed from the declared one."""
        from refractal.resolve import resolve

        with Temp() as root:
            build(root, probe=Probe())
            plan = resolve(root, hardware_profile="rtx5090")
            self.assertEqual({s.scene_hash for s in plan.scenes},
                             {e.scene_hash for e in load_lock(root).scenes})

    def test_editing_the_geometry_after_build_is_still_refused(self):
        from refractal.resolve import resolve
        from refractal.schema import CatalogError

        with Temp() as root:
            build(root, probe=Probe())
            model = next((root / "assets").glob("*.xml"))
            model.write_text(model.read_text(encoding="utf-8") + "\n<!-- moved -->\n", encoding="utf-8")
            with self.assertRaises(CatalogError) as ctx:
                resolve(root, hardware_profile="rtx5090")
            self.assertIn("geometry", str(ctx.exception))

    def test_a_machine_without_the_engine_still_uses_the_declaration(self):
        with Temp() as root:
            build(root)
            self.assertEqual({e.engine_version for e in load_lock(root).scenes}, {"3.2.0"})


ENV = {"response_metadata": {"server_env": {
    "env_digest": "sha256:abc", "source_sha": "102f19d", "source_dirty": True,
    "checkpoint_revision": "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f"}}}


class TestServerIdentity(unittest.TestCase):
    def test_it_reads_the_server_env(self):
        self.assertEqual(server_identity("ws://h:1", fetch=lambda url: ENV), {
            "server_env_digest": "sha256:abc", "server_source_sha": "102f19d+dirty",
            "checkpoint_revision": "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
            "server_boot_id": None})

    def test_an_unreadable_server_is_nulls(self):
        def down(url):
            raise OSError("refused")

        self.assertEqual(set(server_identity("ws://h:1", fetch=down).values()), {None})


class TestServerIdentityReachesEveryRow(LoopCase):
    def test_every_row_carries_the_server_identity(self):
        import refractal.execute.vla_eval_runner as runner

        saved = runner.server_identity
        runner.server_identity = lambda url: server_identity(url, fetch=lambda u: ENV)
        self.addCleanup(setattr, runner, "server_identity", saved)
        plan = make_plan(scenarios=3, seeds=(0, 1), checkpoints=("pi0",))
        self.run_loop(plan, FakeHarness(), only=("pi0",))
        rows = self.rows(plan)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["server_env_digest"] for r in rows}, {"sha256:abc"})
        self.assertEqual({r["server_source_sha"] for r in rows}, {"102f19d+dirty"})
