"""A Hub checkpoint must name its commit, the commit is part of plan_id, and a run
refuses a server that did not load it (audit/REPORT.md finding 9)."""

import shutil
import tempfile
import types
import unittest
from pathlib import Path

import yaml

from refractal.checkpoints import is_hub_id, pinned_revision, split_revision
from refractal.execute.vla_eval import BridgeError, check_server_revisions
from refractal.resolve import resolve
from refractal.schema import CatalogError

CATALOG = Path(__file__).resolve().parents[1] / "examples" / "catalog"
HARDWARE = "rtx5090"
SHA_A = "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f"
SHA_B = "7152f13e34c6c115d808aea0e5d08aca74eac14b"


class TestTheShapeRule(unittest.TestCase):
    def test_hub_ids_and_local_paths(self):
        self.assertTrue(is_hub_id("lerobot/pi0_libero_finetuned_v044"))
        self.assertTrue(is_hub_id(f"lerobot/pi0@{SHA_A}"))
        for local in ("./checkpoints/baseline", "/abs/model", "~/m", "p", "a/b/c"):
            with self.subTest(path=local):
                self.assertFalse(is_hub_id(local))

    def test_only_a_full_commit_pins(self):
        self.assertEqual(pinned_revision(f"org/m@{SHA_A}"), SHA_A)
        for loose in ("org/m", "org/m@main", "org/m@v1.0", "org/m@45dcc8fc"):
            with self.subTest(path=loose):
                self.assertIsNone(pinned_revision(loose))
        self.assertEqual(split_revision(f"org/m@{SHA_A}"), ("org/m", SHA_A))


class TestPlanTime(unittest.TestCase):
    def plan_with(self, *paths):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        shutil.copytree(CATALOG, root, dirs_exist_ok=True)
        run = yaml.safe_load((root / "run.yaml").read_text(encoding="utf-8"))
        for checkpoint, path in zip(run["run"]["checkpoints"], paths):
            checkpoint["path"] = path
        (root / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
        return resolve(root, hardware_profile=HARDWARE)

    def test_an_unpinned_hub_id_is_refused(self):
        with self.assertRaises(CatalogError) as ctx:
            self.plan_with("lerobot/pi0_libero_finetuned_v044", "./checkpoints/candidate")
        self.assertIn("without a commit", str(ctx.exception))

    def test_a_branch_name_is_not_a_pin(self):
        with self.assertRaises(CatalogError):
            self.plan_with("lerobot/pi0@main", "./checkpoints/candidate")

    def test_pinned_hub_ids_and_local_paths_plan(self):
        self.plan_with(f"lerobot/pi0@{SHA_A}", "./checkpoints/candidate")

    def test_the_revision_enters_plan_id(self):
        a = self.plan_with(f"lerobot/pi0@{SHA_A}", "./checkpoints/candidate")
        b = self.plan_with(f"lerobot/pi0@{SHA_B}", "./checkpoints/candidate")
        self.assertNotEqual(a.plan_id, b.plan_id)


def _ck(cid, path):
    return types.SimpleNamespace(id=cid, path=path)


def _reporting(name, revision):
    return lambda url: {"response_metadata": {"server_env": {"checkpoint": name, "checkpoint_revision": revision}}}


class TestRunTime(unittest.TestCase):
    servers = {"pi0": "ws://127.0.0.1:8000"}

    def test_a_server_that_loaded_the_pinned_commit_passes(self):
        check_server_revisions([_ck("pi0", f"lerobot/pi0@{SHA_A}")], self.servers,
                               fetch=_reporting("lerobot/pi0", SHA_A))

    def test_a_server_on_another_commit_is_refused(self):
        with self.assertRaises(BridgeError) as ctx:
            check_server_revisions([_ck("pi0", f"lerobot/pi0@{SHA_A}")], self.servers,
                                   fetch=_reporting("lerobot/pi0", SHA_B))
        self.assertIn(SHA_B, str(ctx.exception))

    def test_a_server_that_does_not_report_is_refused(self):
        with self.assertRaises(BridgeError) as ctx:
            check_server_revisions([_ck("pi0", f"lerobot/pi0@{SHA_A}")], self.servers,
                                   fetch=lambda url: {"config": {}})
        self.assertIn("does not report", str(ctx.exception))

    def test_a_local_checkpoint_is_not_fetched(self):
        def fetch(url):
            raise AssertionError("fetched")

        check_server_revisions([_ck("pi0", "./checkpoints/baseline")], self.servers, fetch=fetch)
