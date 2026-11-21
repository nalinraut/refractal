"""plan_id recomputes from plan.json alone (audit finding 9).

The stored `identity` was built without the task hashes `build` folds in from
the provider, while plan_id was computed with them, so the identity in the file
did not hash to the id beside it. Only visible on a catalog whose tasks have
provider facts, which is every LIBERO catalog.
"""

import json
import tempfile
import unittest
from pathlib import Path

import tests.test_build as test_build
from tests.test_build import HARDWARE, Temp

from refractal.build import build
from refractal.resolve import resolve
from refractal.schema.identity import plan_id


class ProbeWithTaskFacts(test_build.TestExternallyDefinedScenes.Probe):
    def task_facts(self, scene, task):
        return {"goal": ["On", "akita_black_bowl_1", "plate_1"]}


class TestPlanIdRecomputesFromPlanJson(unittest.TestCase):
    def plan_json(self, catalog_setup):
        with Temp() as root:
            tmp = Temp.__new__(Temp); tmp.root = root
            catalog_setup(tmp, root)
            build(root, hardware_profile=HARDWARE, probe=ProbeWithTaskFacts())
            plan = resolve(root, hardware_profile=HARDWARE)
            path = Path(tempfile.mkdtemp()) / "plan.json"
            path.write_text(plan.to_json(), encoding="utf-8")
            return json.loads(path.read_text(encoding="utf-8"))

    def test_an_external_catalog_with_task_facts(self):
        doc = self.plan_json(test_build.TestExternallyDefinedScenes()._catalog)
        self.assertEqual(plan_id(doc["identity"]), doc["plan_id"])
        # and the stored identity names the same task hash its episodes carry
        stored = {t["id"]: t["task_hash"] for t in doc["identity"]["tasks"]}
        for scene in doc["scenes"]:
            for worker in scene["workers"]:
                for episode in worker["episodes"]:
                    self.assertEqual(stored[episode["task_id"]], episode["task_hash"])

    def test_a_catalog_without_provider_facts(self):
        doc = self.plan_json(lambda tmp, root: None)
        self.assertEqual(plan_id(doc["identity"]), doc["plan_id"])
