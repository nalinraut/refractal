"""The MuJoCo version the harness ran is recorded per row and must be the one
`build` locked (audit finding 9: a catalog declared 3.2.0 while 3.2.3 ran)."""

import unittest

from tests.test_build import HARDWARE, Temp
from tests.test_vla_eval_loop import FakeHarness, LoopCase, make_plan

from refractal.build import build
from refractal.execute.vla_eval import BridgeError
from refractal.resolve import resolve


class Probe:
    def version(self, engine):
        return "3.2.3"

    def verify(self, engine, model_path):
        pass


class TestThePlanCarriesTheLockedVersion(unittest.TestCase):
    def test_resolve_copies_the_lock_engine_version_onto_each_scene(self):
        with Temp() as root:
            build(root, probe=Probe())
            plan = resolve(root, hardware_profile=HARDWARE)
        self.assertEqual({s.engine_version for s in plan.scenes}, {"3.2.3"})
        self.assertEqual(plan.plan_schema, 5)


class TestTheRunChecksIt(LoopCase):
    def test_a_harness_on_another_mujoco_is_refused_and_writes_nothing(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",), engine_version="3.2.3")
        with self.assertRaises(BridgeError) as ctx:
            self.run_loop(plan, FakeHarness(mujoco_version="3.2.0"), only=("pi0",))
        self.assertIn("built against mujoco 3.2.3", str(ctx.exception))
        self.assertEqual(self.rows(plan), [])

    def test_the_locked_version_is_recorded_on_every_row(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",), engine_version="3.2.3")
        self.run_loop(plan, FakeHarness(mujoco_version="3.2.3"), only=("pi0",))
        self.assertEqual({r["engine_version"] for r in self.rows(plan)}, {"3.2.3"})
