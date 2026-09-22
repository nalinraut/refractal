"""The perturbation module, with no simulator anywhere.

If this file ever needs MuJoCo, the boundary has moved to the wrong place. The
fake below is not a mock that asserts calls -- it holds state, so the tests can
check what the world became rather than which functions ran. That distinction is
the whole reason the module reads the simulator twice.
"""

import random
import subprocess
import sys
import unittest
from pathlib import Path

from refractal.perturbations import (
    Fired,
    PerturbationError,
    Primitives,
    Timeline,
    rng_for_episode,
)


class FakeSim:
    """A dictionary shaped like the parts of a simulator the protocol touches.

    `reset` replaces its state wholesale and invalidates every handle, because
    that is what LIBERO's `env.reset()` was measured doing -- new sim, new model,
    new data, every write restored. A fake that quietly kept working across a
    reset would let the bug this models sail through.
    """

    def __init__(self, limits=None, poses=None):
        self._initial = (dict(limits or {}), dict(poses or {}))
        self.calls = []
        self.generation = 0
        self.reset()

    def reset(self):
        limits, poses = self._initial
        self.limits = dict(limits)
        self.poses = {k: list(v) for k, v in poses.items()}
        self.wrenches = {}
        self.generation += 1

    # -- the protocol ----------------------------------------------------
    def resolve(self, name):
        if name not in self.limits and name not in self.poses:
            raise KeyError(f"no such target: {name}")
        self.calls.append(("resolve", name))
        # The handle is only valid for this generation. Using a stale one is a
        # KeyError rather than a wrong answer, which is the kinder failure.
        return (self.generation, name)

    def get_actuator_limit(self, name):
        return self.limits[name]

    def scale_actuator(self, name, factor):
        self.calls.append(("scale_actuator", name, factor))
        current = self.limits[name]
        if current is None:
            return  # unlimited stays unlimited: the behaviour being modelled
        self.limits[name] = current * factor

    def get_body_pose(self, name):
        return list(self.poses[name])

    def set_body_pose(self, name, pose):
        self.calls.append(("set_body_pose", name, list(pose)))
        self.poses[name] = list(pose)

    def apply_force(self, name, wrench):
        self.calls.append(("apply_force", name, list(wrench)))
        # Persists, as MuJoCo's xfrc_applied does -- verified on LIBERO: still
        # set after a step, with qvel moving because of it.
        self.wrenches[name] = list(wrench)

    def get_applied_wrench(self, name):
        return list(self.wrenches.get(name, [0.0] * 6))


def sim():
    return FakeSim(
        limits={"gripper": 20.0, "arm-j2": None},
        poses={"bowl": [0.1, 0.2, 0.3]},
    )


def spec(type="scale_actuator", target="gripper", at_step=10, **args):
    return {"at_step": at_step, "type": type, "target": target, "args": args}


class TestTheFakeIsEnoughOfASimulator(unittest.TestCase):
    def test_it_satisfies_the_protocol(self):
        self.assertIsInstance(sim(), Primitives)


class TestTheTimelineFires(unittest.TestCase):
    def test_a_trigger_fires_once_at_its_step(self):
        world = sim()
        line = Timeline([spec(at_step=3, factor=0.5)], world, random.Random(0))
        fired = [line.step(i) for i in range(6)]
        self.assertEqual([len(f) for f in fired], [0, 0, 0, 1, 0, 0])
        self.assertEqual(world.limits["gripper"], 10.0)

    def test_a_trigger_past_the_end_does_not_fire_and_is_not_an_error(self):
        """An episode that succeeded at step 150 never reached step 200. The
        spec is unfired, the episode is valid, and only a sweep where nothing
        fired anywhere is wrong."""
        world = sim()
        line = Timeline([spec(at_step=200, factor=0.5)], world, random.Random(0))
        for i in range(150):
            line.step(i)
        self.assertEqual(line.fired, [])
        self.assertEqual(len(line.unfired()), 1)
        self.assertEqual(world.limits["gripper"], 20.0)

    def test_it_records_what_was_asked_and_what_happened(self):
        """A loop that does not visit every step fires late, correctly. Keeping
        only one of the two numbers turns that into a mystery."""
        world = sim()
        line = Timeline([spec(at_step=5, factor=0.5)], world, random.Random(0))
        fired = line.step(9)          # first visit is already past the trigger
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].specified_at, 5)
        self.assertEqual(fired[0].fired_at, 9)

    def test_an_unknown_effect_is_refused_before_the_episode_runs(self):
        with self.assertRaises(PerturbationError) as ctx:
            Timeline([spec(type="teleport")], sim(), random.Random(0))
        self.assertIn("teleport", str(ctx.exception))

    def test_an_unknown_target_is_refused_before_the_episode_runs(self):
        """Resolving at construction turns step 200 of 400 into step zero."""
        with self.assertRaises(KeyError):
            Timeline([spec(target="elbow")], sim(), random.Random(0))


class TestTheReceiptRecordsEffectNotIntent(unittest.TestCase):
    """The finding that changed this design.

    Every arm actuator on LIBERO's Panda declares `forcerange = [0, 0]`, which
    MuJoCo reads as unlimited. Scaling unlimited by 0.3 is unlimited. A receipt
    that logged the call would report a perturbation that did nothing, and a
    whole sweep could come back looking like clean unperturbed runs.
    """

    def test_scaling_an_unlimited_actuator_is_refused_not_recorded(self):
        world = sim()
        line = Timeline([spec(target="arm-j2", factor=0.3)], world, random.Random(0))
        with self.assertRaises(PerturbationError) as ctx:
            line.step(10)
        message = str(ctx.exception)
        self.assertIn("arm-j2", message)
        self.assertIn("unlimited", message)

    def test_the_log_carries_the_value_either_side(self):
        world = sim()
        line = Timeline([spec(at_step=0, factor=0.25)], world, random.Random(0))
        (fired,) = line.step(0)
        self.assertEqual(fired.before, 20.0)
        self.assertEqual(fired.after, 5.0)
        self.assertTrue(fired.changed)

    def test_the_after_value_is_read_back_not_predicted(self):
        """A backend whose scale_actuator silently does nothing must produce a
        log showing it. Computing `after` as before*factor would report the
        change that was intended and miss the one that happened."""
        world = sim()

        def deaf(name, factor):
            world.calls.append(("scale_actuator", name, factor))  # and no more

        world.scale_actuator = deaf
        line = Timeline([spec(at_step=0, factor=0.25)], world, random.Random(0))
        (fired,) = line.step(0)
        self.assertEqual(fired.before, 20.0)
        self.assertEqual(fired.after, 20.0)
        self.assertFalse(fired.changed, "a silent no-op must read as unchanged")

    def test_a_wrench_is_verified_against_the_field_it_lands_in(self):
        """The pose was the wrong observable, not an impossible one. MuJoCo puts
        an applied wrench in data.xfrc_applied, readable at the write with no
        physics step -- measured on LIBERO. So this effect is guarded like the
        others and there is no exemption to remember."""
        world = sim()
        line = Timeline(
            [spec(type="apply_force", target="bowl", at_step=0,
                  wrench=[0, 0, 5, 0, 0, 0])],
            world, random.Random(0),
        )
        (fired,) = line.step(0)
        self.assertEqual(list(fired.before), [0.0] * 6)
        self.assertEqual(list(fired.after), [0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
        self.assertTrue(fired.changed)

    def test_every_effect_reads_the_world_back(self):
        """No effect is exempt. If one ever cannot be read back, this fails and
        the exemption has to be argued rather than assumed."""
        from refractal.perturbations import effects

        world = sim()
        cases = {
            "scale_actuator": spec(at_step=0, factor=0.5),
            "displace_body": spec(type="displace_body", target="bowl", at_step=0,
                                  delta=[0.01, 0, 0]),
            "apply_force": spec(type="apply_force", target="bowl", at_step=0,
                                wrench=[0, 0, 5, 0, 0, 0]),
        }
        self.assertEqual(set(cases), set(effects()), "an effect has no case here")
        for name, one in cases.items():
            with self.subTest(effect=name):
                world.reset()
                (fired,) = Timeline([one], world, random.Random(0)).step(0)
                self.assertTrue(fired.changed, f"{name} did not read back a change")


class TestComposition(unittest.TestCase):
    """Two perturbations on one actuator, which is the fixture the invariant
    needs -- one perturbation cannot show multiplicative composition, and a
    test with one would pass against an implementation where the last write
    wins."""

    def test_two_scalings_compose_multiplicatively(self):
        world = sim()
        line = Timeline(
            [spec(at_step=1, factor=0.5), spec(at_step=2, factor=0.6)],
            world, random.Random(0),
        )
        line.step(1)
        self.assertEqual(world.limits["gripper"], 10.0)
        line.step(2)
        self.assertAlmostEqual(world.limits["gripper"], 6.0)   # 20 * 0.5 * 0.6

    def test_each_step_logs_the_resulting_value(self):
        """Never let the last one silently win: both values are in the log, so
        the composition is visible in the artifact and not only in the code."""
        world = sim()
        line = Timeline(
            [spec(at_step=1, factor=0.5), spec(at_step=2, factor=0.6)],
            world, random.Random(0),
        )
        line.step(1)
        line.step(2)
        self.assertEqual([(f.before, f.after) for f in line.fired],
                         [(20.0, 10.0), (10.0, 6.0)])


class TestResolveRunsPerEpisode(unittest.TestCase):
    """`env.reset()` replaces the sim, model and data objects and restores every
    write -- measured on LIBERO by object identity. A handle cached in episode
    one addresses an orphan in episode two, and the write lands where nothing
    reads it.

    A single-episode test cannot see this. That is the point of the fixture.
    """

    def test_the_perturbation_fires_in_the_second_episode_too(self):
        world = sim()

        for episode in (1, 2):
            world.reset()
            line = Timeline([spec(at_step=0, factor=0.5)], world, random.Random(0))
            line.step(0)
            self.assertEqual(
                world.limits["gripper"], 10.0,
                f"episode {episode} did not perturb: reset restored the limit "
                "and nothing re-applied it",
            )

    def test_resolve_is_called_again_for_the_second_episode(self):
        world = sim()
        resolves = lambda: [c for c in world.calls if c[0] == "resolve"]

        world.reset()
        Timeline([spec()], world, random.Random(0))
        first = len(resolves())

        world.reset()
        Timeline([spec()], world, random.Random(0))
        self.assertEqual(len(resolves()), first * 2,
                         "resolve must run per episode, never cached across one")

    def test_a_handle_from_the_previous_episode_is_stale(self):
        """The fake models the invalidation, so a test that wrongly reused one
        would fail here rather than passing quietly."""
        world = sim()
        stale = world.resolve("gripper")
        world.reset()
        self.assertNotEqual(stale, world.resolve("gripper"))


class TestTheSeedIsItsOwn(unittest.TestCase):
    def test_same_episode_id_gives_the_same_stream(self):
        a = rng_for_episode("sha256:abc")
        b = rng_for_episode("sha256:abc")
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])

    def test_different_episode_ids_give_different_streams(self):
        a = rng_for_episode("sha256:abc")
        b = rng_for_episode("sha256:abd")
        self.assertNotEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])

    def test_it_reproduces_across_processes(self):
        """The failure this guards is silent: a stream seeded off `hash()` is
        salted per process, so two runs of one episode inject differently, the
        run completes, and nothing reports why the numbers moved. Two
        interpreters, because one cannot see it.
        """
        snippet = (
            "from refractal.perturbations import rng_for_episode;"
            "r = rng_for_episode('sha256:abc');"
            "print([round(r.random(), 12) for _ in range(5)])"
        )
        root = Path(__file__).resolve().parents[1]
        outs = [
            subprocess.run(
                [sys.executable, "-c", snippet],
                capture_output=True, text=True, check=True, cwd=root,
                env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin",
                     "PYTHONHASHSEED": seed},
            ).stdout.strip()
            for seed in ("0", "1")
        ]
        self.assertEqual(outs[0], outs[1])
        self.assertTrue(outs[0].startswith("["))


class TestTheModuleNeedsNoSimulator(unittest.TestCase):
    def test_it_imports_nothing_heavy(self):
        """The property that makes everything above testable on a laptop."""
        import ast

        source = Path(__file__).resolve().parents[1] / "src/refractal/perturbations/__init__.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        for heavy in ("mujoco", "numpy", "torch", "robosuite", "libero", "pyarrow"):
            self.assertNotIn(heavy, imported)


if __name__ == "__main__":
    unittest.main()
