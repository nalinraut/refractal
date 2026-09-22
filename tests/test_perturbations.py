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
    effects,
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
        self.alternatives = {("state", "joints"): [9.0, 9.0]}
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

    def alternative(self, kind, name):
        """What the adapter can still produce that the observation no longer
        contains. Keyed by kind so one primitive serves any such need, rather
        than a new one per effect."""
        try:
            return self.alternatives[(kind, name)]
        except KeyError:
            raise KeyError(f"no {kind} named {name!r}") from None

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

    def test_every_MUTATION_reads_the_world_back(self):
        """No mutation is exempt. A wrapper's evidence is a digest pair rather
        than values off the simulator, and is checked in the obligations walk --
        one obligation, two shapes, because the declaration is a commitment
        about implementation rather than a classification."""
        from refractal.perturbations import effects, temporality_of

        world = sim()
        cases = {
            "scale_actuator": spec(at_step=0, factor=0.5),
            "displace_body": spec(type="displace_body", target="bowl", at_step=0,
                                  delta=[0.01, 0, 0]),
            "apply_force": spec(type="apply_force", target="bowl", at_step=0,
                                wrench=[0, 0, 5, 0, 0, 0]),
        }
        mutations = {n for n in effects() if temporality_of(n) == "mutation"}
        self.assertEqual(set(cases), mutations, "a mutation has no case here")
        for name, one in cases.items():
            with self.subTest(effect=name):
                world.reset()
                (fired,) = Timeline([one], world, random.Random(0)).step(0)
                self.assertTrue(fired.changed, f"{name} did not read back a change")


class TestSustainedPerturbations(unittest.TestCase):
    """`until_step` ends a perturbation by applying the effect's INVERSE.

    Not by restoring a remembered value, and that is the whole design. The
    realistic weak gripper is a servo that browns out and recovers, and a
    recovery that clobbers whatever else is acting on the same actuator is the
    last-write-wins failure multiplicative composition exists to prevent.
    """

    def test_it_reverts_at_the_step_it_names(self):
        world = sim()
        line = Timeline([dict(spec(at_step=2, factor=0.25), until_step=5)],
                        world, random.Random(0))
        for i in range(8):
            line.step(i)
        self.assertEqual(world.limits["gripper"], 20.0, "the limit came back")
        self.assertEqual([(e.fired_at, e.ends) for e in line.fired],
                         [(2, False), (5, True)])

    def test_it_is_weakened_only_inside_the_window(self):
        world = sim()
        line = Timeline([dict(spec(at_step=2, factor=0.25), until_step=5)],
                        world, random.Random(0))
        seen = []
        for i in range(8):
            line.step(i)
            seen.append(world.limits["gripper"])
        self.assertEqual(seen, [20.0, 20.0, 5.0, 5.0, 5.0, 20.0, 20.0, 20.0])

    def test_ending_one_does_not_clobber_another(self):
        """The fixture the design needs: two perturbations on one actuator,
        one ending while the other is still in force. Restoring a remembered
        value would put the limit back to 20 and lose the second entirely."""
        world = sim()
        line = Timeline([
            dict(spec(at_step=1, factor=0.5), until_step=4),
            spec(at_step=2, factor=0.6),
        ], world, random.Random(0))
        for i in range(6):
            line.step(i)
        self.assertAlmostEqual(world.limits["gripper"], 12.0,
                               msg="20 x 0.5 x 0.6 x 2.0 -- the second survives")

    def test_the_receipt_distinguishes_a_start_from_an_end(self):
        world = sim()
        line = Timeline([dict(spec(at_step=0, factor=0.25), until_step=3)],
                        world, random.Random(0))
        for i in range(5):
            line.step(i)
        start, end = line.fired
        self.assertFalse(start.ends)
        self.assertEqual((start.before, start.after), (20.0, 5.0))
        self.assertTrue(end.ends)
        self.assertEqual((end.before, end.after), (5.0, 20.0))

    def test_an_effect_with_no_inverse_is_refused(self):
        """apply_force ASSIGNS its wrench rather than adding to it, so undoing
        it would restore a previous value and clobber anything since."""
        with self.assertRaises(PerturbationError) as ctx:
            Timeline([{"at_step": 0, "until_step": 5, "type": "apply_force",
                       "target": "bowl", "args": {"wrench": [1, 0, 0, 0, 0, 0]}}],
                     sim(), random.Random(0))
        self.assertIn("no inverse", str(ctx.exception))

    def test_a_window_that_closes_before_it_opens_is_refused(self):
        with self.assertRaises(PerturbationError) as ctx:
            Timeline([dict(spec(at_step=5, factor=0.5), until_step=5)],
                     sim(), random.Random(0))
        self.assertIn("not after", str(ctx.exception))

    def test_an_episode_ending_inside_the_window_leaves_it_unfired(self):
        """Real and expected: the end never came due. It is reported, not
        treated as an error, exactly as an unfired start is."""
        world = sim()
        line = Timeline([dict(spec(at_step=1, factor=0.5), until_step=99)],
                        world, random.Random(0))
        for i in range(5):
            line.step(i)
        self.assertEqual(len(line.fired), 1)
        self.assertEqual(len(line.unfired()), 1)
        self.assertEqual(world.limits["gripper"], 10.0)


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


class TestEveryEffectMeetsItsObligations(unittest.TestCase):
    """Four promises, checked by walking the registry rather than by review.

    These are what make a declaration mean something. A protocol field with no
    obligation to read the world back is a label -- and once the registry opens
    to effects written elsewhere, a promise nothing checks is a promise nobody
    keeps.

    Each maps to a bug this project has already had.
    """

    #: One case per effect. Mutations carry a world spec; wrappers carry a spec
    #: and the value they transform, because their evidence is a digest pair
    #: rather than values read off the simulator.
    CASES = {
        "scale_actuator": (spec(at_step=0, factor=0.5), "gripper"),
        "displace_body": (spec(type="displace_body", target="bowl", at_step=0,
                               delta=[0.01, 0, 0]), "bowl"),
        "apply_force": (spec(type="apply_force", target="bowl", at_step=0,
                             wrench=[0, 0, 5, 0, 0, 0]), "bowl"),
        "drop_observation": ({"at_step": 0, "type": "drop_observation",
                              "target": "states", "args": {"fill": "zeros"}},
                             {"states": [1.0, 2.0]}),
        "substitute_observation": ({"at_step": 0, "type": "substitute_observation",
                                    "target": "states",
                                    "args": {"kind": "state", "name": "joints"}},
                                   {"states": [1.0, 2.0]}),
    }

    @staticmethod
    def _mutations():
        from refractal.perturbations import temporality_of

        return {n for n in effects() if temporality_of(n) == "mutation"}

    def test_every_effect_has_a_case_here(self):
        """So adding one to the registry and not to this file is a failure
        rather than a silent gap in the obligations."""
        self.assertEqual(set(self.CASES), set(effects()))

    def test_1_every_effect_declares_a_protocol_and_its_primitives(self):
        """The planner has nothing else. It cannot inspect an unfamiliar effect
        and work out what it calls."""
        from refractal.perturbations import PROTOCOLS, declaration

        for name in effects():
            with self.subTest(effect=name):
                declared = declaration(name)
                self.assertIsNotNone(declared, f"{name} declares nothing")
                protocol, needs = declared
                self.assertIn(protocol, PROTOCOLS)
                self.assertTrue(needs, f"{name} claims to need no primitives")

    def test_1a_an_effect_cannot_register_without_them(self):
        from refractal.perturbations import effect as register

        with self.assertRaises(ValueError):
            register("bogus", protocol="nowhere", needs=("x",))(lambda *a: (0, 0))
        with self.assertRaises(ValueError) as ctx:
            register("bogus", protocol="world", needs=())(lambda *a: (0, 0))
        self.assertIn("cannot be refused before it runs", str(ctx.exception))

    def test_2_every_effect_shows_that_it_happened(self):
        """One obligation, one purpose. The EVIDENCE differs by temporality --
        values either side of the write for a mutation, a digest pair and a
        count for a wrapper -- which is why this branches rather than asserting
        one shape on everything.

        This is the one that caught scaling an unlimited actuator: the call
        succeeded and nothing changed.
        """
        from refractal.perturbations import temporality_of

        for name, (one, subject) in self.CASES.items():
            with self.subTest(effect=name):
                line = Timeline([one], sim(), random.Random(0))
                if temporality_of(name) == "mutation":
                    (fired,) = line.step(0)
                    self.assertIsNotNone(fired.before, f"{name} read no before")
                    self.assertIsNotNone(fired.after, f"{name} read no after")
                else:
                    protocol, _ = __import__(
                        "refractal.perturbations", fromlist=["declaration"]
                    ).declaration(name)
                    line.apply(0, protocol, subject)
                    (fired,) = line.fired
                    self.assertIsNotNone(fired.before, f"{name} digested no before")
                    self.assertNotEqual(fired.before, fired.after,
                                        f"{name} passed its input through")
                    self.assertEqual(fired.applications, 1)

    def test_3_every_effect_is_pure_given_its_rng(self):
        """No global random, no clock. Same episode, same effect, same result --
        and getting this wrong destroys reproducibility with no error anywhere."""
        for name, (one, target) in self.CASES.items():
            if name not in self._mutations():
                continue     # a wrapper's output is checked below, not the world
            with self.subTest(effect=name):
                runs = []
                for _ in range(2):
                    world = sim()
                    Timeline([one], world, rng_for_episode("sha256:same")).step(0)
                    runs.append((dict(world.limits), {k: list(v) for k, v
                                                      in world.poses.items()},
                                 dict(world.wrenches)))
                self.assertEqual(runs[0], runs[1], f"{name} is not reproducible")

    def test_3b_every_wrapper_is_idempotent(self):
        """Declare wrapper and you owe idempotence across repeated application.
        It is applied on every active step, so a second application must produce
        what the first did -- otherwise a window of 200 steps is 200 different
        transforms and the digest pair describes only the first."""
        from refractal.perturbations import _EFFECTS, temporality_of

        for name, (one, subject) in self.CASES.items():
            if temporality_of(name) != "wrapper":
                continue
            with self.subTest(effect=name):
                world = sim()
                once = _EFFECTS[name](world, subject, one.get("target"),
                                      dict(one.get("args", {})), random.Random(0))
                twice = _EFFECTS[name](world, once, one.get("target"),
                                       dict(one.get("args", {})), random.Random(0))
                self.assertEqual(once, twice, f"{name} is not idempotent")

    def test_4_no_effect_mutates_its_spec(self):
        """The spec is hashed. An effect that edits it changes the episode's
        identity from inside the episode."""
        import copy

        for name, (one, _) in self.CASES.items():
            with self.subTest(effect=name):
                original = copy.deepcopy(one)
                line = Timeline([one], sim(), random.Random(0))
                if name in self._mutations():
                    line.step(0)
                else:
                    protocol, _ = __import__(
                        "refractal.perturbations", fromlist=["declaration"]
                    ).declaration(name)
                    line.apply(0, protocol, self.CASES[name][1])
                self.assertEqual(one, original, f"{name} mutated its spec")


class TestTheScheduleIsProtocolAgnostic(unittest.TestCase):
    """`active_at` answers when, for any protocol. The hooks answer what.

    Sustained is the DEFAULT for a transform and the special case for a state
    mutation, exactly inverted -- an observation is produced fresh every step,
    so a transform applied once corrupts one frame and the next arrives clean,
    while a state mutation persists because the state does.

    So the schedule answers with a window rather than an event, and the timeline
    never branches on protocol.
    """

    def test_a_spec_with_no_end_is_active_to_the_end(self):
        """This asserted "active for exactly one step" and was wrong.

        It encoded the world protocol's reading of `at_step` -- the instant a
        mutation fires, after which the change persists on its own -- into a
        schedule that answers for every protocol. The mutation path does not
        even ask this question; it walks its own pointer. So nothing noticed.

        A wrapper has nothing that persists. Under the old answer, a
        perturbation declared with no end transformed exactly one observation
        and every later one arrived clean, while the catalog documented it as
        lasting to the end of the episode.
        """
        line = Timeline([spec(at_step=3)], sim(), random.Random(0))
        self.assertEqual([i for i in range(8) if line.active_at(i)],
                         [3, 4, 5, 6, 7])

    def test_a_window_is_active_for_every_step_in_it(self):
        line = Timeline([dict(spec(at_step=2, factor=0.5), until_step=5)],
                        sim(), random.Random(0))
        self.assertEqual([i for i in range(8) if line.active_at(i)], [2, 3, 4])

    def test_the_window_excludes_its_end(self):
        """Half-open, so a perturbation ending at 5 and one starting at 5 do not
        overlap for a step."""
        line = Timeline([dict(spec(at_step=2, factor=0.5), until_step=5)],
                        sim(), random.Random(0))
        self.assertTrue(line.active_at(4))
        self.assertFalse(line.active_at(5))

    def test_it_reports_the_spec_not_an_expansion(self):
        """A window is ONE spec here, whatever the world protocol does with it
        internally. A transform hook needs the window, not two triggers it would
        have to reassemble."""
        one = dict(spec(at_step=2, factor=0.5), until_step=5)
        line = Timeline([one], sim(), random.Random(0))
        (active,) = line.active_at(3)
        self.assertEqual(active["until_step"], 5)
        self.assertEqual(active["args"]["factor"], 0.5)

    def test_overlapping_windows_are_both_reported(self):
        line = Timeline([
            dict(spec(at_step=0, factor=0.5), until_step=6),
            dict(spec(at_step=3, factor=0.6), until_step=9),
        ], sim(), random.Random(0))
        self.assertEqual([len(line.active_at(i)) for i in range(10)],
                         [1, 1, 1, 2, 2, 2, 1, 1, 1, 0])

    def test_the_world_protocol_still_acts_on_transitions(self):
        """Not on every active step. Firing a state mutation repeatedly would
        apply it once per step -- 20 x 0.5 four times over, not once."""
        world = sim()
        line = Timeline([dict(spec(at_step=1, factor=0.5), until_step=4)],
                        world, random.Random(0))
        for i in range(6):
            line.step(i)
        self.assertEqual(world.limits["gripper"], 20.0)
        self.assertEqual(len(line.fired), 2, "entering and leaving, not four")


class TestWrappersApplyOnEveryActiveStep(unittest.TestCase):
    """The thing a mutation-shaped timeline could not express.

    An observation is produced fresh each step, so a transform applied once
    corrupts one frame and the next arrives intact. Sustained is the DEFAULT
    here and the special case for a state mutation -- exactly inverted.
    """

    OBS = {"images": {"main": b"\x01\x01", "wrist": b"\x02\x02"},
           "states": [1.0, 2.0, 3.0]}

    def _line(self, **over):
        one = {"at_step": 1, "type": "drop_observation",
               "target": "images.wrist", "args": {"fill": "zeros"}}
        one.update(over)
        return Timeline([one], sim(), random.Random(0))

    def test_it_transforms_every_step_of_its_window(self):
        line = self._line(until_step=4)
        seen = [line.apply(i, "observation", self.OBS)["images"]["wrist"]
                for i in range(6)]
        blanked = [s == b"\x00\x00" for s in seen]
        self.assertEqual(blanked, [False, True, True, True, False, False])

    def test_a_wrapper_with_no_end_transforms_every_later_step(self):
        """This asserted one step, matching the schedule's old answer.

        Both were the world protocol's reading of `at_step`. A mutation fires
        once and the change stays; a wrapper transforms one frame and the next
        arrives clean -- so "no end" meant a single corrupted observation while
        the catalog promised the rest of the episode.
        """
        line = self._line()
        blanked = [line.apply(i, "observation", self.OBS)["images"]["wrist"]
                   == b"\x00\x00" for i in range(4)]
        self.assertEqual(blanked, [False, True, True, True])

    def test_one_step_is_available_and_has_to_be_asked_for(self):
        """`until_step` one past `at_step`, since the window is half-open."""
        line = self._line(until_step=2)
        blanked = [line.apply(i, "observation", self.OBS)["images"]["wrist"]
                   == b"\x00\x00" for i in range(4)]
        self.assertEqual(blanked, [False, True, False, False])

    def test_the_receipt_is_one_event_per_window(self):
        """200 steps of corruption is not 200 events."""
        line = self._line(until_step=8)
        for i in range(10):
            line.apply(i, "observation", self.OBS)
        (event,) = line.fired
        self.assertEqual(event.applications, 7, "steps 1 through 7")
        self.assertEqual(event.fired_at, 1)
        self.assertEqual(event.specified_at, 1)

    def test_the_evidence_is_a_digest_pair_from_the_first_application(self):
        from refractal.perturbations import digest_observation

        line = self._line(until_step=4)
        for i in range(5):
            line.apply(i, "observation", self.OBS)
        (event,) = line.fired
        self.assertEqual(event.before, digest_observation(self.OBS))
        self.assertNotEqual(event.before, event.after)

    def test_a_passthrough_transform_reads_as_equal_digests(self):
        """The 'applied to nothing' case. A count alone would say it worked."""
        from refractal.perturbations import _DECLARED, _EFFECTS, _TEMPORALITY

        _EFFECTS["noop"] = lambda p, obs, t, a, r: obs
        _DECLARED["noop"] = ("observation", ("transform_observation",))
        _TEMPORALITY["noop"] = "wrapper"
        try:
            line = Timeline([{"at_step": 0, "until_step": 3, "type": "noop",
                              "target": None, "args": {}}],
                            sim(), random.Random(0))
            for i in range(4):
                line.apply(i, "observation", self.OBS)
            (event,) = line.fired
            self.assertEqual(event.applications, 3)
            self.assertEqual(event.before, event.after,
                             "equal digests with a non-zero count is the "
                             "signature of a transform that did nothing")
        finally:
            for table in (_EFFECTS, _DECLARED, _TEMPORALITY):
                table.pop("noop", None)

    def test_it_does_not_mutate_what_it_was_handed(self):
        """Otherwise the 'before' digest would be a digest of the after."""
        import copy

        original = copy.deepcopy(self.OBS)
        line = self._line(until_step=3)
        line.apply(1, "observation", self.OBS)
        self.assertEqual(self.OBS, original)

    def test_a_world_wrapper_would_use_the_same_path(self):
        """Nothing here is observation-specific: `apply` filters by the
        protocol it was asked for, and the protocol is the effect's
        declaration rather than anything the timeline knows."""
        line = self._line(until_step=4)
        untouched = line.apply(2, "world", self.OBS)
        self.assertIs(untouched, self.OBS, "wrong protocol, not applied")


class TestTheCallContractDiffersByTemporality(unittest.TestCase):
    """Protocol decides where an effect hooks and what it is handed.
    Temporality decides what it returns.

    A mutation acts and reports: `(before, after)`. A wrapper transforms and
    reports: the transformed value, and nothing else -- its evidence is derived
    by the framework.

    One signature cannot be both. The transformed value is the wrapper's whole
    function, and a mutation has none to give.
    """

    def test_the_framework_digests_never_the_effect(self):
        """A wrapper returns the value; Refractal digests it. An effect
        reporting its own digest would be the subject writing its own receipt --
        the failure corrected three times already."""
        from refractal.perturbations import digest_observation

        obs = {"states": b"\x01\x02", "task": "pick"}
        self.assertEqual(digest_observation(obs), digest_observation(dict(obs)))
        self.assertNotEqual(digest_observation(obs),
                            digest_observation({"states": b"\x01\x03",
                                                "task": "pick"}))

    def test_digestible_is_defined_and_the_rest_refused(self):
        """Bounded rather than attempted. A digest of a repr would be stable for
        the wrong reasons -- two identical observations differing by a memory
        address would read as a change."""
        from refractal.perturbations import digest_observation

        with self.assertRaises(PerturbationError) as ctx:
            digest_observation({"odd": object()})
        self.assertIn("cannot digest", str(ctx.exception))

    def test_nested_structures_digest_by_content(self):
        from refractal.perturbations import digest_observation

        a = {"images": {"main": b"xy", "wrist": b"zw"}, "states": [1.0, 2.0]}
        b = {"states": [1.0, 2.0], "images": {"wrist": b"zw", "main": b"xy"}}
        self.assertEqual(digest_observation(a), digest_observation(b),
                         "key order is not content")


class TestTargetIsProtocolSpecific(unittest.TestCase):
    """Required for world, optional for observation, absent for action.

    Forcing it everywhere would make a state substitution carry a field the
    reader has to ignore, and a field a reader has to ignore is worse than no
    field. The protocol knows the rule, so the protocol declares it.
    """

    def test_the_rules_are_declared_per_protocol(self):
        from refractal.perturbations import PROTOCOL_TARGET, PROTOCOLS

        self.assertEqual(set(PROTOCOL_TARGET), set(PROTOCOLS))
        self.assertEqual(PROTOCOL_TARGET["world"], "required")
        self.assertEqual(PROTOCOL_TARGET["action"], "absent")

    def test_the_schema_no_longer_forces_one(self):
        """So an action spec does not have to invent a target to validate."""
        from refractal.schema.models import PerturbationSpec

        spec = PerturbationSpec(at_step=0, type="add_action_noise",
                                args={"sigma": 0.1})
        self.assertIsNone(spec.target)


class TestTemporalityIsOrthogonalToProtocol(unittest.TestCase):
    """Two declarations, cutting across each other.

    Protocol says WHERE an effect hooks. Temporality says HOW it relates to
    time. World effects are mostly mutations and observation effects mostly
    wrappers, but that is a tendency rather than a rule -- a world effect
    clamping a joint every step while active is a wrapper.

    So the timeline is protocol-blind and temporality-aware: it reports active
    spans, and the declared temporality decides whether that means two calls or
    many.
    """

    def test_every_effect_declares_both(self):
        from refractal.perturbations import (
            PROTOCOLS, TEMPORALITIES, declaration, temporality_of,
        )

        for name in effects():
            with self.subTest(effect=name):
                protocol, _ = declaration(name)
                self.assertIn(protocol, PROTOCOLS)
                self.assertIn(temporality_of(name), TEMPORALITIES)

    def test_an_unknown_temporality_cannot_register(self):
        from refractal.perturbations import effect as register

        with self.assertRaises(ValueError) as ctx:
            register("bogus", protocol="world", needs=("x",),
                     temporality="permanent")(lambda *a: (0, 0))
        self.assertIn("mutation", str(ctx.exception))

    def test_a_wrapper_is_not_expanded_into_two_triggers(self):
        """It has nothing to undo. Expanding one would enqueue an inverse that
        does not exist, and the world protocol's refusal would fire on an effect
        that never needed it."""
        from refractal.perturbations import _EFFECTS, _DECLARED, _TEMPORALITY

        _EFFECTS["blur"] = lambda p, t, a, r: ("d0", "d1")
        _DECLARED["blur"] = ("observation", ("transform_observation",))
        _TEMPORALITY["blur"] = "wrapper"
        try:
            world = sim()
            one = {"at_step": 1, "until_step": 4, "type": "blur",
                   "target": "gripper", "args": {"severity": 3}}
            line = Timeline([one], world, random.Random(0))
            # Active across the span, and no inverse trigger was enqueued.
            self.assertEqual([i for i in range(6) if line.active_at(i)], [1, 2, 3])
            self.assertEqual(len(line.unfired()), 1,
                             "one spec, not a start and an end")
        finally:
            for table in (_EFFECTS, _DECLARED, _TEMPORALITY):
                table.pop("blur", None)

    def test_a_wrapper_may_not_supply_an_inverse(self):
        """The declaration is a commitment, so the obligations enforce what was
        declared rather than a generic shape. A wrapper ends by not being
        applied; an inverse says it leaves something behind, which is a
        mutation."""
        from refractal.perturbations import effect as register

        with self.assertRaises(ValueError) as ctx:
            register("blurry", protocol="observation", needs=("x",),
                     temporality="wrapper",
                     inverse=lambda args: args)(lambda *a: (0, 0))
        self.assertIn("nothing to undo", str(ctx.exception))

    def test_a_mutation_without_an_inverse_is_legitimate(self):
        """`apply_force` assigns a wrench that persists, which makes it a
        mutation, and has no inverse because undoing an assignment would clobber
        anything written since.

        So "declare mutation and you owe an inverse" is one step too strong. The
        debt is owed only if the effect is ever SUSTAINED, which registration
        cannot know -- it is enforced where sustaining happens.
        """
        from refractal.perturbations import has_inverse, temporality_of

        self.assertEqual(temporality_of("apply_force"), "mutation")
        self.assertFalse(has_inverse("apply_force"))
        # Instantaneous: fine. Sustained: refused, at the point it is sustained.
        Timeline([spec(type="apply_force", target="bowl", at_step=0,
                       wrench=[0, 0, 1, 0, 0, 0])], sim(), random.Random(0))
        with self.assertRaises(PerturbationError):
            Timeline([{"at_step": 0, "until_step": 5, "type": "apply_force",
                       "target": "bowl", "args": {"wrench": [0, 0, 1, 0, 0, 0]}}],
                     sim(), random.Random(0))

    def test_a_mutation_still_expands(self):
        line = Timeline([dict(spec(at_step=1, factor=0.5), until_step=4)],
                        sim(), random.Random(0))
        self.assertEqual(len(line.unfired()), 2, "the effect and its inverse")


class TestSweepableIsDeclaredNotGuessed(unittest.TestCase):
    """Not every perturbation has a level, and that is not a failure to find one.

    Torque scale is continuous, so a sweep is natural. A state source is
    categorical -- correct, wrong, maybe a third -- and cannot be swept: an axis
    with two points is a comparison, not a curve, and `compare` already does
    matched comparisons.

    Both are legitimate. What is not legitimate is the two being
    indistinguishable, which is what guessing at argument names produced: an
    effect naming its argument anything unexpected silently became categorical,
    losing the column a curve groups by with nothing to say so.
    """

    def test_a_continuous_effect_declares_its_level_argument(self):
        from refractal.perturbations import is_sweepable, level_arg_of

        self.assertEqual(level_arg_of("scale_actuator"), "factor")
        self.assertTrue(is_sweepable("scale_actuator"))

    def test_a_categorical_effect_declares_none(self):
        from refractal.perturbations import is_sweepable, level_arg_of

        for name in ("displace_body", "apply_force"):
            with self.subTest(effect=name):
                self.assertIsNone(level_arg_of(name))
                self.assertFalse(is_sweepable(name))

    def test_the_level_comes_from_the_declaration_not_the_argument_name(self):
        """An effect whose level argument is called something unexpected must
        still be swept. Guessing could not do this."""
        from refractal.execute.vla_eval_runner import _declared_level
        from refractal.perturbations import _LEVEL_ARGS

        _LEVEL_ARGS["odd_effect"] = "severity"
        try:
            self.assertEqual(
                _declared_level([{"type": "odd_effect", "args": {"severity": 3}}]),
                3.0,
            )
        finally:
            _LEVEL_ARGS.pop("odd_effect")

    def test_a_categorical_perturbation_has_a_count_but_no_level(self):
        """The row that would otherwise be indistinguishable from a bug: one
        perturbation, no level, and that is the right answer."""
        from refractal.execute.vla_eval_runner import _declared_level

        specs = [{"type": "apply_force", "args": {"wrench": [0, 0, 1, 0, 0, 0]}}]
        self.assertEqual(len(specs), 1, "it IS perturbed")
        self.assertIsNone(_declared_level(specs), "and correctly has no level")

    def test_an_unknown_effect_yields_no_level(self):
        """Rather than raising. An unknown effect is refused by name elsewhere;
        this function's job is the level, and it should not be the thing that
        decides an effect exists."""
        from refractal.execute.vla_eval_runner import _declared_level

        self.assertIsNone(_declared_level([{"type": "nope", "args": {"factor": 1}}]))


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


class TestAWrapperCanBeIdentityOnSomeSteps(unittest.TestCase):
    """The evidence is how many applications CHANGED the value, not the first
    pair -- and the first real effect is the reason.

    Swapping a rotation convention is identity whenever the quaternion already
    lies in the hemisphere the other convention normalizes to, and differs by a
    full turn when it does not. Measured on LIBERO's reset pose, the wrist sits
    within 2e-4 of that boundary, so which side a given episode starts on is
    decided by floating-point noise.

    Judging by the first pair would therefore call a correctly applied effect a
    no-op on roughly half of episodes. `applications_changed` is what makes the
    receipt say what actually happened.
    """

    def _timeline(self, alternatives):
        from refractal.perturbations import Timeline

        fake = FakeSim()
        fake.alternatives = alternatives
        return Timeline(
            specs=[{"at_step": 0, "type": "substitute_observation",
                    "target": "states",
                    "args": {"kind": "state", "name": "other"}}],
            primitives=fake,
            rng=random.Random(0),
        )

    def test_changed_counts_only_the_applications_that_differed(self):
        """Identity on the first step and different on the second. Both counts
        must move independently."""
        same = {"states": [1.0, 2.0]}
        timeline = self._timeline({("state", "other"): [1.0, 2.0]})
        timeline.apply(0, "observation", dict(same))
        timeline.apply(1, "observation", {"states": [7.0, 7.0]})

        (event,) = [e for e in timeline.fired if e.applications]
        self.assertEqual(event.applications, 2, "applied on both steps")
        self.assertEqual(event.applications_changed, 1,
                         "changed on the second only -- the first returned "
                         "what was already there")

    def test_the_first_pair_can_be_equal_on_a_working_effect(self):
        """Which is why the first pair is an exemplar and not a verdict."""
        timeline = self._timeline({("state", "other"): [1.0, 2.0]})
        timeline.apply(0, "observation", {"states": [1.0, 2.0]})
        timeline.apply(1, "observation", {"states": [7.0, 7.0]})

        (event,) = [e for e in timeline.fired if e.applications]
        self.assertEqual(event.before, event.after,
                         "step one was identity, so the exemplar pair matches")
        self.assertTrue(event.applications_changed,
                        "and the effect still demonstrably did something")

    def test_never_changing_anything_leaves_the_count_at_zero(self):
        """The real 'applied to nothing', which `check_receipts` refuses."""
        timeline = self._timeline({("state", "other"): [1.0, 2.0]})
        for index in range(3):
            timeline.apply(index, "observation", {"states": [1.0, 2.0]})

        (event,) = [e for e in timeline.fired if e.applications]
        self.assertEqual(event.applications, 3)
        self.assertEqual(event.applications_changed, 0)


class TestTheTwoPathsDoNotRunEachOthersSpecs(unittest.TestCase):
    """One timeline, both hooks, and a spec belongs to exactly one of them.

    `step` walks a single sorted list holding every spec, because ordering has
    to be one thing. So it has to skip what is not its own: a wrapper called
    through the mutation path gets the wrong arity, and if the signatures ever
    happened to line up it would apply a second time on top of its own hook.

    An episode that perturbs the observation calls both hooks every step, so
    this is the ordinary path rather than an exotic one -- it was simply never
    tested with both a wrapper present and `step` being called.
    """

    def _line(self):
        from refractal.perturbations import Timeline

        return Timeline(
            specs=[{"at_step": 0, "type": "substitute_observation",
                    "target": "states",
                    "args": {"kind": "state", "name": "joints"}}],
            primitives=sim(),
            rng=random.Random(0),
        )

    def test_stepping_never_fires_a_wrapper(self):
        line = self._line()
        for index in range(4):
            self.assertEqual(line.step(index), [],
                             "the mutation path has nothing to do here")

    def test_and_the_wrapper_still_applies_exactly_once_per_step(self):
        """The skip must not cost the wrapper its own application."""
        line = self._line()
        for index in range(4):
            line.step(index)
            line.apply(index, "observation", {"states": [1.0, 2.0]})
        (event,) = [e for e in line.fired if e.applications]
        self.assertEqual(event.applications, 4)

    def test_a_mixed_timeline_runs_each_spec_on_its_own_path(self):
        line = __import__("refractal.perturbations", fromlist=["x"]).Timeline(
            specs=[
                spec(at_step=0, factor=0.5),
                {"at_step": 0, "type": "substitute_observation",
                 "target": "states", "args": {"kind": "state", "name": "joints"}},
            ],
            primitives=sim(),
            rng=random.Random(0),
        )
        (fired,) = line.step(0)
        self.assertEqual(fired.type, "scale_actuator",
                         "only the mutation fires through step")
        line.apply(0, "observation", {"states": [1.0, 2.0]})
        self.assertEqual(
            sorted(e.type for e in line.fired),
            ["scale_actuator", "substitute_observation"],
            "and both appear in the receipt, each from its own path",
        )
