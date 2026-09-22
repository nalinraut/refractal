"""The physics surface: robosuite's layer, made visible.

`harness_surface` exists because the harness can change behaviour without
changing anything Refractal hashes. robosuite is the same kind of layer one
level down -- it owns the robot model, and a release can edit a gripper's
forcerange without touching a BDDL file, a suite name or an engine version.

Tested with dictionaries. Nothing here imports MuJoCo or robosuite, which is
what makes the module usable for any engine that can describe its actuators.
"""

import unittest

from refractal.execute.physics import (
    ABSENT,
    actuator_facts,
    compare_facts,
    physics_digest,
    physics_manifest,
)


class Model:
    """The five attributes the reader touches, and nothing else."""

    def __init__(self, names, limited, ranges):
        self._names, self.actuator_forcelimited, self.actuator_forcerange = (
            names, limited, ranges)
        self.nu = len(names)

    def actuator_id2name(self, i):
        return self._names[i]


def libero():
    """As measured: two limited gripper actuators, seven unlimited arm ones."""
    names = [f"robot0_torq_j{i}" for i in range(1, 8)] + [
        "gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"]
    return Model(names, [0] * 7 + [1, 1],
                 [[0.0, 0.0]] * 7 + [[-20.0, 20.0], [-20.0, 20.0]])


class TestReadingTheModel(unittest.TestCase):
    def test_it_reports_every_actuator(self):
        facts = actuator_facts(libero())
        self.assertEqual(len(facts), 9)
        self.assertTrue(facts["gripper0_gripper_finger_joint1"]["force_limited"])
        self.assertFalse(facts["robot0_torq_j2"]["force_limited"])

    def test_the_flag_is_read_not_inferred(self):
        """`forcelimited = True` with range [0, 0] is legal. Inferring the limit
        from the range calls it unlimited, which is a second implementation of
        MuJoCo's rule -- and this fixture is the case that tells them apart."""
        model = Model(["odd"], [1], [[0.0, 0.0]])
        self.assertTrue(actuator_facts(model)["odd"]["force_limited"])

    def test_a_model_with_no_actuators_is_empty_not_an_error(self):
        self.assertEqual(actuator_facts(Model([], [], [])), {})


class TestTheDigestIsOverValues(unittest.TestCase):
    """The distinction the whole design rests on: a version bump that changed
    nothing must not block a comparison, and a forcerange edited in a patch
    release must."""

    def test_the_same_model_digests_the_same(self):
        self.assertEqual(physics_digest(actuator_facts(libero())),
                         physics_digest(actuator_facts(libero())))

    def test_a_changed_forcerange_moves_the_digest(self):
        before = physics_digest(actuator_facts(libero()))
        weaker = libero()
        weaker.actuator_forcerange[-1] = [-12.0, 12.0]
        self.assertNotEqual(before, physics_digest(actuator_facts(weaker)))

    def test_a_changed_limit_flag_moves_the_digest(self):
        """The arm becoming limited is a different machine, even at the same
        numbers -- unlimited and limited-to-zero are not the same actuator."""
        before = physics_digest(actuator_facts(libero()))
        after = libero()
        after.actuator_forcelimited[0] = 1
        self.assertNotEqual(before, physics_digest(actuator_facts(after)))

    def test_nothing_read_is_absent_not_a_hash_of_nothing(self):
        """A digest of an empty model would collide with 'nobody looked' and
        claim a fact that was never established."""
        self.assertEqual(physics_digest({}), ABSENT)


class TestTheManifestNamesWhatMoved(unittest.TestCase):
    def test_it_lists_every_actuator(self):
        manifest = physics_manifest(actuator_facts(libero()))
        self.assertEqual(len(manifest), 9)
        self.assertIn("limited", manifest["gripper0_gripper_finger_joint1"])
        self.assertIn("unlimited", manifest["robot0_torq_j2"])

    def test_a_diff_names_the_actuator_and_both_values(self):
        """The digest says something moved; months later nobody has both engines
        installed to find out what."""
        before = physics_manifest(actuator_facts(libero()))
        weaker = libero()
        weaker.actuator_forcerange[-1] = [-12.0, 12.0]
        diff = compare_facts(before, physics_manifest(actuator_facts(weaker)))
        self.assertEqual(len(diff["changed"]), 1)
        self.assertIn("gripper0_gripper_finger_joint2", diff["changed"][0])
        self.assertIn("20", diff["changed"][0])
        self.assertIn("12", diff["changed"][0])

    def test_added_and_removed_actuators_are_named(self):
        before = physics_manifest(actuator_facts(libero()))
        fewer = physics_manifest(actuator_facts(
            Model(["gripper0_gripper_finger_joint1"], [1], [[-20.0, 20.0]])))
        diff = compare_facts(before, fewer)
        self.assertEqual(diff["added"], [])
        self.assertIn("robot0_torq_j1", diff["removed"])


if __name__ == "__main__":
    unittest.main()
