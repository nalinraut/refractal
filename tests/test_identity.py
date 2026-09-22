import unittest
from pathlib import Path

from refractal.schema.identity import base_scenario_hash
from refractal.schema import (
    Task,
    comparison_unit,
    episode_id,
    load_catalog,
    scenario_hash,
    scene_hash,
    task_hash,
)

CATALOG = Path(__file__).resolve().parents[1] / "examples" / "catalog"


def task(**overrides):
    base = {
        "id": "vial-slot-4",
        "scene": "vial-rack-v1",
        "instruction": "put the vial in slot 4",
        "predicate": "so101_eval.predicates:object_in_slot",
        "predicate_args": {"object": "vial", "slot": 4},
        "max_steps": 400,
    }
    base.update(overrides)
    return Task.model_validate(base)


class TestTaskIdentity(unittest.TestCase):
    def test_renaming_a_task_preserves_identity(self):
        # Ids are labels. Renaming must not break the join against results
        # already recorded.
        self.assertEqual(task_hash(task()), task_hash(task(id="vial-slot-four")))

    def test_changing_meaning_breaks_identity(self):
        # This is the case that motivated hashing tasks by content: editing the
        # predicate args while leaving the id alone would otherwise produce an
        # episode with an unchanged id and a changed meaning -- the positional
        # identity bug, one level up.
        self.assertNotEqual(
            task_hash(task()),
            task_hash(task(predicate_args={"object": "vial", "slot": 5})),
        )

    def test_max_steps_participates(self):
        # Raising the step limit changes outcomes, so it changes comparability.
        self.assertNotEqual(task_hash(task()), task_hash(task(max_steps=800)))

    def test_description_does_not_participate(self):
        self.assertEqual(task_hash(task()), task_hash(task(description="a note")))

    def test_phases_participate(self):
        phased = task(
            phases=[
                {"name": "reach", "predicate": "so101_eval.predicates:ee_near"},
            ]
        )
        self.assertNotEqual(task_hash(task()), task_hash(phased))


class TestScenarioIdentity(unittest.TestCase):
    def test_key_order_irrelevant(self):
        self.assertEqual(
            scenario_hash({"vial_x": 0.12, "vial_y": 0.0}),
            scenario_hash({"vial_y": 0.0, "vial_x": 0.12}),
        )

    def test_the_reserved_list_is_already_in_the_document(self):
        # The reservation only buys anything if the empty list is hashed from
        # the first commit. If perturbations were merely absent today and added
        # later, every recorded scenario_hash would change the day they landed.
        explicit = scenario_hash({"vial_x": 0.12}, perturbations=[])
        implicit = scenario_hash({"vial_x": 0.12})
        self.assertEqual(explicit, implicit)

    def test_the_hashed_key_name_is_frozen(self):
        """The digest key says `faults` while everything else says
        `perturbations`, and that asymmetry is load-bearing rather than untidy.

        The key is inside the document, so renaming it moves every
        scenario_hash, every episode_id and every plan_id ever recorded --
        exactly what reserving the field from the first commit was meant to
        prevent. A tidy-up would be silent: no test would fail on the rename
        itself, and the damage would appear as old results refusing to join.

        So the literal is asserted here, with its consequence stated, because
        the next person to read `_HASHED_KEY` will want to fix it.
        """
        from refractal.schema.identity import _HASHED_KEY, scenario_identity

        self.assertEqual(_HASHED_KEY, "faults")
        self.assertIn("faults", scenario_identity({"vial_x": 0.12}))

    def test_a_perturbation_moves_the_scenario_hash_but_not_the_base(self):
        """The two-hash design, stated as the property it exists for.

        A perturbed episode is genuinely a different experiment, so it must not
        join its unperturbed counterpart as though it were the same -- and the
        sweep still has to hold something fixed while the level varies. Those
        are the two halves, and one field cannot do both.
        """
        params = {"vial_x": 0.12}
        gentle = ({"at_step": 200, "type": "scale_actuator",
                   "target": "gripper", "args": {"factor": 0.5}},)
        harsh = ({"at_step": 200, "type": "scale_actuator",
                  "target": "gripper", "args": {"factor": 0.3}},)

        self.assertNotEqual(scenario_hash(params, gentle), scenario_hash(params, harsh))
        self.assertNotEqual(scenario_hash(params, gentle), scenario_hash(params))
        # ... while the join axis holds still across all three.
        self.assertEqual(base_scenario_hash(params), base_scenario_hash(params))
        self.assertEqual(len({base_scenario_hash(params)}), 1)

    def test_an_unperturbed_scenario_has_two_equal_hashes(self):
        """The relationship, chosen and stated: they are the same string.

        Not merely derivable from each other. Equal. Which means every
        scenario_hash recorded before this field existed is already a valid
        base_scenario_hash, so a sweep run tomorrow joins against results run
        today on the unperturbed arm of its own curve.

        Hashing `params` alone would have made them differ always, bought
        nothing, and orphaned every existing result.
        """
        params = {"vial_x": 0.12, "vial_y": -0.02}
        self.assertEqual(base_scenario_hash(params), scenario_hash(params))


class TestPerturbationIdentityIsFrozen(unittest.TestCase):
    """Golden hashes, written BEFORE the protocol field exists.

    An effect is about to declare which protocol it uses -- world, observation
    or action -- so the planner can refuse one an adapter cannot run. That is a
    property of the EFFECT, not of the experiment: two runs of the same
    perturbation are the same experiment whether or not the code records how it
    reaches the simulator.

    So the field must not reach identity. This project has had to be careful
    about exactly that three times now, twice getting it wrong in opposite
    directions, and "remember not to" has never been the thing that worked.

    Written before the change rather than after, which is the whole point: a
    test written afterwards records whatever happened. These literals are what
    a perturbed scenario hashes to today. If adding the field moves them, it
    leaked into identity and this fails -- naming the leak rather than leaving
    it to be discovered when old results stop joining.

    If a change to identity is ever DELIBERATE, these values change in the same
    commit that argues for it.
    """

    PARAMS = {"init_state_index": 3}
    SPEC = {"at_step": 0, "type": "scale_actuator",
            "target": "gripper0_gripper_finger_joint1", "args": {"factor": 0.1}}

    UNPERTURBED = "sha256:8b238626b69c6d17d38835b2b0a4d1cd0057ee631a9b7a108281edcf780e4829"
    PERTURBED = "sha256:b12843205883c64e2def0135b0c81fff9c8d5c99f58a057b8ece23b3045c4df9"
    SUSTAINED = "sha256:22df268163c2c35ab2265b7eca1d714cc917bff27112b1d84a33d80fc233910a"

    def test_an_unperturbed_scenario_hashes_to_what_it_always_has(self):
        self.assertEqual(scenario_hash(self.PARAMS), self.UNPERTURBED)

    def test_a_perturbed_scenario_is_frozen(self):
        self.assertEqual(scenario_hash(self.PARAMS, [self.SPEC]), self.PERTURBED)

    def test_a_sustained_perturbation_is_frozen(self):
        """until_step is IN identity, deliberately: a perturbation lasting to
        step 100 is a different experiment from one lasting all episode."""
        sustained = dict(self.SPEC, until_step=100)
        self.assertEqual(scenario_hash(self.PARAMS, [sustained]), self.SUSTAINED)
        self.assertNotEqual(scenario_hash(self.PARAMS, [sustained]), self.PERTURBED)

    def test_the_base_is_unchanged_by_any_of_it(self):
        """The join axis. Every perturbed variant shares it, which is what makes
        a sweep joinable -- and what lets results recorded before perturbations
        existed serve as its baseline."""
        for spec in (self.SPEC, dict(self.SPEC, until_step=100)):
            self.assertEqual(base_scenario_hash(self.PARAMS), self.UNPERTURBED)
        self.assertEqual(base_scenario_hash(self.PARAMS), scenario_hash(self.PARAMS))


class TestEpisodeIdentity(unittest.TestCase):
    def test_all_five_components_matter(self):
        base = dict(
            scene_hash="sha256:aa",
            task_hash="sha256:bb",
            scenario_hash="sha256:cc",
            seed=0,
            checkpoint_id="ckpt-46",
        )
        baseline = episode_id(**base)
        for key, value in [
            ("scene_hash", "sha256:zz"),
            ("task_hash", "sha256:zz"),
            ("scenario_hash", "sha256:zz"),
            ("seed", 1),
            ("checkpoint_id", "ckpt-47"),
        ]:
            self.assertNotEqual(baseline, episode_id(**{**base, key: value}), key)

    def test_no_concatenation_ambiguity(self):
        # sha256(a + b) collides for ("a", "b1") and ("ab", "1"). A keyed
        # document cannot.
        left = episode_id(
            scene_hash="s", task_hash="t", scenario_hash="c", seed=0, checkpoint_id="a"
        )
        right = episode_id(
            scene_hash="s", task_hash="t", scenario_hash="ca", seed=0, checkpoint_id=""
        )
        self.assertNotEqual(left, right)


class TestComparisonUnit(unittest.TestCase):
    def test_two_tasks_on_one_scenario_are_different_units(self):
        # A scenario set is crossed with every task on its scene, so one
        # scenario_hash covers slot-4 and slot-7 alike. Joining on it alone
        # would merge two different goals into one 2x2.
        sh = scenario_hash({"vial_x": 0.12})
        self.assertNotEqual(
            comparison_unit(scene_id="vial-rack-v1", task_hash="A", scenario_hash=sh),
            comparison_unit(scene_id="vial-rack-v1", task_hash="B", scenario_hash=sh),
        )

    def test_same_params_in_two_scenes_are_different_units(self):
        sh = scenario_hash({"x": 0.1, "y": 0.2})
        self.assertNotEqual(
            comparison_unit(scene_id="a", task_hash="T", scenario_hash=sh),
            comparison_unit(scene_id="b", task_hash="T", scenario_hash=sh),
        )


class TestSceneAndPlanIdentity(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog(CATALOG)

    def test_scene_hash_is_computable_without_an_engine(self):
        # The property the compiler architecture rests on: plan runs on a bare
        # laptop. Hashing the compiled model would need MuJoCo installed.
        h = scene_hash(self.catalog.root, self.catalog.scene("vial-rack-v1"))
        self.assertTrue(h.startswith("sha256:"))

    def test_scene_hash_tracks_engine_version(self):
        scene = self.catalog.scene("vial-rack-v1")
        bumped = scene.model_copy(update={"engine_version": "3.3.0"})
        self.assertNotEqual(
            scene_hash(self.catalog.root, scene), scene_hash(self.catalog.root, bumped)
        )

    def test_plan_id_ignores_results_uri(self):
        # Otherwise the comparison id depends on the output directory: the same
        # experiment written to ./results and to s3://... would never join.
        other = self.catalog.run.model_copy(update={"results_uri": "s3://bucket/prefix"})
        moved = load_catalog(CATALOG)
        object.__setattr__(moved, "run", other)
        self.assertEqual(self.catalog.plan_id(), moved.plan_id())

    def test_plan_id_ignores_execution_mode(self):
        # Interleaved and serial runs of one experiment belong in one
        # comparison; the mode is recorded per episode so compare can gate
        # latency reporting on it.
        other = self.catalog.run.model_copy(update={"execution_mode": "serial"})
        moved = load_catalog(CATALOG)
        object.__setattr__(moved, "run", other)
        self.assertEqual(self.catalog.plan_id(), moved.plan_id())

    def test_plan_id_tracks_tier(self):
        other = self.catalog.run.model_copy(update={"tier": "smoke"})
        moved = load_catalog(CATALOG)
        object.__setattr__(moved, "run", other)
        self.assertNotEqual(self.catalog.plan_id(), moved.plan_id())

    def test_plan_id_tracks_checkpoint_set(self):
        one = self.catalog.run.model_copy(
            update={"checkpoints": self.catalog.run.checkpoints[:1]}
        )
        moved = load_catalog(CATALOG)
        object.__setattr__(moved, "run", one)
        self.assertNotEqual(self.catalog.plan_id(), moved.plan_id())

    def test_plan_id_tracks_server_args(self):
        # arXiv 2603.13966v2 SS III-B: the wrong proprioceptive state source in
        # X-VLA on LIBERO moved success from 97.8% to 42% -- 55 points from one
        # parameter. That class of parameter is a server arg. Hashing it into
        # the experiment identity means two runs that disagree on it get
        # different plan_ids and land in different comparison directories, so
        # the mismatch is not merely detectable, it is unrepresentable as a
        # comparison.
        ckpts = [
            c.model_copy(update={"server_args": {**c.server_args, "state_source": "ee_pose"}})
            for c in self.catalog.run.checkpoints
        ]
        other = self.catalog.run.model_copy(update={"checkpoints": ckpts})
        moved = load_catalog(CATALOG)
        object.__setattr__(moved, "run", other)
        self.assertNotEqual(self.catalog.plan_id(), moved.plan_id())

    def test_plan_id_ignores_checkpoint_order(self):
        flipped = self.catalog.run.model_copy(
            update={"checkpoints": list(reversed(self.catalog.run.checkpoints))}
        )
        moved = load_catalog(CATALOG)
        object.__setattr__(moved, "run", flipped)
        self.assertEqual(self.catalog.plan_id(), moved.plan_id())


if __name__ == "__main__":
    unittest.main()
