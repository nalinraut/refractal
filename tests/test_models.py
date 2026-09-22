import unittest

from pydantic import ValidationError

from refractal.schema import ParamSpec, Run, Task
from refractal.schema.models import ScenarioSet, ScenesFile


def scenes_doc(**scene_overrides):
    scene = {
        "id": "vial-rack-v1",
        "engine": "mujoco",
        "model": "assets/vial_rack.xml",
    }
    scene.update(scene_overrides)
    return {"apiVersion": "refractal.dev/v1alpha1", "scenes": [scene]}


class TestStrictness(unittest.TestCase):
    def test_unknown_key_is_an_error(self):
        # The single highest-value validation rule: a typo in a YAML key is the
        # most common failure mode, and silently ignoring it costs an afternoon.
        with self.assertRaises(ValidationError) as ctx:
            ScenesFile.model_validate(scenes_doc(engien="mujoco"))
        self.assertIn("engien", str(ctx.exception))

    def test_wrong_api_version_rejected(self):
        doc = scenes_doc()
        doc["apiVersion"] = "refractal.dev/v1"
        with self.assertRaises(ValidationError):
            ScenesFile.model_validate(doc)

    def test_bad_id_rejected(self):
        with self.assertRaises(ValidationError):
            ScenesFile.model_validate(scenes_doc(id="Vial_Rack"))

    def test_import_string_needs_a_colon(self):
        with self.assertRaises(ValidationError) as ctx:
            Task.model_validate(
                {
                    "id": "t",
                    "scene": "s",
                    "instruction": "do it",
                    "predicate": "so101_eval.predicates.in_slot",  # dot, not colon
                }
            )
        self.assertIn("import string", str(ctx.exception))

    def test_duplicate_resource_shape_profile_rejected(self):
        shape = {
            "hardware_profile": "rtx5090",
            "envs_per_process": 1,
            "vram_per_env_mb": 0,
            "cpu_cores": 1,
            "sec_per_1k_steps": 10.0,
            "startup_sec": 3,
        }
        with self.assertRaises(ValidationError):
            ScenesFile.model_validate(scenes_doc(resource_shape=[shape, dict(shape)]))


class TestParamSpec(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(ParamSpec(range=[0.0, 1.0], steps=5).form, "range")
        self.assertEqual(ParamSpec(choices=[0.4, 0.6]).form, "choices")
        self.assertEqual(ParamSpec(value=0.0).form, "constant")
        self.assertEqual(
            ParamSpec(range=[0.0, 1.0], samples=4, distribution="uniform").form, "random"
        )

    def test_explicit_null_is_a_constant(self):
        # Detection is by `model_fields_set`, so `value: null` is a constant
        # rather than an absent field.
        self.assertEqual(ParamSpec(value=None).form, "constant")

    def test_mixing_forms_rejected(self):
        with self.assertRaises(ValidationError):
            ParamSpec(range=[0.0, 1.0], steps=5, choices=[1, 2])

    def test_normal_requires_mean_and_std(self):
        with self.assertRaises(ValidationError):
            ParamSpec(range=[0.0, 1.0], samples=4, distribution="normal")

    def test_inverted_range_rejected(self):
        with self.assertRaises(ValidationError):
            ParamSpec(range=[1.0, 0.0], steps=3)

    def test_cardinality(self):
        self.assertEqual(ParamSpec(range=[0.0, 1.0], steps=5).cardinality(), 5)
        self.assertEqual(ParamSpec(choices=[1, 2, 3]).cardinality(), 3)
        self.assertEqual(ParamSpec(value=1).cardinality(), 1)


class TestReservedFields(unittest.TestCase):
    def test_non_empty_perturbations_are_accepted_now(self):
        """They were refused from the moment the identity was settled until the
        moment a receipt could reach the row. A spec the schema accepts and the
        backend drops records as though it fired and never did, so the refusal
        lifted in the same change that closed the route -- there was no window
        where the schema accepted what nothing could run.

        What refuses now is the planner, per target: an actuator the scene does
        not have, or one with no torque limit to scale.
        """
        scenario_set = ScenarioSet.model_validate(
            {
                "id": "s",
                "scene": "sc",
                "generator": "refractal.generators:linspace_grid",
                "generator_seed": 1,
                "params": {"x": {"value": 1}},
                "perturbations": [
                    {"at_step": 200, "type": "scale_actuator", "target": "gripper"}
                ],
            }
        )
        (spec,) = scenario_set.perturbations
        self.assertEqual((spec.at_step, spec.type, spec.target),
                         (200, "scale_actuator", "gripper"))

    def test_the_old_key_names_its_replacement(self):
        """`extra="forbid"` would say "Extra inputs are not permitted", and send
        the reader hunting for a typo in a key they copied from the docs. Every
        catalog written before this carries `faults: []`."""
        with self.assertRaises(Exception) as ctx:
            ScenarioSet.model_validate(
                {
                    "id": "s",
                    "scene": "sc",
                    "generator": "refractal.generators:linspace_grid",
                    "generator_seed": 1,
                    "params": {"x": {"value": 1}},
                    "faults": [],
                }
            )
        message = str(ctx.exception)
        self.assertIn("perturbations", message)
        self.assertIn("faults", message)

    def test_empty_perturbations_accepted(self):
        ss = ScenarioSet.model_validate(
            {
                "id": "s",
                "scene": "sc",
                "generator": "refractal.generators:linspace_grid",
                "generator_seed": 1,
                "params": {"x": {"value": 1}},
                "perturbations": [],
            }
        )
        self.assertEqual(ss.perturbations, [])


class TestRun(unittest.TestCase):
    def test_seeds_are_shared_across_checkpoints(self):
        run = Run.model_validate(
            {
                "checkpoints": [
                    {"id": "a", "path": "./a", "server": "m:S"},
                    {"id": "b", "path": "./b", "server": "m:S"},
                ],
                "results_uri": "./results",
                "seeds": 3,
                "seed_base": 100,
            }
        )
        # Pairing is not an analysis choice; it is a property of the plan.
        self.assertEqual(run.seed_values(), [100, 101, 102])

    def test_duplicate_checkpoint_ids_rejected(self):
        with self.assertRaises(ValidationError):
            Run.model_validate(
                {
                    "checkpoints": [
                        {"id": "a", "path": "./a", "server": "m:S"},
                        {"id": "a", "path": "./b", "server": "m:S"},
                    ],
                    "results_uri": "./results",
                }
            )


if __name__ == "__main__":
    unittest.main()


class TestPassthroughPredicate(unittest.TestCase):
    """A decision recorded in the catalog, not a field someone forgot."""

    def test_it_defers_to_the_benchmark(self):
        from refractal.predicates import from_benchmark

        self.assertTrue(from_benchmark({"benchmark_success": True}, {}))
        self.assertFalse(from_benchmark({"benchmark_success": 0}, {}))

    def test_a_missing_verdict_raises_rather_than_guessing(self):
        from refractal.predicates import from_benchmark

        with self.assertRaises(KeyError) as ctx:
            from_benchmark({}, {})
        self.assertIn("nothing to fall back to", str(ctx.exception))

    def test_it_has_predicate_arity(self):
        """A filter takes (scenario); a predicate takes (state, args)."""
        from refractal.predicates import from_benchmark
        from refractal.schema import check_arity

        check_arity(from_benchmark, "predicate", "refractal.predicates:from_benchmark")
