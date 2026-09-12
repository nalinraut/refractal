import unittest

from refractal.generators import latin_hypercube, linspace_grid, random_sample
from refractal.schema import ParamSpec
from refractal.schema.errors import GeneratorError


GRID = {
    "vial_x": ParamSpec(range=[0.08, 0.16], steps=5),
    "vial_y": ParamSpec(range=[-0.08, 0.08], steps=8),
    "friction": ParamSpec(choices=[0.4, 0.6, 0.8]),
}

SAMPLED = {
    "cube_x": ParamSpec(range=[0.05, 0.20], samples=24, distribution="uniform"),
    "cube_y": ParamSpec(range=[-0.10, 0.10], samples=24, distribution="uniform"),
    "bowl_yaw": ParamSpec(value=0.0),
}


class TestLinspaceGrid(unittest.TestCase):
    def test_full_cross_product(self):
        rows = linspace_grid(GRID, seed=12345)
        self.assertEqual(len(rows), 5 * 8 * 3)

    def test_endpoints_exact(self):
        rows = linspace_grid({"x": ParamSpec(range=[0.08, 0.16], steps=5)}, seed=0)
        self.assertEqual(rows[0]["x"], 0.08)
        self.assertEqual(rows[-1]["x"], 0.16)

    def test_single_step_yields_min(self):
        rows = linspace_grid({"x": ParamSpec(range=[0.08, 0.16], steps=1)}, seed=0)
        self.assertEqual(rows, [{"x": 0.08}])

    def test_deterministic(self):
        self.assertEqual(linspace_grid(GRID, seed=12345), linspace_grid(GRID, seed=12345))

    def test_output_is_normalized(self):
        # Every value is a quantized float, so identity cannot fork on a last-bit
        # difference between this machine and the next.
        for row in linspace_grid(GRID, seed=0):
            for v in row.values():
                self.assertIsInstance(v, float)

    def test_size_guard(self):
        huge = {f"p{i}": ParamSpec(range=[0.0, 1.0], steps=100) for i in range(4)}
        with self.assertRaises(GeneratorError):
            linspace_grid(huge, seed=0)


class TestSamplingGenerators(unittest.TestCase):
    def test_random_sample_count_and_determinism(self):
        rows = random_sample(SAMPLED, seed=999)
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows, random_sample(SAMPLED, seed=999))

    def test_seed_changes_output(self):
        self.assertNotEqual(random_sample(SAMPLED, seed=1), random_sample(SAMPLED, seed=2))

    def test_constants_stay_constant(self):
        for row in latin_hypercube(SAMPLED, seed=999):
            self.assertEqual(row["bowl_yaw"], 0.0)

    def test_latin_hypercube_covers_every_stratum(self):
        # The reason to prefer it at low sample counts: 24 independent draws
        # routinely leave a quarter of an axis untouched.
        rows = latin_hypercube(SAMPLED, seed=999)
        lo, hi = 0.05, 0.20
        occupied = {int((row["cube_x"] - lo) / (hi - lo) * 24) for row in rows}
        self.assertEqual(len(occupied), 24)

    def test_values_within_range(self):
        for row in latin_hypercube(SAMPLED, seed=7):
            self.assertGreaterEqual(row["cube_x"], 0.05)
            self.assertLessEqual(row["cube_x"], 0.20)

    def test_disagreeing_sample_counts_rejected(self):
        params = {
            "a": ParamSpec(range=[0.0, 1.0], samples=10, distribution="uniform"),
            "b": ParamSpec(range=[0.0, 1.0], samples=12, distribution="uniform"),
        }
        with self.assertRaises(GeneratorError):
            random_sample(params, seed=0)

    def test_no_sample_count_rejected(self):
        with self.assertRaises(GeneratorError):
            random_sample({"a": ParamSpec(value=1)}, seed=0)

    def test_normal_is_clamped_to_range(self):
        params = {
            "a": ParamSpec(
                range=[-0.01, 0.01], samples=200, distribution="normal", mean=0.0, std=1.0
            )
        }
        for row in random_sample(params, seed=3):
            self.assertGreaterEqual(row["a"], -0.01)
            self.assertLessEqual(row["a"], 0.01)




class TestNumericTypePreservation(unittest.TestCase):
    """The authored type decides, and it is uniform across the parameter.

    Not a new ParamSpec form the author picks: if the author chose, two catalogs
    expressing the same grid differently would hash differently. The type follows
    from the bounds, so there is no decision to get wrong.
    """

    def _values(self, **kw):
        return [row["p"] for row in linspace_grid({"p": ParamSpec(**kw)}, seed=0)]

    def test_integer_bounds_that_divide_evenly_stay_integers(self):
        # The LIBERO case: init-state indices, not quantities.
        values = self._values(range=[0, 49], steps=50)
        self.assertTrue(all(isinstance(v, int) for v in values))
        self.assertEqual(values[:4], [0, 1, 2, 3])
        self.assertEqual(values[-1], 49)

    def test_integer_bounds_with_a_coarser_step_still_stay_integers(self):
        self.assertEqual(self._values(range=[0, 100], steps=5), [0, 25, 50, 75, 100])

    def test_integer_bounds_that_do_not_divide_become_float_throughout(self):
        """Uniform across the parameter, never per-value.

        [0, 1] over 3 steps yields 0.5, so the whole axis is float. Mixing 0 and
        0.5 would make the canonical form depend on which grid point you landed
        on.
        """
        values = self._values(range=[0, 1], steps=3)
        self.assertTrue(all(isinstance(v, float) for v in values))
        self.assertEqual(values, [0.0, 0.5, 1.0])

    def test_float_bounds_stay_float_even_at_integral_values(self):
        values = self._values(range=[0.0, 2.0], steps=3)
        self.assertTrue(all(isinstance(v, float) for v in values))

    def test_the_divisibility_test_is_structural_not_a_tolerance(self):
        """Never ask whether 2.9999999999999996 is close enough to an integer.

        A tolerance would eventually call a genuinely fractional grid integral,
        and identity must not depend on a judgement call.
        """
        values = self._values(range=[0, 10], steps=4)
        self.assertTrue(all(isinstance(v, float) for v in values))
        self.assertAlmostEqual(values[1], 10 / 3)

    def test_choices_preserve_ints_and_widen_when_mixed(self):
        self.assertTrue(all(isinstance(v, int) for v in self._values(choices=[1, 2, 3])))
        widened = self._values(choices=[1, 2.5])
        self.assertTrue(all(isinstance(v, float) for v in widened))
        self.assertEqual(widened, [1.0, 2.5])

    def test_constants_preserve_their_type(self):
        self.assertIsInstance(self._values(value=0)[0], int)
        self.assertIsInstance(self._values(value=0.0)[0], float)

    def test_non_numeric_choices_are_untouched(self):
        self.assertEqual(sorted(self._values(choices=["red", "blue"])), ["blue", "red"])

    def test_sampled_axes_are_float(self):
        rows = random_sample(
            {"p": ParamSpec(range=[0, 10], samples=5, distribution="uniform")}, seed=1
        )
        self.assertTrue(all(isinstance(r["p"], float) for r in rows))

    def test_an_index_axis_survives_the_whole_pipeline(self):
        """What this change exists for: plan.json must not say 3.0 for an index."""
        import json

        rows = linspace_grid({"init_state_index": ParamSpec(range=[0, 9], steps=10)}, seed=0)
        rendered = json.dumps(rows[3])
        self.assertIn('"init_state_index": 3', rendered)
        self.assertNotIn("3.0", rendered)

if __name__ == "__main__":
    unittest.main()
