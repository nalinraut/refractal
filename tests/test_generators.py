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


if __name__ == "__main__":
    unittest.main()
