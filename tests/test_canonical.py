import math
import unittest

from refractal.schema import canonical_json, hash_obj, normalize_params, round_significant
from refractal.schema.canonical import normalize_number
from refractal.schema.errors import CanonicalizationError


class TestFloatNormalisation(unittest.TestCase):
    def test_trivially_equal_literals_agree(self):
        # The case the spec calls out. Free in Python -- both parse to one double.
        self.assertEqual(canonical_json({"x": 0.1}), canonical_json({"x": 0.10}))

    def test_arithmetic_noise_collapses(self):
        # The case that actually bites: a grid point computed by accumulation
        # versus one computed by interpolation. Without quantization these are
        # two different scenarios that a human would swear are one.
        accumulated = 0.08
        for _ in range(2):
            accumulated += 0.02
        interpolated = 0.08 + 2 * (0.16 - 0.08) / 4
        self.assertNotEqual(accumulated, interpolated)  # genuinely different doubles
        self.assertEqual(
            hash_obj(normalize_params({"vial_x": accumulated})),
            hash_obj(normalize_params({"vial_x": interpolated})),
        )

    def test_integers_survive_normalisation(self):
        """Integers are already canonical, so they pass through unchanged.

        This reverses an earlier decision. The old rule coerced every scenario
        number to float on the grounds that scenario parameters are physical
        quantities, so `0` and `0.0` are the same thing. True of a vial position,
        false of an array index -- and it made `plan.json` print
        `init_state_index: 3.0` for an index into a fixed list of init states.

        `plan.json` exists to be read and checked, and "looks wrong but works" is
        the category this project keeps finding real bugs in.
        """
        self.assertIsInstance(normalize_params({"i": 3})["i"], int)
        self.assertIsInstance(normalize_params({"x": 3.0})["x"], float)
        # The accepted cost: these are now different scenarios. Visible in the
        # YAML diff and in catalog_hash, which the silent coercion was not.
        self.assertNotEqual(
            hash_obj(normalize_params({"friction": 1})),
            hash_obj(normalize_params({"friction": 1.0})),
        )

    def test_negative_zero_folds(self):
        self.assertEqual(
            hash_obj(normalize_params({"y": -0.0})), hash_obj(normalize_params({"y": 0.0}))
        )

    def test_int_and_float_stay_distinct_in_authored_data(self):
        # predicate_args are authored, not computed: {slot: 4} and {slot: 4.0}
        # read differently to a human and are not normalized together.
        self.assertNotEqual(canonical_json({"slot": 4}), canonical_json({"slot": 4.0}))

    def test_real_distinctions_survive_quantization(self):
        self.assertNotEqual(round_significant(0.120000001), round_significant(0.12))

    def test_nan_and_inf_rejected(self):
        for bad in (float("nan"), math.inf, -math.inf):
            with self.assertRaises(CanonicalizationError):
                canonical_json({"x": bad})


class TestCanonicalForm(unittest.TestCase):
    def test_key_order_irrelevant(self):
        self.assertEqual(canonical_json({"a": 1, "b": 2}), canonical_json({"b": 2, "a": 1}))

    def test_list_order_relevant(self):
        self.assertNotEqual(canonical_json([1, 2]), canonical_json([2, 1]))

    def test_nested_keys_sorted(self):
        self.assertEqual(
            canonical_json({"o": {"z": 1, "a": 2}}), canonical_json({"o": {"a": 2, "z": 1}})
        )

    def test_bool_is_not_a_number(self):
        # bool is an int subclass, so the numeric paths have to reject it
        # explicitly or `True` would normalize to `1.0`.
        self.assertNotEqual(canonical_json({"x": True}), canonical_json({"x": 1}))
        self.assertIs(normalize_params({"x": True})["x"], True)
        with self.assertRaises(CanonicalizationError):
            normalize_number(True)

    def test_unsupported_type_rejected(self):
        with self.assertRaises(CanonicalizationError):
            canonical_json({"x": object()})

    def test_hash_is_prefixed(self):
        self.assertTrue(hash_obj({"a": 1}).startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()


class TestFileSetIsSetShaped(unittest.TestCase):
    """Here deduplication is correct, and the contrast is the point.

    The question is *which files define this scene*, which is set-valued. A scene
    whose `assets` glob happens to match its own `model` is the same scene as one
    whose glob does not, and it used to hash differently -- same geometry, two
    identities, decided by how a glob was written.

    The rule is not "always preserve duplicates". It is that the container has to
    match the question.
    """

    def test_listing_a_file_twice_does_not_change_the_hash(self):
        import tempfile
        from pathlib import Path

        from refractal.schema.canonical import hash_file_set

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene.xml").write_text("model", encoding="utf-8")
            (root / "mesh.stl").write_text("mesh", encoding="utf-8")
            once = hash_file_set(root, [root / "scene.xml", root / "mesh.stl"])
            # `model:` plus an `assets:` glob that also matches it.
            twice = hash_file_set(
                root, [root / "scene.xml", root / "scene.xml", root / "mesh.stl"]
            )
            self.assertEqual(once, twice)

    def test_but_a_different_file_still_changes_it(self):
        import tempfile
        from pathlib import Path

        from refractal.schema.canonical import hash_file_set

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.xml").write_text("a", encoding="utf-8")
            (root / "b.stl").write_text("b", encoding="utf-8")
            self.assertNotEqual(
                hash_file_set(root, [root / "a.xml"]),
                hash_file_set(root, [root / "a.xml", root / "b.stl"]),
            )
