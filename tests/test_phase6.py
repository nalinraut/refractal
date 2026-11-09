"""Phase 6: refusals that name the mismatch, and none with an override."""

import pathlib
import unittest

from refractal.compare.analysis import Family
from refractal.compare.verdict import (
    apparatus_rates_differ,
    ghost_penalties_differ,
    identical_arm_pairs,
    termination_maps_differ,
)


class TestApparatusRates(unittest.TestCase):
    """Dropping apparatus rows is safe only when the loss is arm-independent."""

    def test_similar_rates_pass(self):
        self.assertIsNone(apparatus_rates_differ({"A": (1, 100), "B": (3, 100)}))

    def test_an_arm_dependent_loss_is_refused(self):
        said = apparatus_rates_differ({"A": (1, 100), "B": (20, 100)})
        self.assertIsNotNone(said)
        self.assertIn("20.0%", said)
        self.assertIn("1.0%", said)
        self.assertIn("different subsets", said)

    def test_it_says_why_the_difference_is_not_noise(self):
        said = apparatus_rates_differ({"A": (0, 50), "B": (25, 50)})
        self.assertIn("property of that arm", said)

    def test_one_arm_cannot_differ_from_itself(self):
        self.assertIsNone(apparatus_rates_differ({"A": (50, 50)}))


class TestGhostPenalties(unittest.TestCase):
    def test_one_convention_passes(self):
        self.assertIsNone(ghost_penalties_differ({"A": 300.0, "B": 300.0}))

    def test_two_conventions_are_refused(self):
        said = ghost_penalties_differ({"A": 300.0, "B": 250.0})
        self.assertIn("convention rather than a measurement", said)
        self.assertIn("not on one scale", said)

    def test_a_per_basis_mapping_differs_from_a_scalar(self):
        said = ghost_penalties_differ({"A": 300.0, "B": {"TASK": 300.0}})
        self.assertIsNotNone(said)


class TestTerminationMaps(unittest.TestCase):
    def test_one_map_passes(self):
        self.assertIsNone(termination_maps_differ({"A": "sha256:x", "B": "sha256:x"}))

    def test_two_maps_are_refused_with_no_override(self):
        said = termination_maps_differ({"A": "sha256:aaa", "B": "sha256:bbb"})
        self.assertIn("changes the numbers", said)
        self.assertIn("no override", said)

    def test_it_explains_the_mechanism(self):
        said = termination_maps_differ({"A": "sha256:aaa", "B": "sha256:bbb"})
        self.assertIn("SUCCESS", said)
        self.assertIn("TERMINAL", said)


class TestDeclaredIdentityPrunesTheFamily(unittest.TestCase):
    """Naming a duplicate while still charging the family for it would be
    reporting the problem and continuing to pay for it."""

    def test_the_pairs_feed_the_family(self):
        arm = {"path": "p", "server": "s:S", "server_args": {}}
        arms = {k: dict(arm) for k in "ABCDEF"}
        arms["B"]["path"] = "q"
        arms["C"]["path"] = "r"
        arms["D"]["path"] = "t"
        arms["E"]["path"] = "u"
        # F is a duplicate of A.
        pairs = identical_arm_pairs(arms, list("ABCDEF"))
        self.assertEqual(pairs, [("A", "F")])

        family = Family(arms=tuple("ABCDEF"))
        self.assertEqual(family.m, 15)
        family.prune_duplicates(pairs)
        self.assertEqual(family.m, 10,
                         "five distinct arms deserve m = 10, not 15")

    def test_no_duplicates_leaves_the_family_alone(self):
        arms = {k: {"path": k} for k in "ABC"}
        self.assertEqual(identical_arm_pairs(arms, list("ABC")), [])
        f = Family(arms=("A", "B", "C"))
        f.prune_duplicates([])
        self.assertEqual(f.m, 3)


class TheDefaultNoCallerEverOmitted(unittest.TestCase):
    """`evaluate(checkpoints=None)` derives them from the eligibility.

    Measured: 58 calls through the suite, every one of them naming the
    checkpoints. The branch that reads them off the data had never run, and it
    is the one a caller with no opinion about ordering gets.
    """

    def test_checkpoints_default_to_what_the_data_holds(self):
        from refractal.compare import build_units, evaluate

        rows = []
        for arm in ("A", "B"):
            for i in range(12):
                rows.append({
                    "scene_id": "s", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sc{i}",
                    "checkpoint_id": arm, "seed": 0, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": False, "success": i % 2 == 0,
                    "termination_map_digest": "sha256:a",
                })
        units = build_units(rows, checkpoints=["A", "B"])
        verdict = evaluate(units, resamples=200, seed=1)
        self.assertEqual(verdict.checkpoints, ["A", "B"])
        self.assertEqual(verdict.baseline, "A")

    def test_build_units_discovers_the_arms_when_not_told(self):
        """238 calls in the suite, every one naming them. This is the branch
        that decides what gets compared, so an untested default here is the
        one that silently produces a comparison nobody specified.
        """
        from refractal.compare import build_units

        rows = []
        for arm in ("zeta", "alpha", "mu"):
            for i in range(12):
                rows.append({
                    "scene_id": "s", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sc{i}",
                    "checkpoint_id": arm, "seed": 0, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": False, "success": i % 2 == 0,
                })
        discovered = build_units(rows)
        self.assertEqual(discovered.checkpoints, ["alpha", "mu", "zeta"])
        told = build_units(rows, checkpoints=["alpha", "mu", "zeta"])
        self.assertEqual(discovered.checkpoints, told.checkpoints)
        self.assertEqual(len(discovered.units), len(told.units))

    def test_the_cli_and_build_units_discover_the_same_arms(self):
        """The rule is written twice -- once in `cli.compare` and once as
        `build_units`'s default -- and the codebase's own argument against
        that is in the probe contract: two implementations of one rule is how
        identities fork without anyone noticing. Pinned rather than merged,
        because merging them is a refactor and this is the property that
        matters either way.
        """
        import re

        from refractal.compare import build_units

        source = pathlib.Path("src/refractal/cli.py").read_text()
        # The CLI's rule, lifted from its source so an edit to it fails here.
        match = re.search(r"args\.checkpoints or (sorted\(\{[^}]*\}\))", source)
        self.assertIsNotNone(match, "cli.py no longer discovers checkpoints this way")

        rows = [{"checkpoint_id": c} for c in ("b", "a", "b", "c")]
        cli_rule = eval(match.group(1), {"sorted": sorted}, {"rows": rows})

        full = [dict(r, scene_id="s", scene_hash="h", task_id="t", task_hash="th",
                     scenario_hash="sc", seed=0, session_id="x",
                     harness_version="v", harness_surface="v",
                     is_infra_failure=False, success=True) for r in rows]
        self.assertEqual(cli_rule, build_units(full).checkpoints)



if __name__ == "__main__":
    unittest.main()


class TestTheRefusalsAreWiredIntoEvaluate(unittest.TestCase):
    """The refusals exist and are reachable from `evaluate`.

    They are currently DARK CODE: built, tested in isolation above, and
    never called from the one entry point a comparison goes through. A
    refusal nothing reaches is a refusal that does not exist, and the way
    that rots is quietly -- the unit tests stay green while the behaviour is
    absent.

    These were written expecting three new parameters -- per-arm apparatus
    counts, ghost penalties and termination-map digests. Two of the three were
    the wrong shape, and wiring them is what showed it: apparatus losses and
    termination maps are per-ROW facts, so a caller passing them in would be
    passing back what it had just read out. They arrive on `Eligibility`
    instead, counted while the rows are walked, and the refusals take nothing.

    Apparatus has to be counted there rather than derived later: an apparatus
    episode is excluded before a unit is built, so by the time anyone holds
    units the evidence that the arms lost different shares is gone.

    What is left genuinely cannot come from the rows -- a convention chosen at
    analysis time, and the provenance of the environment -- and that is one
    parameter, `context`.
    """

    def test_evaluate_takes_only_what_the_rows_cannot_say(self):
        import inspect

        from refractal.compare import evaluate

        params = set(inspect.signature(evaluate).parameters)
        self.assertIn("context", params)
        # The two that would have been a caller handing back what it read.
        for derived in ("apparatus", "termination_maps"):
            self.assertNotIn(derived, params)


    def test_differing_apparatus_rates_block_without_being_passed_in(self):
        """The refusal reads the data. Nothing is handed to it."""
        from refractal.compare import build_units, evaluate

        rows = []
        for arm, infra_through in (("A", 0), ("B", 6)):
            for i in range(12):
                rows.append({
                    "scene_id": "s", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sc{i}",
                    "checkpoint_id": arm, "seed": 0, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": i < infra_through,
                    "success": i % 2 == 0, "termination_map_digest": "sha256:a",
                })
        verdict = evaluate(build_units(rows, checkpoints=["A", "B"]), ["A", "B"],
                           resamples=200, seed=1)
        self.assertTrue(any("apparatus" in b.lower() for b in verdict.blocking),
                        verdict.blocking)

    def test_differing_ghost_penalties_block_through_context(self):
        """This one does have to be passed: it is a choice, not a row."""
        from refractal.compare import build_units, evaluate
        from refractal.compare.verdict import ArmContext

        rows = []
        for arm in ("A", "B"):
            for i in range(12):
                rows.append({
                    "scene_id": "s", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sc{i}",
                    "checkpoint_id": arm, "seed": 0, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": False, "success": i % 2 == 0,
                    "termination_map_digest": "sha256:a",
                })
        verdict = evaluate(
            build_units(rows, checkpoints=["A", "B"]), ["A", "B"],
            resamples=200, seed=1,
            context={"A": ArmContext(ghost_penalty=300),
                     "B": ArmContext(ghost_penalty=500)},
        )
        self.assertTrue(any("penalt" in b.lower() for b in verdict.blocking),
                        verdict.blocking)

    def test_an_unrecorded_map_is_said_rather_than_assumed_to_agree(self):
        """Pre-v2 rows carry no digest, and silence would read as agreement."""
        from refractal.compare import build_units, evaluate

        rows = []
        for arm in ("A", "B"):
            for i in range(12):
                rows.append({
                    "scene_id": "s", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sc{i}",
                    "checkpoint_id": arm, "seed": 0, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": False, "success": i % 2 == 0,
                })
        verdict = evaluate(build_units(rows, checkpoints=["A", "B"]), ["A", "B"],
                           resamples=200, seed=1)
        self.assertTrue(any("not compared" in n for n in verdict.notes),
                        verdict.notes)
        self.assertFalse(any("termination map" in b for b in verdict.blocking))

    def test_an_editable_install_is_flagged_and_never_refused(self):
        """The rule the taxonomy now states: refuse on a substantive
        difference, flag on unverifiability. Identical code from a working
        tree is the same experiment that cannot be proved to be the same
        experiment, so refusing would fragment the result space for no gain.
        """
        from refractal.compare import build_units, evaluate
        from refractal.compare.verdict import ArmContext

        rows = []
        for arm in ("A", "B"):
            for i in range(12):
                rows.append({
                    "scene_id": "s", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sc{i}",
                    "checkpoint_id": arm, "seed": 0, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": False, "success": i % 2 == 0,
                    "termination_map_digest": "sha256:a",
                })
        units = build_units(rows, checkpoints=["A", "B"])
        verdict = evaluate(
            units, ["A", "B"], resamples=200, seed=1,
            context={"A": ArmContext(editable=("libero",), installs_checked=True),
                     "B": ArmContext(installs_checked=True)},
        )
        self.assertTrue(any("Editable installs" in n for n in verdict.notes),
                        verdict.notes)
        self.assertFalse(any("editable" in b.lower() for b in verdict.blocking))

    def test_unchecked_installs_are_distinguished_from_none(self):
        from refractal.compare import build_units, evaluate
        from refractal.compare.verdict import ArmContext

        rows = []
        for arm in ("A", "B"):
            for i in range(12):
                rows.append({
                    "scene_id": "s", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sc{i}",
                    "checkpoint_id": arm, "seed": 0, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": False, "success": i % 2 == 0,
                    "termination_map_digest": "sha256:a",
                })
        units = build_units(rows, checkpoints=["A", "B"])
        unchecked = evaluate(units, ["A", "B"], resamples=200, seed=1)
        self.assertTrue(any("not checked" in n for n in unchecked.notes))
        checked = evaluate(units, ["A", "B"], resamples=200, seed=1,
                           context={"A": ArmContext(installs_checked=True),
                                    "B": ArmContext(installs_checked=True)})
        self.assertTrue(any("published release" in n for n in checked.notes))

    def test_a_differing_termination_map_blocks_through_evaluate(self):
        from refractal.compare import build_units, evaluate

        rows = []
        for arm, digest in (("A", "sha256:aaa"), ("B", "sha256:bbb")):
            for i in range(12):
                rows.append({
                    "scene_id": "s", "scene_hash": "h", "task_id": "t",
                    "task_hash": "th", "scenario_hash": f"sc{i}",
                    "checkpoint_id": arm, "seed": 0, "session_id": "x",
                    "harness_version": "v", "harness_surface": "v",
                    "is_infra_failure": False, "success": i % 2 == 0,
                    "termination_map_digest": digest,
                })
        verdict = evaluate(build_units(rows, checkpoints=["A", "B"]), ["A", "B"],
                           resamples=200, seed=1)
        self.assertTrue(any("termination map" in b for b in verdict.blocking))
