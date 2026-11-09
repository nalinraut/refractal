"""Phase 4: a phase order the catalog constrains and the policy chooses."""

import unittest

from refractal.compare import build_units, evaluate
from refractal.compare.analysis import AnalysisSpec
from refractal.compare.compose import compose_report
from refractal.compare.phases import (
    PhaseError,
    conditional_rates,
    orders_differ,
    traversal_of,
)
from refractal.schema.identity import task_hash, task_identity
from refractal.schema.models import Phase, Task
from refractal.schema.outcome import OutcomeInvariantError, validate_phase_records


def phase(name, after=()):
    return Phase(name=name, predicate="builtins:bool", after=after)


def task(phases):
    return Task(id="t", scene="sc", instruction="i", predicate="builtins:bool",
                max_steps=100, phases=phases)


def episode(order, *, satisfied=(), ends_held=(), attempted=None):
    """One episode's phase records, in a chosen traversal order."""
    records = []
    for i, name in enumerate(order):
        was = name in (attempted if attempted is not None else order)
        records.append({
            "phase_id": name,
            "attempted": was,
            "start_step": i * 10 if was else None,
            "end_step": i * 10 + 9 if was else None,
            "first_satisfied_step": i * 10 + 3 if name in satisfied else None,
            "outcome": "SUCCESS" if name in ends_held else "BUDGET_EXHAUSTED",
        })
    return {"phase_records": records}


class PrerequisitesArePerPhase(unittest.TestCase):
    """Adjacency list, not a top-level edge list: the same graph in the one
    form that inherits the hash-stability argument already made for extending
    a phase entry."""

    def test_a_phase_may_require_several(self):
        """Open the drawer AND turn on the light before picking. A single
        predecessor would force a false linearisation of two independent
        preconditions, and the traversal would then look like a policy choice
        when it was the catalog's."""
        t = task([phase("pick", ("drawer", "light")), phase("drawer"),
                  phase("light")])
        self.assertEqual(t.phases[0].after, ("drawer", "light"))

    def test_an_absent_after_leaves_the_task_hash_alone(self):
        """The reason it is per-phase. Measured against every catalog on disk
        as well: 313 identity values unchanged, 63/63 task hashes still
        reproduced."""
        plain = task([phase("a"), phase("b")])
        self.assertEqual(sorted(task_identity(plain)["phases"][0]),
                         ["name", "predicate", "predicate_args"])

    def test_a_declared_order_is_hashed(self):
        """Because it is a different task. "Pick after opening the drawer"
        and "pick" are not the same goal, and results recorded against one
        must not join results recorded against the other."""
        plain = task([phase("a"), phase("b")])
        ordered = task([phase("a"), phase("b", ("a",))])
        self.assertNotEqual(task_hash(plain), task_hash(ordered))
        self.assertIn("after", task_identity(ordered)["phases"][1])


class ACycleIsRefusedAtLoad(unittest.TestCase):
    """Compiler, not runtime. A cycle is a property of the declaration, so
    discovering it at step 200 of the first episode wastes a simulator and
    reports as a policy failure."""

    def _refusal(self, build):
        with self.assertRaises(Exception) as caught:
            build()
        return str(caught.exception)

    def test_the_cycle_is_named_not_merely_reported(self):
        """"There is a cycle" leaves the author to find it. Kahn's algorithm
        leaves exactly the cycle behind, which is why it is worth more here
        than a visited-set walk."""
        message = self._refusal(
            lambda: task([phase("a", ("c",)), phase("b", ("a",)),
                          phase("c", ("b",))]))
        self.assertIn("cycle", message)
        for name in ("'a'", "'b'", "'c'"):
            self.assertIn(name, message)

    def test_a_long_chain_is_not_mistaken_for_a_cycle(self):
        """The guard has to admit the legal case it most resembles."""
        t = task([phase("a"), phase("b", ("a",)), phase("c", ("b",)),
                  phase("d", ("c",))])
        self.assertEqual(len(t.phases), 4)

    def test_a_diamond_is_legal(self):
        t = task([phase("start"), phase("left", ("start",)),
                  phase("right", ("start",)),
                  phase("join", ("left", "right"))])
        self.assertEqual(len(t.phases), 4)

    def test_a_prerequisite_the_task_does_not_declare(self):
        message = self._refusal(lambda: task([phase("a", ("ghost",))]))
        self.assertIn("ghost", message)
        self.assertIn("does not declare", message)

    def test_a_phase_cannot_require_itself(self):
        self.assertIn("its own prerequisite",
                      self._refusal(lambda: phase("a", ("a",))))


class OneAttemptPerPhase(unittest.TestCase):
    """The conditional product is what this buys. P(all) = prod P(k | prefix)
    is a product over phases, and a second attempt at one makes the left side
    a different quantity from the right."""

    def test_a_repeated_phase_is_refused_not_merged(self):
        with self.assertRaises(OutcomeInvariantError) as c:
            validate_phase_records(
                [{"phase_id": "a", "attempted": True},
                 {"phase_id": "a", "attempted": True}], steps=10)
        self.assertIn("more than one record", str(c.exception))
        self.assertIn("distinct phase", str(c.exception))

    def test_a_record_with_no_id_is_refused(self):
        """Phases pair by id and never by position, so a record without one
        cannot be paired with anything."""
        with self.assertRaises(OutcomeInvariantError):
            validate_phase_records([{"attempted": True}], steps=10)


class BothStepFactsAreRecorded(unittest.TestCase):
    """"Did it demonstrate the skill" and "did it deliver" are different
    questions. A policy can put the item in the bin and later knock it out,
    and a single field has to silently pick one of the two answers."""

    def test_held_and_then_stopped_holding_is_legal(self):
        """The case the second field exists for. Refusing it would make the
        pair pointless."""
        validate_phase_records(
            [{"phase_id": "a", "attempted": True, "first_satisfied_step": 5,
              "end_step": 9, "outcome": "BUDGET_EXHAUSTED"}], steps=20)

    def test_what_cannot_have_happened_in_time_is_refused(self):
        for record, expected in (
            ({"phase_id": "a", "attempted": False, "first_satisfied_step": 3},
             "never ran"),
            ({"phase_id": "a", "attempted": True, "first_satisfied_step": 3,
              "start_step": 7}, "before its start_step"),
            ({"phase_id": "a", "attempted": True, "first_satisfied_step": 99},
             "step(s)"),
            ({"phase_id": "a", "attempted": True, "first_satisfied_step": 9,
              "end_step": 4}, "after its end_step"),
        ):
            with self.subTest(record=record):
                with self.assertRaises(OutcomeInvariantError) as c:
                    validate_phase_records([record], steps=20)
                self.assertIn(expected, str(c.exception))

    def test_the_credit_choice_changes_the_rate(self):
        """So it cannot be a default. Here the phase held mid-episode and not
        at the end: one reading gives 1.0 and the other 0.0, over identical
        rows."""
        rows = [episode(["a"], satisfied=("a",), ends_held=())]
        self.assertEqual(conditional_rates(rows, credit="first_satisfied")[0].rate, 1.0)
        self.assertEqual(conditional_rates(rows, credit="at_end")[0].rate, 0.0)

    def test_the_credit_choice_is_in_the_analysis_hash(self):
        """Two per-phase reports of one run under different credit are not
        comparable, and nothing else in the identity would say why."""
        self.assertNotEqual(AnalysisSpec().analysis_hash(),
                            AnalysisSpec(phase_credit="first_satisfied").analysis_hash())

    def test_an_unknown_credit_refuses_rather_than_defaulting(self):
        with self.assertRaises(PhaseError):
            conditional_rates([episode(["a"])], credit="whatever")


class TheOrderTakenIsAMeasurement(unittest.TestCase):
    def test_it_is_read_from_start_step_not_the_declaration(self):
        """The declaration constrains which orders are legal; the policy
        picks one from those."""
        t = traversal_of("A", [episode(["light", "open", "pick"])])
        self.assertEqual(t.order, ("light", "open", "pick"))

    def test_a_phase_never_reached_has_no_position(self):
        """Giving it one would invent an order the run did not take."""
        t = traversal_of("A", [episode(["open", "light", "pick"],
                                       attempted=("open", "light"))])
        self.assertEqual(t.order, ("open", "light"))


class RuleFourIsNarrow(unittest.TestCase):
    """Order affects the three readings differently, so one refusal for all
    three would be wrong in two of them."""

    def _rows(self, orders):
        rows = []
        for arm, order in orders.items():
            for sc in range(6):
                for seed in range(3):
                    ok = sc > 0
                    rows.append({
                        "scene_id": f"s{sc}", "scene_hash": "h", "task_id": "t",
                        "task_hash": "th", "scenario_hash": f"sh{sc}",
                        "checkpoint_id": arm, "seed": seed, "session_id": "x",
                        "harness_version": "v", "harness_surface": "v",
                        "is_infra_failure": False, "success": ok,
                        "outcome": "SUCCESS" if ok else "BUDGET_EXHAUSTED",
                        "step_to_success": 60 + sc if ok else None,
                        "step_budget": 300, "termination_map_digest": "sha256:a",
                        **episode(order, satisfied=order,
                                  ends_held=order if ok else order[:2]),
                    })
        return rows

    def _report(self, orders):
        rows = self._rows(orders)
        e = build_units(rows, checkpoints=list(orders))
        v = evaluate(e, list(orders), resamples=200, seed=1)
        return v, compose_report(
            v, e, plan_id="p", tier="full",
            episodes={c: [r for r in rows if r["checkpoint_id"] == c]
                      for c in orders})

    def test_the_same_order_reports_chained_rates(self):
        _, report = self._report({"A": ["open", "light", "pick"],
                                  "B": ["open", "light", "pick"]})
        phases = [s for s in report.arms[0].scalars if s.name.startswith("phase ")]
        self.assertEqual(len(phases), 3)

    def test_a_different_order_refuses_only_the_chained_reading(self):
        """P(phase 3 | prefix) conditions on different prefixes, so the two
        arms are not estimating the same quantity."""
        verdict, report = self._report({"A": ["open", "light", "pick"],
                                        "B": ["light", "open", "pick"]})
        self.assertTrue(any("chained per-phase" in d for d in report.declined))
        self.assertFalse([s for s in report.arms[0].scalars
                          if s.name.startswith("phase ")])

        # And the episode-level comparison proceeds: both arms got the same
        # scene, task and budget, and the order is how each spent it.
        self.assertFalse(any("order" in b for b in verdict.blocking))
        self.assertTrue(verdict.tasks)
        self.assertTrue(any(s.value is not None for s in report.arms[0].scalars))

    def test_the_order_is_reported_even_when_nothing_is_refused(self):
        """Choosing a good order is part of what is being evaluated, so two
        arms picking different ones is a finding about the policies. If it
        only ever appeared inside a refusal, every comparison that proceeded
        would hide it."""
        _, same = self._report({"A": ["open", "light", "pick"],
                                "B": ["open", "light", "pick"]})
        self.assertEqual(len(same.traversals), 2)
        self.assertIn("open -> light -> pick", same.traversals[0])
        self.assertIn("phase order taken", same.render())

    def test_one_arm_alone_is_not_an_order_disagreement(self):
        self.assertIsNone(orders_differ([traversal_of("A", [episode(["a", "b"])])]))


if __name__ == "__main__":
    unittest.main()
