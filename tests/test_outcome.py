"""The four-valued outcome, its invariants, and the maps that produce it.

Each invariant in the Phase 1 brief gets a test, and each test names the
reading that would be wrong if the invariant did not hold. An invariant whose
test only asserts that valid input is accepted is not tested: the question is
always what it REJECTS.
"""

import unittest

from refractal.schema.outcome import (
    DERIVED,
    Outcome,
    OutcomeInvariantError,
    TerminalBasis,
    TerminationMap,
    TerminationMapError,
    derived_flags,
    validate_row,
)


class TestDerivedFlags(unittest.TestCase):
    def test_every_outcome_declares_its_flags(self):
        # Total, so a new member cannot default to "an ordinary policy
        # failure" -- which is what a .get(o, (False, False)) would give it,
        # and is wrong for anything apparatus-shaped.
        for outcome in Outcome:
            self.assertIn(outcome, DERIVED, f"{outcome} has no derived flags")

    def test_only_apparatus_is_an_infra_failure(self):
        for outcome in Outcome:
            _, infra = derived_flags(outcome)
            self.assertEqual(infra, outcome is Outcome.APPARATUS)

    def test_only_success_is_a_success(self):
        for outcome in Outcome:
            ok, _ = derived_flags(outcome)
            self.assertEqual(ok, outcome is Outcome.SUCCESS)


class TestInvariants(unittest.TestCase):
    def ok(self, **kw):
        validate_row(**kw)

    def refuses(self, **kw):
        return self.assertRaises(OutcomeInvariantError)

    def test_success_requires_a_step(self):
        # Without it the episode contributes nothing to a time-to-success
        # distribution while still counting as a success, which is the one
        # combination that biases every functional downward.
        with self.refuses():
            validate_row(outcome=Outcome.SUCCESS, step_to_success=None,
                         step_budget=300, steps=120, decisions=120)

    def test_success_cannot_be_after_the_budget(self):
        with self.refuses():
            validate_row(outcome=Outcome.SUCCESS, step_to_success=301,
                         step_budget=300, steps=300, decisions=300)

    def test_success_at_exactly_the_budget_is_allowed(self):
        validate_row(outcome=Outcome.SUCCESS, step_to_success=300,
                     step_budget=300, steps=300, decisions=300)

    def test_budget_exhausted_carries_no_step(self):
        with self.refuses():
            validate_row(outcome=Outcome.BUDGET_EXHAUSTED, step_to_success=120,
                         step_budget=300, steps=300, decisions=300)

    def test_budget_exhausted_must_reach_the_budget(self):
        # An episode that stopped at 114 of 400 did not run out of budget;
        # something else ended it, and recording it as censoring at 400 would
        # put a wrong lower bound into the distribution.
        with self.refuses():
            validate_row(outcome=Outcome.BUDGET_EXHAUSTED, step_to_success=None,
                         step_budget=400, steps=114, decisions=114)

    def test_terminal_carries_no_step(self):
        with self.refuses():
            validate_row(outcome=Outcome.TERMINAL, step_to_success=40,
                         step_budget=300, steps=40, decisions=40)

    def test_apparatus_carries_no_step(self):
        with self.refuses():
            validate_row(outcome=Outcome.APPARATUS, step_to_success=5,
                         step_budget=300, steps=5, decisions=5)

    def test_step_budget_is_required(self):
        # Never inferred from the plan at analysis time: every functional is
        # bounded by the smallest budget in the cell, and that bound must come
        # from the row that ran.
        with self.refuses():
            validate_row(outcome=Outcome.TERMINAL, step_to_success=None,
                         step_budget=None, steps=40, decisions=40)


class TestTerminationMap(unittest.TestCase):
    def test_an_unmapped_flag_raises(self):
        m = TerminationMap("x", {"done": (Outcome.SUCCESS, None)})
        with self.assertRaises(TerminationMapError):
            m.resolve("exploded")

    def test_totality_is_checked_against_the_declared_vocabulary(self):
        # Against what the adapter CAN emit, not what it has emitted. A flag
        # that never fired is exactly the one nobody has thought about.
        m = TerminationMap("x", {"done": (Outcome.SUCCESS, None)})
        with self.assertRaises(TerminationMapError):
            m.covers(("done", "never_yet_seen"))

    def test_terminal_must_declare_a_basis(self):
        with self.assertRaises(TerminationMapError):
            TerminationMap("x", {"crash": (Outcome.TERMINAL, None)})

    def test_only_terminal_may_declare_a_basis(self):
        with self.assertRaises(TerminationMapError):
            TerminationMap("x", {"done": (Outcome.SUCCESS, TerminalBasis.TASK)})

    def test_the_basis_is_inside_the_digest(self):
        # It changes no statistic, but it changes what can be claimed, and a
        # sensitivity test keys off it -- so two runs that disagree about it
        # must be mechanically distinguishable.
        task = TerminationMap("x", {"c": (Outcome.TERMINAL, TerminalBasis.TASK)})
        sim = TerminationMap("x", {"c": (Outcome.TERMINAL, TerminalBasis.SIMULATOR)})
        self.assertNotEqual(task.digest(), sim.digest())

    def test_the_digest_is_order_independent(self):
        a = TerminationMap("x", {"a": (Outcome.SUCCESS, None),
                                 "b": (Outcome.BUDGET_EXHAUSTED, None)})
        b = TerminationMap("x", {"b": (Outcome.BUDGET_EXHAUSTED, None),
                                 "a": (Outcome.SUCCESS, None)})
        self.assertEqual(a.digest(), b.digest())


if __name__ == "__main__":
    unittest.main()


class TestAgainstARealRun(unittest.TestCase):
    """The invariants, against rows a run actually wrote."""

    def test_step_budget_equals_the_plans_max_steps(self):
        # Recorded rather than inferred, and asserted equal rather than
        # assumed equal. A divergence means the episode ran under a budget
        # other than the one planned, which is a bug to catch at the row
        # rather than a value to record quietly and reason over later.
        import tempfile
        from refractal.execute import read_episodes, run_local
        from refractal.execute.fake import FakeProvider
        from tests.test_execute import small_plan

        catalog_tmp, catalog_root, plan = small_plan()
        try:
            with tempfile.TemporaryDirectory() as out:
                run_local(plan, out, session_id="s", catalog_root=str(catalog_root),
                          provider=FakeProvider(success_rate=0.5))
                rows = read_episodes(out, plan.plan_id).to_pylist()
        finally:
            catalog_tmp.cleanup()
        budgets = {
            e.episode_id: e.max_steps
            for s in plan.scenes for w in s.workers for e in w.episodes
        }
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["step_budget"], budgets[row["episode_id"]])

    def test_every_written_row_satisfies_the_invariants(self):
        import tempfile
        from refractal.execute import read_episodes, run_local
        from refractal.execute.fake import FakeProvider
        from tests.test_execute import small_plan

        catalog_tmp, catalog_root, plan = small_plan()
        try:
            with tempfile.TemporaryDirectory() as out:
                run_local(plan, out, session_id="s", catalog_root=str(catalog_root),
                          provider=FakeProvider(success_rate=0.6, infra_failure_rate=0.1))
                rows = read_episodes(out, plan.plan_id).to_pylist()
        finally:
            catalog_tmp.cleanup()
        for row in rows:
            validate_row(
                outcome=Outcome(row["outcome"]),
                step_to_success=row["step_to_success"],
                step_budget=row["step_budget"],
                steps=row["steps"],
                decisions=row["decisions"],
            )
            ok, infra = derived_flags(Outcome(row["outcome"]))
            self.assertEqual(row["success"], ok)
            self.assertEqual(row["is_infra_failure"], infra)


class TestPhaseHashKeysAreDecided(unittest.TestCase):
    """The hashed phase keys are exactly these, not a subset of them.

    Phase 4 extends `phases` with optional `after` and `budget`, emitted only
    when set, which keeps every existing task_hash. That only holds while the
    hashed key set is deliberate: a field added to the Phase model either
    belongs in task identity or does not, and both answers are fine while
    somebody decides. What is not fine is the decision happening by accident,
    in either direction -- a field that silently joins the hash orphans every
    recorded result, and one that silently misses it makes two different
    experiments share an id.

    Asserted exactly rather than as a subset, so adding a field fails here
    until it is decided. That will occasionally be annoying, and silent hash
    drift is an identity-category bug, which is the worst kind this codebase
    can carry.
    """

    HASHED = {"name", "predicate", "predicate_args"}
    OPTIONAL = {"after", "budget"}

    def _task_with_phases(self):
        import pathlib
        import yaml
        from refractal.schema.models import TasksFile

        raw = yaml.safe_load(
            pathlib.Path("examples/catalog/tasks.yaml").read_text())
        tasks = TasksFile(**raw).tasks
        return max(tasks, key=lambda t: len(t.phases))

    def test_hashed_phase_keys_are_exactly_the_declared_set(self):
        from refractal.schema.identity import task_identity

        task = self._task_with_phases()
        self.assertTrue(task.phases, "fixture has no phases to check")
        for phase in task_identity(task)["phases"]:
            self.assertEqual(
                set(phase), self.HASHED,
                "the hashed phase document gained or lost a key. If a field was "
                "added to Phase, decide whether it belongs in task_hash and "
                "update HASHED/OPTIONAL here deliberately.")

    def test_every_phase_model_field_is_classified(self):
        from refractal.schema.models import Phase

        self.assertEqual(
            set(Phase.model_fields), self.HASHED | self.OPTIONAL & set(Phase.model_fields),
            "a Phase field is neither hashed nor declared optional-and-unhashed. "
            "Classify it rather than letting it default either way.")


class TestDecisions(unittest.TestCase):
    """The realized chunk length, measured rather than declared."""

    def test_decisions_is_required(self):
        with self.assertRaises(OutcomeInvariantError):
            validate_row(outcome=Outcome.SUCCESS, step_to_success=10,
                         step_budget=400, steps=10)

    def test_an_episode_that_stepped_made_a_decision(self):
        with self.assertRaises(OutcomeInvariantError):
            validate_row(outcome=Outcome.TERMINAL, step_to_success=None,
                         step_budget=400, steps=10, decisions=0)

    def test_a_crash_before_the_first_step_decides_nothing(self):
        # What an APPARATUS failure on start-up looks like: it inferred
        # nothing and stepped nothing, and zero is the honest count.
        validate_row(outcome=Outcome.APPARATUS, step_to_success=None,
                     step_budget=400, steps=0, decisions=0)

    def test_a_chunked_budget_exhaustion_is_refused_not_guessed(self):
        # 404 steps over 51 decisions is a realized chunk length near 8, which
        # is legitimate for a chunking adapter and unhandled here. It refuses
        # with the general form written out rather than accepting a row whose
        # censoring point this version cannot interpret.
        with self.assertRaises(OutcomeInvariantError) as caught:
            validate_row(outcome=Outcome.BUDGET_EXHAUSTED, step_to_success=None,
                         step_budget=400, steps=404, decisions=51)
        self.assertIn("chunk", str(caught.exception))

    def test_realized_chunk_length_is_one_for_every_adapter_here(self):
        import tempfile
        from refractal.execute import read_episodes, run_local
        from refractal.execute.fake import FakeProvider
        from tests.test_execute import small_plan

        catalog_tmp, catalog_root, plan = small_plan()
        try:
            with tempfile.TemporaryDirectory() as out:
                run_local(plan, out, session_id="s", catalog_root=str(catalog_root),
                          provider=FakeProvider(success_rate=0.5))
                rows = read_episodes(out, plan.plan_id).to_pylist()
        finally:
            catalog_tmp.cleanup()
        self.assertTrue(rows)
        for row in rows:
            if row["steps"]:
                self.assertEqual(row["decisions"], row["steps"])


class TestArmHashKeysAreDecided(unittest.TestCase):
    """The hashed arm descriptor is exactly these keys.

    Same ratchet as the phase test, on the other structure Phase 1 extends.
    `chunk_length` is emitted only when declared, which keeps every plan_id
    recorded before it existed -- verified against a real catalog, byte for
    byte, and against the 20 distinct plan_ids on disk, none of which declares
    one.

    `vram_mb` is the useful counterexample: it is a Checkpoint field that is
    deliberately NOT hashed, because it describes where the arm fits rather
    than what the arm is. So "every model field is hashed" would be the wrong
    rule, and the classification has to be explicit.
    """

    HASHED = {"id", "path", "server", "server_args"}
    OPTIONAL = {"chunk_length"}       # hashed only when declared
    UNHASHED = {"vram_mb", "action_hash_log"}   # placement, never hashed

    def test_every_checkpoint_field_is_classified(self):
        from refractal.schema.models import Checkpoint

        self.assertEqual(
            set(Checkpoint.model_fields),
            self.HASHED | self.OPTIONAL | self.UNHASHED,
            "a Checkpoint field is unclassified. Decide whether it is identity "
            "(hashed), placement (never hashed), or optional-and-hashed-when-set, "
            "rather than letting it default either way.")

    def test_an_undeclared_chunk_length_is_absent_from_the_hash(self):
        from refractal.schema.canonical import hash_obj
        from refractal.schema.models import Checkpoint

        base = dict(id="a", path="p", server="s:S", server_args={})
        plain = Checkpoint(**base)
        self.assertIsNone(plain.chunk_length)
        descriptor = {k: getattr(plain, k) for k in sorted(self.HASHED)}
        with_field = dict(descriptor)
        if plain.chunk_length is not None:
            with_field["chunk_length"] = plain.chunk_length
        self.assertEqual(hash_obj(descriptor), hash_obj(with_field))

    def test_declaring_one_changes_the_experiment(self):
        from refractal.schema.canonical import hash_obj

        base = {"id": "a", "path": "p", "server": "s:S", "server_args": {}}
        self.assertNotEqual(
            hash_obj({**base, "chunk_length": 50}),
            hash_obj({**base, "chunk_length": 8}),
            "k=50 and k=8 drive the arm differently and are different "
            "experiments.")


class TestDeclaredIdentity(unittest.TestCase):
    """Two arms declared the same thing, which the behavioural check may miss.

    The behavioural detector needs replicates and >= 10 scenarios; below
    either it declines. This one needs neither, so it is the half that still
    works exactly where the other cannot -- which is the point of it being a
    separate site rather than a widening.
    """

    def arms(self, **kw):
        base = {"path": "p", "server": "s:S", "server_args": {}}
        return {k: {**base, **v} for k, v in kw.items()}

    def test_two_arms_differing_only_in_id_are_blocked(self):
        from refractal.compare.verdict import _arms_declared_identical

        msgs = _arms_declared_identical(self.arms(E={}, F={}), ["E", "F"])
        self.assertEqual(len(msgs), 1)
        self.assertIn("declared identically", msgs[0])
        self.assertIn("correction family", msgs[0],
                      "says why a duplicate costs more than one junk line")

    def test_different_weights_pass(self):
        from refractal.compare.verdict import _arms_declared_identical

        arms = self.arms(E={}, F={"path": "other"})
        self.assertEqual(_arms_declared_identical(arms, ["E", "F"]), [])

    def test_a_declared_chunk_length_distinguishes_arms(self):
        from refractal.compare.verdict import _arms_declared_identical

        arms = self.arms(E={}, F={"chunk_length": 8})
        self.assertEqual(
            _arms_declared_identical(arms, ["E", "F"]), [],
            "same weights at two declared chunk lengths is a legitimate sweep")

    def test_the_same_declared_chunk_length_still_blocks(self):
        from refractal.compare.verdict import _arms_declared_identical

        arms = self.arms(E={"chunk_length": 8}, F={"chunk_length": 8})
        self.assertEqual(len(_arms_declared_identical(arms, ["E", "F"])), 1)

    def test_it_declines_rather_than_passing_when_the_plan_is_absent(self):
        from refractal.compare.verdict import _arms_declared_identical

        self.assertEqual(_arms_declared_identical(None, ["E", "F"]), [])

    def test_the_descriptor_comes_from_one_definition(self):
        # Derived from the hashed arm descriptor minus the id, rather than a
        # second list of fields. A second list is a second place the identity
        # decision lives, and the place it drifts.
        from refractal.schema.identity import arm_descriptor
        from refractal.schema.models import Checkpoint

        c = Checkpoint(id="a", path="p", server="s:S", server_args={"x": 1},
                       vram_mb=16, chunk_length=4)
        full = arm_descriptor(c)
        without = arm_descriptor(c, with_id=False)
        self.assertEqual(set(full) - set(without), {"id"})
        self.assertNotIn("vram_mb", full, "placement is not identity")
        self.assertIn("chunk_length", without)
