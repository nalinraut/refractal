"""Declared chunk length against realized, and the promotion hole it closes."""

import unittest

from refractal.compare import build_units, evaluate
from refractal.compare.verdict import chunk_length_mismatch
from refractal.schema.identity import episode_id


def rows(by_arm, *, steps=110):
    out = []
    for arm, decisions in by_arm.items():
        for i in range(12):
            out.append({
                "scene_id": "s", "scene_hash": "h", "task_id": "t",
                "task_hash": "th", "scenario_hash": f"sc{i}",
                "checkpoint_id": arm, "seed": 0, "session_id": "x",
                "harness_version": "v", "harness_surface": "v",
                "is_infra_failure": False, "success": i % 2 == 0,
                "steps": steps, "decisions": decisions,
                "termination_map_digest": "sha256:a",
            })
    return out


class TheCheckIsNotCircular(unittest.TestCase):
    """Both sides come from different places: the realized side is counted
    from the model server's own per-call log, the declared side from the
    catalog. Derived from the declared chunk size, `steps / decisions` would
    be a restatement of the declaration and would always agree."""

    def test_agreement_is_silent(self):
        self.assertEqual(
            chunk_length_mismatch({"a": (10, 10)}, {"a": {"chunk_length": 10}}), [])

    def test_a_mismatch_names_both_sides(self):
        message = chunk_length_mismatch(
            {"a": (5, 5)}, {"a": {"chunk_length": 10}})[0]
        self.assertIn("chunk_length=10", message)
        self.assertIn("consistent with 5", message)
        self.assertIn("No override", message)

    def test_an_undeclared_arm_is_not_checked(self):
        """Not passed -- the caller reports it as unchecked. An arm that
        declares nothing has no declaration to contradict."""
        self.assertEqual(chunk_length_mismatch({"a": (10, 10)}, {}), [])

    def test_an_ambiguous_episode_does_not_refuse_a_true_declaration(self):
        """81 steps over 9 calls is consistent with 9 AND 10. Inverting the
        ratio picks one and refuses the other; membership admits both."""
        from refractal.compare.pairing import _chunks_consistent_with

        self.assertEqual(_chunks_consistent_with(81, 9), (9, 10))
        self.assertEqual(
            chunk_length_mismatch({"a": (9, 10)}, {"a": {"chunk_length": 10}}), [])

    def test_the_ratio_itself_is_not_the_chunk_length(self):
        """The error real data caught. The last chunk is partial, so the ratio
        lands between integers and reads as a different chunk length on every
        episode whose step count is not a multiple."""
        from refractal.compare.pairing import _chunks_consistent_with

        self.assertAlmostEqual(104 / 11, 9.4545, places=3)
        self.assertEqual(_chunks_consistent_with(104, 11), (10, 10))


class OneArmIsOneCondition(unittest.TestCase):
    def test_rows_no_single_chunk_length_explains_refuse(self):
        """What promoting a run with a different chunk size looks like from
        inside: the feasible set intersects to empty."""
        message = chunk_length_mismatch({"a": None}, {})[0]
        self.assertIn("no single chunk length explains", message)
        self.assertIn("episode_id does not cover it", message)

    def test_pooling_two_chunk_lengths_empties_the_feasible_set(self):
        pooled = rows({"pi0": 11}) + rows({"pi0": 22})
        eligibility = build_units(pooled, checkpoints=["pi0"])
        self.assertIsNone(eligibility.realized_chunk["pi0"])


class ThePromotionHoleIsReal(unittest.TestCase):
    def test_two_plans_sharing_an_arm_label_mint_the_same_episode_ids(self):
        """`episode_id` covers scene, task, scenario, seed and checkpoint id.
        It does not cover chunk size, which lives in `plan_id`. So two plans
        differing only in `server_args.chunk_size`, both labelling an arm
        `pi0`, produce identical episode ids -- and promotion's nesting check
        asks only whether each promoted id is one the target plan contains.
        """
        from refractal.promote import check_nesting, duplicated_in

        common = dict(scene_hash="sha256:scene", task_hash="sha256:task",
                      scenario_hash="sha256:sc0", seed=3)
        chunk10 = episode_id(**common, checkpoint_id="pi0")
        chunk5 = episode_id(**common, checkpoint_id="pi0")
        self.assertEqual(chunk10, chunk5)

        class Ep: episode_id = chunk10
        class W: episodes = [Ep()]
        class S: workers = [W()]
        class Plan: scenes = [S()]

        promoted = [{"episode_id": chunk5}]
        self.assertEqual(check_nesting(Plan(), promoted), set())
        self.assertEqual(duplicated_in(promoted), {})

    def test_distinct_arm_labels_do_not_collide(self):
        """Which is why the probe runs as one plan with two arms rather than
        as two plans: the collision is removed by construction instead of
        being detected afterwards."""
        common = dict(scene_hash="sha256:scene", task_hash="sha256:task",
                      scenario_hash="sha256:sc0", seed=3)
        self.assertNotEqual(
            episode_id(**common, checkpoint_id="pi0-chunk10"),
            episode_id(**common, checkpoint_id="pi0-chunk5"))

    def test_the_check_catches_the_pooled_rows(self):
        """Even promoted, the rows carry twice the decisions at the same step
        count, so no single k explains them."""
        pooled = rows({"pi0": 11}) + rows({"pi0": 22})
        eligibility = build_units(pooled, checkpoints=["pi0"])
        self.assertIsNone(eligibility.realized_chunk["pi0"])


class ItReachesTheVerdict(unittest.TestCase):
    def test_a_mismatched_arm_blocks(self):
        data = rows({"a": 11, "b": 11})
        eligibility = build_units(data, checkpoints=["a", "b"])
        verdict = evaluate(eligibility, ["a", "b"], resamples=200, seed=1,
                           arms={"a": {"chunk_length": 10, "path": "x"},
                                 "b": {"chunk_length": 5, "path": "y"}})
        self.assertTrue(any("chunk_length=5" in b for b in verdict.blocking),
                        verdict.blocking)

    def test_rows_without_decisions_are_reported_as_unchecked(self):
        """Silence here is indistinguishable from a check that ran and
        passed, and without `decisions` the realized side does not exist."""
        data = [dict(r, decisions=None) for r in rows({"a": 11, "b": 11})]
        eligibility = build_units(data, checkpoints=["a", "b"])
        verdict = evaluate(eligibility, ["a", "b"], resamples=200, seed=1,
                           arms={"a": {"chunk_length": 10}})
        self.assertTrue(any("Chunk length not checked" in n for n in verdict.notes),
                        verdict.notes)
        self.assertFalse(any("chunk_length" in b for b in verdict.blocking))


class OneDecisionLeavesTheUpperBoundOpen(unittest.TestCase):
    """`ceil(steps/k) == 1` for every k at least as long as the episode.

    The enumerated-set version truncated that to {steps}, which would have
    falsely refused a correct declaration on any suite with episodes shorter
    than its chunk length. It does not arise in the recorded data -- the
    shortest episode is 63 steps at chunk 10, so 7 decisions -- and it would
    arise on a suite with short episodes.
    """

    def test_the_range_is_unbounded_above(self):
        from refractal.compare.pairing import _chunks_consistent_with

        self.assertEqual(_chunks_consistent_with(5, 1), (5, None))

    def test_a_longer_declaration_is_not_refused(self):
        from refractal.compare.pairing import _chunks_consistent_with

        self.assertEqual(
            chunk_length_mismatch({"a": _chunks_consistent_with(5, 1)},
                                  {"a": {"chunk_length": 10}}), [])

    def test_it_still_intersects_to_something_tight(self):
        from refractal.compare.pairing import _chunks_consistent_with, _intersect

        short = _chunks_consistent_with(5, 1)        # (5, None)
        long = _chunks_consistent_with(104, 11)      # (10, 10)
        self.assertEqual(_intersect(short, long), (10, 10))


class WideIsNotVerified(unittest.TestCase):
    """A declaration inside a wide range has not been verified, it has merely
    not been contradicted."""

    def test_a_wide_range_is_reported(self):
        from refractal.compare.verdict import chunk_length_weakly_constrained

        note = chunk_length_weakly_constrained(
            {"a": (9, 10)}, {"a": {"chunk_length": 10}})[0]
        self.assertIn("only to 9..10", note)
        self.assertIn("not verified either", note)

    def test_a_pinned_range_says_nothing(self):
        from refractal.compare.verdict import chunk_length_weakly_constrained

        self.assertEqual(chunk_length_weakly_constrained(
            {"a": (10, 10)}, {"a": {"chunk_length": 10}}), [])

    def test_an_unbounded_range_reads_as_wide(self):
        from refractal.compare.verdict import chunk_length_weakly_constrained

        note = chunk_length_weakly_constrained({"a": (5, None)}, {})[0]
        self.assertIn("unbounded", note)


class TheTwoErrorsDoNotShareAMessage(unittest.TestCase):
    def test_contradiction_and_non_membership_differ(self):
        """They point at different fixes: rows that cannot all have come from
        one chunk length, versus rows that agree with each other and disagree
        with the catalog."""
        contradiction = chunk_length_mismatch({"a": None}, {})[0]
        non_member = chunk_length_mismatch(
            {"a": (5, 5)}, {"a": {"chunk_length": 10}})[0]
        self.assertIn("no single chunk length explains", contradiction)
        self.assertNotIn("declares chunk_length", contradiction)
        self.assertIn("declares chunk_length=10", non_member)
        self.assertNotIn("no single chunk length explains", non_member)


if __name__ == "__main__":
    unittest.main()


class PromotionInheritsTheComparisonsRefusals(unittest.TestCase):
    """Everything that must match within one comparison must match across a
    promoted boundary. The only exceptions are the seed and plan_id, because
    varying those is what promotion is for."""

    TARGET = {"pi0": {"path": "p", "server": "s:S",
                      "server_args": {"chunk_size": 10}}}

    def test_identical_declarations_pass(self):
        from refractal.promote import arms_differ

        self.assertEqual(arms_differ(self.TARGET, self.TARGET), [])

    def test_a_different_chunk_size_refuses_at_the_boundary(self):
        """The case that reached this through `episode_id`: both guards passed
        and nothing downstream could tell the rows apart."""
        from refractal.promote import arms_differ

        source = {"pi0": {"path": "p", "server": "s:S",
                          "server_args": {"chunk_size": 5}}}
        message = arms_differ(self.TARGET, source)[0]
        self.assertIn("server_args", message)
        self.assertIn("covers the checkpoint's ID and nothing in its declaration",
                      message)

    def test_it_is_not_specific_to_chunk_size(self):
        """Chunk size was the field that happened to be invisible. The
        exposure is every behaviour-determining key, which is why the whole
        descriptor is compared rather than a list of fields."""
        from refractal.promote import arms_differ

        for key, value in (("path", "other-weights"), ("server", "other:Server"),
                           ("chunk_length", 5)):
            with self.subTest(key=key):
                source = {"pi0": dict(self.TARGET["pi0"], **{key: value})}
                self.assertIn(key, arms_differ(self.TARGET, source)[0])

    def test_an_absent_declaration_declines_rather_than_passing(self):
        """A results tree someone else wrote has no plan in reach. The
        comparison's own declared-identity check is what reports that."""
        from refractal.promote import arms_differ

        self.assertEqual(arms_differ(None, self.TARGET), [])
        self.assertEqual(arms_differ(self.TARGET, None), [])

    def test_it_runs_before_nesting_and_duplication(self):
        """Those ask whether these particular rows belong, which is a question
        about a comparison that should not be happening."""
        from refractal.promote import PromotionError, promote

        source = {"pi0": {"path": "p", "server": "s:S",
                          "server_args": {"chunk_size": 5}}}
        # Rows that would ALSO fail duplication, to show which refusal wins.
        doubled = [{"episode_id": "e1"}, {"episode_id": "e1"}]
        with self.assertRaises(PromotionError) as caught:
            promote(None, [], doubled, "sha256:other",
                    target_arms=self.TARGET, source_arms=source)
        self.assertIn("declared differently", str(caught.exception))
        self.assertNotIn("duplicated episode id", str(caught.exception))


class ServerArgsIsAnUnverifiedDeclaration(unittest.TestCase):
    """A field in `plan_id` that describes behaviour without causing it.

    `build_eval_config` sends a server only its URL, so Refractal's execute
    path never reads `server_args`. Every other field in the identity chain
    covers something that happens -- `scene_hash` covers files that load,
    `task_hash` covers the predicate evaluated -- and this one covers a label.
    `plan_id` can be confidently wrong about how a policy was driven.

    Keeping it in `plan_id` is still right: two runs declaring different
    arguments are different experiments, and the declaration is the only
    record of intent. What would not be right is a verdict silent about
    whether anyone checked it.
    """

    def test_the_execute_path_does_not_read_it(self):
        """Pinned, so a future change that starts honouring it fails here and
        the verdict's wording is revisited rather than left stale.

        By AST, not by searching the text. The text version fired on a comment
        that merely EXPLAINED why nothing reads `server_args` -- so it was
        asserting that nobody mentions the field rather than that nobody reads
        it, and it would have gone off on any passing remark while masking a
        real read behind a rewording. Checks that match prose instead of code
        are another way of not exercising the thing they name.
        """
        import ast
        import pathlib

        offenders = []
        for name in ("vla_eval.py", "vla_eval_runner.py", "local.py", "results.py"):
            source = pathlib.Path("src/refractal/execute") / name
            if not source.exists():
                continue
            tree = ast.parse(source.read_text())
            # Every docstring, so prose about the field is not a read of it.
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef))
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            }
            for node in ast.walk(tree):
                reads = (
                    isinstance(node, ast.Attribute) and node.attr == "server_args"
                ) or (
                    isinstance(node, ast.Constant) and node.value == "server_args"
                    and id(node) not in docstrings
                )
                if reads:
                    offenders.append(f"{name}:{getattr(node, 'lineno', '?')}")
        self.assertEqual(
            offenders, [],
            "the execute path now reads server_args; the verdict says nothing "
            "verifies it, and that wording needs revisiting")

    def test_the_verdict_says_it_is_unverified(self):
        from refractal.compare.compose import SERVER_ARGS_UNVERIFIED
        from refractal.compare.report import Reproducible

        line = Reproducible(installs_checked=True,
                            server_args_verified=SERVER_ARGS_UNVERIFIED).render()
        self.assertIn("NOT verified", line)
        self.assertIn("meant to be driven rather than how they were", line)

    def test_silence_is_distinguishable_from_verified(self):
        """A report built without the field says nothing about server_args,
        which must not read as a pass."""
        from refractal.compare.report import Reproducible

        self.assertNotIn("server_args", Reproducible(installs_checked=True).render())

    def test_the_hello_handshake_would_narrow_it_not_close_it(self):
        """Recorded as a test because it decides whether building the check is
        worth it. The harness's HELLO reply carries the model server's class
        name and `observation_params`; the LeRobot server's
        `get_observation_params` returns `send_state` and `send_wrist_image`
        and nothing else. `chunk_size` is absent, so a handshake check could
        verify the server class and two observation flags, and the chunk
        length would still be post hoc only.
        """
        import pathlib

        harness = pathlib.Path("/home/nalin/Develop/vla-eval-0.6.0/src/vla_eval")
        if not harness.exists():
            self.skipTest("the harness checkout is not here")
        params = harness / "model_servers" / "lerobot.py"
        body = params.read_text()
        start = body.index("def get_observation_params")
        reported = body[start:body.index("def get_action_spec", start)]
        self.assertIn("send_state", reported)
        self.assertIn("send_wrist_image", reported)
        self.assertNotIn("chunk_size", reported)


class TheCanariesCheckTheConfigNotJustTheLog(unittest.TestCase):
    """Two unrelated derivations agreeing, applied before the GPU time.

    The log canary proves the server is writing where the catalog reads. It
    says nothing about WHICH config the server came up on, and `server_args`
    is a declaration Refractal never sends -- so a chunk-5 arm can come up on
    a chunk-10 config and nothing notices until the post-hoc check.
    """

    def _first(self, steps, success=True):
        class Out:
            def __init__(self):
                self.episode = type("E", (), {"episode_id": "e1"})()
                self.steps, self.success = steps, success
        return Out()

    def test_one_episode_separates_five_from_ten(self):
        """The only distinction a replan-interval probe needs."""
        from refractal.schema.outcome import chunk_lengths_consistent_with as f

        self.assertEqual(f(104, 11), (10, 10))
        self.assertEqual(f(104, 21), (5, 5))

    def test_a_declared_five_against_a_ten_window_warns(self):
        import io
        from contextlib import redirect_stdout

        from refractal.execute.vla_eval_runner import _chunk_canary

        out = io.StringIO()
        with redirect_stdout(out):
            _chunk_canary("pi0-chunk5", self._first(104), 11, 5)
        printed = out.getvalue()
        self.assertIn("WARNING", printed)
        self.assertIn("chunk_length=5", printed)
        self.assertIn("only consistent with 10", printed)
        self.assertIn("launched on another config", printed)

    def test_agreement_is_reported_rather_than_silent(self):
        """Silence would leave a reader unable to tell a check that ran from
        one that was skipped -- the same rule as everywhere else here."""
        import io
        from contextlib import redirect_stdout

        from refractal.execute.vla_eval_runner import _chunk_canary

        out = io.StringIO()
        with redirect_stdout(out):
            _chunk_canary("pi0-chunk10", self._first(104), 11, 10)
        self.assertIn("consistent with chunk length 10", out.getvalue())
        self.assertNotIn("WARNING", out.getvalue())

    def test_an_ambiguous_episode_does_not_warn(self):
        """81 steps over 9 calls is consistent with 9 and 10. It will not
        always pin one value; it always separates 5 from 10."""
        import io
        from contextlib import redirect_stdout

        from refractal.execute.vla_eval_runner import _chunk_canary

        out = io.StringIO()
        with redirect_stdout(out):
            _chunk_canary("arm", self._first(81), 9, 10)
        self.assertNotIn("WARNING", out.getvalue())
        self.assertIn("9..10", out.getvalue())


class TheNewestMeasurementIsCheckedAgainstTheOldest(unittest.TestCase):
    """`step_to_success` is measured from the per-step buffer, which is the
    least exercised code in the write path. `steps` comes from the harness's
    aggregate and has been written for far longer. On a harness that ends an
    episode at success they must agree on a success row.
    """

    # NOT `_outcome`: `unittest.TestCase` keeps its own `self._outcome`, an
    # `_Outcome` instance, so a helper by that name is shadowed by the
    # framework at run time and the call fails with "object is not callable".
    def _episode(self, steps, success=True):
        return type("Out", (), {"steps": steps, "success": success})()

    def test_agreement_is_stated(self):
        import io
        from contextlib import redirect_stdout

        from refractal.execute.vla_eval_runner import _step_canary

        out = io.StringIO()
        with redirect_stdout(out):
            _step_canary("arm", [self._episode(104), self._episode(63)],
                         [{"step_to_success": 104}, {"step_to_success": 63}])
        self.assertIn("agrees with steps on all 2", out.getvalue())

    def test_disagreement_names_both_readings(self):
        """Either the per-step attribution is off, or this harness keeps
        stepping past success -- which would change what `steps` means on
        every row written to date."""
        import io
        from contextlib import redirect_stdout

        from refractal.execute.vla_eval_runner import _step_canary

        out = io.StringIO()
        with redirect_stdout(out):
            _step_canary("arm", [self._episode(220)], [{"step_to_success": 104}])
        printed = out.getvalue()
        self.assertIn("104 vs 220", printed)
        self.assertIn("keeps stepping past success", printed)

    def test_failures_are_not_compared(self):
        """A failure has no success step, and `steps` is its budget."""
        import io
        from contextlib import redirect_stdout

        from refractal.execute.vla_eval_runner import _step_canary

        out = io.StringIO()
        with redirect_stdout(out):
            _step_canary("arm", [self._episode(220, success=False)],
                         [{"step_to_success": None}])
        self.assertEqual(out.getvalue(), "")
