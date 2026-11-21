"""Which server process produced a row is recorded from that episode's own
responses; an episode served by two processes is refused, and a run warns when
replicates of one (checkpoint, scenario) span processes (audit finding 7)."""

import contextlib
import io
import unittest

from tests.test_vla_eval_loop import FakeHarness, LoopCase, make_plan

from refractal.execute.vla_eval import BridgeError, boot_id_warnings, server_identity

A, B = "a" * 32, "b" * 32


class TestBootIdWarnings(unittest.TestCase):
    def row(self, boot, scenario="s0", checkpoint="pi0"):
        return {"checkpoint_id": checkpoint, "scenario_hash": scenario, "server_boot_id": boot}

    def test_one_process_per_state_is_silent(self):
        self.assertEqual(boot_id_warnings([self.row("a"), self.row("a"), self.row("b", "s1")]), [])

    def test_two_processes_for_one_state_is_named(self):
        (line,) = boot_id_warnings([self.row(A), self.row(B)])
        self.assertIn("2 server processes", line)

    def test_unknown_alongside_known_is_named(self):
        (line,) = boot_id_warnings([self.row("a"), self.row(None)])
        self.assertIn("cannot be ruled out", line)


class TestTheEpisodesOwnBootId(LoopCase):
    def run_quietly(self, plan, harness):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            summary = self.run_loop(plan, harness, only=("pi0",))
        return summary, out.getvalue()

    def test_the_row_takes_the_episodes_boot_id_not_config(self):
        import refractal.execute.vla_eval_runner as runner

        saved = runner.server_identity
        runner.server_identity = lambda url: server_identity(
            url, fetch=lambda u: {"boot_id": B, "response_metadata": {}})
        self.addCleanup(setattr, runner, "server_identity", saved)
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        self.run_quietly(plan, FakeHarness(boot_ids=lambda call, index: [A]))
        self.assertEqual({r["server_boot_id"] for r in self.rows(plan)}, {A})

    def test_an_episode_served_by_two_processes_is_refused_and_writes_nothing(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        with self.assertRaises(BridgeError) as ctx:
            self.run_quietly(plan, FakeHarness(boot_ids=lambda call, index: [A, B] if index == 1 else [A]))
        self.assertIn("more than one model-server process", str(ctx.exception))
        self.assertEqual(self.rows(plan), [])

    def test_an_unreported_boot_id_is_null(self):
        plan = make_plan(scenarios=2, checkpoints=("pi0",))
        self.run_quietly(plan, FakeHarness(boot_ids=lambda call, index: None))
        self.assertEqual({r["server_boot_id"] for r in self.rows(plan)}, {None})

    def test_a_restart_between_replicates_warns_loudly(self):
        plan = make_plan(scenarios=2, seeds=(0, 1), checkpoints=("pi0",))
        summary, printed = self.run_quietly(plan, FakeHarness(boot_ids=lambda call, index: [A if call == 1 else B]))
        self.assertEqual({r["server_boot_id"] for r in self.rows(plan)}, {A, B})
        self.assertEqual(len(summary.boot_id_warnings), 2)
        self.assertIn("did not all run on one server process", printed)

    def test_one_process_throughout_is_silent(self):
        plan = make_plan(scenarios=2, seeds=(0, 1), checkpoints=("pi0",))
        summary, printed = self.run_quietly(plan, FakeHarness(boot_ids=lambda call, index: [A]))
        self.assertEqual(summary.boot_id_warnings, [])
        self.assertNotIn("WARNING", printed)
