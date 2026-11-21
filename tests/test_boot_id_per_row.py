"""Which server process produced a row is recorded, and a run warns when replicates
of one (checkpoint, scenario) span processes (audit finding 7: a restarted server
returns different chunks for the same observation and seed)."""

import contextlib
import io
import itertools
import unittest

from tests.test_vla_eval_loop import FakeHarness, LoopCase, make_plan

from refractal.execute.vla_eval import boot_id_warnings, server_identity


def _config(boot):
    return {"boot_id": boot, "response_metadata": {"server_env": {"env_digest": "sha256:e"}}}


class TestBootIdWarnings(unittest.TestCase):
    def row(self, boot, scenario="s0", checkpoint="pi0"):
        return {"checkpoint_id": checkpoint, "scenario_hash": scenario, "server_boot_id": boot}

    def test_one_process_per_state_is_silent(self):
        self.assertEqual(boot_id_warnings([self.row("a"), self.row("a"), self.row("b", "s1")]), [])

    def test_two_processes_for_one_state_is_named(self):
        (line,) = boot_id_warnings([self.row("a" * 32), self.row("b" * 32)])
        self.assertIn("2 server processes", line)

    def test_unknown_alongside_known_is_named(self):
        (line,) = boot_id_warnings([self.row("a"), self.row(None)])
        self.assertIn("cannot be ruled out", line)

    def test_server_identity_records_the_boot_id(self):
        self.assertEqual(server_identity("ws://h:1", fetch=lambda u: _config("abc"))["server_boot_id"], "abc")


class TestTheRunWarnsAcrossProcesses(LoopCase):
    def run_with_boot_ids(self, boots):
        import refractal.execute.vla_eval_runner as runner

        ids = iter(boots)
        saved = runner.server_identity
        runner.server_identity = lambda url: server_identity(url, fetch=lambda u: _config(next(ids)))
        self.addCleanup(setattr, runner, "server_identity", saved)
        plan = make_plan(scenarios=2, seeds=(0, 1), checkpoints=("pi0",))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            summary = self.run_loop(plan, FakeHarness(), only=("pi0",))
        return plan, summary, out.getvalue()

    def test_a_server_restart_between_replicates_warns_loudly(self):
        plan, summary, printed = self.run_with_boot_ids(["a" * 32, "b" * 32])  # one per seed's invocation
        self.assertEqual({r["server_boot_id"] for r in self.rows(plan)}, {"a" * 32, "b" * 32})
        self.assertEqual(len(summary.boot_id_warnings), 2)  # both states
        self.assertIn("did not all run on one server process", printed)

    def test_one_process_throughout_is_silent(self):
        plan, summary, printed = self.run_with_boot_ids(itertools.repeat("a" * 32))
        self.assertEqual(summary.boot_id_warnings, [])
        self.assertNotIn("WARNING", printed)
