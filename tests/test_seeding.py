"""The policy seed: the paper's independent variable.

These assert the properties a result depends on, not that the functions run.
q(c) is the success probability from one fixed initial state over K seeded
rollouts, where the seed varies the policy's sampling noise and nothing else.
Every test here is one half of that sentence.
"""

import json
import os
import pathlib
import subprocess
import sys
import unittest

from refractal.schema.canonical import hash_obj
from refractal.schema.identity import episode_id, external_scene_ref_key
from refractal.schema.models import ScenesFile
from refractal.seeding import SCHEMES, call_seed, episode_seed

V1, V2 = "episode-id-sha256/v1", "episode-id-sha256/v2"
EID = "sha256:" + "ab" * 32

REAL_CATALOG = pathlib.Path(
    "/home/nalin/Develop/refractal-libero/experiments/repro/q-predicts/"
    "catalogs/nominal"
)


class TheSeedIsAFunctionOfTheEpisodeAndTheReplicate(unittest.TestCase):
    def test_nothing_else_reaches_it(self):
        """Not the clock, not the environment, not the order of calls.

        The failure this guards is a derivation that picks up ambient state:
        the seed would then be unreproducible while every row still claimed a
        scheme that implies it is reproducible.
        """
        first = episode_seed(EID, V2, replicate=3)
        os.environ["REFRACTAL_IRRELEVANT"] = "x"
        try:
            for _ in range(5):
                episode_seed("sha256:" + "cd" * 32, V2, replicate=9)
            self.assertEqual(episode_seed(EID, V2, replicate=3), first)
        finally:
            os.environ.pop("REFRACTAL_IRRELEVANT", None)

    def test_the_replicate_changes_it(self):
        """Otherwise K rollouts would share one noise draw and q(c) would be
        an estimate of a single sample repeated K times."""
        seeds = {episode_seed(EID, V2, replicate=r) for r in range(64)}
        self.assertEqual(len(seeds), 64)

    def test_the_episode_changes_it(self):
        """Under BOTH schemes, because both are declared and both have run.

        This covered v2 only at first, and the gap was found the way the rest
        of these were: dropping `episode_id` from the v1 branch left the test
        passing. v1 is still in SCHEMES and still identifies recorded pilot
        data, so an unguarded v1 derivation is an unguarded experiment.
        """
        other = "sha256:" + "cd" * 32
        self.assertNotEqual(episode_seed(EID, V1), episode_seed(other, V1))
        self.assertNotEqual(episode_seed(EID, V2, replicate=0),
                            episode_seed(other, V2, replicate=0))

    def test_it_survives_a_fresh_process(self):
        """SHA-256 rather than `hash()`, which Python salts per process.

        Run under a different PYTHONHASHSEED, because that is the variable
        that would expose a `hash()` based derivation -- and a seed that
        changes per process makes every recorded policy_seed a number that
        cannot be replayed.
        """
        script = (
            "from refractal.seeding import episode_seed;"
            f"print(episode_seed({EID!r}, {V2!r}, replicate=3))"
        )
        env = dict(os.environ, PYTHONHASHSEED="12345",
                   PYTHONPATH="src")
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env, cwd=pathlib.Path.cwd())
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(int(out.stdout.strip()),
                         episode_seed(EID, V2, replicate=3))


class TheTwoSchemesNeverShareASeed(unittest.TestCase):
    def test_no_collision_across_a_range_that_matters(self):
        """v1 and v2 are different experiments by plan_id, so a shared seed
        would mean two experiments drew identical noise while claiming
        different derivations."""
        v1 = {episode_seed(f"sha256:{i:064x}", V1) for i in range(256)}
        v2 = {episode_seed(f"sha256:{i:064x}", V2, replicate=r)
              for i in range(256) for r in range(4)}
        self.assertEqual(len(v1), 256)
        self.assertEqual(len(v2), 1024)
        self.assertEqual(v1 & v2, set())

    def test_the_version_prefix_is_what_separates_them(self):
        """Not the replicate: v2 at any replicate must differ from v1 on the
        same episode, including whatever replicate a reader might assume is
        "the same as v1"."""
        v1 = episode_seed(EID, V1)
        for r in range(64):
            self.assertNotEqual(v1, episode_seed(EID, V2, replicate=r))


class TheRefusalsFire(unittest.TestCase):
    def test_v1_refuses_a_replicate(self):
        with self.assertRaises(ValueError) as c:
            episode_seed(EID, V1, replicate=0)
        self.assertIn("takes no replicate", str(c.exception))

    def test_v2_refuses_a_missing_replicate(self):
        """Refused rather than defaulted to 0: a silent default would make
        every replicate of an episode share one seed, which is the exact
        failure the scheme exists to prevent."""
        with self.assertRaises(ValueError) as c:
            episode_seed(EID, V2)
        self.assertIn("needs the replicate", str(c.exception))

    def test_an_unknown_scheme_names_the_known_ones(self):
        with self.assertRaises(ValueError) as c:
            episode_seed(EID, "episode-id-sha256/v3", replicate=0)
        self.assertIn("v3", str(c.exception))
        for scheme in SCHEMES:
            self.assertIn(scheme, str(c.exception))


class TheEnvSeedIsHeldFixed(unittest.TestCase):
    """What the paper needs from the environment side: one fixed initial
    state across the K rollouts.

    CORRECTION to the claim this was written against. "Changing the env seed
    does not move the policy seed" is false, and measurably so: the env seed
    is `external.params.seed`, params are hashed into `scene_hash`, and
    `scene_hash` is hashed into `episode_id`, which is the policy seed's
    input. Changing it moves every policy seed in the run.

    That is the correct behaviour rather than a leak -- a different env seed
    is a different experiment and gets a different plan_id, so its results
    cannot pool with the old ones. The property the paper actually relies on
    is the one below: within one experiment the env seed is a constant, so it
    cannot vary with the replicate.
    """

    def _scene(self, env_seed):
        return ScenesFile.model_validate({
            "apiVersion": "refractal.dev/v1alpha1",
            "scenes": [{
                "id": "s", "engine": "mujoco", "engine_version": "3.2.0",
                "external": {
                    "provider": "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark",
                    "ref": {"suite": "libero_spatial"},
                    "params": {"suite": "libero_spatial", "seed": env_seed,
                               "send_state": True},
                },
            }],
        }).scenes[0]

    def _policy_seed(self, scene, replicate):
        digest = hash_obj({"facts": {"provider": "libero"},
                           "catalog_ref": external_scene_ref_key(scene)})
        eid = episode_id(scene_hash=digest, task_hash="th", scenario_hash="sh",
                         seed=replicate, checkpoint_id="pi0")
        return episode_seed(eid, V2, replicate=replicate)

    def test_the_env_seed_does_not_vary_with_the_replicate(self):
        """The direction that matters. One scene, ten replicates: one env
        seed, ten different policy seeds. If the env seed moved with the
        replicate, the K rollouts would be K different initial states and
        q(c) would not be a per-state quantity at all.
        """
        scene = self._scene(7)
        self.assertEqual(scene.external.params["seed"], 7)
        digests = {hash_obj({"facts": {"provider": "libero"},
                             "catalog_ref": external_scene_ref_key(scene)})
                   for _ in range(10)}
        self.assertEqual(len(digests), 1)
        self.assertEqual(len({self._policy_seed(scene, r) for r in range(10)}), 10)

    def test_a_different_env_seed_is_a_different_experiment(self):
        """Stated as a test because it is the part that was mis-stated. The
        env seed reaches the policy seed through episode_id, and the
        protection is identity, not independence."""
        self.assertNotEqual(
            external_scene_ref_key(self._scene(7)),
            external_scene_ref_key(self._scene(8)))
        self.assertNotEqual(self._policy_seed(self._scene(7), 0),
                            self._policy_seed(self._scene(8), 0))

    @unittest.skipUnless(REAL_CATALOG.exists(), "the q-predicts catalog is not here")
    def test_the_recorded_experiment_holds_the_env_seed_at_seven(self):
        """Against the catalog that produced the evidence, not a fixture."""
        import yaml

        scene = ScenesFile.model_validate(
            yaml.safe_load((REAL_CATALOG / "scenes.yaml").read_text())).scenes[0]
        self.assertEqual(scene.external.params["seed"], 7)
        run = yaml.safe_load((REAL_CATALOG / "run.yaml").read_text())["run"]
        self.assertEqual(run["policy_seeding"], V2)
        self.assertEqual(run["seeds"], 10)
        self.assertEqual(run["seed_base"], 0)


class TheCallSeedIsPerEpisodeAndStep(unittest.TestCase):
    def test_the_step_changes_it(self):
        """So a draw depends on (episode, step) and never on request order or
        batch composition."""
        base = episode_seed(EID, V2, replicate=0)
        self.assertEqual(len({call_seed(base, s) for s in range(256)}), 256)

    def test_the_episode_seed_changes_it(self):
        self.assertNotEqual(call_seed(1, 5), call_seed(2, 5))

    def test_it_stays_inside_the_signed_64_bit_range(self):
        """`torch.Generator.manual_seed` is what consumes these, and it wraps
        above 2**63 - 1 rather than refusing."""
        limit = 2 ** 63 - 1
        for r in range(32):
            seed = episode_seed(EID, V2, replicate=r)
            self.assertLessEqual(seed, limit)
            self.assertLessEqual(call_seed(seed, r), limit)


if __name__ == "__main__":
    unittest.main()


class DecisionsAreMeasuredNotDeclared(unittest.TestCase):
    """`steps / decisions` is the realized chunk length, and it is only a
    check if the two sides come from different places.

    Derived from the declared chunk size it would be a restatement of the
    declaration, and a check comparing a declaration against itself always
    passes. The server's log is the only per-inference-call record this
    backend produces -- the harness records per environment step.
    """

    def test_the_counter_reads_forward_and_does_not_double_count(self):
        import json
        import tempfile

        from refractal.execute.vla_eval import CallCounter

        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "h.jsonl"
            log.write_text("".join(
                json.dumps({"episode_id": "e1", "step": i * 10}) + "\n"
                for i in range(11)))
            counter = CallCounter()
            counter.update(str(log))
            self.assertEqual(counter.calls("e1"), 11)
            with open(log, "a") as handle:
                for i in range(11, 22):
                    handle.write(json.dumps({"episode_id": "e1", "step": i * 10}) + "\n")
            counter.update(str(log))
            self.assertEqual(counter.calls("e1"), 22)

    def test_a_half_written_line_is_kept_for_the_next_pass(self):
        """The server appends while this reads. Dropping the partial line
        would silently lose a call; parsing it would lose the same call with a
        swallowed exception."""
        import json
        import tempfile

        from refractal.execute.vla_eval import CallCounter

        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "h.jsonl"
            whole = json.dumps({"episode_id": "e1", "step": 0}) + "\n"
            second = json.dumps({"episode_id": "e1", "step": 10})
            log.write_text(whole + second[:20])
            counter = CallCounter()
            counter.update(str(log))
            self.assertEqual(counter.calls("e1"), 1)
            with open(log, "a") as handle:
                handle.write(second[20:] + "\n")
            counter.update(str(log))
            self.assertEqual(counter.calls("e1"), 2)

    def test_a_missing_log_is_not_an_error(self):
        """A server launched without one is the normal case, and `decisions`
        is then null with that reason rather than guessed."""
        from refractal.execute.vla_eval import CallCounter

        counter = CallCounter()
        counter.update("/nonexistent/hashes.jsonl")
        counter.update(None)
        self.assertIsNone(counter.calls("e1"))

    def test_the_log_path_is_placement_and_leaves_plan_id_alone(self):
        """Two runs of one experiment logging to different paths are one
        experiment."""
        from refractal.schema.canonical import hash_obj
        from refractal.schema.identity import arm_descriptor
        from refractal.schema.models import Checkpoint

        base = dict(id="a", path="p", server="s:S", server_args={})
        self.assertEqual(
            hash_obj(arm_descriptor(Checkpoint(**base))),
            hash_obj(arm_descriptor(Checkpoint(**base, action_hash_log="/tmp/h"))))

    def test_the_row_records_which_route_decisions_took(self):
        from refractal.execute.vla_eval_runner import _distributional

        class Row:
            success, steps = True, 104

        measured = dict(_distributional(Row(), {"success_samples": 0}, 220, 11)
                        ["field_provenance"])
        self.assertEqual(_distributional(Row(), {"success_samples": 0}, 220, 11)
                         ["decisions"], 11)
        self.assertIn("measured:", measured["decisions"])

        absent = dict(_distributional(Row(), {"success_samples": 0}, 220, None)
                      ["field_provenance"])
        self.assertIn("unmeasured:", absent["decisions"])
        self.assertIn("action_hash_log", absent["decisions"])

    def test_an_unrotated_log_refuses_rather_than_doubling(self):
        """The hazard of append mode. Re-running a plan without rotating the
        log puts the same episode ids in the file twice and every count
        doubles -- silently, and in the direction that makes the realized
        chunk length look half what it was. A wrong count is worse than none:
        null reads as "nobody measured it", a doubled one reads as a
        measurement.
        """
        import json
        import tempfile

        from refractal.execute.vla_eval import CallCounter

        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "h.jsonl"
            run = "\n".join(json.dumps({"episode_id": "e1", "step": i * 10})
                            for i in range(11))
            log.write_text(run + "\n")
            counter = CallCounter()
            counter.update(str(log))
            self.assertEqual(counter.calls("e1"), 11)

            with open(log, "a") as handle:
                handle.write(run + "\n")
            again = CallCounter()
            again.update(str(log))
            self.assertIsNone(again.calls("e1"))
            self.assertEqual(again.ambiguous(), ("e1",))

    def test_attribution_is_keyed_not_positional(self):
        """Two episodes interleaving in one log is the concurrency case.
        Segmenting by step reset would split them wrongly; the episode id
        Refractal itself sent does not care about interleaving.
        """
        import json
        import tempfile

        from refractal.execute.vla_eval import CallCounter

        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "h.jsonl"
            lines = []
            for step in range(0, 50, 10):          # interleaved, not grouped
                lines.append(json.dumps({"episode_id": "a", "step": step}))
                lines.append(json.dumps({"episode_id": "b", "step": step}))
            log.write_text("\n".join(lines) + "\n")
            counter = CallCounter()
            counter.update(str(log))
            self.assertEqual(counter.calls("a"), 5)
            self.assertEqual(counter.calls("b"), 5)
            self.assertEqual(counter.ambiguous(), ())
