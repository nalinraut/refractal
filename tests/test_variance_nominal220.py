"""The variance split on real data: LIBERO-Spatial nominal-220 (audit finding 13).

Per-state success counts out of 10 replicates, 100 states, in the same state
order for both policies, copied from the q-predicts nominal-220 rows
(refractal-libero experiments/repro/q-predicts/runs/nominal-220). pi0 succeeded
on 707 of 1,000 episodes, pi0.5 on 978.

With the s/(s-1) correction in both terms, pi0's within-state share of outcome
variance is 0.8676: within = 1.8011e-1, between = 2.7495e-2. The old estimator
corrected the between term only and reported 0.855.
"""

import unittest

from refractal.compare.pairing import build_units
from refractal.compare.variance import measure_variance

PI0 = [9, 6, 0, 9, 8, 7, 9, 8, 8, 9, 7, 10, 7, 7, 4, 6, 9, 7, 5, 6, 9, 10, 9, 10, 8, 9, 9, 10, 7, 9,
       10, 8, 5, 10, 8, 7, 9, 5, 9, 9, 7, 6, 4, 4, 8, 6, 9, 8, 5, 9, 2, 4, 7, 8, 8, 8, 7, 8, 4, 6,
       10, 6, 7, 4, 10, 9, 9, 10, 5, 7, 8, 7, 8, 8, 9, 8, 10, 6, 7, 8, 7, 5, 3, 4, 0, 7, 8, 6, 7, 7,
       6, 5, 6, 5, 7, 8, 5, 5, 3, 8]
PI05 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 7, 10, 10, 10, 10, 5, 10, 8, 10, 9, 10, 9,
        10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 6, 10,
        10, 10, 10, 10, 10, 10, 9]
K = 10


def nominal_220_rows():
    rows = []
    for state, counts in enumerate(zip(PI0, PI05)):
        for checkpoint, successes in zip(("pi0", "pi05"), counts):
            for seed in range(K):
                rows.append({
                    "scene_id": "libero-spatial", "scene_hash": "h", "task_id": f"task-{state // 10}",
                    "task_hash": f"th{state // 10}", "scenario_hash": f"s{state}",
                    "checkpoint_id": checkpoint, "seed": seed, "session_id": "q-nominal-220",
                    "harness_version": "vla-eval 0.6.0", "harness_surface": "v",
                    "is_infra_failure": False, "success": seed < successes,
                })
    return rows


class TestTheVarianceSplitOnNominal220(unittest.TestCase):
    def setUp(self):
        self.assertEqual((len(PI0), sum(PI0), len(PI05), sum(PI05)), (100, 707, 100, 978))
        units = build_units(nominal_220_rows(), checkpoints=["pi0", "pi05"]).units
        self.report = measure_variance(units, "pi0", "pi05")

    def test_pi0_within_state_share_is_86_8_percent(self):
        self.assertAlmostEqual(self.report.within_share("pi0"), 0.8676, places=4)
        self.assertEqual(round(100 * self.report.within_share("pi0"), 1), 86.8)

    def test_both_terms_are_unbiased(self):
        self.assertAlmostEqual(self.report.within_var["pi0"], 0.180111, places=6)
        self.assertAlmostEqual(self.report.between_var["pi0"], 0.027495, places=6)
