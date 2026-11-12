"""The mechanical properties a backend must have, and that vla-eval has them.

The protection against a slice-holding worker lived in one adapter by good
fortune. The planner's floor is the primary fix; this is the second layer, and
it needs to be REQUIRED rather than present.
"""

import unittest

from refractal.conformance import (
    REQUIREMENTS,
    ConformanceError,
    require_index_contract,
)
from refractal.execute.vla_eval import check_index_contract


class TheVlaEvalBridgeConforms(unittest.TestCase):
    def test_it_passes_every_requirement(self):
        """Run the suite rather than the one requirement, so a new entry
        applies to the shipped backend without anyone remembering to."""
        for name, requirement in REQUIREMENTS:
            with self.subTest(requirement=name):
                requirement(check_index_contract)


class TheRequirementCanFail(unittest.TestCase):
    """Both directions, because a requirement that only confirms refusals is
    satisfied by a backend that refuses everything."""

    def test_a_backend_that_checks_nothing_is_caught(self):
        with self.assertRaises(ConformanceError) as caught:
            require_index_contract(lambda episodes, scenarios: None)
        message = str(caught.exception)
        self.assertIn("[5, 6, 7, 8, 9]", message)
        self.assertIn("never ran", message)

    def test_a_backend_that_refuses_everything_is_caught(self):
        """Otherwise `raise` would be a conforming implementation."""
        def refuse_all(episodes, scenarios):
            raise ValueError("no")

        with self.assertRaises(ConformanceError) as caught:
            require_index_contract(refuse_all)
        self.assertIn("exactly [0, 1, 2, 3]", str(caught.exception))
        self.assertIn("is not conforming, it is unusable", str(caught.exception))

    def test_each_bad_case_is_caught_on_its_own(self):
        """A backend might catch the obvious slice and miss the near-miss, so
        every case has to fail independently rather than as a set."""
        for bad in ([5, 6, 7, 8, 9], [0, 1, 2, 4], [1, 2, 3]):
            with self.subTest(indices=bad):
                def admit(episodes, scenarios, allow=tuple(bad)):
                    wanted = sorted(
                        scenarios[e.scenario_hash]["init_state_index"]
                        for e in episodes
                    )
                    # Refuses everything EXCEPT this one bad case.
                    if tuple(wanted) != allow:
                        raise ValueError("refused")

                with self.assertRaises(ConformanceError):
                    require_index_contract(admit)

    def test_a_requirement_that_stops_being_collected_would_be_visible(self):
        """REQUIREMENTS is a named tuple rather than a prefix scan, because a
        requirement that is no longer collected passes by not running."""
        self.assertIn("index contract", dict(REQUIREMENTS))
        self.assertTrue(all(callable(fn) for _, fn in REQUIREMENTS))


if __name__ == "__main__":
    unittest.main()
