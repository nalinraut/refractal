"""A filter for the build tests.

Deliberately trivial, and deliberately *not* pure geometry: it stands in for
`so101_eval.filters:reachable`, which needs forward kinematics and is therefore
the reason filters are evaluated by `build` rather than by `resolve`.
"""


def reachable(scenario):
    """Drop poses outside a plausible 5-DOF workspace."""
    return abs(scenario.get("vial_y", 0.0)) <= 0.05


def rejects_everything(scenario):
    return False


def explodes(scenario):
    raise RuntimeError("kinematics solver did not converge")
