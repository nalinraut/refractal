"""The policy's sampling seed, derived from what an episode IS.

`seed` on a planned episode is a repeat label on the vla-eval backend: the
harness has no seeding of its own, and a flow-matching policy draws its noise
from an unseeded global generator. Two runs sharing a ``plan_id`` therefore
shared every declared input and still drew different noise.

A catalog that declares ``run.policy_seeding`` gets a seed per episode, sent to
the model server with the episode. The scheme is named and versioned because it
is part of what the experiment is: it enters ``plan_id``, so changing the
derivation is a new experiment rather than a silent change of meaning.

``episode-id-sha256/v1``
    episode_seed = first 8 bytes, big-endian, of
                   sha256(b"refractal.policy_seed/v1\\0" + episode_id), >> 1
    call_seed    = first 8 bytes, big-endian, of
                   sha256(b"refractal.policy_call_seed/v1\\0"
                          + f"{episode_seed}:{step}"), >> 1

``episode_seed`` is computed here. ``call_seed`` is computed by the model
server, which knows the step a request was made at; it builds a fresh
``torch.Generator`` from it for that call alone, so the draw depends on
(episode, step) and never on request order or batch composition. Both halves
are defined here so they have one written definition; the server's copy is
checked against ``call_seed`` below by the verification script.

SHA-256 rather than ``hash()``: Python's string hash is salted per process.
The ``>> 1`` keeps the value inside a signed 64-bit range, which is what
``torch.Generator.manual_seed`` accepts without wrapping.
"""

from __future__ import annotations

import hashlib

SCHEMES = ("episode-id-sha256/v1",)


def episode_seed(episode_id: str, scheme: str = "episode-id-sha256/v1") -> int:
    if scheme != "episode-id-sha256/v1":
        raise ValueError(f"unknown policy seeding scheme {scheme!r}; known: {SCHEMES}")
    digest = hashlib.sha256(b"refractal.policy_seed/v1\0" + episode_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def call_seed(episode_seed_value: int, step: int) -> int:
    payload = f"{episode_seed_value}:{step}".encode()
    digest = hashlib.sha256(b"refractal.policy_call_seed/v1\0" + payload).digest()
    return int.from_bytes(digest[:8], "big") >> 1


__all__ = ["SCHEMES", "call_seed", "episode_seed"]
