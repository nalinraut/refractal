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
are defined here so they have one written definition, and that is enforced
rather than asserted: ``scripts/verify_harness_claims.py`` lifts the server's
``_call_seed`` out of ``model_servers/lerobot.py`` by AST, runs it, and refuses
if it disagrees with ``call_seed`` below on any of 35 (seed, step) pairs. It
also refuses if Refractal names a scheme the server does not implement.

Checked numerically rather than by reading the source, because two spellings
of the same arithmetic are equal while a changed constant inside an
unchanged-looking line is not -- and nothing in a run would notice the two
drifting apart, since the server computes ``call_seed`` and only it knows the
step. The rows would record a ``policy_seed`` and a scheme, and the noise
would come from a different rule.

``episode-id-sha256/v2``
    episode_seed = first 8 bytes, big-endian, of
                   sha256(b"refractal.policy_seed/v2\\0" + episode_id
                          + b"\\0" + str(replicate)), >> 1
    call_seed    = as v1.

    ``replicate`` is the planned episode's ``seed`` -- the replicate index the
    plan already carries (``seed_base .. seed_base + seeds - 1``, both in
    ``plan_id``). v1 already gave distinct noise per replicate, because
    ``episode_id`` covers the seed; v2 names the replicate in the derivation so
    a replicate is addressable on its own, and so v1 and v2 never share a seed
    for the same episode.

SHA-256 rather than ``hash()``: Python's string hash is salted per process.
The ``>> 1`` keeps the value inside a signed 64-bit range, which is what
``torch.Generator.manual_seed`` accepts without wrapping.
"""

from __future__ import annotations

import hashlib

SCHEMES = ("episode-id-sha256/v1", "episode-id-sha256/v2")


def episode_seed(
    episode_id: str, scheme: str = "episode-id-sha256/v1", replicate: int | None = None
) -> int:
    if scheme == "episode-id-sha256/v1":
        if replicate is not None:
            raise ValueError("episode-id-sha256/v1 takes no replicate; use v2")
        payload = b"refractal.policy_seed/v1\0" + episode_id.encode()
    elif scheme == "episode-id-sha256/v2":
        if replicate is None:
            raise ValueError("episode-id-sha256/v2 needs the replicate index")
        payload = (b"refractal.policy_seed/v2\0" + episode_id.encode()
                   + b"\0" + str(int(replicate)).encode())
    else:
        raise ValueError(f"unknown policy seeding scheme {scheme!r}; known: {SCHEMES}")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 1


def call_seed(episode_seed_value: int, step: int) -> int:
    payload = f"{episode_seed_value}:{step}".encode()
    digest = hashlib.sha256(b"refractal.policy_call_seed/v1\0" + payload).digest()
    return int.from_bytes(digest[:8], "big") >> 1


__all__ = ["SCHEMES", "call_seed", "episode_seed"]
