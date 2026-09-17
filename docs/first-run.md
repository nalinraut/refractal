# The first real run — what the loop found on contact

Environment, measured 2026-09-17:

| thing | value |
|---|---|
| harness | vla-eval v0.6.0 (`c233542`), all 6 claims hold |
| LIBERO | 0.1.0 at `8f1084e`, all 9 claims hold, 624 asset files |
| arms | `lerobot/pi0_libero_finetuned_v044` vs `pi05_..._v044`, both bf16 per their own config |
| GPU | RTX 5090, 32,607 MiB. **Two servers resident: 18,950 MiB** |
| suite | libero_spatial, 10 tasks x 50 init states of 92 dims |

## Python 3.8 versus 3.11

LIBERO's reference image pins Python 3.8; Refractal requires >=3.11. The bridge
subclasses `Orchestrator` and the orchestrator imports the benchmark in-process,
so they cannot be in different interpreters. LIBERO runs on 3.11 here, on numpy
2.4.6 rather than the image's 1.22.4 — it works, but this is not the reference
environment and absolute success rates should not be compared with published
numbers. Both arms see the identical environment, which is what a *paired*
question needs.

## The seed does not reach the policy

`predict_action_chunk` -> `sample_actions` -> `sample_noise` ->
`torch.normal(...)` with **no generator**. So:

* pi0 **is** genuinely stochastic at inference. There is real within-scenario
  variance, which is what makes a design effect and an ICC measurable at all.
* Nothing Refractal can set controls it. The benchmark's `seed` seeds the LIBERO
  environment, which is then overwritten by an explicit init state; the model
  server has no seed parameter at all.

So **a Refractal seed is a replicate index here, not a reproducibility
guarantee**. Repeats differ; re-running "seed 0" does not reproduce it. Pairing
is unaffected — `compare` joins on (scene, task, scenario) and treats seeds as
replicates — but nothing should claim these runs are reproducible episode by
episode. Worth an upstream issue: the model server should accept a seed and use
a `torch.Generator`.

## Warm-up is not optional, and the harness knows it

First inference triggers torch.compile inductor autotuning, which takes far
longer than the 30s act timeout. Every episode of the first run errored with
`TimeoutError (act timeout=30.0s)`.

The harness's own `robodojo_pi05` server calls `_warmup()` at load for exactly
this reason, with the comment "the first infer() JIT-compiles the flow-matching
sampler (>30s), which would blow the harness act timeout and error the first
episode". `lerobot.py` does not. So the servers are warmed externally before the
measured run.

This is the warm-up asymmetry made concrete: had the servers been started by the
run, the first arm would have paid a multi-minute compile *inside the thing being
measured*, and the second would not.

## The guard that was wrong

The recorder receipt read the step buffer: empty buffer meant "the injection
failed". Every episode timed out before its first step, so the buffer was empty
and the check reported the recorder had never been consulted. It had been
consulted twice.

"Nothing was recorded" and "the injection failed" are different, and only the
second is a bridge problem. The receipt now counts recorder *constructions*.
