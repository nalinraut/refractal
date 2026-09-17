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

## The loop works

Smoke, 4 episodes, 2 harness invocations, warm servers:

```
  pi0   2d9030fe8b9c success=False steps= 220    9.3s infra=False reason=policy_failure
  pi0   bee8e5a8e5bf success=False steps= 220    8.3s infra=False reason=policy_failure
  pi05  2d9030fe8b9c success=True  steps=  74    3.6s infra=False reason=None
  pi05  bee8e5a8e5bf success=True  steps=  76    3.2s infra=False reason=None
```

Everything the design asks for is in the rows: distinct `episode_id`s, Hive
partitioning by checkpoint and scene, deterministic part names, `server_url`
naming which server answered, `harness_version` and `harness_surface` both
populated, and `is_infra_failure=False` with `failure_reason="policy_failure"` —
so the denominator logic has a real case to work on rather than a simulated one.

Both failures hit the 220-step cap, which is `MAX_STEP_MAPPING["libero_spatial"]`
and matches the catalog's `max_steps`.

## The thing worth being careful about

pi0 at 0/2 against pi05 at 2/2 is the shape of a result, and on two episodes it
is not one. It is also the exact ambiguity this project exists to resolve: "pi0
is worse here" and "pi0 is misconfigured here" produce identical rows.

What has been ruled out by reading rather than by hoping:

* Both checkpoints declare the same `input_features`, `n_action_steps: 50`,
  `chunk_size: 50` and `empty_cameras: 1`. The two arms are configured
  identically — same `image_keys`, same `state_key`, same `chunk_size: 10`
  override — so nothing here advantages one over the other.
* Both get `send_wrist_image: true` and `send_state: true`, which the harness
  defaults to false and which both checkpoints need.

What is not ruled out: that `chunk_size: 10` against a checkpoint trained with
50 costs pi0 more than pi05. The harness's own configs differ on exactly this
point — `pi05_libero.yaml` sets 10, `pi0.yaml` sets null — which is a hint, not
an answer.

## It was the state convention, and it is in `scene_hash`

pi0 on the same task and the same eight init states:

| configuration | success |
|---|---|
| `chunk_size: 10` (the catalog's) | **0.0% (0/8)** |
| `chunk_size: 50` (the policy's own default) | **0.0% (0/8)** |
| `quat_no_antipodal: true` | **50.0% (2/4)** |

`LIBEROBenchmark` converts the end-effector quaternion to axis-angle for the
proprioceptive state, and ships two implementations of that conversion:

```python
self._quat_to_aa = _quat_to_axisangle_robosuite if quat_no_antipodal else quat_to_axisangle
```

`_quat_to_axisangle_robosuite` does no antipodal normalisation; the default does.
A quaternion and its negation are the same rotation, so both are "correct" — and
they hand the policy different numbers. The flag defaults to `False` and appears
in none of the shipped configs.

This is the 97.8%-to-42% failure from arXiv 2603.13966v2 SS III-B, which the
catalog cites as the reason `server_args` are in `plan_id` — reproduced here at
full strength, from the other side: not a server argument but a benchmark one.
A working policy reads as a 0% policy, and the run completes, reports cleanly,
and writes 8 rows that say `policy_failure`.

### And the identity scheme did not actually cover it

`quat_no_antipodal` is a benchmark constructor argument, so in this catalog it
lives in `external.params`. The claim in the paragraph above — that it is
therefore in `scene_hash` — was **false when written**.

`build` computed an external scene's hash as `hash_obj(facts)`: the probe's facts
alone. The probe reports what LIBERO contains and knows nothing about how this
catalog asks for it. So adding the flag left `plan_id` **unchanged**, and a run
with it and a run without it would have joined into one comparison and been
averaged.

Caught by adding the flag and checking `plan_id`, which did not move. Not by the
test suite: one test recomputed the formula (`assertEqual(scene_hash,
hash_obj({provider, digest}))`), which agrees with whatever the formula omits,
and another asserted this exact property correctly — against
`external_scene_ref_key`, the helper, while `build` hashed something else.

Fixed: `hash_obj({"facts": facts, "catalog_ref": external_scene_ref_key(scene)})`.
The two configurations now produce `plan_id` `1b441de8…` and `0cf5cbc3…`. Written
up in `patterns.md` as "A test that restates the implementation".

So the property held in the end — but it held because the run went looking, not
because the design was already right. Worth being exact about which of those
happened.

### pi05 under the corrected convention

4/4, against 2/2 at the default. The flag is right for both arms, which matters:
it is a scene-level setting and cannot be varied per arm without making the two
arms different scenes.

What it does not settle: 50% on four episodes is not 96%, and this is not the
reference environment. There may be more wrong. But the difference between 0/8
and 2/4 is not a small-sample artefact, and "pi0 is worse than pi05" would have
been the wrong conclusion to draw from the first run.

## `compare` on real rows

Exercised on the smoke results before the measurement run, because a path that
has never run on real data is not a path:

```
  libero-spatial-0 / libero-spatial-0-task   (2 scenarios)
    rates:  pi0: 0.0%   pi05: 100.0%    (baseline pi0)
                        pi05 fails   pi05 succeeds
      pi0 fails                0             2
      pi0 succeeds             0             0
    design effect on the paired difference: 1.00 (observed Var 0.00000 vs
      binomial 0.00000, 0 scenarios, 0.0 seeds each)
    pi0 -> pi05: +1.000 [+1.000, +1.000]  p=0.0004 holm=0.0004
      McNemar p=0.5000  -> improved
      note: 2 scenarios is below 200, ...
      note: the rate moved but few scenarios flipped their majority
        (McNemar p=0.500). A uniform shift, not a set of scenarios breaking --
        read the 2x2 before acting.
```

Two things worth noting in that output.

**"0 scenarios, 0.0 seeds each"** is correct and is the argument for the
measurement run. With one replicate per scenario there is no within-cell
variance to estimate, so the design effect is 1.00 by construction rather than by
measurement. A design effect needs repeats; that is what `seeds: 3` buys.

**The McNemar note fires.** The rate moved by a full 100 points and McNemar says
p=0.50, because with two concordant-direction pairs there is nothing to test.
The note says to read the 2x2 before acting, which is right, and it is the case
the note was written for — arriving unprompted on the first real data.

## The measurement run

120 episodes: 2 scenes x 10 init states x 3 replicates x 2 arms, one worker per
scene, servers warm, `quat_no_antipodal: true`.

Per-group success (each group is one scene, one arm, one replicate, 10 init
states) came out with a shape worth naming before the statistics:

* **pi0 varies across replicates** — 80%, 60%, 50% on the same ten init states.
  Same scene, same task, same starting configurations, three runs. That spread is
  the flow-matching noise draw, and it is the thing a design effect is built to
  account for.
* **pi05 does not vary** — 10/10 in every group.

Which sets up the one asymmetry that matters for reading the output: the paired
difference has real variance contributed by one arm and none by the other. A
design effect computed on that is measuring pi0's stochasticity, not a shared
scenario effect, and an ICC for pi05 is estimated on a column with no variance at
all.
