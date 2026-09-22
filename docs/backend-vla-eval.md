# Running against vla-eval

`--backend vla-eval` drives
[**`allenai/vla-evaluation-harness`**](https://github.com/allenai/vla-evaluation-harness)
against running model servers and writes Parquet.

The harness is the Allen Institute for AI's, under Apache-2.0, and it does the
work: the model-server protocol, the benchmark adapters, the observation and
action specifications, and the episode loop are all theirs. Refractal chooses
which episodes to run, gives them identities and compares the results. Cite the
harness in anything you publish.

```console
$ pip install refractal[vla-eval]
```

Requires `vla-eval>=0.6.0`.

## Before the run

**Start the model servers yourself and warm them.** One per checkpoint, each on
its own address.

```console
$ vla-eval serve -c configs/model_servers/<family>/<config>.yaml --address 0.0.0.0:8000
$ vla-eval serve -c configs/model_servers/<family>/<config>.yaml --address 0.0.0.0:8001
```

Then send at least one episode through each before the measured run. A first
inference can take minutes (model load, CUDA context, JIT compilation), and the
default per-step timeout is 30 seconds, so an unwarmed server errors every
episode of whichever checkpoint starts first.

Refractal does not start servers. A supervised server would pay its warm-up
inside the first episode of whichever checkpoint ran second, which is inside the thing
being measured.

## The run

```console
$ refractal run plan.json --backend vla-eval \
    --server baseline=ws://localhost:8000 \
    --server candidate=ws://localhost:8001 \
    --catalog catalog -o results
```

Every checkpoint in the plan needs a `--server`. Two checkpoints sharing one URL
is refused: that is a comparison against itself, and it produces a difference
near zero with a tight interval and every internal check passing.

| flag | |
|---|---|
| `--server CKPT=URL` | repeat once per checkpoint |
| `--harness-output DIR` | scratch for the harness's own files, default `./vla-eval-output` |
| `--worker ID` | run only one worker's episodes |
| `--no-resume` | re-run episodes that already have results |

One harness invocation runs per worker, task, checkpoint and seed. Resume is at
that granularity: a partly-finished group re-runs whole, and its part file is
replaced rather than added to.

## What the plan must satisfy

**Scenario indices start at zero and are contiguous.** The harness counts
episodes from zero within a task, so a worker holding a later slice of a scenario
range would run the early ones while every row claimed the late ones. `refractal
run` refuses this rather than recording it.

Set `partition_unit: task` on the scene's resource shape and the planner will not
produce such a worker. See [writing a catalog](writing-a-catalog.md).

**One task per harness invocation.** The harness's work list can be constrained
to one task at a time. The loop handles this by splitting, but a worker holding
two tasks with the same instruction is refused.

## What gets recorded

One Parquet row per episode, partitioned by comparison, checkpoint and scene.
Beyond the identity fields:

| column | |
|---|---|
| `server_url` | which server answered |
| `harness_version` | what was installed |
| `harness_surface` | digest of the harness modules Refractal depends on |
| `is_infra_failure` | a crash or timeout, not a policy failure; excluded from success denominators |
| `concurrent_with` | other checkpoints running at the same time, under `execution_mode: concurrent` |

`harness_surface` gates comparisons. If it differs between checkpoints, `compare` exits
2 and names the files that changed. Override with `compare --allow-harness-mismatch`
after reading them.

## Frames

`--video` asks the harness to record, and writes one WebP sprite strip per
episode into `frames/` beside the results -- every `--frame-every`-th frame, laid
left to right in one image. `--frame-every` defaults to 10.

Decide before the run. Recording is off by default and cannot be added
afterwards without running the episodes again, so the choice of rate and of which
episodes get video is made when the run starts.

A worker asked for frames that produces none **fails**. The frames come back
from the harness keyed by its own episode numbering while the rows are keyed by
content-addressed ids, and an empty `frames/` is indistinguishable from a run
nobody asked to record -- which is the sort of thing found after the GPU time is
spent. The receipt is checked instead.

Recording does not change what the experiment is. A recorded run and an
unrecorded one of the same plan share a `plan_id` and join in one comparison,
which is why this is a flag and not a catalog field.

## Keeping the pin safe

Refractal makes six assumptions about the harness that are checked rather than
assumed:

```console
$ python scripts/verify_harness_claims.py
Verifying harness claims against installed vla-eval 0.6.0

  [ok  ] the recorder gate is `self._store is None`               found in orchestrator.py
  [ok  ] no model server reads db_path                            no callers
  [ok  ] _ALL_RECORD_FIELDS is per-benchmark and advisory         validated via getattr, 17 benchmark(s) declare one
  [ok  ] spec cross-validation is inline in _run_benchmark_inner  36 lines, inline
  [ok  ] the work-item loop is inline, not an overridable method  inline
  [ok  ] the model server address is configurable, not hardcoded  overridable from YAML

All 6 claims hold.
```

Run it on every version bump. Each failure names what it was holding up. Exit 0
if all hold, 1 if any broke, 2 if the harness is not importable.

It accepts a source checkout too:

```console
$ python scripts/verify_harness_claims.py /path/to/vla-evaluation-harness
```

## Environment notes

The benchmark is imported into the same process as Refractal, so both must be
installed for one interpreter. If the benchmark you are wrapping pins an older
Python than Refractal requires (3.11+), you will need to install it into a newer
one rather than using its reference environment.

Headless rendering needs `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` and
`EGL_PLATFORM=device` for MuJoCo-based benchmarks.

## When every episode fails

In order of likelihood:

1. **An observation the policy needs is not being sent.** Benchmark constructors
   often default optional cameras and proprioception to off. Check the
   checkpoint's expected input features against the scene's `params`.
2. **A state or action convention differs.** Quaternion sign, delta versus
   absolute actions, gripper polarity. These produce a policy that moves but
   never succeeds.
3. **Scenario parameter names do not match what the adapter reads.**
4. **The servers were not warm**, and the failures are `is_infra_failure: true`
   timeouts rather than policy failures.

The first three produce a clean run with a wrong answer. Check
`is_infra_failure` first to rule out the fourth.
