# `--backend compose`

A renderer over the `--worker` entrypoint. Nothing about how episodes execute
changes: both backends call the same loop, and a test asserts they produce
identical episode ids.

## What was already there, and what was not

The brief said the loop, `--worker`, and the plan's worker assignments were all
built and tested. Two of three: **`--worker` did not exist.** `run --help` had no
such flag and the backends were `{local, vla-eval}`, so step 1 would have
rendered a command nothing could execute.

Built as `restrict_to_worker`, a **filter** rather than a different plan:
`plan_id`, `catalog_hash`, every scene hash and every episode id are untouched.
Accounting *is* recomputed — leaving `total_episodes` describing the whole plan
would make a per-worker progress line lie — and scenes keeping no workers are
dropped, because a backend iterating scenes would construct one for nothing and a
LIBERO scene costs twelve seconds.

## The renderer refuses plans it cannot run

The same move as `plan` refusing a VRAM overcommit: a precondition that is a pure
function of an artifact, checked at the cheapest point that can see it, before
anything is spent. There the artifact is the catalog and the cost is a GPU; here
it is the plan and the cost is an 8 GB pull plus a simulator construction, twelve
times over.

The index contract is a pure function of the plan, which makes the renderer the
cheapest place it can possibly be checked. On the real LIBERO plan: **540 of 600
groups refused across twelve workers.** Without the check the renderer emits
twelve services, eleven of which start a container, pull an 8 GB image and
construct a simulator before dying on preflight — with the answer having been in
the plan the whole time.

## Where parallelism comes from, which is not where I expected

**The planner splits a scene's *scenarios* across workers.** The vla-eval backend
needs a worker to own whole zero-based ranges, so a multi-worker single-scene plan
is refused — and LIBERO-Spatial is now correctly **one** scene, so a single-suite
run has exactly one worker and Compose buys nothing.

Parallelism for this backend therefore comes from **multiple scenes**, not multiple
workers per scene. The demonstration below uses two suites for that reason.

**The fan-out was never real, and calling its removal a regression would be
wrong.** The ten-scenes-one-task shape split one compiled model into ten scenes,
and the twelve services were an artifact of that error. Fixing the model removed
something that should not have existed; nothing was lost.

Which matters for how the open question below is read. It is not "how do we get
the parallelism back". It is **"may a scene's episodes be split at all"** — a
question that was always there and that the wrong scene model was hiding, because
ten fake scenes never had to answer it.

This is a genuine tension between two correct models rather than a defect —
Refractal's worker unit is a scene, Compose's is a container, and a single-suite
run has one of each. Written up as an open planner question in
[planner-question-worker-unit.md](planner-question-worker-unit.md), including the
measured reason task-sharding would give up nothing (the harness's environment
reuse unit is already the task, not the scene) and the reason the obvious fix is
not obviously right (it would make the worker unit engine-dependent, and `resolve`
must never branch on engine).

## Four permission failures, in order

Every one of them was a non-root container meeting a root-built image, and every
one reads as something else.

| symptom | cause |
|---|---|
| `EOFError` while LIBERO prompts on stdin | `HOME=/tmp` moved where LIBERO looks for `config.yaml`; the base image wrote it to `/root/.libero`, which a non-root user cannot read anyway |
| `Could not find platform independent libraries` | `uv venv --python 3.11` installed a managed CPython under `/root/.local`, mode 700, so the venv's `python` symlinked into an unreadable directory |
| `PermissionError: vla_eval/runners/action_buffer.py` | the base image's `/workspace/src` files are mode `600`, and the harness is installed editable |
| `PermissionError: '/scratch'` | `--harness-output /scratch` named a directory that did not exist, and a non-root process cannot create one at `/` |

None of these is interesting individually. Together they are the concrete argument
for a decision taken abstractly much earlier: that the uid is a **render-time**
setting, emitted as a literal, and that a container runs as a real user rather
than as root. Three of the four are invisible to a root run, so a container tested
as root and given `user:` afterwards would have failed on somebody else's machine
with no clue attached.

## Answers to the three questions

### Does the progress-file fix hold across containers?

**It did not hold, and containers are not why.** The scratch path was
`output_dir/checkpoint/task/seedN` with no worker id, so two workers on one scene
that split scenarios share the key. The default LIBERO plan has twelve workers per
scene, and one container per worker would have collided at exactly that boundary.
The worker id is now in the path.

It needs no per-worker *mount*: `/scratch` is container-local and holds only the
harness's own progress files, never Refractal's results.

`ResultWriter` is safe. Staging names are `uuid.uuid4().hex` — not a pid, not a
timestamp — so two containers writing into one mounted `/results` cannot collide.

### Measured wall clock

Same plan, 8 episodes, two scenes, two arms, warm servers:

| | wall clock |
|---|---|
| single process, `--backend vla-eval` | **45 s** |
| two containers, `--backend compose` | **34 s** |

A 1.32× speedup on two workers, not 2×. The workers are unequal — one scene is
LIBERO-Object at 280 max steps and the other Spatial at 220 — so the makespan is
the slower container, and both arms still serialise against two shared model
servers. The number is small enough that it is a demonstration rather than a
benchmark; it is reported because an assertion would have been worse.

### Anything in the plan the renderer needed and could not find

**Nothing.** Every field the renderer reads — worker ids, cpusets, engine,
`resource_shape.memory_mb`, checkpoint ids — was present and populated. This is
the first consumer of `cpuset` and `device`, which were written into every plan
and never read, and they were right.

Which is the opposite of the `max_steps` and `external.params` findings, and worth
saying: those were fields covered by a hash and never exercised by a consumer.
The placement fields were also never exercised by a consumer, and they were fine.
Being unexercised is a risk, not a defect.

## Verified

- 8 episodes, two containers, 0 infra failures, `docker compose` exit 0.
- **The same episode ids as one process**, and the same scene and task hashes. The
  backend changes where episodes run, not which exist.
- One `session_id` per container, which is correct: a session is provenance of the
  observation, and two containers are two sessions.
- 17 renderer tests, none of which need Docker.
