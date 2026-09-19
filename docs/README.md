# Refractal

Refractal runs the same experiment against two checkpoints and tells you whether
the difference is real.

It sits on top of an evaluation harness rather than replacing one. The harness
answers *what did this checkpoint score*; Refractal answers *did my change help*,
which needs paired statistics over episodes that are content-addressed rather
than positional.

**New here?** [getting-started.md](getting-started.md) — five commands, about a
minute, ending on a real verdict. No GPU, no simulator, no checkpoints.

## Using it

| | read it when |
|---|---|
| [getting-started.md](getting-started.md) | First. Install to verdict, start to finish. |
| [writing-a-catalog.md](writing-a-catalog.md) | Describing your own experiment. What goes in each of the five files, how to decide what is a scene and what is a task, and which edits invalidate existing results. |
| [reading-a-comparison.md](reading-a-comparison.md) | Looking at a verdict. Every number in the output, what it means, and when not to trust it. |
| [execution-mode.md](execution-mode.md) | Choosing between `serial` and `concurrent`. |

## Connecting a simulator

| | read it when |
|---|---|
| [adapter-contract.md](adapter-contract.md) | Writing an adapter. Five methods, their signatures, and what each owes. |
| [adapter-example.md](adapter-example.md) | Alongside it. A complete adapter with the catalog that drives it. |

## Running at scale

| | read it when |
|---|---|
| [backend-vla-eval.md](backend-vla-eval.md) | Driving `vla-eval` against real model servers. |
| [backend-compose.md](backend-compose.md) | One container per worker. |

## Maintaining it

| | read it when |
|---|---|
| [releasing.md](releasing.md) | Publishing. |

## Commands

| | |
|---|---|
| `refractal init DIR` | write a working example catalog |
| `refractal build CATALOG` | establish facts that need the engine; writes `build.lock` |
| `refractal plan CATALOG --hardware ID` | compile a catalog into `plan.json` |
| `refractal run PLAN` | execute it |
| `refractal render PLAN` | write a deployment file from a plan |
| `refractal compare RESULTS PLAN_ID` | the verdict |
| `refractal explain BEFORE AFTER` | why two plans are different experiments |

## Scripts

| | |
|---|---|
| `scripts/test.sh` | the test command, used by CI too |
| `scripts/verify_harness_claims.py` | six assumptions about `vla-eval`; run on every version bump |
| `scripts/verify_wheel.py` | the built wheel against the source tree |
| `scripts/lint_provenance_claims.py` | refuses a comment claiming a value was measured when `measured_at` is empty |
