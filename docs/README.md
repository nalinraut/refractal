# Refractal documentation

Fourteen documents, in the order someone new should read them.

Refractal is a scene-coherent placement and paired-comparison layer over
[`allenai/vla-evaluation-harness`](https://github.com/allenai/vla-evaluation-harness).
The harness answers *what did this checkpoint score*. Refractal answers
**did my change help** — which is a different question, and the difference is
paired statistics over content-addressed episodes.

## Start here

| | |
|---|---|
| [first-run.md](first-run.md) | The system pointed at two real pi0 servers for the first time. Read this before anything else: it is the shortest path from "what is this" to "what does it do", and everything it found is a thing the design exists to prevent. |
| [ten-task-run.md](ten-task-run.md) | 600 episodes, the design effect measured against real stochasticity, and the step-budget run whose result was predicted before it was run. The closest thing here to a result rather than a capability. |
| [statistics-measurements.md](statistics-measurements.md) | Why a clustered bootstrap and not a two-proportion z-test, measured rather than argued. Includes the circularity that makes the measurement partly self-referential, and how a correction to it should be stated. |

## The design, and where it was wrong

| | |
|---|---|
| [schema-review.md](schema-review.md) | The original spec, reviewed field by field, with the places it was wrong and why. |
| [design-part-12-revised.md](design-part-12-revised.md) | Part 12 of the design doc, rewritten after the review. |
| [scene-task-split.md](scene-task-split.md) | A scene is the compiled model; a goal belongs to the task. Corrects a catalog that turned the design doc's canonical cheap case into its expensive one, and closes a gap the spec did not see: a task whose goal is opaque to `instruction`, `predicate` and `predicate_args`. |
| [execution-mode.md](execution-mode.md) | Two modes, `serial` and `concurrent`, defined by episode ordering rather than by deployment. Records why `interleaved` was not built: the drift it existed to remove measured at zero. |

## Driving someone else's harness

| | |
|---|---|
| [harness-integration.md](harness-integration.md) | The contract with vla-eval. Which hooks, why no fork, what changed under the pin, and §9 — the three fields the plan was missing, all of them covered by a hash and never exercised by a consumer. |
| [compose-backend.md](compose-backend.md) | `--backend compose` as a renderer over `--worker`. Four permission failures, three of them invisible to a root run. |
| [upstream-issues/recorder-injection.md](upstream-issues/recorder-injection.md) | A draft issue for the harness: `_store` is unreachable, so `_build_recorder` cannot be used. Ready to file. |

## Open questions — two of them block a release

| | |
|---|---|
| [planner-question-worker-unit.md](planner-question-worker-unit.md) | **Blocking.** What must a worker own *whole*, as distinct from how many episodes may share a process at once. Proposes `partition_unit` as an enum. Needs an MJX scene to design against honestly. |
| [question-tier-in-identity.md](question-tier-in-identity.md) | **Blocking.** Is a smoke run the same experiment as a full run? Smaller, and it changes what "the same experiment" means. |

Both move `plan_id`, which is why they come before publishing rather than after.

## Practice

| | |
|---|---|
| [patterns.md](patterns.md) | The longest document here, and the one with the most reuse outside this project. Tests and fixtures that looked like they worked and did not — a fixture that overlapped nothing, a test that restated the implementation, a check that read a downstream symptom, a comparison at the wrong aggregation level. Each entry has the failure, the tell, and how it was caught. |
| [releasing.md](releasing.md) | The PyPI checklist, and what has to land first. Not yet: the trigger is a second person needing to install it. |

## Scripts that check claims about code we do not control

An assertion about somebody else's code does not fail when it becomes wrong. It
sits there being wrong — which is how a claim that the harness hardcodes its
server address survived nine days and four documents. So the assertions are
executable:

* `scripts/verify_harness_claims.py` — six claims about vla-eval, each carrying
  what it holds up. Run it on every pin move.
* `scripts/verify_wheel.py` — the artifact against the source tree, derived
  rather than listed.
* `scripts/check_session_drift.py` — whether within-session drift exists at the
  timescale a run takes. It does not, which is why there are two execution modes.
* `refractal-libero/scripts/verify_libero_claims.py` — nine claims about LIBERO,
  including an asset digest per version.
