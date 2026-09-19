# Refractal documentation

Refractal is a scene-coherent placement and paired-comparison layer over
[`allenai/vla-evaluation-harness`](https://github.com/allenai/vla-evaluation-harness).
The harness answers *what did this checkpoint score*. Refractal answers
**did my change help** — a different question, and the difference is paired
statistics over content-addressed episodes.

Seven documents, in the order they become useful.

## Using it

| | when to read it |
|---|---|
| [statistics-measurements.md](statistics-measurements.md) | Before trusting a `compare` verdict. Why a clustered bootstrap and not a two-proportion z-test, measured rather than argued — including the design effect, what it means when it is below 1, and the circularity in the measurement that a real run has since partly resolved. |
| [scene-task-split.md](scene-task-split.md) | Before writing a catalog. A scene is the compiled model; the goal belongs to the task. Explains `partition_unit`, `provider_ref`, and why a benchmark whose goal is opaque to `predicate` needs a content hook. |
| [execution-mode.md](execution-mode.md) | Before choosing `serial` or `concurrent`. Defined by episode ordering rather than by deployment, with the reason a third mode was measured and not built. |

## Running it against a real harness

| | when to read it |
|---|---|
| [harness-integration.md](harness-integration.md) | Before using `--backend vla-eval`. The contract with the harness: which hooks, why no fork, what each claim holds up, and the three plan fields that were covered by a hash and never exercised by a consumer. `scripts/verify_harness_claims.py` re-checks six of its claims on every pin move. |
| [compose-backend.md](compose-backend.md) | Before using `--backend compose`. One container per worker over the `--worker` entrypoint, what stays outside it (the model servers), and four permission failures that are invisible to a root run. |

## Maintaining it

| | when to read it |
|---|---|
| [releasing.md](releasing.md) | Before publishing. The checklist, and why `dist/` is deleted rather than reused. |

## Scripts that check claims about code we do not control

An assertion about someone else's code does not fail when it becomes wrong — it
sits there being wrong, which is how a claim that the harness hardcodes its
server address survived nine days and four documents. So the assertions execute:

* `scripts/verify_harness_claims.py` — six claims about vla-eval, each carrying
  what it holds up. Run on every pin move.
* `scripts/verify_wheel.py` — the built artefact against the source tree,
  derived rather than listed. Has caught two real bugs.
* `scripts/lint_provenance_claims.py` — refuses a comment claiming a value was
  measured when `measured_at` is empty. In CI.

`refractal-libero/scripts/verify_libero_claims.py` does the same for LIBERO,
including an asset digest per version.

## The development record

Nine documents that are *not* here, in `refractal-archive/` alongside this
repository: the engineering-practice log, the two run write-ups, the original
spec review, and two questions that are now closed. None is needed to use or
maintain Refractal. They are kept because the reasoning is harder to reconstruct
than the code — `patterns.md` in particular is the most reusable thing written
here and the least about Refractal.
