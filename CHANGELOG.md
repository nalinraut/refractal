# Changelog

What changed in each released version. Newest first.

This is not the commit history. The history records why a decision was made;
this records what a user gets in a version they can install. Different
audiences, and the history is the better read for the first question.

## 0.1.0a1 (unreleased)

First alpha. Every stage is implemented and tested against real model servers
as well as the synthetic backend.

**`plan_id` may move between alpha versions.** Episode identity is a hash of
the catalog's content, and the fields that feed it are still settling: they
changed three times this month. Results written by `0.1.0a1` may therefore fail
to join with results from a later alpha, and `refractal compare` will refuse
the join rather than pool them, which is the intended behaviour and not a bug.
If you are about to spend GPU hours, this is the sentence to read first. The
`a` in the version is doing real work.

`plan_schema` is 3. It is deliberately not part of `plan_id`: a format bump
does not make an experiment a different experiment.

### Added

- `refractal init`, write a working example catalog that runs end to end with
  no GPU, no simulator and no checkpoints.
- `refractal build`, compute scene hashes, evaluate filters and record resource
  shapes into `catalog/build.lock`.
- `refractal plan`, compile a catalog into `plan.json`. Runs with none of the
  optional dependencies installed, which is enforced by a test rather than by
  convention.
- `refractal run`, execute a plan. Backends: `local` (simulates outcomes, no
  infrastructure), `vla-eval` (drives the harness against running model
  servers), `compose` (one container per worker).
- `refractal render`, write a deployment description from a plan without Docker
  or a cluster installed. Targets: `compose` and `k8s`.
- `refractal compare`, compare checkpoints in a results directory. Paired
  McNemar on scenario-level outcomes, a clustered bootstrap that resamples
  whole scenarios, Cochran's Q as a screen across three or more checkpoints,
  and Holm correction across the whole contrast family.
- `refractal explain`, why two plans are different experiments.
- Perturbations: timed changes to the world, the observation or the action
  during an episode, hashed into `scenario_hash` so a perturbed episode is a
  different experiment from its unperturbed counterpart.
- Exit codes for CI: 0 no regression, 1 regression, 2 cannot be answered.
  The third is never reported as the second.
- Optional extras: `execute`, `compare`, `video`, `vla-eval`, `all`, `test`.
  Comparison needs no numerical stack.

### Not built yet

Stated because a gap a user discovers is worse than one they were told about.

- **Trajectory quality metrics.** The `steps.parquet` schema is declared and
  nothing writes it. Writing it against synthetic data would bake in guesses
  about what a real adapter can record.
- **`refractal run --backend k8s`.** Kubernetes is reached through `render`:
  Refractal writes one Job per worker and your cluster schedules them. Nothing
  submits them, watches them or collects their exit codes.
- **Resource-shape probing.** `refractal build` carries hand-declared shapes
  forward and records whether they were measured. A lint refuses a comment
  claiming a measurement when nothing recorded one.
- **Partial credit.** `phase_outcomes` exists in the schema and the LIBERO
  adapter never populates it, so success is binary.
