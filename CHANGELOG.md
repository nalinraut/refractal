# Changelog

What changed in each released version. Newest first.

This is not the commit history. The history records why a decision was made;
this records what a user gets in a version they can install. Different
audiences, and the history is the better read for the first question.

## 0.1.0a2 (2026-10-01)

Still alpha, and the `a` is doing its job: the identity-bearing fields moved
again this month. `task_hash` now covers a success rule and the extractors that
feed it, so a task declaring one hashes differently than it did under 0.1.0a1.
Results from the two are different experiments and `compare` will say so.

**Your existing results are not orphaned.** That only applies to a task that
declares a rule, which nothing could do before this version. A catalog without
one produces a byte-identical `plan_id` under both — checked, not assumed, by
planning the same catalog with 0.1.0a1 and 0.1.0a2 installed side by side.
`plan_schema` does move, 3 to 4, which is the file format rather than the
experiment's identity.

### Added

- **Metrics.** A scene can declare extractors that turn live simulator state
  into named numbers, which land in the `metrics` column of `episodes.parquet`
  as `map<string, double>`. The names are declared in the catalog rather than
  discovered, because `refractal plan` runs with no simulator and cannot import
  an extractor to ask it.
- **A success rule in the catalog.** A task can declare `success` as a
  conjunction of thresholds over those metrics, and the verdict is computed
  when the row is written. Two consequences worth stating: a missing metric
  fails the threshold rather than passing it, and re-scoring an existing result
  under a different threshold is impossible by construction rather than
  discouraged -- the rule is inside `task_hash`, so changing it makes a
  different experiment.
- Where no rule is declared, nothing changes: the provider's own boolean still
  decides, which is how every wrapped suite keeps owning its definition of
  success.
- **Two refusals, at different scopes.** One reads only the rows and refuses a
  pool whose episodes disagree about which extractors ran; the other reads only
  the declaration and refuses a session naming a metric no extractor produces.
  Neither is written in terms of the other, so neither can mask the other.

### Changed

- `benchmark` was vla-eval's word for its own interface and had spread into
  language with nothing to do with that backend. Refractal's vocabulary is
  scene, task, scenario, seed, episode, checkpoint.
  - `refractal run --benchmark` is now `--provider`. The old spelling is still
    accepted and no longer shown in `--help`.
  - The `benchmark_class` results column is now `provider_class`. Files written
    under the old name still read: the rename is applied on read, so a results
    directory can hold part files from either side of the change.
  - `refractal.predicates:from_benchmark` is now `from_provider`.
    **`from_benchmark` remains as an alias and existing catalogs should keep
    using it**: `predicate` is inside `task_hash`, so changing the string in a
    catalog changes the experiment and orphans results already recorded.
  - `FakeProvider`, formerly `FakeBenchmark`, and no longer exported from
    `refractal.execute`. It is the local backend's default and a test fixture,
    not API. Import it from `refractal.execute.fake` if you need it.

- The vla-eval pin stays at `>=0.6.0`. It is a capability floor -- the oldest
  release carrying the two upstream fixes Refractal needs -- and 0.7.0 and
  0.8.0 add nothing it requires, so raising it would exclude working installs.
  Both were verified on 2026-10-01: all six claims in
  `scripts/verify_harness_claims.py` hold against each, and the suite passes on
  each.
- **If you have results from more than one harness version, `compare` will now
  block.** vla-eval 0.8.0 changed `orchestrator.py`, `recording.py` and the
  three runners, which moves the surface digest (`8ab641f1` -> `7b1d850b`).
  That is the gate doing its job rather than a regression: two harness versions
  may not be measuring the same thing. Pass `--allow-harness-mismatch` once you
  have decided they are.

### Documentation

- **How to run a perturbation.** Four documents described the feature and none
  contained a command. `--provider` — the only way to run one — was named in
  no document at all. [Perturbations](perturbations/index.md) now shows the
  invocation, why the class is a flag rather than a hashed catalog field, and
  what happens if you omit it.
- **Declaring your own success rule**, in `writing-a-catalog.md`, beside the
  section on letting a wrapped suite decide. They are two halves of one choice
  and only one was written.
- **Where the `success` bit came from**, in `reading-a-comparison.md`. The
  statistics are identical whether a provider reported the verdict or a
  catalog rule computed it, which is exactly why the page should say which.
- `refractal plan -v` is documented.

### Fixed

- `getting-started.md` showed `plan_schema 1` and `README.md` showed
  `plan_schema 3`; the code emits `4`. It is the first number a new user sees.
- The only worked example for `metrics[].extractor` named a package that is not
  published. Copying it produced `ModuleNotFoundError` from the documentation.
- `Scenario.perturbations` claimed sustained perturbations were unimplemented.
  `until_step` has worked since perturbations landed; it is refused only for
  effects with no registered inverse.

## 0.1.0a1 (2026-09-25)

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
