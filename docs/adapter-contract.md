# Adapter contract

An adapter wraps a simulator so Refractal can drive it. It is a class with five
methods, loaded by import string from your catalog.

```yaml
scenes:
  - id: bench-v1
    external:
      provider: my_package.adapter:BenchAdapter
```

A worked implementation is in [adapter-example.md](adapter-example.md).

## The five methods

### `reset(task) -> raw_observation`

Called once per episode. Build or reconfigure the environment, apply the
scenario, return the first raw observation.

`task` is a dict. The scenario's parameters are in it under the names your
catalog used.

- Store the environment on `self`; the next four methods will need it.
- Rebuild only when you must. `reset` is called for every episode, and rebuilding
  a simulator each time is usually the difference between minutes and hours.
- **Do not mutate the scenario dict.** It is hashed; a mutated copy is a
  different scenario from the one recorded.

### `step(action) -> StepResult`

Apply one action. Return `StepResult(obs, reward, done, info)`.

`action` is whatever the model server sent. Convert it here, including any
discretisation your environment needs.

### `make_obs(raw_obs, task) -> Observation`

Convert your environment's observation into the wire format the model server
expects.

```python
{"images": {"front": array}, "states": array, "task_description": str}
```

The keys must match what the server was configured to read. A mismatch here is
silent: the policy receives zeros or defaults, scores badly, and nothing reports
an error.

### `get_step_result(step_result) -> EpisodeResult`

Called once, on the final step. Return `{"success": bool}` and any metrics worth
keeping.

Whatever you return is the outcome. Nothing downstream re-derives it or
second-guesses it.

### `check_done(step_result) -> bool`

Whether the episode should stop. Defaults to `step_result.done`; override if your
environment signals termination differently.

## What your adapter owes

**The scenario dict is the vocabulary.** Refractal never looks inside it. The
names are whatever your catalog's `params` used, and nothing checks that your
adapter reads the same ones. If the catalog says `cup_x` and the adapter reads
`cube_x`, every episode starts from your default, the comparison is between two
identical conditions, and no error is raised anywhere.

**Anything you vary that is not in the scenario is invisible.** Randomising a
distractor's position inside `reset` means two episodes with the same
`scenario_hash` were not the same episode. Put it in the scenario, or seed it
from something already there.

**Record what you want to keep.** Fields you do not pass to `record_step` are not
recoverable later. The episode ran and the value is gone.

**Count from zero.** If your adapter selects from a fixed array of start states,
the index it receives runs `0, 1, 2, …` per task. A catalog whose indices start
elsewhere is refused by `refractal run` rather than recorded.

## Registering it

The provider is an import string resolved at run time, so the adapter does not
need to be installed where you run `refractal plan`.

```yaml
external:
  provider: my_package.adapter:BenchAdapter
  ref:    {suite: kitchen}                        # which scene
  params: {send_state: true, num_steps_wait: 10}  # constructor arguments
```

`params` is passed to your `__init__`. `ref` is not; it identifies the scene.
Both are hashed into the scene's identity.

## Checking it works

```console
$ refractal build catalog --probe my_package.probe:MyProbe
$ refractal plan catalog --hardware laptop -o plan.json
$ refractal run plan.json -o results --backend vla-eval --server baseline=ws://localhost:8000
```

If episodes run but every one fails, check the observation keys first and the
scenario parameter names second. Those are the two failures that produce a
complete run and a meaningless result.
