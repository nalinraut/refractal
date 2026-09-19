# Adapter example

A complete adapter for a tabletop pick-and-place simulator. Every comment answers
why a line is there or what breaks without it.

The method signatures are in [adapter-contract.md](adapter-contract.md).

```python
"""Adapter for the Bench simulator."""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.types import Action, EpisodeResult, Observation, Task


class BenchAdapter(StepBenchmark):

    def __init__(
        self,
        suite: str = "kitchen",
        send_state: bool = True,
        num_steps_wait: int = 10,
        max_steps: int | None = None,
    ) -> None:
        # These come from `params` in the catalog's scene block, and they are
        # hashed into the scene's identity. Two runs that disagree about
        # `send_state` are not comparable, and because it is in the hash they
        # land in different result directories instead of being averaged.
        super().__init__()
        self.suite = suite
        self.send_state = send_state
        self.num_steps_wait = num_steps_wait
        self._max_steps = max_steps

        # Built lazily in reset() and reused. Constructing a simulator costs
        # seconds; doing it per episode instead of per task turns a twenty
        # minute run into a two hour one.
        self._env = None
        self._current_task_id: int | None = None

    def get_tasks(self) -> list[Task]:
        # One entry per goal this scene offers. `name` is what a catalog's
        # `instruction` must match EXACTLY -- that string is the selector, and a
        # catalog naming a goal this list does not contain selects nothing and
        # runs zero episodes.
        return [
            {"name": "put the cup on the shelf", "suite": self.suite, "task_id": 0},
            {"name": "put the bowl on the shelf", "suite": self.suite, "task_id": 1},
        ]

    def reset(self, task: Task) -> Any:
        from bench_sim import BenchEnv  # imported here: plan must run without it

        task_id = task["task_id"]

        # Rebuild only when the goal changes. Reusing the environment across
        # episodes of one task is the single largest speed decision in an
        # adapter.
        if self._env is None or self._current_task_id != task_id:
            if self._env is not None:
                self._env.close()
            self._env = BenchEnv(suite=self.suite, task_id=task_id)
            self._current_task_id = task_id

        self._env.reset()

        # THE SCENARIO. These keys are whatever the catalog's `params` declared.
        # Refractal never inspects this dict. It hashes it and passes it
        # through. If the catalog says `cup_x` and this reads `cube_x`, every
        # episode silently starts from the default, the two arms face identical
        # conditions, and the comparison is between a thing and itself. Nothing
        # anywhere raises. Grep both files for the names.
        #
        # `.get` with a default would hide exactly that mistake, so index
        # directly and let a missing key be a KeyError on episode one.
        obs = self._env.place("cup", x=task["cup_x"], y=task["cup_y"])

        # Do not write back into `task`. It is the hashed scenario; a mutated
        # copy is a different scenario from the one recorded against this row.

        # Anything random here that is NOT derived from the scenario makes two
        # episodes with the same scenario_hash different episodes. Seed from the
        # scenario so the variation is part of the identity.
        rng = np.random.default_rng(abs(hash((task["cup_x"], task["cup_y"]))) % 2**32)
        self._env.jitter_distractors(rng)

        # Let the arm settle before the policy sees anything. Without it the
        # first observation catches the scene mid-drop and the policy acts on a
        # state that never really existed.
        for _ in range(self.num_steps_wait):
            obs, *_ = self._env.step(np.zeros(7))

        return obs

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        # These key names must match what the model server was configured to
        # read. A mismatch is the other silent failure: the policy gets zeros,
        # scores badly, and the run completes with no error to find.
        out: Observation = {
            "images": {"front": raw_obs["front_rgb"]},
            "task_description": task["name"],
        }
        if self.send_state:
            # Optional because some policies are trained without proprioception.
            # It is a constructor argument rather than a hardcoded choice so it
            # lands in the scene's identity; sending state to a policy trained
            # without it costs tens of points and looks like a policy result.
            out["states"] = np.asarray(raw_obs["joint_positions"], dtype=np.float32)
        return out

    def step(self, action: Action) -> StepResult:
        raw = action["actions"]

        # Convert to whatever the environment wants. The gripper here is
        # binary; passing the continuous value through makes the arm grip
        # weakly on every episode, which reads as a policy that cannot hold
        # things.
        command = list(raw[:-1]) + [1.0 if raw[-1] >= 0 else -1.0]

        obs, reward, done, info = self._env.step(np.asarray(command))

        # Everything worth having later goes through the recorder. A field not
        # passed here cannot be recovered: the episode ran and the value is
        # gone.
        if self._recorder is not None:
            self._recorder.record_step(
                reward=float(reward),
                gripper=float(command[-1]),
                object_height=float(info["cup_z"]),
            )

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def check_done(self, step_result: StepResult) -> bool:
        # Default is `step_result.done`. Override when the environment signals
        # termination some other way. Here, success is a flag in `info`.
        return bool(step_result.info.get("goal_reached", False))

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        # This IS the outcome. Nothing downstream re-derives or second-guesses
        # it, so a bug here is a wrong success rate with no other symptom.
        return {
            "success": bool(step_result.info.get("goal_reached", False)),
            "final_height": float(step_result.info["cup_z"]),
        }

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self._max_steps or 300, "suite": self.suite}

    def cleanup(self) -> None:
        # Called when the worker finishes. Without it a long run leaks one
        # simulator per task.
        if self._env is not None:
            self._env.close()
            self._env = None
```

## The catalog that drives it

```yaml
# scenes.yaml
scenes:
  - id: bench-kitchen
    engine: mujoco
    engine_version: "3.2.0"
    external:
      provider: my_package.adapter:BenchAdapter
      ref:    {suite: kitchen}
      params: {suite: kitchen, send_state: true, num_steps_wait: 10}

# tasks.yaml -- `instruction` must match a `name` from get_tasks() exactly
tasks:
  - id: cup-on-shelf
    scene: bench-kitchen
    instruction: "put the cup on the shelf"
    predicate: refractal.predicates:from_benchmark
    provider_ref: {task_id: 0}
    max_steps: 300

# scenarios.yaml: these names must match what reset() reads
scenario_sets:
  - id: cup-positions
    scene: bench-kitchen
    generator: refractal.generators:linspace_grid
    generator_seed: 0
    params:
      cup_x: {range: [-0.15, 0.15], steps: 6}
      cup_y: {range: [0.30, 0.60], steps: 6}
```

## The two mistakes that produce a clean run and a meaningless result

**Parameter names that do not match.** `scenarios.yaml` says `cup_x`; `reset`
reads `cup_x`. If those diverge, every episode starts from the default position.
The run completes, the rates look plausible, and both arms faced the same
conditions.

**Observation keys that do not match.** `make_obs` returns `front`; the model
server must be configured to read `front`. If it is not, the policy sees nothing
and scores near zero, which looks like a bad checkpoint.

Both are checkable in about a minute: run two episodes, print the scenario dict
`reset` receives, and print the keys `make_obs` returns.
