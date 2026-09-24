"""``refractal init`` -- a catalog that actually runs.

The seconds test is ``pip install refractal``, ``refractal init``,
``refractal plan``, ``refractal run``, ``refractal compare``, on a laptop with
nothing else installed. What makes the claim true is that ``init`` produces
something that *works* -- nobody should have to author a catalog before seeing
one run.

**The example declares no filter, deliberately.** A filter is evaluated by
``refractal build``, which needs the engine, so a filtered example would make
``refractal plan`` fail on the very first command. Shipping a committed
``build.lock`` instead would be worse: it is a lock nobody can regenerate, and
the first thing anyone does after ``init`` is widen the grid -- which invalidates
it and drops them into a staleness error before they have run anything.

So the example plans immediately, and filters appear in the docs as the thing
that requires ``build``.

Templates are literals here rather than package data, so ``init`` works from a
wheel without any packaging configuration to get wrong.
"""

from __future__ import annotations

from pathlib import Path

from .schema.errors import RefractalError


class InitError(RefractalError):
    """Refused to write an example over something that already exists."""


SCENES = """\
apiVersion: refractal.dev/v1alpha1
scenes:
  # A scene is the physical world: which bodies, joints and geoms exist. It is
  # the affinity key -- work cannot move between scenes, because vectorized
  # environments require identical structure.
  #
  # Two scenarios are the same scene if and only if they compile to the same
  # physics model and differ only in that model's data. Position, friction and
  # mass live in the data, so they do not split a scene. A distractor object
  # does.
  - id: cube-bowl-v1
    engine: mujoco
    model: assets/cube_bowl.xml
    engine_version: "3.2.0"     # `refractal build` records this where the engine exists
    resource_shape:
      - hardware_profile: laptop
        envs_per_process: 1     # classic MuJoCo: one env per process
        vram_per_env_mb: 0      # CPU-only, so the model server is the sole GPU consumer
        cpu_cores: 1
        sec_per_1k_steps: 41
        startup_sec: 3
    description: contact-light and forgiving; a good first scene
"""

TASKS = """\
apiVersion: refractal.dev/v1alpha1
tasks:
  # A task is the goal plus the predicate that decides success. It is NOT in the
  # physics model, which is why tasks are cheap to vary and scenes are not:
  # 200 tasks over 10 scenes runs far faster than 10 tasks over 200 scenes.
  #
  # Two tasks on one scene share a batch.
  - id: cube-in-bowl
    scene: cube-bowl-v1
    instruction: "put the cube in the bowl"
    predicate: refractal.example:cube_in_bowl
    predicate_args: {tolerance: 0.05}
    max_steps: 300

  - id: cube-on-table
    scene: cube-bowl-v1
    instruction: "put the cube on the left of the table"
    predicate: refractal.example:cube_left_of
    predicate_args: {x: 0.10}
    max_steps: 300
"""

SCENARIOS = """\
apiVersion: refractal.dev/v1alpha1
scenario_sets:
  # Scenarios are generated, not enumerated: you describe a grid once and the
  # description is what gets stored as provenance.
  #
  # No `filter` here on purpose. Filters are evaluated by `refractal build`,
  # which needs the engine, so a filtered example could not be planned until you
  # had one installed. Add one once you have a real scene -- see the docs.
  - id: cube-grid
    scene: cube-bowl-v1
    generator: refractal.generators:linspace_grid
    generator_seed: 12345
    params:
      cube_x:  {range: [0.05, 0.20], steps: 6}
      cube_y:  {range: [-0.10, 0.10], steps: 5}
      bowl_yaw: {value: 0.0}
    # Empty here because this example has no simulator behind it, and a
    # perturbation acts on one: it changes the world mid-episode -- a weaker
    # gripper, a displaced object -- or what the policy is shown of it.
    #
    # Declared as data so it is hashed into the scenario, which means a
    # perturbed episode is a DIFFERENT experiment from its unperturbed
    # counterpart and cannot silently join it. Each episode also carries a
    # base identity covering the unperturbed scenario, so results recorded
    # before you started perturbing become the baseline end of a later sweep.
    #
    # See docs/perturbations/ once you have a real scene.
    perturbations: []
"""

RUN = """\
apiVersion: refractal.dev/v1alpha1
run:
  # A list, always. Two is the common case, not a special one -- a single
  # checkpoint is a comparison of size one.
  checkpoints:
    - id: baseline
      path: ./checkpoints/baseline
      server: refractal.example:EchoServer
      vram_mb: 0
    - id: candidate
      path: ./checkpoints/candidate
      server: refractal.example:EchoServer
      vram_mb: 0
  seeds: 3          # separates "is this scenario hard" from "did the policy get lucky"
  tier: full        # smoke | regression | full
  execution_mode: serial
  results_uri: ./results
"""

HARDWARE = """\
apiVersion: refractal.dev/v1alpha1
hardware_profiles:
  # What the machine provides, as opposed to what one worker needs. The planner
  # divides one into the other and refuses to emit a plan that will not fit.
  - id: laptop
    cpu_cores: 4
    memory_mb: 8192
    devices: []
    max_workers: 4
"""

MODEL = """\
<!-- A placeholder scene. Enough to be a real file with a real hash, not enough
     to be a real task. Replace it with your own MJCF; `refractal build` will
     notice the geometry changed and say so. -->
<mujoco model="cube_bowl">
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.1"/>
    <body name="bowl" pos="0.25 0 0">
      <geom name="bowl_body" type="cylinder" size="0.06 0.03"/>
    </body>
    <body name="cube" pos="0.1 0 0.05">
      <freejoint/>
      <geom name="cube_body" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""

README = """\
# An example Refractal catalog

It runs as-is:

    refractal plan   catalog/ --hardware laptop
    refractal run    plan.json --backend local
    refractal compare results <plan_id>

## What is here

| file | what it declares |
|---|---|
| `scenes.yaml` | the physical world. **The affinity key** -- work cannot cross scenes |
| `tasks.yaml` | goals and success predicates. Not in the physics model, so cheap to vary |
| `scenarios.yaml` | parameter grids. Generated, never enumerated |
| `run.yaml` | which checkpoints to compare, how many seeds, where results go |
| `hardware.yaml` | what the machine provides, so the planner can refuse to overcommit |

## What is deliberately missing

**A filter.** Filters drop scenarios a robot cannot reach, and evaluating one
needs the engine -- so they are the job of `refractal build`, not
`refractal plan`. An example with a filter could not be planned until you had a
simulator installed, which would defeat the point of this file.

Add one when you have a real scene:

    # scenarios.yaml
    filter: your_package.filters:reachable

then run `refractal build catalog/ --hardware laptop` before planning. The
survivors are recorded in `catalog/build.lock`, keyed to the grid, the geometry
and the filter's body -- change any of the three and planning refuses rather than
answering from a stale answer.

## Next

Edit the grid in `scenarios.yaml` and re-plan; the episode count and the cost
estimate move, and `plan.json` diffs cleanly. Nothing is spent until
`refractal run`.
"""

FILES = {
    "catalog/scenes.yaml": SCENES,
    "catalog/tasks.yaml": TASKS,
    "catalog/scenarios.yaml": SCENARIOS,
    "catalog/run.yaml": RUN,
    "catalog/hardware.yaml": HARDWARE,
    "catalog/assets/cube_bowl.xml": MODEL,
    "README.md": README,
}


def init(target: str | Path, *, force: bool = False) -> list[Path]:
    """Write a working example catalog under ``target``."""
    root = Path(target)
    existing = [root / name for name in FILES if (root / name).exists()]
    if existing and not force:
        raise InitError(
            f"{len(existing)} file(s) already exist under {root} "
            f"(first: {existing[0]}). Use --force to overwrite, or pick an empty directory."
        )

    written = []
    for name, content in FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


__all__ = ["FILES", "InitError", "init"]
