"""``refractal`` command line.

Only ``plan`` exists so far, which is the point: the step-1 gate is that
``refractal plan catalog/`` runs on a bare laptop -- no Docker, no GPU, no
simulator -- prints what the run will cost, and writes a plan you can read
before spending anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .resolve import resolve
from .schema.errors import RefractalError


def _fmt_duration(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def cmd_plan(args: argparse.Namespace) -> int:
    plan = resolve(
        args.catalog,
        hardware_profile=args.hardware,
        pack_below_startup_sec=args.pack_below_startup_sec,
    )

    for scene in plan.scenes:
        shape = scene.resource_shape
        packed = any(w.packed_scenes for w in scene.workers)
        print(
            f"  {scene.scene_id:<16} {len(scene.scenarios):>5} scenarios"
            f"  {scene.episode_count:>6} episodes"
            f"  {len(scene.workers):>3} worker(s){' [packed]' if packed else ''}"
            f"  ~{_fmt_duration(scene.estimated_seconds)}"
        )
        if args.verbose:
            for worker in scene.workers:
                print(
                    f"      {worker.worker_id:<20} {worker.device:<8}"
                    f" cpuset={worker.cpuset or '-':<8}"
                    f" {len(worker.episodes):>5} episodes"
                    f"  ~{_fmt_duration(worker.estimated_seconds)}"
                    f"  ({shape.envs_per_process} env/proc)"
                )

    workers = sum(len(s.workers) for s in plan.scenes)
    print(
        f"  {'-' * 66}\n"
        f"  {plan.total_episodes} episodes across {len(plan.scenes)} scene(s), "
        f"{workers} worker(s), est. {_fmt_duration(plan.estimated_seconds)}"
    )
    print(
        f"  tier={plan.tier}  seeds={plan.seeds}  "
        f"checkpoints={[c.id for c in plan.checkpoints]}  mode={plan.execution_mode}"
    )

    for warning in plan.warnings:
        print(f"  warning: {warning}", file=sys.stderr)

    out = Path(args.output)
    plan.write(out)
    print(f"  wrote {out}  (plan_schema {plan.plan_schema}, plan_id {plan.plan_id[:19]}...)")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Exit code is the gate. 0 clean, 1 regression, 2 unanswerable."""
    from .compare import build_units, evaluate, render
    from .execute import read_episodes

    rows = read_episodes(args.results, args.plan_id).to_pylist()
    if not rows:
        print(f"error: no episodes under {args.results} for {args.plan_id}", file=sys.stderr)
        return 2

    checkpoints = args.checkpoints or sorted({r["checkpoint_id"] for r in rows})
    if len(checkpoints) < 2:
        print(f"error: need at least two checkpoints, found {checkpoints}", file=sys.stderr)
        return 2

    eligibility = build_units(rows, checkpoints=checkpoints, min_seeds=args.min_seeds)
    verdict = evaluate(
        eligibility,
        checkpoints[0],
        checkpoints[1],
        rule=args.dichotomy,
        resamples=args.resamples,
    )
    print(render(verdict))
    return verdict.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refractal", description=__doc__)
    parser.add_argument("--version", action="version", version=f"refractal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="compile a catalog into plan.json")
    plan.add_argument("catalog", help="path to the catalog directory")
    plan.add_argument("--hardware", required=True, help="hardware profile id, e.g. rtx5090")
    plan.add_argument("-o", "--output", default="plan.json")
    plan.add_argument("-v", "--verbose", action="store_true", help="show per-worker placement")
    plan.add_argument(
        "--pack-below-startup-sec",
        type=int,
        default=30,
        help="pack starved scenes sequentially when startup_sec is below this "
        "(at 90s load, packing eleven scenes costs 16 minutes of pure loading)",
    )
    plan.set_defaults(func=cmd_plan)

    compare = sub.add_parser(
        "compare",
        help="compare checkpoints in a results directory",
        description="Exit codes: 0 no regression, 1 regression detected, 2 cannot be answered.",
    )
    compare.add_argument("results", help="results URI, local path or s3://...")
    compare.add_argument("plan_id", help="plan_id / comparison_id of the run")
    compare.add_argument(
        "-c", "--checkpoints", nargs="+", help="which to compare (default: all present)"
    )
    compare.add_argument(
        "--dichotomy",
        choices=("majority", "all"),
        default="majority",
        help="how several seeds at one scenario collapse to pass/fail for McNemar",
    )
    compare.add_argument("--min-seeds", type=int, default=None, help="seed floor per scenario")
    compare.add_argument("--resamples", type=int, default=5000)
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RefractalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
