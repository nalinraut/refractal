"""``refractal`` command line.

``plan`` still sets the tone: it runs on a bare laptop -- no Docker, no GPU, no
simulator -- prints what the run will cost, and writes a plan you can read before
spending anything.

Everything else is arranged so that stays true. ``build`` is the only verb that
needs an engine, ``run --backend vla-eval`` is the only one that needs the harness
and a GPU, and both import their dependencies inside the branch that uses them.
``compare`` reads Parquet and needs neither.
"""

from __future__ import annotations

import argparse
import os
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
            f"  <={_fmt_duration(scene.estimated_seconds)}"
        )
        if args.verbose:
            for worker in scene.workers:
                print(
                    f"      {worker.worker_id:<20} {worker.device:<8}"
                    f" cpuset={worker.cpuset or '-':<8}"
                    f" {len(worker.episodes):>5} episodes"
                    f"  <={_fmt_duration(worker.estimated_seconds)}"
                    f"  ({shape.envs_per_process} env/proc)"
                )

    workers = sum(len(s.workers) for s in plan.scenes)
    print(
        f"  {'-' * 66}\n"
        f"  {plan.total_episodes} episodes across {len(plan.scenes)} scene(s), "
        f"{workers} worker(s), at most {_fmt_duration(plan.estimated_seconds)}"
    )
    # A BOUND, said so because a reader assumes the other thing. The makespan
    # model costs every episode at its step cap; episodes that succeed finish
    # earlier, so the truth is lower by exactly the rate at which they do.
    print(
        "  that is an upper bound: every episode is costed at its full step "
        "limit, and episodes that succeed finish sooner."
    )

    # The second number, when a prior run can supply one. Two numbers, each
    # labelled with what it is -- rather than one number a reader has to guess
    # the meaning of. It is printed and never written to plan.json: it is fitted
    # to particular hardware and particular checkpoints, which makes it
    # render-time by the project's own rule.
    if args.expect_from:
        from .execute import read_episodes
        from .expect import expected_seconds

        try:
            prior = read_episodes(args.expect_from, args.expect_plan or plan.plan_id)
            rows = prior.to_pylist()
        except Exception as exc:
            print(f"  warning: no prior run to expect from: {exc}", file=sys.stderr)
            rows = []
        expectation = expected_seconds(plan, rows) if rows else None
        if expectation is None:
            print("  no usable prior episodes; expected duration not computed",
                  file=sys.stderr)
        else:
            print(
                f"  expected {_fmt_duration(expectation.seconds)}, from "
                f"{expectation.sample} prior episode(s) of "
                f"{expectation.learned_from}"
            )
            for note in expectation.notes:
                print(f"    caution: {note}", file=sys.stderr)
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


def cmd_init(args: argparse.Namespace) -> int:
    from .init import init

    written = init(args.target, force=args.force)
    catalog = Path(args.target) / "catalog"
    print(f"  wrote {len(written)} file(s) under {args.target}")
    print(f"  next:  refractal plan {catalog} --hardware laptop")
    return 0


def _servers(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``--server ckpt=url`` into the mapping the bridge preflights.

    Split on the *first* ``=`` only: a URL may contain one in a query string, and
    silently truncating it would point an arm at the wrong server -- which is the
    failure the whole pairing design exists to make unrepresentable.
    """
    def bad(message: str) -> "SystemExit":
        # Exit 2, matching argparse's own usage-error code and every other
        # refusal in this CLI. `SystemExit(str)` would print and exit 1.
        print(f"error: {message}", file=sys.stderr)
        return SystemExit(2)

    servers: dict[str, str] = {}
    for pair in pairs or []:
        name, sep, url = pair.partition("=")
        if not sep or not name or not url:
            raise bad(f"--server takes checkpoint=url, got {pair!r}")
        if name in servers:
            raise bad(f"--server {name} given twice")
        servers[name] = url
    return servers


def cmd_run(args: argparse.Namespace) -> int:
    import uuid

    from .schema.plan import PlanSchemaError, read_plan, restrict_to_worker

    plan = read_plan(args.plan)
    if args.worker:
        # A filter, not a different plan: plan_id and every episode id are
        # untouched, so this worker's rows join the others' as though one process
        # had written them all.
        try:
            plan = restrict_to_worker(plan, args.worker)
        except PlanSchemaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    session_id = args.session_id or uuid.uuid4().hex

    if args.backend == "compose":
        return _run_compose(args, plan)

    if args.backend == "vla-eval":
        from .execute.vla_eval import BridgeError
        from .execute.vla_eval_runner import run_vla_eval

        try:
            summary = run_vla_eval(
                plan,
                args.results,
                _servers(args.server),
                catalog_root=args.catalog,
                session_id=session_id,
                output_dir=args.harness_output,
                resume=not args.no_resume,
            )
        except BridgeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"  session {summary.session_id[:8]}  {summary.invocations} harness "
            f"invocation(s), {summary.written} episode(s) written, "
            f"{summary.skipped} already done"
        )
        print(f"  next:  refractal compare {args.results} {plan.plan_id}")
        return 0

    from .execute import run_local

    if args.server:
        print(
            "error: --server applies to --backend vla-eval; the local backend "
            "simulates and contacts nothing.",
            file=sys.stderr,
        )
        return 2
    summary = run_local(
        plan,
        args.results,
        catalog_root=args.catalog,
        session_id=session_id,
        resume=not args.no_resume,
    )
    print(
        f"  session {summary.session_id[:8]}  "
        f"{summary.written} episode(s) written, {summary.skipped} already done"
    )
    print(f"  next:  refractal compare {args.results} {plan.plan_id}")
    return 0


def _run_compose(args: argparse.Namespace, plan) -> int:
    """Render, then hand over to `docker compose up`. Deliberately thin.

    Everything that could be wrong about the deployment is wrong in the rendered
    file, and the renderer is a pure function with its own tests. What is left
    here is process handling and one thing the renderer cannot do: create the
    results directory.

    That has to happen on the host, before any container starts. Docker creates a
    missing bind-mount target as root whatever `user:` says, and a non-root
    container then cannot write into it -- and the failure is quiet, because the
    run completes, Parquet lands where the container can write, and `compare`
    fails later on somebody else's machine.
    """
    import subprocess

    from .render import ComposeSettings, RenderError, render_compose

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    try:
        text = render_compose(
            plan,
            ComposeSettings(
                servers=_servers(args.server),
                user=args.user or f"{os.getuid()}:{os.getgid()}",
                plan_file=str(Path(args.plan).resolve()),
                catalog_dir=str(Path(args.catalog).resolve()) if args.catalog else None,
                results_dir=str(results.resolve()),
                images=dict(
                    pair.split("=", 1) for pair in (args.image or []) if "=" in pair
                ),
                source_mounts=list(args.mount or []),
            ),
        )
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = Path(args.compose_file)
    out.write_text(text, encoding="utf-8")
    print(f"  rendered {out}")

    command = ["docker", "compose", "-f", str(out), "up",
               "--abort-on-container-failure"]
    if args.detach:
        command = ["docker", "compose", "-f", str(out), "up", "-d"]
    print(f"  {' '.join(command)}")
    try:
        completed = subprocess.run(command)
    except FileNotFoundError:
        print(
            "error: docker is not on PATH. `--backend compose` shells out to "
            "`docker compose`; the rendered file is written either way, so you can run "
            "it by hand.",
            file=sys.stderr,
        )
        return 2
    if completed.returncode != 0:
        print(
            f"error: docker compose exited {completed.returncode}. The workers' own logs "
            "are the place to look: a container that failed preflight says which server "
            "it could not reach.",
            file=sys.stderr,
        )
        return 2
    print(f"  next:  refractal compare {args.results} {plan.plan_id}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Write a deployment description from a plan. No Docker, no execution."""
    from .render import ComposeSettings, RenderError, render_compose
    from .schema.plan import read_plan

    plan = read_plan(args.plan)
    try:
        text = render_compose(
            plan,
            ComposeSettings(
                servers=_servers(args.server),
                user=args.user,
                plan_file=args.plan_file or args.plan,
                catalog_dir=args.catalog,
                results_dir=args.results,
                images=dict(
                    pair.split("=", 1) for pair in (args.image or []) if "=" in pair
                ),
                host_gateway=not args.no_host_gateway,
                source_mounts=list(args.mount or []),
            ),
        )
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.write_text(text, encoding="utf-8")
    print(f"  wrote {out}  ({text.count(chr(10) + '  ') and ''}"
          f"{sum(len(s.workers) for s in plan.scenes)} service(s))")
    print(f"  next:  mkdir -p {args.results} && docker compose -f {out} up")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Exit code is the gate. 0 clean, 1 regression, 2 unanswerable."""
    from .compare import build_units, evaluate, render
    from .execute import ResultWriter, read_episodes

    rows = read_episodes(args.results, args.plan_id).to_pylist()

    promotion = None
    if args.promote_from:
        # Explicit, and refused when the runs do not nest. `tier` stays in
        # plan_id -- the two runs keep their different ids and this does not
        # merge them, it pools their rows for one comparison and says so.
        from .promote import PromotionError, promote
        from .execute.results import comparison_prefix
        from .schema.plan import locate_plan, read_plan

        try:
            # The comparison directory, not the results root: a results tree with
            # several comparisons in it has several plan.json files, and
            # locate_plan refuses to guess between them. Naming the one we are
            # comparing is the whole point of having the id.
            plan = read_plan(
                locate_plan(comparison_prefix(args.results, args.plan_id))
            )
        except Exception as exc:
            print(
                f"error: --promote-from needs this run's plan.json to know which "
                f"episodes it contains, and it could not be read: {exc}",
                file=sys.stderr,
            )
            return 2
        promoted_rows = read_episodes(args.results, args.promote_from).to_pylist()
        try:
            rows, promotion = promote(plan, rows, promoted_rows, args.promote_from)
        except PromotionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"  promoted {promotion.episodes} episode(s) in {promotion.rows} row(s) from "
            f"{args.promote_from[:19]}... ({promotion.remaining} still unrun)"
        )

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
        checkpoints,
        baseline=args.baseline,
        rule=args.dichotomy,
        resamples=args.resamples,
        correct=not args.no_correction,
        allow_harness_mismatch=args.allow_harness_mismatch,
        surface_manifests=ResultWriter(args.results, args.plan_id).read_harness_manifests(),
    )
    if promotion is not None:
        # On the verdict, not only on stdout. A rendered verdict that does not
        # say it pooled two runs reads as an ordinary one, and somebody reading
        # it later has no way to know -- which is the failure mode promotion was
        # designed to avoid in the first place.
        verdict.notes.append(
            f"{promotion.episodes} episode(s) were PROMOTED from run "
            f"{promotion.from_plan_id} for checkpoint(s) {promotion.checkpoints}. "
            "Those runs are different experiments by plan_id and were pooled on "
            "request; every promoted episode id was verified to be one this plan "
            "contains, which is exact because episode_id does not cover tier."
        )
    print(render(verdict))
    return verdict.exit_code


def cmd_build(args: argparse.Namespace) -> int:
    """Establish the facts that need an engine, and write catalog/build.lock."""
    from .build import build as build_catalog

    probe = None
    if args.probe:
        # Resolved here and nowhere else. An externally-defined scene has no
        # catalog-local model to hash, so the facts that identify it can only come
        # from the package that owns it -- and `build` is the one verb allowed to
        # require that package be installed.
        from .schema.importstr import resolve_import_string

        try:
            probe = resolve_import_string(args.probe)()
        except Exception as exc:
            print(f"error: could not load probe {args.probe!r}: {exc}", file=sys.stderr)
            return 2

    report = build_catalog(args.catalog, hardware_profile=args.hardware, probe=probe)
    for line in report.summary_lines():
        print(line)
    for note in report.notes:
        print(f"  {note}")
    for warning in report.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    print(f"  wrote {Path(args.catalog) / 'build.lock'}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Say why two plans are different experiments."""
    from .schema.plan import explain_identity_difference, read_plan

    before, after = read_plan(args.before), read_plan(args.after)
    if before.plan_id == after.plan_id:
        print(f"  same experiment: {before.plan_id[:26]}...")
        return 0
    print(f"  {before.plan_id[:26]}...\n  {after.plan_id[:26]}...\n")
    for line in explain_identity_difference(before, after):
        print(f"  {line}")
    print("\n  Results under these two ids are separate comparisons and will not join.")
    return 0


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
        "--expect-from", metavar="RESULTS_URI",
        help="also print an EXPECTED duration, learned from a prior run's episode "
             "lengths. The bound is always printed; this is the second number. Never "
             "written to plan.json -- it is fitted to that run's hardware and "
             "checkpoints, so it is render-time, not part of the experiment")
    plan.add_argument(
        "--expect-plan", metavar="PLAN_ID",
        help="plan_id of the prior run, when it is not this one -- which is the "
             "normal case, since changing a step budget changes plan_id")
    plan.add_argument(
        "--pack-below-startup-sec",
        type=int,
        default=30,
        help="pack starved scenes sequentially when startup_sec is below this "
        "(at 90s load, packing eleven scenes costs 16 minutes of pure loading)",
    )
    plan.set_defaults(func=cmd_plan)

    init_cmd = sub.add_parser(
        "init", help="write a working example catalog", description=
        "The example declares no filter on purpose: filters need 'refractal build', "
        "which needs the engine, so a filtered example could not be planned.")
    init_cmd.add_argument("target", nargs="?", default=".", help="directory to write into")
    init_cmd.add_argument("--force", action="store_true", help="overwrite existing files")
    init_cmd.set_defaults(func=cmd_init)

    run_cmd = sub.add_parser("run", help="execute a plan")
    run_cmd.add_argument("plan", help="path to plan.json")
    run_cmd.add_argument("-o", "--results", default="./results", help="results URI")
    run_cmd.add_argument("--catalog", help="catalog to copy in as provenance")
    run_cmd.add_argument(
        "--backend", choices=("local", "vla-eval", "compose"), default="local",
        help="'local' simulates; 'vla-eval' drives the harness in this process "
             "against running model servers; 'compose' renders one container per "
             "worker over that same entrypoint and starts them. The model servers "
             "stay outside in every case")
    run_cmd.add_argument(
        "--server", action="append", metavar="CKPT=URL",
        help="model server for a checkpoint, e.g. --server pi0=http://localhost:8000. "
             "Repeat once per checkpoint. Every checkpoint in the plan needs one, and "
             "two checkpoints sharing a URL is refused -- that is a self-comparison")
    run_cmd.add_argument(
        "--harness-output", default="./vla-eval-output",
        help="scratch directory for the harness's own outputs. Not the results URI: "
             "Refractal's results go to --results as Parquet")
    run_cmd.add_argument(
        "--worker", metavar="WORKER_ID",
        help="run only this worker's episodes, e.g. libero-spatial/0. The ids come "
             "from 'refractal plan -v'. A filter, not a different plan: plan_id and "
             "every episode id are unchanged, so several workers' rows join as one "
             "experiment. This is the entrypoint --backend compose renders against")
    run_cmd.add_argument("--compose-file", default="docker-compose.yml",
                         help="--backend compose: where to write the rendered file")
    run_cmd.add_argument("--user", metavar="UID:GID",
                         help="--backend compose: defaults to the current uid:gid")
    run_cmd.add_argument("--image", action="append", metavar="ENGINE=IMAGE")
    run_cmd.add_argument("--mount", action="append", metavar="HOST:CONTAINER",
                         help="--backend compose: extra read-only mount, repeatable")
    run_cmd.add_argument("--detach", action="store_true",
                         help="--backend compose: 'up -d' instead of waiting")
    run_cmd.add_argument("--session-id", help="fixed session id, for reproducible tests")
    run_cmd.add_argument("--no-resume", action="store_true",
                         help="re-run episodes that already have results")
    run_cmd.set_defaults(func=cmd_run)

    build_cmd = sub.add_parser(
        "build",
        help="compute scene hashes, evaluate filters, record resource shapes",
        description="Runs where the engine is installed. Writes catalog/build.lock; "
        "never rewrites hand-authored YAML.",
    )
    build_cmd.add_argument("catalog", help="path to the catalog directory")
    build_cmd.add_argument("--hardware", help="hardware profile to record shapes for")
    build_cmd.add_argument(
        "--probe", metavar="module:Class",
        help="probe supplying facts only an installed engine can answer, e.g. "
             "refractal_libero.probe:LiberoProbe. Required for externally-defined "
             "scenes, whose geometry lives in someone else's package")
    build_cmd.set_defaults(func=cmd_build)

    explain = sub.add_parser(
        "explain", help="why two plans are different experiments",
        description="A plan_id says two runs differ; this says which field differs.")
    explain.add_argument("before")
    explain.add_argument("after")
    explain.set_defaults(func=cmd_explain)

    render = sub.add_parser(
        "render", help="write a deployment description from a plan (no Docker)")
    render.add_argument("plan", help="path to plan.json")
    render.add_argument(
        "--target", choices=("compose",), default="compose",
        help="only 'compose' exists; k8s is explicitly out of scope")
    render.add_argument("-o", "--output", default="docker-compose.yml")
    render.add_argument(
        "--server", action="append", metavar="CKPT=URL",
        help="model server for a checkpoint. Point at the HOST, not a service name: "
             "the servers stay outside Compose, bare and warm")
    render.add_argument(
        "--user", metavar="UID:GID",
        help="emitted literally, e.g. 1000:1000. Pass \"$(id -u):$(id -g)\" -- NOT "
             "'${UID}:${GID}', which is not exported by default and interpolates to ':'")
    render.add_argument("--catalog", default="./catalog",
                        help="host path to mount read-only as provenance")
    render.add_argument("--results", default="./results", help="host path for results")
    render.add_argument("--plan-file", help="host path to the plan, if not the path given")
    render.add_argument("--image", action="append", metavar="ENGINE=IMAGE")
    render.add_argument(
        "--mount", action="append", metavar="HOST:CONTAINER",
        help="extra read-only mount, repeatable. For iterating on source without "
             "rebuilding the image")
    render.add_argument("--no-host-gateway", action="store_true",
                        help="omit the host.docker.internal mapping")
    render.set_defaults(func=cmd_render)

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
    compare.add_argument(
        "--baseline", help="checkpoint every other is measured against (default: the first)"
    )
    compare.add_argument("--min-seeds", type=int, default=None, help="seed floor per scenario")
    compare.add_argument(
        "--no-correction",
        action="store_true",
        help="skip the Holm correction across the family of contrasts. Only for "
        "inspecting a single contrast; the gate is not calibrated without it.",
    )
    compare.add_argument(
        "--allow-harness-mismatch",
        action="store_true",
        help="compare across harness versions anyway. Blocked by default: a harness "
        "change can alter which observation parameters reach the benchmark.",
    )
    compare.add_argument(
        "--promote-from", metavar="PLAN_ID",
        help="also read episodes from another run and pool them into this "
             "comparison. For reusing a smoke run inside a full one: `tier` is in "
             "plan_id so they are different experiments, and this says so out loud "
             "rather than letting them join silently. Refused unless every promoted "
             "episode id is one this plan contains")
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
