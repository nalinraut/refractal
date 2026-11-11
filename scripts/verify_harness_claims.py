#!/usr/bin/env python3
"""Re-verify the claims Refractal makes about the harness it does not control.

Eight claims, each of which was checked once by reading and would otherwise stay
true by assumption. They are not covered by ``harness_surface``: that digest
catches a *change* to files Refractal subclasses, which is a different question
from "is this thing still absent" or "is this still a default rather than a
constant".

Why this is a script rather than a line in a checklist
-----------------------------------------------------

The claims are dated, which makes them degrade visibly -- but only to someone who
looks. Running them on every pin move turns "remember to re-verify" into "the
build fails until you do", which is the same move as printing the design effect
unconditionally and asserting that the fixture exercises its own invariants.

The failure this guards against, concretely: ``refractal-design.md`` asserted for
nine days that the harness hardcodes the model server address to localhost. It
reached the design doc, the implementation prompt, the API reference and the
Kubernetes discussion, and it decided that multi-host was blocked upstream. It
came from a grep whose only hits were a docstring example and ``--network host``
help text. Nothing tested it because nothing needed multi-host, and an assertion
about someone else's code -- unlike a guard -- does not fail when it is wrong. It
just sits there.

Usage::

    python scripts/verify_harness_claims.py            # against the installed harness
    python scripts/verify_harness_claims.py PATH       # against a source checkout

Exit code 0 if every claim holds, 1 if any has broken, 2 if the harness is not
importable and no path was given.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Claim:
    name: str
    #: What Refractal does because of this claim. If it breaks, this is what
    #: breaks -- stated so a failure is actionable rather than just red.
    relies_on: str
    holds: bool
    detail: str


def _read(root: Path, relative: str) -> str:
    path = root / relative
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _files(root: Path, relative: str) -> list[Path]:
    directory = root / relative
    return sorted(directory.rglob("*.py")) if directory.is_dir() else []


def check_store_gate(root: Path) -> Claim:
    source = _read(root, "orchestrator.py")
    holds = "_store is None" in source and "NullEpisodeRecorder()" in source
    return Claim(
        "the recorder gate is `self._store is None`",
        "the bridge overrides _build_recorder AND the store setup; if the gate moves, "
        "overriding one of them silently records nothing",
        holds,
        "found in orchestrator.py" if holds else "GATE NOT FOUND — re-read _build_recorder",
    )


def check_db_path_unused(root: Path) -> Claim:
    offenders = []
    for path in _files(root, "model_servers"):
        if path.name in {"base.py", "serve.py"}:
            continue  # the plumbing itself, not a caller
        text = path.read_text(encoding="utf-8")
        # Both a call and a bare import. A model server importing StepRecorder
        # has no other purpose, and catching only the call lets the claim break
        # one commit before this notices. (Found by injecting an import to test
        # this check and watching it pass.)
        if re.search(
            r"\brecording_db_path\b|\bStepRecorder\s*\(|import\s+StepRecorder|"
            r"from\s+[\w.]*recording\s+import\s+[^\n]*StepRecorder",
            text,
        ):
            offenders.append(path.name)
    return Claim(
        "no model server reads db_path",
        'ParquetEpisodeRecorder returns "" for db_path. sqlite3.connect("") does not '
        "raise — it makes a temp database that vanishes — so a server that started "
        "reading it would lose its step data silently",
        not offenders,
        "no callers" if not offenders else f"NOW READ BY: {', '.join(offenders)}",
    )


def check_record_fields_advisory(root: Path) -> Claim:
    source = _read(root, "orchestrator.py")
    validated = '_ALL_RECORD_FIELDS' in source and "getattr(" in source
    declaring = sum(
        1 for path in _files(root, "benchmarks")
        if "_ALL_RECORD_FIELDS" in path.read_text(encoding="utf-8")
    )
    return Claim(
        "_ALL_RECORD_FIELDS is per-benchmark and advisory",
        "any adapter declares whatever step fields it wants; if this became a fixed "
        "list, a custom field would be rejected. (This said 'the SO-101 adapter', "
        "which was descoped -- the claim held while its stated reason named nothing.)",
        validated and declaring > 0,
        f"validated via getattr, {declaring} benchmark(s) declare one",
    )


def check_cross_validation_inline(root: Path) -> Claim:
    source = _read(root, "orchestrator.py")
    holds = "Spec cross-validation" in source
    lines = 0
    if holds:
        rows = source.splitlines()
        start = next(i for i, line in enumerate(rows) if "Spec cross-validation" in line)
        end = next(
            (i for i, line in enumerate(rows[start:], start) if "cfg.mode.startswith" in line),
            start,
        )
        lines = end - start
    return Claim(
        "spec cross-validation is inline in _run_benchmark_inner",
        "the bridge subclasses rather than hand-rolling a driver, because this is what "
        "catches an absolute-vs-delta action space — the mismatch that produces 0%",
        holds,
        f"{lines} lines, inline" if holds else "NOT FOUND — it may have been extracted",
    )


def check_work_items_inline(root: Path) -> Claim:
    source = _read(root, "orchestrator.py")
    inline = "work_items = [" in source
    extracted = bool(re.search(r"def\s+_?build_work_items", source))
    return Claim(
        "the work-item loop is inline, not an overridable method",
        "worker_selection expresses a worker's assignment in the harness's own terms "
        "instead of overriding the loop; if this became a method, that could be dropped",
        inline and not extracted,
        "inline" if inline and not extracted else "IT IS NOW A METHOD — simplify the bridge",
    )


def check_server_url_is_a_default(root: Path) -> Claim:
    source = _read(root, "config.py")
    holds = bool(re.search(r'url\s*=\s*data\.get\(\s*["\']url["\']', source))
    return Claim(
        "the model server address is configurable, not hardcoded",
        "multi-host runs need to address a server by hostname. This claim was ASSERTED "
        "WRONGLY for nine days from a grep that hit a docstring",
        holds,
        "overridable from YAML" if holds else "NO LONGER OVERRIDABLE — k3s path is blocked",
    )


def _server_call_seed(root: Path):
    """The harness's own `_call_seed`, lifted out and made callable.

    Extracted by AST and executed with nothing but `hashlib` in scope, rather
    than imported: importing the module pulls torch and a policy. And rather
    than compared as text, because two spellings of the same arithmetic are
    equal and a regex would call them different -- while a changed constant
    inside an unchanged-looking line would slip past.

    Returns None if the function is no longer there, which is itself the
    finding.
    """
    import ast
    import hashlib

    source = _read(root, "model_servers/lerobot.py")
    if not source:
        return None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_call_seed":
            namespace: dict = {"hashlib": hashlib}
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         "<harness>", "exec"), namespace)
            return namespace["_call_seed"]
    return None


def check_call_seed_agrees(root: Path) -> Claim:
    """The claim `refractal/seeding.py` makes in its own docstring.

    Both halves of the derivation are written down in one place so there is
    one definition; this is what makes that true rather than aspirational. The
    server computes `call_seed` because only it knows the step, so nothing in
    a run would notice the two drifting apart -- the rows would record a
    `policy_seed` and a scheme, and the noise would come from a different
    rule.
    """
    from refractal.seeding import call_seed

    theirs = _server_call_seed(root)
    if theirs is None:
        return Claim(
            "the server derives call_seed by refractal's rule",
            "`policy_seeding` claims a reproducible noise draw per (episode, "
            "step). Without the server's half, the scheme named in plan_id "
            "describes a derivation nothing performs",
            False,
            "model_servers/lerobot.py has no _call_seed -- the server's half is gone",
        )
    # Across episode seeds and steps, including the boundaries where a
    # truncation or a shift would differ from ours and nowhere else.
    for seed in (0, 1, 2 ** 31, 2 ** 62, 2 ** 63 - 1):
        for step in (0, 1, 2, 9, 10, 255, 999):
            if theirs(seed, step) != call_seed(seed, step):
                return Claim(
                    "the server derives call_seed by refractal's rule",
                    "`policy_seeding` claims a reproducible noise draw per "
                    "(episode, step). If the two rules differ, every recorded "
                    "policy_seed names a derivation that did not run",
                    False,
                    f"DIVERGES at episode_seed={seed}, step={step}: "
                    f"harness {theirs(seed, step)} vs refractal "
                    f"{call_seed(seed, step)}",
                )
    return Claim(
        "the server derives call_seed by refractal's rule",
        "`policy_seeding` claims a reproducible noise draw per (episode, step)",
        True,
        "identical over 35 (seed, step) pairs including the 2**63 boundary",
    )


def check_seed_schemes_agree(root: Path) -> Claim:
    """Refractal must not name a scheme the server cannot implement.

    The server refuses an unknown scheme at episode start, which is the right
    place for it -- but that refusal lands after a plan is built, a results
    directory is created and the models are loaded. Here it lands before
    anything runs.
    """
    import re as _re

    from refractal.seeding import SCHEMES

    source = _read(root, "model_servers/lerobot.py")
    match = _re.search(r"_SEED_SCHEMES\s*=\s*\(([^)]*)\)", source)
    if not match:
        return Claim(
            "the server implements every scheme refractal can name",
            "a catalog declaring `policy_seeding` is refused at episode start "
            "rather than at plan time if the server does not know the scheme",
            False,
            "model_servers/lerobot.py declares no _SEED_SCHEMES",
        )
    theirs = set(_re.findall(r"[\"\']([^\"\']+)[\"\']", match.group(1)))
    missing = sorted(set(SCHEMES) - theirs)
    return Claim(
        "the server implements every scheme refractal can name",
        "a catalog declaring `policy_seeding` would be refused at episode "
        "start -- after the plan, the results directory and the model load",
        not missing,
        f"both sides declare {sorted(theirs)}" if not missing
        else f"refractal names {missing}, which the server does not implement",
    )


CHECKS = (
    check_store_gate,
    check_db_path_unused,
    check_record_fields_advisory,
    check_cross_validation_inline,
    check_work_items_inline,
    check_server_url_is_a_default,
    check_call_seed_agrees,
    check_seed_schemes_agree,
)


def locate_harness(argv: list[str]) -> Path | None:
    if len(argv) > 1:
        root = Path(argv[1])
        return root / "src" / "vla_eval" if (root / "src" / "vla_eval").is_dir() else root
    try:
        import vla_eval  # type: ignore
    except ImportError:
        return None
    return Path(vla_eval.__file__).parent


def main(argv: list[str]) -> int:
    root = locate_harness(argv)
    if root is None or not root.is_dir():
        print(
            "vla_eval is not importable and no source path was given.\n"
            "  pip install 'refractal[vla-eval]'   or   "
            "python scripts/verify_harness_claims.py /path/to/vla-evaluation-harness",
            file=sys.stderr,
        )
        return 2

    if len(argv) > 1:
        source = f"{root} (source checkout)"
    else:
        import vla_eval  # type: ignore

        source = f"installed vla-eval {getattr(vla_eval, '__version__', 'unknown')}"

    print(f"Verifying harness claims against {source}\n")
    claims = [check(root) for check in CHECKS]
    width = max(len(c.name) for c in claims)
    for claim in claims:
        mark = "ok  " if claim.holds else "FAIL"
        print(f"  [{mark}] {claim.name:<{width}}  {claim.detail}")

    broken = [c for c in claims if not c.holds]
    if not broken:
        print(f"\nAll {len(claims)} claims hold.")
        return 0

    print(f"\n{len(broken)} claim(s) no longer hold. What each one was holding up:\n")
    for claim in broken:
        print(f"  {claim.name}\n    {claim.relies_on}\n")
    print("Record the new finding and a date, and fix")
    print("whatever depended on it, before moving the pin.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
