# Releasing

**Not yet, and the order matters.** See [what has to land first](#what-has-to-land-first)
before running any of this. The trigger for publishing at all is a second person
needing to install it; until then a release solves a distribution problem that
does not exist, and publishing before a change already written down as likely is
choosing the worse order.

A PyPI upload is permanent. You can yank or delete a release, but the
name-plus-version is burned forever — `refractal 0.1.0a1` can never be uploaded
twice. So everything here is about getting it right once rather than recovering.

## Checklist

```console
# 1. The tree is what you think it is.
git status --short                       # must be empty
PYTHONPATH=src python -m unittest discover -s tests -t .

# 2. DELETE dist/ AND REBUILD. Do not reuse it. See below.
rm -rf dist && uv build

# 3. Verify the artifacts, not the source tree.
python scripts/verify_wheel.py dist/*.whl
twine check dist/*

# 4. Install the wheel into a clean venv and run the seconds test from it.
#    Without the execute extra first: `plan` and `build` must work with no
#    pyarrow, no fsspec and no simulator.

# 5. Re-verify the claims against whatever [vla-eval] resolves to NOW.
#    A published version pins a dependency RANGE (>=0.6.0), and the claims are
#    what make that range safe: they are the evidence that the bridge's
#    assumptions hold for the harness a user will actually get, which is not
#    necessarily the one developed against. A release is the moment the range
#    stops being hypothetical.
python scripts/verify_harness_claims.py          # all 6 must hold
#    And the LIBERO claims, if refractal-libero ships in the same cut.

# 6. Rehearse. TestPyPI is free and catches README rendering.
twine upload --repository testpypi dist/*

# 7. Upload.
twine upload dist/*
```

## What has to land first

Five, in order. The first two are gating; the rest is procedure.

1. ~~**Settle `partition_unit`**~~ **DONE** — — what a worker must own *whole*, as distinct from
   how many episodes may share a process at once. It touches `ResourceShape`, the
   most depended-on model in the project, and it is already written down as
   likely to change. See
   the worker-unit write-up in the development record. Needs an
   MJX scene to design against honestly, which is the same prerequisite as scene
   affinity on the unexercised list — one fixture, two items.

2. ~~**Settle `tier` in identity**~~ **DONE** — — whether a smoke run and a full run of one
   catalog are one experiment or two. Smaller than the first, and not a tweak: it
   changes what "the same experiment" means. See
   the tier write-up in the development record.

3. **Run the seconds test from a built wheel in a clean venv**, without the
   `execute` extra first. This is the step that caught both prior artifact bugs.

4. **Rehearse on TestPyPI.** The only way to see the page before it is permanent.

5. **Publish `0.1.0a1`.** Keep the `a`.

Both blockers are closed. What remains is steps 3-5, which are procedure.

Why the order was this rather than publishing first: `plan_id` stability *is* the product,
and both open questions move it. Verified that they move it *safely* — adding a
field to `task_identity` changes `plan_id` (`6212937…` → `6790ee62…`), so results
from two versions land in different comparison directories and cannot silently
join. But "cannot silently join" is the floor, not the goal. Publishing a version
whose hashes are known to be about to move means every result computed under it is
orphaned by design.

## Why step 2 says delete rather than rebuild

**`dist/` gets deleted and rebuilt, never reused.** This is not hygiene, it is the
one failure in this process that is invisible.

The version-permanence argument is about not burning a version number on a wrong
rule. The near-miss came from the direction nobody was watching: not uploading
too early, but having built the artifacts *before* two fixes landed and then not
rebuilding. `dist/refractal-0.1.0a1-py3-none-any.whl` carried a stale
`scenario_hash` rule and a missing scene-freshness check, and the filename was
byte-identical to the correct one. Nothing in `twine check`, `git status` or the
test suite looks at `dist/`.

Uploading that would have burned `0.1.0a1` on a `scenario_hash` rule we had
already decided was wrong — which is exactly the outcome the permanence argument
exists to prevent, arriving through a door the argument did not mention.

## Why the wheel is checked rather than the source tree

A `.gitignore` pattern of `build/` — unanchored — also matches
`src/refractal/build/`, and hatchling honours VCS ignores. The entire
`refractal.build` package was silently absent from both git and the wheel while
`git status` reported clean and the test suite passed against the working tree.

Step 3 exists because of that, and the CI `seconds-test` job installs the built
wheel rather than an editable checkout for the same reason. Two bugs have now
been caught by looking at the artifact instead of the repo.

## Moving the vla-eval pin

```console
python scripts/verify_harness_claims.py        # must exit 0
```

Six claims Refractal makes about the harness, re-checked against the installed
version. CI runs it on every push, so a release that breaks one fails before the
pin moves rather than after something depends on it.

It is separate from `harness_surface` on purpose: the digest catches a *change*
to files Refractal subclasses; these are claims about things being *absent* or
*still configurable*, which a change-detector cannot express.

## Version numbers

- Pre-release (`0.1.0a1`) means `pip install refractal` reports no matching
  distribution until a final release exists. Correct while there is no simulator
  adapter, but note the README's install line assumes a final version.
- If a placeholder release was ever published to hold the name, **yank it** in
  the same sitting as the first real release. A final `0.0.0` outranks a
  pre-release `0.1.0a1` in pip's resolution, so the stub is what users get.

## Licence and author

`Apache-2.0`, and `authors` carries a name with no email. Both are effectively
permanent once published. Confirm them before step 6 rather than after.
