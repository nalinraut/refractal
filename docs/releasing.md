# Releasing

A PyPI upload is permanent. `name` plus `version` can never be uploaded twice,
so this is about getting it right once rather than recovering.

## Checklist

```console
# 1. The tree is what you think it is.
git status --short                       # must be empty
./scripts/test.sh

# 2. Delete dist/ and rebuild. Do not reuse it.
rm -rf dist && uv build

# 3. Verify the artefact, not the source tree.
python scripts/verify_wheel.py dist/*.whl
twine check dist/*

# 4. Install the wheel into a clean venv and run the five-command walkthrough
#    from it. Without the execute extra first: `plan` and `build` must work
#    with no pyarrow, no fsspec and no simulator.

# 5. Re-verify the claims against whatever [vla-eval] resolves to now.
python scripts/verify_harness_claims.py

# 6. Rehearse. TestPyPI is free and catches README rendering.
twine upload --repository testpypi dist/*

# 7. Upload.
twine upload dist/*
```

## Why step 2 deletes rather than rebuilds

`uv build` adds to `dist/` and does not clear it. A stale wheel from an earlier
version stays, `twine upload dist/*` uploads both, and the older one wins on any
resolver that sees it first. Nothing warns.

## Why step 3 checks the artefact

`scripts/verify_wheel.py` compares the wheel's contents against the source tree,
deriving the expected package list rather than hardcoding it.

```console
$ python scripts/verify_wheel.py dist/refractal-0.1.0a1-py3-none-any.whl
  refractal-0.1.0a1-py3-none-any.whl
  source has 6: ['build', 'compare', 'execute', 'render', 'resolve', 'schema']
  wheel has  6: ['build', 'compare', 'execute', 'render', 'resolve', 'schema']

  Wheel matches the source tree (6 subpackages).
```

Exit 1 names what is missing or extra. Missing usually means an unanchored
`.gitignore` pattern — `build/` matches `src/refractal/build/` as well as
`./build/`, and the result is a wheel short an entire subpackage while
`git status` stays clean. Extra means a stale `dist/`.

It does not check that the source tree is what you think it is, which is why
`git status` is step 1.

## Why step 5 re-runs the claims

A published version pins a dependency *range*. The claims are the evidence that
Refractal's assumptions hold for the harness a user will actually get, which is
not necessarily the one you developed against.

## Version numbers

`0.1.0a1` keeps the alpha marker. Identity-bearing fields are still moving, and a
version whose hashes change orphans every result computed under it.
