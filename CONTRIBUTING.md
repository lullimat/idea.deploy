# Contributing to idea.deploy

Thanks for looking. This document covers how to get a working environment, how
to run the tests properly, and the conventions that exist because ignoring them
has caused real problems.

## Getting a development environment

```bash
git clone https://github.com/lullimat/idea.deploy.git
cd idea.deploy
bash idpy-init.sh          # builds py-env/idpy-env, compiles PyCUDA/PyOpenCL where possible
```

The interpreter is then `./py-env/idpy-env/bin/python`. Everything below assumes
you run tests through it, from the repository root.

For use as a library rather than development, `pip install idpy` works on the
current layout and needs none of the above.

## Running the tests

There are two kinds, and they behave differently.

**`unittest` suites** — `idpy/test.py`, `idpy/LBM/test.py`, `idpy/Metal/*`:

```bash
python -m idpy.test
python -m unittest idpy.LBM.test        # NOT python -m idpy.LBM.test
```

> `idpy/LBM/test.py` defines `TestCase` classes but has no `unittest.main()` and
> no `__main__` guard. Running it as a module executes **zero tests and exits 0**.
> It looked green for years while erroring at construction. Always use
> `python -m unittest` for it.

**Print-style suites** — `test_shared`, `test_residency`, `test_residency_policy`,
`test_linkage`, `test_constants`, `test_shared_tiles`, `test_hostmodule`,
`test_storage_bandwidth`, `test_overlap`:

```bash
python -m idpy.IdpyCode.test_residency
```

These print one line per check, skip backends that are absent, and **exit
non-zero on failure** (see `idpy/Utils/TestExit.py`). A script that verified
nothing says so explicitly rather than exiting 0 quietly — "0 checks ran" must
never look like "all checks passed".

## What CI covers, and what it cannot

GitHub Actions runs the CTypes backend on Linux and macOS, plus OpenCL on Linux
via POCL. **CUDA and Metal are not covered** and must be verified by hand on
hardware. If you change anything touching those paths, say so in the PR and
state whether it was run on real hardware or is codegen-only.

The layering lint (`scripts/check_layering.py`) runs first, before dependencies,
and enforces that `idpy` core never imports `idpy` physics. Two violations are
grandfathered in a `KNOWN` allowlist; nothing new may be added.

## Conventions that exist for a reason

**Branch per piece of work, named for what it does.** PRs stay scoped to one
idea. A branch name that has stopped matching its contents is a small ongoing
cost.

**Say what was verified and where.** "Verified on CUDA" and "codegen-only, needs
a CUDA machine" are different claims. Backend-specific work that has not run on
that backend is unverified, and should be labelled so rather than implied to
work.

**An API that claims something must do it.** `H2D(async_=True)` silently ignored
its argument for years; a `DirectPathName()` once named a path that was never
taken. Both passed every correctness test. If a fast path can silently degrade
to a slow one, add a counter that makes the degradation visible, and assert on
it.

**Measurement discipline.** Performance numbers in this repository have been
wrong more often than the code they measured. If you add one:

- Report a **range**, not a single sample. A bandwidth figure here has moved by
  4x between identical runs.
- Include a **control** — a quantity you know should not differ, measured the
  same way. It is what establishes the noise floor.
- Sweep the **transfer size** rather than picking one. A single size is a point
  on a curve reported as though it were the curve.
- State what the number is *of*: a warm page cache is not a disk, and a figure
  above the device rate is reading cache by definition.
- Never gate a build on throughput. Correctness gates; performance reports.

**Precision of constants.** A bare Python float becomes a C `double` and will
silently promote fp32 arithmetic — measured at ~200x slower on a consumer NVIDIA
part, and worse, it makes the same kernel compute at different precision on
different backends. Declare `constants_types={'X': 'FType'}` or pass
`np.float32(...)`; a warning fires if you do neither.

## Design documents

- `STRATEGY.md` — roadmap, phases, positioning
- `docs/residency-probes.md` — the findings record for the residency layer,
  including several conclusions that were recorded, falsified, and corrected.
  Kept that way on purpose: the wrong turns show which class of error produced
  them.

Read the relevant section before re-litigating a decision. Several have been
settled twice because the first answer was wrong, and the reasoning is written
down where it can be checked.
