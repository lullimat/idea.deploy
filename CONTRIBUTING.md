# Contributing to idea.deploy

Thanks for looking. This document covers how to get a working environment, how
to run the tests properly, and the conventions that exist because ignoring them
has caused real problems.

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Getting a development environment

```bash
git clone https://github.com/lullimat/idea.deploy.git
cd idea.deploy
python3 -m venv .venv
.venv/bin/pip install -e ".[physics]"
```

The interpreter is then `.venv/bin/python`. Everything below assumes you run
tests through it, from the repository root.

**Installing is not optional any more.** The package lives at `src/idpy/`, so
the repository root holds nothing importable: `import idpy` resolves to the
installed distribution or it fails. That is the point of the `src/` layout —
under the old flat layout `import idpy` silently picked up `./idpy` for
anything run from the checkout, so a packaging mistake was invisible at home
and broke everyone who installed from an index.

`bash idpy-init.sh` still exists and still works. It builds a ~2 GB
`py-env/idpy-env` and compiles PyCUDA and PyOpenCL from source, which you need
only if you are provisioning those backends from nothing; Phase 0a made the
package pip-installable precisely so that ordinary development does not. For
Metal, `bash scripts/install-pymetallic.sh` — pymetallic comes from a pinned
revision carrying an idpy patch, not from an index.

For use as a library rather than development, `pip install idpy`.

## Running the tests

There are two kinds, and they behave differently.

**`unittest` suites** — `src/idpy/test.py`, `src/idpy/physics/lbm/test.py`,
`src/idpy/core/backends/metal/*`:

```bash
python -m idpy.test
python -m unittest idpy.physics.lbm.test   # NOT python -m idpy.physics.lbm.test
```

> `src/idpy/physics/lbm/test.py` defines `TestCase` classes but has no
> `unittest.main()` and no `__main__` guard. Running it as a module executes
> **zero tests and exits 0**. It looked green for years while erroring at
> construction. Always use `python -m unittest` for it.

**Print-style suites** — `test_shared`, `test_residency`, `test_residency_policy`,
`test_linkage`, `test_constants`, `test_shared_tiles`, `test_hostmodule`,
`test_storage_bandwidth`, `test_overlap`:

```bash
python -m idpy.core.test_residency
```

These print one line per check, skip backends that are absent, and **exit
non-zero on failure** (see `src/idpy/core/utils/TestExit.py`). A script that
verified nothing says so explicitly rather than exiting 0 quietly — "0 checks
ran" must never look like "all checks passed".

That rule is not theoretical, and the checkers are not exempt from it.
`scripts/check_layering.py` used to walk `idpy/<area>/`; when Phase 0b moved
everything under `src/`, those directories stopped existing, its `rglob`
returned nothing and it printed `layering OK` and exited 0 having read zero
files — staying green through the single commit most capable of breaking the
invariant it exists to protect. It now reports how many files it scanned and
fails when that is zero. **A check that cannot find its subject must fail, not
pass.**

## What CI covers, and what it cannot

GitHub Actions runs the CTypes backend on Linux and macOS, plus OpenCL on Linux
via POCL. **CUDA and Metal are not covered** and must be verified by hand on
hardware. If you change anything touching those paths, say so in the PR and
state whether it was run on real hardware or is codegen-only.

The layering lint (`scripts/check_layering.py`) runs first, before dependencies,
and enforces that `src/idpy/core/` never imports `idpy.physics`. It checks the
legacy spellings too — `from idpy.LBM...` inside core still resolves through the
compatibility shims and would violate the invariant while looking like neither.
One violation is grandfathered in a `KNOWN` allowlist; nothing new may be added.

## The consumer surface

`papers/` and `collabs/` import `idpy` by module path, and for a long time
nothing checked that those paths still resolved. Two things make that dangerous:
a rename breaks every consumer at once, and **the consumers are mostly
untracked** — `collabs/` is 1 tracked file out of 80, `papers/` 2 out of 19,
because the `arXiv-*` checkouts are separate gitignored repositories. A fresh
clone cannot see what it must not break, and neither can CI.

So the surface is frozen into two committed fixtures and checked anywhere:

```bash
python3 scripts/check_consumers.py --check-surface   # 47 modules
python3 scripts/check_consumers.py --check-symbols   # 277 (module, symbol) pairs
```

**Run both.** They catch different things: a shim that forwards the module but
drops a symbol passes the first and fails the second. That is how three
pre-existing breakages went unnoticed; they are grandfathered with a leading `!`
in `scripts/consumer-symbols.txt`, on the same principle as the layering lint's
`KNOWN` allowlist — visible every run, failing none, and nothing new may join
them silently.

**Do not regenerate a fixture unless `papers/` and `collabs/` are populated.**
`--freeze` and `--freeze-symbols` refuse an *empty* surface but cannot detect a
*partial* one, so running them in a fresh clone would silently shrink the thing
they protect.

**Publish aggregates, never paths.** Those directories hold unpublished research
and named collaborations, and this repository's history is public and permanent
— a later cleanup commit unpublishes nothing, since forks, existing clones and
cached views keep whatever was pushed. Anything entering a commit, a fixture or
a PR body must be **counts and module paths only**: never a consumer file path,
notebook name or collaboration name. Both fixtures are built this way. Take care
when quoting tool output, which is not: the failure branch of
`check_consumers.py` prints `used by: <path>` by design, because whoever is
fixing a break needs to know who broke.

## The compatibility shims

Phase 0b moved every module. `collabs/` — roughly eighty directories, unpinned,
tracking `master`, with live simulations running against them — was not
migrated, so every old dotted path still resolves:

```bash
python3 scripts/gen_shims.py          # regenerate src/idpy/<old paths>/
python3 scripts/gen_shims.py --check  # CI runs this
```

**Generated from `scripts/consumer-symbols.txt`, never hand-written.** That
fixture is `module <TAB> symbol <TAB> count`, which is already a shim
specification. Fifty-four hand-maintained modules would drift from it the first
time anything moved; a generator plus the fixture cannot, and `--check` in CI
means the two cannot silently disagree.

Four properties, each established by testing rather than by reasoning:

- **`FutureWarning`, not `DeprecationWarning`.** Python's default filter is
  `default::DeprecationWarning:__main__`, so a `DeprecationWarning` surfaces
  only when it fires from `__main__`. A collab reaching an old path through one
  of its own helper `.py` modules would be told *nothing at all*. idpy's users
  are researchers in notebooks, which is the case Python's own guidance points
  at `FutureWarning` for.
- **`__getattr__` raises `AttributeError` for names it does not know.** A shim
  that answers *every* name shadows real submodules: `from pkg import sub`
  consults `__getattr__` first and would return the shim's object instead of the
  module, silently and with nothing raised to notice.
- **The target module is imported eagerly.** Resolving it lazily defers
  `ModuleNotFoundError('pycuda')` from import to attribute access, where it
  escapes `hasattr()` — which catches only `AttributeError` — and takes
  `check_consumers.py` down with it. Eager import reproduces exactly what the
  old module did on a machine without the binding.
- **The grandfathered breakages stay broken.** A shim that quietly resolved one
  would erase the record while looking like an improvement.

### When the shims get removed

**Not on a version schedule.** A date or a version number is a promise made
without knowing whether anything still depends on them:

> Shims are removed when a re-freeze of `consumer-symbols.txt` from a populated
> tree contains no old-path entries. Not before, and not on a schedule.

If a re-freeze still lists `idpy.LBM.LBM`, something still uses it. When it does
not, they are *provably* dead rather than presumed dead.

The shims are for `collabs/`. They are **not** what keeps the papers working:
those never install idpy — they `sys.path.append("../../")` into a checkout —
so a shim inside the installed package cannot reach them. The papers were
migrated to the new paths instead.

## Do the paper notebooks still work?

Import resolution is the fast 90%, and it is not enough. It cannot catch API
drift -- where the module resolves, the symbol exists, and the constructor has
grown a required parameter since the notebook was written. That is how
`Missing 'tau'` sat in `idpy/LBM/test.py` for years while the suite looked green.

```bash
python3 scripts/smoke_papers.py --list          # the inventory, from papers/idpy-papers.py
python3 scripts/smoke_papers.py                 # clone all, smoke all
python3 scripts/smoke_papers.py arXiv-2505.23647
```

Each notebook is executed cell by cell until it errors or a cell exceeds a
wall-clock budget. **Reaching compute is the successful outcome** -- this tests
that the objects can be built, not that the results reproduce. Reproducing them
is hours of GPU time and belongs to the reader.

Two things it gets right that are easy to get wrong:

- **The inventory comes from `papers/idpy-papers.py`, not from `papers/` on
  disk.** They differ: one development machine has six of the seven checked out.
- **Each notebook runs in its own interpreter.** Several paper repositories
  carry local modules with the same names (`TolmanSimulations.py`,
  `LBM_proxy.py`) and different contents. In one process, `sys.modules` caches
  whichever loaded first and every later paper silently gets another paper's
  code -- which surfaces as an `ImportError` naming a different repository's
  file, and reads exactly like drift in the paper under test.

A paper whose default backend is absent is **retried on one this machine has**,
and the backend used is reported in the result line. These notebooks detect
what is available and then discard the answer with a hardcoded `preferred_lang`,
so the backend is a default, not a requirement. Force one with `--lang OCL_T`.

This matters more than convenience: a paper that stops at "no pycuda here" has
told you nothing about whether it constructs, and hides everything after that
cell. Both breakages found so far were behind that wall -- one of them
`text.latex.preview`, a matplotlib rcParam removed upstream, which has nothing
to do with idpy at all.

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
- `docs/architecture.md` — the three diagrams that are not obvious from the
  source tree: four lowerings of one kernel, the three insertion points, and the
  residency layer
- `docs/residency-probes.md` — the findings record for the residency layer,
  including several conclusions that were recorded, falsified, and corrected.
  Kept that way on purpose: the wrong turns show which class of error produced
  them.
- `docs/phase0b-brief.md` — the working brief for the `src/` restructure.
  **Dated by design**: it describes a transition, so it is replaced by a
  retrospective when Phase 0b merges rather than left standing as a description
  of planned work. Anything in it that outlives the restructure belongs here
  instead.

Read the relevant section before re-litigating a decision. Several have been
settled twice because the first answer was wrong, and the reasoning is written
down where it can be checked.
