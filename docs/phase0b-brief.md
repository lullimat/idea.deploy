# Phase 0b working brief — the `src/` restructure

> **Status: active, and dated by design.** This describes a transition, not a
> steady state, so it stops being true the moment the transition completes.
>
> **When Phase 0b merges, replace this file with a retrospective** — what was
> actually done, what the plan below got wrong, and what that cost. Do not leave
> it standing as a description of planned work; `docs/residency-probes.md` is the
> model, and it keeps its falsified conclusions visible on purpose.
>
> The rules here that are *not* transitional — the consumer-surface checks and
> the aggregates-never-paths rule — live in `CONTRIBUTING.md`, which is where
> they survive this file's deletion.

A working brief for whoever implements Phase 0b, human or agent. It exists as a
repository document rather than a handover message so that it arrives with the
clone, stays under version control, and can be corrected when it turns out to be
wrong.

**The task: move `idpy/` to `src/idpy/{core,physics}` without breaking a single
published paper or active collaboration.**

Phase 0a (packaging, `pip install idpy`) is done. 0b is the half that changes
every import path in every consumer, which is why it was deliberately split off
and deferred until packaging landed. See `STRATEGY.md` for the migration mapping
and `docs/architecture.md` for what the layers mean.

Read `STRATEGY.md`, `CONTRIBUTING.md` and `docs/architecture.md` first. They
encode decisions that have already been settled once, sometimes twice.

## Work in a fresh clone

```bash
git clone https://github.com/lullimat/idea.deploy.git ~/github/idea.deploy-0b
cd ~/github/idea.deploy-0b
git checkout -b feature/phase0b-restructure
python3 -m venv .venv
.venv/bin/pip install -e ".[physics]"
```

**Do not run `idpy-init.sh`.** It builds a 2 GB `py-env/` and compiles
PyCUDA/PyOpenCL from source; Phase 0a made the package pip-installable precisely
so this is unnecessary. For Metal, run `scripts/install-pymetallic.sh` —
pymetallic comes from a pinned revision with an idpy patch, not from an index.

A clone rather than a `git worktree`: merging goes through a GitHub PR either
way, so the worktree's shared-history advantage buys nothing, and a fully
separate `.git` cannot disturb the primary repository's state at all.

**Never touch the author's primary tree.** Under `papers/` there are twelve
separate gitignored git checkouts carrying uncommitted work that exists nowhere
else, and `collabs/` is 1 tracked file out of 80, with live simulations running
against it. Changing import paths under a running simulation is the failure this
separation exists to prevent. Your clone contains none of that material, because
it is all gitignored. Keep it that way.

**Publish aggregates, never paths.** Counts and module paths yes; consumer file
paths, notebook names and collaboration names never. This repository's history is
public and permanent, and the consumer directories hold unpublished research. Be
careful when quoting tool output — `check_consumers.py`'s failure branch prints
`used by: <path>` by design, and Phase 0b is the work that will trigger it. The
rule is stated in full in `CONTRIBUTING.md`, which is where it survives this
file's deletion.

## The shim surface is already measured

Two committed fixtures record what must not break, captured from a tree where
the consumers exist. **Both must stay green at every commit:**

```bash
python3 scripts/check_consumers.py --check-surface   # 47 modules
python3 scripts/check_consumers.py --check-symbols   # 277 (module, symbol) pairs
python3 scripts/check_layering.py                    # core never imports physics
```

They are complementary, and the second exists because the first is not enough: a
shim that forwards the module but drops a symbol passes `--check-surface` and
fails `--check-symbols`. That is not hypothetical — it is how the three existing
breakages below went unnoticed.

What the measurement says about the size of the job:

- **277 symbols across 46 modules.** Mechanical, not architectural.
- **Zero star imports.** The best case: every shim can enumerate what it
  re-exports rather than reproduce a module's whole public surface.
- **Three breakages already exist**, grandfathered with a leading `!` in
  `scripts/consumer-symbols.txt`:

  | symbol | what happened | consumers |
  |---|---|---|
  | `idpy.Utils.IdpyHardware` | moved to `idpy.IdpyCode` | 4 |
  | `idpy.LBM.Equilibria.Equilibria` | renamed to `HermiteEquilibria` | 1 |
  | `IdpyUnroll._get_single_neighbor_pos_in_code_out_sym` | removed | 1 |

  All six consumers are in `collabs/`; no published paper is affected. **They
  predate this work — do not fix them as part of 0b, and do not remove the
  marks.** Grandfathering follows `check_layering.py`'s `KNOWN` convention:
  pre-existing breakage stays visible without failing every run, and nothing new
  may join it silently.

The consumer counts are the priority ordering: `idpy.IdpyCode` has 275
consumers, `idpy.Utils.SimpleTiming` 95, `idpy.Utils.ManageData` 45. A shim that
is wrong for one of those is wrong for a lot of live work at once.

**Do not regenerate either fixture in your clone.** `--freeze` and
`--freeze-symbols` refuse an *empty* surface but cannot detect a *partial* one,
so running them where the consumers are absent would silently shrink the thing
you are protecting. If you believe a fixture is wrong, ask the author to
re-freeze from the primary tree.

### What the fixtures do not cover

The 277 pairs cover `from idpy... import name`, the dominant form here. Not
covered, both degrading gracefully:

- `import idpy.X.Y as z` then `z.Something` — the *module* is still checked by
  `--check-surface`, so only a missing attribute slips through.
- Dotted access on an imported package, e.g. `idpy.Utils.ManageData.Foo`.

If you find a symbol neither fixture protects, add it to `consumer-symbols.txt`
by hand with a count of `0` and say so in the PR.

## How to build the shims

**Generate them from the fixture, do not hand-write them.** The fixture is
`module <TAB> symbol <TAB> count`, which is exactly a shim specification. Forty-six
hand-written shim modules will drift; a generator plus the fixture cannot.

**Use lazy module-level `__getattr__` (PEP 562), not eager re-export.** The
already-broken `idpy.Utils.IdpyHardware` shows why: the symbol now lives in
`idpy.IdpyCode`, and `idpy/Utils/__init__.py` doing
`from idpy.IdpyCode import IdpyHardware` at module level risks an import cycle,
since core modules import each other. A module-level `__getattr__` resolves on
first access, which has no cycle.

Emit a `DeprecationWarning` naming the new path. **Do not remove the old paths in
this phase.**

Shims are what make this tractable: with them, the six paper repositories need
no changes and nothing merges in lockstep. Without them, six external repos must
branch and merge simultaneously with this one, and nothing can land until
everything is verified at once. With shims the papers become the *acceptance
test for the shims*, which is the job you actually want them doing.

Move in small commits, running all three checks after each. A commit that breaks
one is a commit to fix before continuing, not after.

## Tag the paper repositories before anything is rewritten

Six distinct repositories, twelve checkout directories:

| repo | also cloned as |
|---|---|
| `lullimat/arXiv-2009.12522` | `doi-10.1103-PhysRevE.103.063309`, `*_safe` |
| `lullimat/arXiv-2105.08772` | `doi-10.1103-PhysRevE.105.015301`, `*-safe` |
| `lullimat/arXiv-2212.07848` | `*-safe` |
| `lullimat/arXiv-2310.03632` | `*-safe` |
| `lullimat/arXiv-2503.05743` | — |
| `lullimat/arXiv-2505.23647` | — (**the only one with any tags**) |

**Five of six have no tags at all.** There is therefore no reachable published
state for them: rewrite imports on `main`/`master` and the version a paper cites
becomes findable only by commit SHA. This is Phase 0c work pulled forward,
because 0b is what makes it urgent.

## Verify against fresh paper clones

Clone each of the six into your own scratch directory — never the author's
`papers/`. Point `check_consumers.py` at your clones. Then pick two or three
notebooks that actually *construct* a simulation and run them far enough to reach
a first kernel dispatch.

Import resolution is the fast 90%, but it would not have caught the missing `tau`
API drift that sat in `idpy/LBM/test.py` for years. Only construction catches
that class.

Paper branches — updating imports to the new paths and dropping the shims — are
**optional cleanup that can land whenever**. They are not a blocker.

## Checkpoints that need the author

Stop and ask; do not proceed on an assumption:

1. **Before starting** — `#11` and `#7` must be merged, so that `master` has
   `pyproject.toml` and both fixtures. Without them the clone cannot
   `pip install -e .` and has nothing to check against.
2. **The tag scheme**, and pushing the tags. These are the author's published
   artifacts; the naming is their call and the push is theirs to make.
3. **CUDA verification.** The development machine is an M1 Max: Metal and CTypes
   are testable there, CUDA is not. The author has a separate two-RTX-5060
   machine and will run checks by hand — write the commands out to be pasted.
4. **If the shim layer starts looking unmaintainable**, say so rather than
   pressing on. Reporting that 0b is more expensive than the mapping implies is a
   useful result, not a failure.

## Conventions that exist because ignoring them caused real problems

- **`python -m unittest idpy.LBM.test`**, never `python -m idpy.LBM.test`. That
  file has no `unittest.main()` and no `__main__` guard: run as a module it
  executes **zero tests and exits 0**. It looked green for years while erroring
  at construction.
- **"Verified on CUDA" and "codegen-only" are different claims.** Say which. CI
  covers CTypes on Linux and macOS plus OpenCL on Linux via POCL; CUDA and Metal
  are hand-verified only.
- **An API that claims something must do it.** `H2D(async_=True)` silently
  ignored its argument for years and passed every correctness test. If a fast
  path can degrade silently, add a counter and assert on it.
- **Measurement discipline**, if you produce numbers at all: report a range not a
  sample, include a control, sweep the transfer size, and state what the number
  is *of*. Performance figures here have been wrong more often than the code they
  measured — `docs/residency-probes.md` keeps the falsified conclusions visible
  on purpose.
- **Constants**: a bare Python float becomes a C `double` and silently promotes
  fp32 arithmetic (~200x slower on consumer NVIDIA). Use
  `constants_types={'X': 'FType'}` or `np.float32(...)`.
- **Correctness gates; performance reports.** Never gate a build on throughput.
- **Branch per piece of work**, PR scoped to one idea.

## Done looks like

1. Tags pushed by the author on all six paper repositories.
2. `src/idpy/{core,physics}` in place, generated shims for every old path.
3. `--check-surface`, `--check-symbols` and `check_layering.py` all green, CI
   green, the three grandfathered breakages still marked and still reported.
4. Fresh clones of all six paper repositories importing successfully, with at
   least two notebooks run to first dispatch.
5. A PR stating plainly what was verified on which hardware, and what was not.
