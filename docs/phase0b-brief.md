# Phase 0b working brief — the `src/` restructure and the paper migration

> **Status: active, and dated by design.** This describes a transition, not a
> steady state, so it stops being true the moment the transition completes.
>
> **When the coordinated release lands, replace this file with a retrospective**
> — what was actually done, what the plan below got wrong, and what that cost.
> Do not leave it standing as a description of planned work;
> `docs/residency-probes.md` is the model, and it keeps its falsified conclusions
> visible on purpose.
>
> The rules here that are *not* transitional — the consumer-surface checks, the
> paper smoke check, and the aggregates-never-paths rule — live in
> `CONTRIBUTING.md`, which is where they survive this file's deletion.

A working brief for whoever implements Phase 0b, human or agent. It lives in the
repository rather than in a handover message so that it arrives with the clone,
stays under version control, and can be corrected when it turns out to be wrong.
It has been corrected twice already.

**The task: move `idpy/` to `src/idpy/{core,physics}`, migrate all seven paper
repositories onto the new structure, and release the two together as `v0.2.0`.**

Phase 0a (packaging, `pip install idpy`, tagged `v0.1.0`) is done.

## Why this is one coordinated release and not eight independent ones

An earlier version of this brief said the opposite — shims would keep the papers
working, so nothing needed to merge in lockstep. **That was wrong, and the reason
is worth understanding before you start.**

The paper repositories do not install idpy. Every one of them does this:

```python
sys.path.append("../../")          # in the notebook
```
```bash
curl -fsSL .../idea.deploy/refs/heads/master/idpy-bootstrap.sh   # in install.sh
```

They assume they are sitting at `idea.deploy/papers/<repo>/` and reach up into a
checkout of `master`. So after the restructure, `../../` is the repository root
and `idpy` lives at `src/idpy/` — **`import idpy` fails at the `sys.path` line,
before any import path matters.** Shims live inside the installed package and are
unreachable to a consumer that never installs it.

Two consequences:

- **The papers must be touched.** There is no shim trick that spares them.
- **Holding 0b unmerged protects `collabs/`.** Eighty directories, unpinned,
  tracking `master`. While the restructure sits on a branch, live simulations
  keep working. Merging 0b early is what would break them.

So: one branch here, one branch per paper repository, and everything merges at
the end.

## Papers get migrated; collabs get shims

Different consumers, different mechanisms. Do not conflate them.

| consumer | count | mechanism | why |
|---|---|---|---|
| paper repos | 7 | **migrated** — new import paths, `pip install`, no `sys.path` | they never install idpy, so shims cannot reach them |
| `collabs/` | ~80 dirs | **shims** — old paths keep working | not migrating in this pass; the author's live work |

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
separate `.git` cannot disturb the primary repository's state.

**Never touch the author's primary tree.** Under `papers/` there are twelve
separate gitignored checkouts carrying uncommitted work that exists nowhere else,
and `collabs/` is 1 tracked file out of 80 with live simulations running against
it. Your clone contains none of that material, because it is all gitignored. Keep
it that way.

**Publish aggregates, never paths.** Counts and module paths yes; consumer file
paths, notebook names and collaboration names never. This repository's history is
public and permanent, and the consumer directories hold unpublished research. Be
careful when quoting tool output — `check_consumers.py`'s failure branch prints
`used by: <path>` by design. The rule is stated in full in `CONTRIBUTING.md`.

## Make the branch installable early — this is the unblocking move

The moment `src/idpy/` imports cleanly, push the branch. Every paper repository
can then be worked on **in parallel** rather than waiting for the restructure to
be finished:

```bash
pip install "git+https://github.com/lullimat/idea.deploy@feature/phase0b-restructure"
```

Without this, seven paper migrations are serially blocked on one branch being
complete. With it, they are blocked only on it being *importable*. Do this before
the restructure is polished, not after.

## The surface is already measured

Two committed fixtures record what must not break. **Both must stay green at
every commit:**

```bash
python3 scripts/check_consumers.py --check-surface   # 47 modules
python3 scripts/check_consumers.py --check-symbols   # 277 (module, symbol) pairs
python3 scripts/check_layering.py                    # core never imports physics
```

They are complementary, and the second exists because the first is not enough: a
shim that forwards the module but drops a symbol passes `--check-surface` and
fails `--check-symbols`. That is not hypothetical — it is how the three existing
breakages went unnoticed.

- **277 symbols across 46 modules.** Mechanical, not architectural.
- **Zero star imports.** Every shim can enumerate what it re-exports rather than
  reproduce a whole public surface.
- **Three breakages already exist**, grandfathered with a leading `!` in
  `scripts/consumer-symbols.txt`:

  | symbol | what happened | consumers |
  |---|---|---|
  | `idpy.Utils.IdpyHardware` | moved to `idpy.IdpyCode` | 4 |
  | `idpy.LBM.Equilibria.Equilibria` | renamed to `HermiteEquilibria` | 1 |
  | `IdpyUnroll._get_single_neighbor_pos_in_code_out_sym` | removed | 1 |

  All six consumers are in `collabs/`; no published paper is affected. **They
  predate this work — do not fix them here, and do not remove the marks.**

Consumer counts are the priority ordering: `idpy.IdpyCode` has 275 consumers,
`idpy.Utils.SimpleTiming` 95, `idpy.Utils.ManageData` 45.

**Do not regenerate either fixture in your clone.** `--freeze` and
`--freeze-symbols` refuse an *empty* surface but cannot detect a *partial* one,
so running them where the consumers are absent would silently shrink the thing
you are protecting. Ask the author to re-freeze from the primary tree.

### What the fixtures do not cover

The 277 pairs cover `from idpy... import name`, the dominant form. Not covered,
both degrading gracefully: `import idpy.X.Y as z` then `z.Something` (the module
is still checked), and dotted access like `idpy.Utils.ManageData.Foo`. If you
find a symbol neither fixture protects, add it to `consumer-symbols.txt` by hand
with a count of `0` and say so in the PR.

## Building the shims (for `collabs/`, not the papers)

**Generate them from the fixture; do not hand-write them.** The fixture is
`module <TAB> symbol <TAB> count`, which is exactly a shim specification.
Forty-six hand-written shim modules will drift; a generator plus the fixture
cannot.

Three things that were established by testing them, not by reasoning:

**1. Use a lazy module-level `__getattr__` (PEP 562), not eager re-export.** The
already-broken `idpy.Utils.IdpyHardware` shows why: the symbol now lives in
`idpy.IdpyCode`, and `idpy/Utils/__init__.py` doing
`from idpy.IdpyCode import IdpyHardware` at module level risks an import cycle,
since core modules import each other. `__getattr__` resolves on first access.

**2. The `__getattr__` must raise `AttributeError` for names it does not know.**
A shim that answers *every* name shadows real submodules — `from pkg import sub`
returns the shim's value instead of the module, silently:

```python
_MOVED = {...}                    # from the fixture
def __getattr__(name):
    if name not in _MOVED:
        raise AttributeError(name)   # let the import system find submodules
    ...
```

Only 1 of the 277 recorded pairs is a submodule import, so this is not about
volume — it is that the failure is silent and returns a wrong object rather than
raising.

**3. Use `FutureWarning`, not `DeprecationWarning`.** Measured:

| where the old path is used | `DeprecationWarning` | `FutureWarning` |
|---|---|---|
| inside a helper `.py` module | **invisible** | shown |
| directly at top level / a notebook cell | shown | shown |

Python's default filter is `default::DeprecationWarning:__main__`, so it only
surfaces when triggered from `__main__`. A collab importing an old path through
its own helper module would be told nothing at all. Python's own guidance:
`DeprecationWarning` targets library developers, `FutureWarning` targets end
users. idpy's users are researchers in notebooks.

### When the shims get removed

**Not on a version schedule.** A date or a version number is a promise made
without knowing whether anything still depends on them. The condition is
measurable with machinery that already exists:

> Shims are removed when a re-freeze of `consumer-symbols.txt` from a populated
> tree contains no old-path entries. Not before, and not on a schedule.

If a re-freeze still lists `idpy.LBM.LBM`, something still uses it. When it does
not, the shims are *provably* dead rather than presumed dead.

## Per paper repository, in this order

Seven repositories, per `papers/idpy-papers.py` — which is the inventory, **not**
whatever is cloned on a development machine. A filesystem survey of one machine
found six, because `arXiv-2112.02574` is not checked out there.

```bash
python3 scripts/smoke_papers.py --list     # the authoritative list
```

| repo | tagged? |
|---|---|
| `lullimat/arXiv-2009.12522` | no |
| `lullimat/arXiv-2105.08772` | no |
| `lullimat/arXiv-2112.02574` | no |
| `lullimat/arXiv-2212.07848` | no |
| `lullimat/arXiv-2310.03632` | no |
| `lullimat/arXiv-2503.05743` | **yes** |
| `lullimat/arXiv-2505.23647` | **yes** |

**Five of seven have no tags at all**, so there is no reachable published state
for them. Once a repackaging branch exists, "what did the paper actually cite"
gets harder to answer every day.

For each repository, in this order:

1. **Smoke it as it is.** `python3 scripts/smoke_papers.py <id>` — record whether
   it constructs *before* you change anything. Otherwise you cannot tell your
   breakage from pre-existing breakage.
2. **Tag the published state.** The author pushes the tags; propose a scheme and
   let them choose. These are their published artifacts.
3. **Fix the drift found in step 1.** Two are known, and neither needed CUDA
   hardware to find:

   | notebook | failure | kind |
   |---|---|---|
   | `arXiv-2009.12522/arXiv-2009.12522v1/…` | `Missing … 'psi_sym'` | idpy API drift |
   | `arXiv-2105.08772/MesoscopicTolmanLength` | `text.latex.preview` rcParam | matplotlib, removed upstream |

   The first is in an **archived v1** directory — an archive edited to work
   against current code is no longer an archive of anything. Raise that one
   rather than silently fixing it. The second is ordinary dependency rot and
   should just be fixed.
4. **Adapt to the new packaging**: new import paths, `pip install` in place of
   the `sys.path.append("../../")` cell, and `install.sh` pointing at a tag
   rather than `refs/heads/master`. Add the dependency declaration none of the
   seven currently has.
5. **Smoke it again** against the restructured branch:
   `python3 scripts/smoke_papers.py --dir /path/to/repo`. That is the acceptance
   test for the branch — a concrete pass, not a judgement call.

**The backend is a default, not a requirement.** Two repositories set
`preferred_lang = CUDA_T`, but every notebook already detects what is available
and then discards the answer by overriding it. `smoke_papers.py` retries on a
backend the machine has and reports which one it used, so **no CUDA hardware is
needed to smoke any paper**. Force one with `--lang OCL_T`.

Both breakages found so far were *behind* that wall: the CUDA failure stopped
execution before the cell that actually fails. One of them,
`text.latex.preview`, is a matplotlib rcParam removed upstream and has nothing
to do with idpy.

**Fix this in the paper repositories as part of the migration.** A hardcoded
`preferred_lang` in a framework whose central claim is backend portability is
the wrong default: honour the detection block already present, or read an
environment variable, so the notebook runs wherever it lands. Running on real
CUDA hardware remains worth doing before release, but it is a confirmation step
rather than a prerequisite for finding drift.

## Checkpoints that need the author

Stop and ask; do not proceed on an assumption.

1. **The tag scheme**, and pushing the tags — their published artifacts.
2. **CUDA verification** — write out commands to be pasted, do not assume.
3. **The archived-v1 drift** — fix, or pin to the idpy of its era?
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
- **Isolate what shares a name.** Several paper repositories carry local modules
  called `TolmanSimulations.py` and `LBM_proxy.py` with *different contents*. Run
  two in one interpreter and `sys.modules` gives the second one the first one's
  code — surfacing as an `ImportError` naming another repository's file, which
  reads exactly like drift. `smoke_papers.py` uses a subprocess per notebook for
  this reason.
- **Measurement discipline**, if you produce numbers: report a range not a
  sample, include a control, sweep the transfer size, and state what the number
  is *of*. `docs/residency-probes.md` keeps the falsified conclusions visible.
- **Constants**: a bare Python float becomes a C `double` and silently promotes
  fp32 arithmetic (~200x slower on consumer NVIDIA). Use
  `constants_types={'X': 'FType'}` or `np.float32(...)`.
- **Correctness gates; performance reports.** Never gate a build on throughput.
- **Read the inventory, do not survey the filesystem.** Both counting errors in
  this document's history came from looking at one machine's disk instead of the
  registry: six papers when there are seven, and a tag survey that missed two
  tagged repositories because shallow clones do not fetch tags.

## Done looks like

1. `src/idpy/{core,physics}` in place, generated shims for every old path,
   emitting `FutureWarning`.
2. `--check-surface`, `--check-symbols` and `check_layering.py` green, CI green,
   the three grandfathered breakages still marked and still reported.
3. Tags pushed by the author on all seven paper repositories.
4. Seven paper branches, each smoking green against the restructured idpy, each
   declaring its dependency, none using `sys.path.append("../../")`.
5. The CUDA-preferring papers confirmed on real CUDA hardware by the author —
   a confirmation step, not the way drift is found.
6. A coordinated release: idea.deploy merges and tags **`v0.2.0`**, the seven
   paper branches merge, and each pins to `idpy==0.2.0`.
7. A PR stating plainly what was verified on which hardware, and what was not.
