# idpy — Strategy & Roadmap

> **Living document.** Update as milestones are reached and decisions are made.
> Last updated: 2026-08-06

---

## 1. Vision

**idpy** is a lightweight, transparent Python framework for GPU-accelerated
scientific computing across heterogeneous hardware. Its mission is to
**democratize access to GPU HPC** by enabling multi-node simulations on
commodity hardware — especially Apple Silicon clusters — while preserving
full scientific reproducibility.

### Core principles

- **Transparency:** researchers write near-native kernels and see exactly
  what executes. No hidden JIT magic, no opaque graph transformations.
- **Hardware portability:** same kernel logic dispatches to CUDA, OpenCL,
  Metal, and CPU (CTypes/OCaml) through thin, protocol-defined wrappers.
- **Reproducibility:** paper-linked repositories, environment capture, and
  cross-backend result verification as first-class workflows.
- **Accessibility:** runs on a desk, plugs into a power strip, administered
  by the researcher who uses it.

### The opportunity

Most GPU frameworks optimize for **performance** or **ease of use**. idpy
optimizes for **scientific transparency and reproducibility** across
heterogeneous hardware, keeping the researcher close to the computation
while eliminating backend lock-in.

Apple Silicon offers a unique convergence: capable GPU compute, unified
memory, excellent FLOPS-per-dollar and FLOPS-per-watt — all in globally
available commodity hardware. The missing piece is HPC software
infrastructure. idpy fills that gap.

---

## 2. Current State Assessment

### What exists

- Multi-backend concept: `CUDA`, `OpenCL`, `CTypes` backends with
  `Tenet` runtime handles and shared kernel dispatch via `IdpyCode`.
- Domain-specific modules: LBM, stencils, spin networks, PRNGs.
- Paper-repo workflow: `papers/idpy-papers.py` clones and links published
  experiment repositories.
- Shell-based environment setup: `idpy-init.sh`, `idpy-bootstrap.sh`.
- Test coverage via `unittest` (`idpy/test.py`, `idpy/LBM/test.py`) plus the
  cross-backend suites added with the residency work (`test_residency`,
  `test_residency_policy`, `test_linkage`, `test_constants`, `test_shared`,
  `test_overlap`, `test_hostmodule`), all with meaningful exit codes.
- Tutorials: Ising 2D, Metal test notebook.

### What needs improvement

- **No installable package** — no `pyproject.toml` or `setup.py`.
- **Backend contracts are implicit** — no protocol/ABC; CUDA, OpenCL,
  CTypes already drift in behavior and metadata shape.
- **Known bugs:**
  - OpenCL device aggregation overwrites per platform (should extend).
  - CUDA `GetTenet` can proceed with no device selected.
  - CTypes `device_name` is a string; CUDA/OpenCL use dicts.
  - `TenetNew` in OpenCL is broken/dead code.
  - Windows detection bug in `idpy/__init__.py` (`==` vs `=`).
- **Fragile setup:** `sys.path` hacks in every submodule `__init__.py`.
- ~~**No CI pipeline.**~~ Resolved 2026-08-06: GitHub Actions runs CTypes on
  Linux and macOS plus OpenCL on Linux (POCL). The prerequisite was that every
  print-style test script exited 0 regardless of what it printed -- as gates
  they were theatre; see `idpy/Utils/TestExit.py`.
- **Core and physics layers are entangled** — difficult to use backends
  without pulling in LBM.

### Stability rating

- Research velocity: **medium-high** (works well for the author).
- Maintainability: **medium** (implicit contracts remain; CI now runs
  CTypes + OpenCL on every push).
- Extensibility: **low** (adding a backend requires reverse-engineering
  existing ones).
- Adoption readiness: **low** (not installable, high onboarding friction).

---

## 3. Architecture Plan

### Package structure (Option B — single package, layered submodules)

```
idea.deploy/
├── pyproject.toml                  # pip install idpy
├── src/idpy/
│   ├── core/
│   │   ├── backends/
│   │   │   ├── base.py             # Protocol / ABC
│   │   │   ├── cuda.py
│   │   │   ├── opencl.py
│   │   │   ├── ctypes_backend.py
│   │   │   ├── metal.py
│   │   │   └── ocaml.py
│   │   ├── mpi/
│   │   │   ├── distenv.py          # DistributedEnvironment
│   │   │   ├── topology.py         # Cartesian decomposition
│   │   │   ├── exchange.py         # Halo packing, async send/recv
│   │   │   └── collectives.py      # Reductions, broadcasts
│   │   ├── kernel.py
│   │   ├── memory.py
│   │   ├── hardware.py
│   │   └── utils/
│   ├── physics/
│   │   ├── lbm/
│   │   ├── stencils/
│   │   ├── spin_networks/
│   │   └── prngs/
│   └── papers/
├── tutorials/
├── collabs/
├── STRATEGY.md                     # this document
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

### Architectural invariant

> **`idpy.core` never imports from `idpy.physics`. Ever.**

This one rule enforces the layer separation. Enforced via CI lint.

### Migration mapping

| Current location             | New location                        | Layer   |
|------------------------------|-------------------------------------|---------|
| `idpy/CUDA`                  | `src/idpy/core/backends/cuda.py`    | core    |
| `idpy/OpenCL`                | `src/idpy/core/backends/opencl.py`  | core    |
| `idpy/CTypes`                | `src/idpy/core/backends/ctypes_backend.py` | core |
| `idpy/Metal`                 | `src/idpy/core/backends/metal.py`   | core    |
| `idpy/IdpyCode`              | `src/idpy/core/`                    | core    |
| `idpy/Utils`                 | `src/idpy/core/utils/`              | core    |
| `idpy/LBM`                   | `src/idpy/physics/lbm/`             | physics |
| `idpy/IdpyStencils`          | `src/idpy/physics/stencils/`        | physics |
| `idpy/SpinNetworks`          | `src/idpy/physics/spin_networks/`   | physics |
| `idpy/PRNGS`                 | `src/idpy/physics/prngs/`           | physics |
| `papers/`                    | stays at repo root (or `src/idpy/papers/`) | supporting |
| `tutorials/`                 | stays at repo root                  | supporting |

### Optional dependency extras

```toml
[project.optional-dependencies]
physics = ["scikit-learn", "einsteinpy", "python-sat", "networkx"]
mpi     = ["mpi4py"]
metal   = ["pymetallic"]   # or chosen Metal bridge
cuda    = ["pycuda"]
opencl  = ["pyopencl"]
all     = ["idpy[physics,mpi,metal,cuda,opencl]"]
```

---

## 4. Competitive Positioning

### Landscape

| Framework | Strengths                | idpy differentiator                      |
|-----------|--------------------------|------------------------------------------|
| JAX       | Autodiff, XLA, TPU       | Opaque compilation; no Metal; no MPI transparency |
| Taichi    | DSL, differentiable      | Own language; heavier abstraction        |
| CuPy      | NumPy-like, CUDA         | Single backend; no Metal/OpenCL          |
| Numba     | JIT for Python           | CPU/CUDA only; no Metal; no MPI layer    |
| PETSc     | Mature MPI+solvers       | C/Fortran core; heavy; no Metal          |
| PyTorch   | ML ecosystem             | ML-focused; opaque; no Metal compute     |

### idpy's unique niche

**Transparent, lightweight, heterogeneous, multi-node GPU computing in
Python — including Apple Silicon clusters.**

No other lightweight Python framework attempts Metal + CUDA + OpenCL + CPU
backends with multi-node MPI and cross-backend reproducibility.

---

## 5. The Apple Silicon Cluster Argument

### Why it matters

Most computational researchers worldwide don't have access to NVIDIA HPC
clusters. Apple Silicon offers:

- **~3-5x better cost** for FP32 compute vs. NVIDIA datacenter GPUs
  (including total infrastructure cost).
- **~4x better FLOPS/watt** (M4 Ultra vs. A100 at FP32).
- **Unified memory** — GPU-accessible RAM without explicit D2H/H2D copies;
  192 GB on M4 Ultra at ~$4,000.
- **Zero infrastructure** — desk power, no cooling, no rack, no IT staff.
- **Global availability** — commodity hardware, purchasable anywhere.

### The Beowulf parallel

In the late 1990s, Beowulf clusters showed that commodity PCs + Linux + MPI
could replace expensive supercomputers for a class of problems. The parallel:

- **Then:** commodity x86 + Linux + MPI → affordable HPC.
- **Now:** commodity Apple Silicon + macOS + Metal + idpy → affordable GPU HPC.

The difference: this time commodity hardware comes with capable GPUs built
in, unified memory, and dramatically better power efficiency.

### Unified memory advantage for MPI

On discrete-GPU clusters, halo exchange requires:
`GPU → D2H copy → MPI send → MPI recv → H2D copy → GPU`

On Apple Silicon with unified memory:
`GPU output (already CPU-visible) → MPI send → MPI recv → available to GPU`

This eliminates two copy steps per exchange, partially compensating for
lower inter-node bandwidth vs. InfiniBand.

### Honest limitations

- **FP64:** Apple GPUs have limited double-precision throughput. Not
  suitable for FP64-dominant workloads (some CFD, quantum chemistry).
- **Scale ceiling:** commodity cluster, not petascale. Fits problems in
  tens-to-hundreds of GB.
- **Software maturity:** Metal compute is younger than CUDA. MPI on macOS
  is functional but not first-class.
- **Inter-node bandwidth:** Thunderbolt (~40-120 Gbps) is not InfiniBand
  (~400 Gbps); doesn't scale to large switch fabrics.

---

## 6. MPI Layer Design

### Conceptual model

```
Rank 0 (node A)              Rank 1 (node B)
┌──────────────────┐         ┌──────────────────┐
│  idpy.core       │         │  idpy.core       │
│  ┌─────────────┐ │         │  ┌─────────────┐ │
│  │ Tenet(CUDA) │ │◄──MPI──►│  │Tenet(Metal) │ │
│  └─────────────┘ │         │  └─────────────┘ │
│  local kernels   │         │  local kernels   │
│  local memory    │         │  local memory    │
└──────────────────┘         └──────────────────┘
```

MPI sits **beside** backends, not above or below. Each rank uses its own
Tenet (potentially a different backend). MPI handles inter-rank
communication only.

### Implementation plan

- **Transport:** start with staged transfers (D2H → MPI → H2D); add
  zero-copy shortcut for Metal unified memory; GPU-aware MPI as optional
  future optimization.
- **Decomposition:** Cartesian first (covers LBM, stencils); unstructured
  partitioning (METIS) as future plugin.
- **Testing:** `mpirun -np 4` with CTypes on any laptop — no GPU required
  for MPI development and CI.

---

## 7. OCaml Extension Design

OCaml is treated as a **compiled-kernel backend**, parallel to
`CTypesKernelModule`:

- OCaml functions expose C-compatible entry points via the OCaml FFI.
- `OCamlKernelModule` compiles to shared library, loads via `ctypes.CDLL`.
- Same backend protocol as all other backends.
- Demonstrates that the protocol is language-agnostic.

---

## 8. Phased Roadmap

### Phase 0: Foundation

> **Split into 0a and 0b (2026-08-06).** This phase originally bundled
> "make it installable" with "move everything into `src/idpy/{core,physics}`".
> Those have very different costs and serve different goals, and bundling them
> made the cheap half look as expensive as the dear one.
>
> **Installability does not require the move.** `pyproject.toml` works on the
> current flat layout — `packages = ["idpy"]` and `pip install idpy` works. The
> `src/` layout and the core/physics split are conventions serving architectural
> clarity; JOSS asks for installability, documentation, tests and a contribution
> guide, and **none of those touch a single import path**.
>
> Only the move changes module paths, and module paths are what the published
> paper notebooks import. So 0a is paper-safe and 0b is not — which is the whole
> reason to separate them.

#### Phase 0a — packaging and JOSS readiness (paper-safe)

No import path changes; nothing in `papers/` is affected.

- [x] `pyproject.toml` on the **current layout**, with optional dependency
      extras (`cuda`, `opencl`, `metal`, `physics`, `mpi`, `storage`)
- [x] GitHub Actions CI — CTypes on Linux and macOS, plus OpenCL on Linux via
      POCL, which also makes the LBM suite runnable there. CUDA and Metal stay
      manual on the development machines.
- [x] Rewrite README with a "Statement of Need"
- [x] `CONTRIBUTING.md`
- [ ] `CODE_OF_CONDUCT.md`
- [x] Architecture diagram — `docs/architecture.md`

The blocker was not `pyproject.toml`. **`idpy` could not be imported outside a
source checkout at all**: it read `.idpy-env`, `py-env/cuda_path_found` and
`LICENSE` at import time and raised `FileNotFoundError` before any backend was
consulted. Unnoticed because every existing user works in a checkout — the same
shape as the LBM suite that ran zero tests and exited 0.

**Milestone: reached** — `pip install idpy` works, verified in a fresh
virtualenv from a clean clone, CI green on both runners. **Tag v0.1.0 still
pending**, and it is the prerequisite for 0c's tag-pinning below.

#### Phase 0b — restructure (breaks import paths)

- [ ] Restructure into `src/idpy/core/` and `src/idpy/physics/`
- [ ] Define backend protocol (`core/backends/base.py`)
- [ ] Eliminate `sys.path` hacks — package-relative imports throughout
- [ ] Backend conformance test suite
- [ ] Fix the remaining known bugs (OpenCL aggregation, CUDA guard, CTypes
      metadata, remove `TenetNew`)

**Compatibility shims are the mechanism, not paper-repo edits.** When the move
happens, `idpy/LBM/LBM.py` and friends become two-line re-exports at the new
locations with a `DeprecationWarning`. Published notebooks keep working
untouched and the shims retire on our schedule. "Update every paper repository"
is not a requirement of this phase, and treating it as one is what made 0b look
prohibitive.

**The shim surface is now measured rather than guessed.** `scripts/check_consumers.py`
freezes what 0b must not break into two committed fixtures — 47 modules and
**277 (module, symbol) pairs, with zero star imports**. Both are checkable from
a fresh clone, which matters because the consumers are mostly untracked
(`collabs/` is 1 tracked file out of 80). Two fixtures rather than one because a
shim that forwards a module but drops a symbol passes the module check and fails
the symbol check.

That measurement says 0b is **mechanical, not architectural**: 277 re-exports
with no star imports, generatable from the fixture rather than hand-written.
Working brief in `docs/phase0b-brief.md`, including why the shims must use lazy
`__getattr__` rather than eager re-export.

**Restructurability has been preserved meanwhile:** the architectural invariant
already almost holds — as of 2026-08-06 there are exactly **two** core→physics
imports, both function-local, both in `idpy/Utils/IdpySymbolic.py` (lines 1182
and 1240, importing `idpy.IdpyStencils.IdpyConvolution`), so `idpy.Utils`
already loads cleanly without physics. `scripts/check_layering.py` now holds
that line in CI, grandfathering the single `(file, module)` pair those two
imports collapse to and refusing any new one. Deferring the restructure is
reasonable; letting the layering rot while deferring is what would have made it
expensive.

#### Phase 0c — versioning the reproducibility promise

This is the item the packaging discussion actually surfaced, and it is
independent of both 0a and 0b.

The promise is currently **unversioned**. The README says "keep pulling the
updates" *and* "backwards compatibility will be assured". Together those are an
unbounded obligation: every future change must preserve every past notebook,
forever, against a moving `master`. That, rather than any particular
restructure, is what makes structural change feel expensive.

- [x] Check that every module and symbol the consumers import still resolves —
      `scripts/check_consumers.py`, with the surface frozen into
      `scripts/consumer-surface.txt` and `scripts/consumer-symbols.txt` so it is
      verifiable from a clone that does not contain the consumers
- [ ] Tag each of the six paper repositories at its **published** state, before
      any import rewrite. Five of six currently have **no tags at all**, so
      there is no reachable published state for them — rewrite `main` and the
      version a paper cites is findable only by SHA
- [ ] Pin each paper repository to a **tagged** idea.deploy release, so
      "reproducible" means "works against `v0.2.0`" — checkable — instead of
      "works against whatever master is today", which is not
- [ ] Periodic (not per-push) smoke job that constructs the simulations from
      each paper notebook and checks they still build

The import check is the cheap 90% and runs anywhere in seconds. The smoke job is
what turns the promise into something with teeth: it reports that a paper has
broken *before a reader finds out*, and it catches the class the import check
cannot — API drift, where the module resolves and the constructor has grown a
required parameter. Periodic rather than per-push because these are slow.

The import check has already earned its place: it found **three symbols that had
silently stopped resolving** — `idpy.Utils.IdpyHardware` (moved to
`idpy.IdpyCode`), `idpy.LBM.Equilibria.Equilibria` (renamed
`HermiteEquilibria`), and a removed `IdpyUnroll` helper. All six consumers are
in `collabs/`; no published paper is affected. They are grandfathered, visible
on every run, and nothing new may join them silently.

Worth recording that the notebooks were checked on 2026-08-06 and are **fine** —
they pass the parameters `SetupRoot` requires. It was the repository's own
`idpy/LBM/test.py` that had gone stale, not the papers. The discipline has been
applied to the published work; what is missing is the machinery to keep proving
it.

### Phase 1: Metal Backend (Weeks 4–6)

- [x] Bridge chosen: **pymetallic** (`METAL_T = "pymetallic"`, optional via `IsModuleThere`)
- [ ] Implement Metal Tenet / memory / IdpyCode deploy (in progress on `feature/metal-backend`)
- [ ] Exploit unified memory for zero-copy buffer access
- [ ] Conformance tests pass on Metal
- [ ] Tutorial notebook (single-node Metal, e.g. diffusion)
- [ ] Benchmark: Metal vs. CTypes vs. CUDA/OpenCL (single node)

**Requires:** Apple Silicon (or Metal-capable macOS), Xcode/Swift toolchain, `pip install pymetallic`.

**Milestone:** Metal backend passes full conformance suite. **Tag v0.2.0.**

### Phase 2: MPI Layer (Weeks 7–10)

- [ ] Implement `DistributedEnvironment` (`src/idpy/core/mpi/distenv.py`)
- [ ] Cartesian topology and neighbor maps (`topology.py`)
- [ ] Halo packing/unpacking, async send/recv (`exchange.py`)
- [ ] Collective operations (`collectives.py`)
- [ ] Staged transfer path (universal) + zero-copy for Metal
- [ ] MPI conformance tests (`mpirun -np 4`, CTypes-only in CI)
- [ ] LBM multi-node example with result verification

**Milestone:** Multi-node LBM runs on Mac cluster. **Tag v0.3.0.**

### Phase 3: Apple Silicon Cluster Benchmarking (Weeks 11–14)

- [ ] Build test cluster (4x Mac Mini M4 Pro or Mini/Studio mix)
- [ ] LBM benchmark suite (2D/3D, multiple resolutions)
- [ ] Measure: time-to-solution, FLOPS, memory, power consumption
- [ ] Compare: single Mac / Mac cluster / NVIDIA GPU / NVIDIA multi-GPU
- [ ] Reproducibility validation: cross-backend checksums
- [ ] Generate publication-quality tables and figures

**Milestone:** Benchmark data complete. CPC paper draft ready.

### Phase 4: OCaml Extension (Weeks 12–15, parallel with Phase 3)

- [ ] Implement `src/idpy/core/backends/ocaml.py`
- [ ] C ABI bridge for OCaml shared libraries
- [ ] Conformance tests on OCaml backend
- [ ] Example notebook: same kernel in C and OCaml, identical results

**Milestone:** OCaml backend passes conformance suite.

### Dependency graph

```
Phase 0 (foundation)
  ├──► Phase 1 (Metal backend)
  │      ├──► Paper A: JOSS submission
  │      └──► Phase 2 (MPI layer)
  │             └──► Phase 3 (benchmarks)
  │                    └──► Paper B: CPC submission
  └──► Phase 4 (OCaml) ──► feeds into Paper A revision
```

---

## 9. Publication Strategy

### Paper A: JOSS Software Paper

**Target submission:** End of Phase 1 (~Week 7)

| Section                 | Source material              |
|-------------------------|------------------------------|
| Statement of Need       | Phase 0 README               |
| Architecture            | Phase 0 protocol + diagram   |
| Backend implementations | Phases 0–1 (CUDA, OpenCL, CTypes, Metal) |
| Extensibility           | Phase 4 (OCaml — add in revision if needed) |
| Tests and CI            | Phase 0 conformance suite    |
| Example                 | Phase 1 tutorial notebook    |

**Format:** ~2 pages + repository review.
**Expected review cycle:** 4–8 weeks.

JOSS reviewers check: installability, documentation, tests, contribution
guidelines, statement of need. The Phase 0 restructuring produces exactly
what they require.

### Paper B: Main Research Paper

**Target submission:** End of Phase 3 (~Week 15)
**Primary venue:** Computer Physics Communications

| Section | Content |
|---------|---------|
| 1. Introduction | Democratization argument, Beowulf parallel, cost/access barriers to GPU HPC |
| 2. Apple Silicon opportunity | FLOPS/$/W analysis, unified memory architecture, global hardware availability |
| 3. idpy framework | Architecture overview (cite JOSS paper), backend protocol, Metal backend design |
| 4. MPI layer | Distributed environment, halo exchange, heterogeneous backend support |
| 5. Benchmarks | Phase 3 data: Mac cluster vs. NVIDIA, weak/strong scaling, power efficiency |
| 6. Reproducibility | Cross-backend result verification, same physics everywhere |
| 7. Discussion | FP64 limitations, scaling ceiling, future directions (RDMA, larger clusters) |

**Alternative venues (ranked by fit):**
1. Computer Physics Communications — method + benchmark + code
2. SIAM Journal on Scientific Computing — if performance analysis is deep
3. IEEE Computing in Science & Engineering — practice/perspective framing

### Paper C: Networking Study (Optional, Later)

**Content:** Inter-node communication characterization on Apple hardware —
TCP vs. Thunderbolt networking vs. emerging RDMA paths. Latency/bandwidth
measurements, comparison with InfiniBand.

**Venues:** IEEE Cluster, SC Workshop papers, CPC technical note.

Even negative results (documenting what Apple hardware can and can't do
for RDMA) have publication value.

---

## 10. Key Milestones Summary

| Week | Milestone                                    | Tag / Deliverable      |
|------|----------------------------------------------|------------------------|
| 3    | `pip install idpy` works, CTypes tests green | v0.1.0                 |
| 6    | Metal backend passes conformance suite       | v0.2.0                 |
| 7    | JOSS paper submitted                         | arXiv preprint         |
| 10   | Multi-node LBM on Mac cluster                | v0.3.0                 |
| 14   | Benchmark data complete                      | CPC draft              |
| 15   | CPC paper submitted                          | arXiv preprint         |

---

## 11. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Metal Python bridge instability | Blocks Phase 1 | Evaluate multiple bridges early; fall back to direct `objc` FFI |
| FP64 weakness on Apple GPUs | Limits physics applicability | Be explicit in paper; focus benchmarks on FP32 workloads |
| `mpi4py` on macOS rough edges | Slows Phase 2 | Test early; document workarounds; CTypes path avoids GPU complications |
| Apple RDMA not publicly available | Paper C may be negative-result | Frame as characterization study; negative results are still valuable |
| Scope creep in MPI layer | Delays Phase 3 | Keep MPI thin — communication only, not a scheduler |
| JOSS reviewer requests major changes | Delays Paper A | Front-load quality (tests, docs, contribution guide) |
| Phase 3 hardware procurement | Can't benchmark without cluster | Start with 2 Macs (even laptops); scale up for final data |

---

## 12. Success Criteria

- [ ] Any researcher can `pip install idpy` and run a simulation on CPU
      within 5 minutes, on any OS.
- [ ] The same simulation code produces verified-identical results on
      CUDA, OpenCL, Metal, and CPU backends.
- [ ] A 4-node Mac cluster runs multi-node LBM through idpy with no
      code changes vs. single-node (only configuration).
- [ ] Two peer-reviewed publications (JOSS + CPC or equivalent).
- [ ] At least one external contributor successfully adds or modifies a
      backend using the documented protocol.
