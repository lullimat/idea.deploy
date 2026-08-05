# Phase 0 findings — residency & host-capability layer

Companion to `idea.deploy-extension.md`. Records the Phase 0 answers and the
test baseline that Phase 1's acceptance criterion ("existing tests pass
unchanged") is measured against.

**Tree:** `master` @ `e49f997` **Date:** 2026-08-05
**Machine:** Apple M1 Max, macOS. Backends live: OpenCL, CTypes, Metal. **No CUDA.**

---

## 1. Probe status

| id | question | method | answer |
|----|----------|--------|--------|
| F2 | can `IdpyMemory` express the residency op? | inspection | **no — settled** |
| F4 | can `IDPY_T` express sub-byte packed types? | inspection | **no — settled** |
| P1 | sustained streaming bandwidth under overlap | measurement | **not yet run** |
| P3 | is one dynamic shared buffer enough? | inspection + kernel design | **not yet run** |

Two of the four original probes were answerable by reading the tree rather than
by measurement. Both came back negative. See §4 of `idea.deploy-extension.md`
for the full write-up; the evidence is summarized below.

### F2 — no sub-range, no async, and a drain on every host touch

Across all four `IdpyArray*` classes in `idpy/IdpyCode/IdpyMemory.py`: no
`__getitem__`, no slicing, no view, no `set_async`, no `memcpy_htod_async`. The
`async_` keyword is accepted and discarded on every backend — `IdpyArrayCUDA.H2D`
(line 62) forwards to the synchronous `super().set(ary=ary)`.

On Metal, `IdpyArrayMETAL.H2D`, `D2H` and `SetConst` each open with
`_sync_tenet()` → `tenet.Finish()`, a **full GPU drain**. The required operation
— write slot *k* asynchronously while the GPU reads slots *j≠k* — is therefore
structurally excluded on Metal, not merely unimplemented.

**Consequence:** new Phase 2b (memory-layer prerequisites), blocking the CUDA and
Metal rows of Phase 3.

### F4 — the type model has no layout concept

`CustomTypes` (`idpy/Utils/CustomTypes.py:32`) is a `{alias: c_type_string}` dict
with `Push`/`Set`/`Pop`/`ToList`. No width, no packing, no layout, no accessor
generation. 4-bit affine weights at group 64 are outside the model, not merely
awkward in it.

**Consequence:** Phase 5 gated off by default; primary residency test case moved
from the LLM workload to the lattice.

---

## 2. Test baseline

Run from the repo root with `./py-env/idpy-env/bin/python`.

| suite | command | result |
|-------|---------|--------|
| core | `-m idpy.test` | **40 tests, OK** (4.6 s) |
| convolution | `-m idpy.test_convolution` | **21 tests, OK**, 7 skipped (8.2 s) |
| shared memory | `-m idpy.IdpyCode.test_shared` | **OK** — OpenCL and Metal exact (`max|out-ref| = 0`), CUDA skipped |
| LBM | `-m unittest idpy.LBM.test` | **5 tests, 1 ERROR** (see below) |

### Known-failing before any work starts

`idpy.LBM.test.TestShanChenMultiPhase.test_ShanChenMultiPhase` →
`Exception: Missing 'tau'` raised at `idpy/LBM/LBM.py:1238`, because the test at
`idpy/LBM/test.py:72` constructs `ShanChenMultiPhase` without passing `tau`.
Stale test against a drifted API, unrelated to this work. **Phase 1 acceptance
means "40 / 21 / shared-OK / LBM 5-with-this-one-error", not "everything green".**

### Trap: the LBM suite silently runs nothing

`idpy/LBM/test.py` defines `unittest.TestCase` classes but has **no
`unittest.main()` and no `__main__` guard**. So:

```
python -m idpy.LBM.test        # runs ZERO tests, prints nothing, exits 0
python -m unittest idpy.LBM.test   # actually runs the 5 tests
```

The first form looks like a pass. Any CI or acceptance script must use the
second form. Worth fixing with a `__main__` guard in a stray-cleanup PR.

---

## 3. Standing constraints

- **No CUDA on the development machine.** The CUDA path of the merged
  shared-memory layer (`e49f997`) is codegen-only and still unvalidated at
  runtime; Phase 2b's acceptance criterion ("verified correct on CUDA and
  Metal") inherits the same gap. Two CUDA-shaped debts now outstanding —
  settle via borrowed hardware or a cloud instance, or record the deferral
  explicitly.
- **No CI.** The four suites above are the entire safety net for Phase 1's
  refactor, and they must be run by hand.
- **`docs/planning/` is a separate git repository** with its own remote
  (`idea.deploy-planning.git`); it is deliberately not tracked by this repo.

---

## 4. Remaining Phase 0 work

- **P1** — measure achieved streaming bandwidth with overlap enabled vs.
  disabled, using a model-free stub: real cache-planning logic, real bounded
  parallel `pread` against a representative file, concurrent dummy GPU
  workload. Collect per-layer planning latency as a secondary number.
- **P3** — determine whether the target kernels need more than one dynamic
  shared buffer. Evaluate against the **lattice** kernels (tiled stencil/halo)
  first; the MoE answer only matters if F4 is reopened.
