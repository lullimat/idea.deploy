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

## 2b. Findings from starting the work

### pymetallic exposes enough — no Swift shim needed for synchronization

The Swift bridge has 39 `@_cdecl` exports and **no `MTLEvent`, no `MTLFence`, no
completion handlers**. It does have, per command buffer:

- `metal_command_buffer_get_status` (`1=completed, 0=in-progress, -1=error`)
- `metal_command_buffer_wait_until_completed`
- `metal_blit_command_encoder_copy_buffer(src, src_off, dst, dst_off, size)` —
  offset-based sub-range GPU copies
- `metal_buffer_get_contents` — direct pointer into unified storage

That is sufficient to replace `_sync_tenet()`: track in-flight command buffers
with the ranges they touched, and wait only on those overlapping the target
sub-range. The obstacle is in idpy, not the binding — `Tenet` keeps a single
`last_command_buffer` (`idpy/Metal/Metal.py:117`), discarding exactly the
history per-region waiting needs.

**Ordering consequence:** Phase 1 is *not* a prerequisite of Phase 2b. Phase 2b
is self-contained, so it goes first and alone. (The I/O-queue half of Phase 3 —
`MTLIOCommandQueue` for storage→device — is a separate question and still
likely needs the shim.)

### Two latent defects found while implementing

**Duplicate class definitions.** `IdpyMemory.py` defined `IdpyArrayCUDA`,
`IdpyArrayOCL` and `IdpyArrayCTYPES` twice (lines 39-266 and 393-620,
byte-identical). The second block silently shadowed the first, so ~230 lines
were unreachable while reading as live — editing the first copy is a no-op.
Removed in its own commit, verified behaviour-neutral against the baseline.
`IdpyArrayMETAL` was never duplicated.

**`IdpyArrayOCL` could not be sliced.** pyopencl constructs derived arrays via
`self.__class__(..., _fast=True, ...)`; the idpy subclass's `__init__` did not
accept unknown kwargs, so `arr[a:b]` raised
`TypeError: unexpected keyword argument '_fast'`. Fixed by forwarding `**kwargs`
to `super().__init__`.

### Phase 2b status

| backend | primitives | verified |
|---------|-----------|----------|
| OpenCL | `SubView`, `H2DSub`, `D2HSub`, `Sync`, real `async_` on `H2D` | **yes** — T1/T2/T3 exact on M1 Max *and* on RTX 5060 |
| CUDA | same surface, plus `_pinned_host_CUDA` | **yes** — T1/T2/T3 exact on RTX 5060 |
| Metal | same surface, on range-scoped waiting (§2e) | **yes** — T1/T2/T3 exact on M1 Max |
| CTypes | not yet (trivially unified) | — |

Test: `python -m idpy.IdpyCode.test_residency`.

**CUDA-specific caveat now encoded in the API:** an async H2D only overlaps with
compute when the host buffer is page-locked. `H2DSub(async_=True)` accepts a
pageable array and stays correct, but will not overlap — `_pinned_host_CUDA()`
exists for that. An API that claims async and behaves synchronously is the same
failure mode as F2, so it is documented in the method rather than left to be
discovered.

**What T3 does and does not establish:** on a single in-order queue the runtime
may order the copy after the kernel, so T3 shows the async partial write is
*correct when issued against in-flight work* — not that the two overlapped.
Proving overlap needs a second queue (OpenCL) or a non-default stream plus
pinned memory (CUDA). That is P1's measurement; the two-queue variant is
follow-up work.

---

## 2c. CUDA validation run — both debts cleared

**Date:** 2026-08-05 **Host:** `id` (ssh alias), Linux, AMD Ryzen 5 3600
**GPUs:** 2x NVIDIA GeForce RTX 5060, ~8 GB each, driver 580.159.03 / CUDA (13,1,0)
**Backends live there:** pycuda, pyopencl, ctypes (no Metal)

| suite | result |
|-------|--------|
| `test_shared` (static **and** dynamic) | **CUDA exact**, `max|out-ref| = 0`; OpenCL exact; Metal skipped |
| `test_residency` T1/T2/T3 | **CUDA exact** on all three; OpenCL exact on all three |
| `idpy.test` | **41 tests, OK** (41 not 40 — one more backend class runs with CUDA present) |
| `idpy.test_convolution` | **21 tests, OK**, 3 skipped (Metal); Mac skips 7 (CUDA) |
| `idpy.LBM.test` | 5 tests, same single pre-existing `Missing 'tau'` error |

Three things this settles:

1. **The `idpy_shared`/`idpy_sync` CUDA codegen from PR #4 is now runtime-verified**,
   both the static `__shared__` and the dynamic `extern __shared__` + launch-bytes
   paths. It was codegen-only since it merged.
2. **The Phase 2b CUDA primitives are verified.** `SubView` was the flagged risk —
   whether `IdpyArrayCUDA(gpudata=..., base=...)` would borrow the pointer or
   allocate a new one. pycuda honours `gpudata`, so the view aliases correctly
   (T2 exact). `memcpy_htod_async` with an integer destination also works.
3. **The `IdpyMemory.py` de-duplication is validated on the CUDA path**, which the
   Mac cannot exercise (`TestIdpyArrayCU` passed).

Two facts worth carrying forward:

- **FP64 is available on the RTX 5060** (`Double: 63`); the LBM suite kept
  `double` throughout rather than downcasting. The Mac downcasts everything to
  fp32 (no fp64 on Apple GPUs). So `id` is the only host where fp64 physics runs
  natively — relevant to any cross-backend numerical-agreement claim, which must
  either fix precision or state the asymmetry.
- **`id` has a working OpenCL CPU device** (POCL, `pthread-AMD Ryzen 5 3600`),
  which the Mac does not ("There is some issue for pyopencl to list cpu's"). That
  is a fourth execution target available for cross-backend verification.

**T3 caveat is unchanged on CUDA:** on the default stream the runtime may order
the copy after the kernel, so this remains correctness-under-concurrent-issue,
not demonstrated overlap. With two GPUs and real streams available, the stronger
two-stream + pinned-memory variant is now buildable — see §4.

---

## 2d. T3-overlap results, and a latent metalanguage trap

**Overlap is real on both backends and both machines.** `test_overlap.py` times
the kernel alone (tK), the transfer alone (tC), and both issued together (tB),
reporting `(tK + tC - tB) / min(tK, tC)` — 1.0 fully concurrent, 0.0 fully
serialized. 128 MB transfer:

All rows below are post-fp32-fix (see the trap section), so the filler kernel is
genuinely fp32 everywhere and the numbers are mutually consistent:

| host / backend | kernel | copy | both | serial would be | overlap | bandwidth |
|---|---|---|---|---|---|---|
| M1 Max / OpenCL | 38.12 ms | 4.81 ms | 38.90 ms | 42.93 ms | **0.84** | 27.9 GB/s |
| M1 Max / Metal | 38.48 ms | 2.80 ms | 38.54 ms | 41.28 ms | **0.98** | 48.0 GB/s |
| RTX 5060 / CUDA | 27.51 ms | 9.33 ms | 27.41 ms | 36.84 ms | **1.01** | 14.4 GB/s |
| RTX 5060 / OpenCL | 26.30 ms | 15.04 ms | 26.22 ms | 41.34 ms | **1.00** | 8.9 GB/s |

Correctness exact everywhere, on both the transferred and the computed half.

Reading these honestly:

- **The bandwidths are not comparable across hosts.** On Apple the "transfer" is
  a memcpy into unified storage; on NVIDIA it crosses PCIe from pinned memory.
  Same API, physically different operations. The *overlap ratio* is comparable;
  the GB/s figure is not.
- **1.01 is noise, not superlinearity.** tB came in marginally under tK alone.
  Values at or slightly above 1.0 mean the copy was fully hidden and the
  measurement has hit its noise floor.
- CUDA at 0.98 with pinned memory on a non-default stream is close to ideal:
  the transfer is essentially free when there is compute to hide it behind.

### The trap: float constants emit as double literals

`_format_c_macro_value` (`idpy/IdpyCode/IdpyCode.py:60`) falls through to
`str(value)` for floats, so `constants['X'] = 0.99999` becomes
`#define X 0.99999` — a **double** in C. Any surrounding `float` arithmetic then
promotes, and the expression evaluates in fp64 wherever fp64 exists.

Caught because the calibrated ITERS disagreed wildly between machines: 723 on the
M1 Max versus 60 on the RTX 5060 for the same 39 ms of kernel. The Mac has no
fp64 to promote to, so it ran fp32; the GeForce ran the chain on its fp64 path.

**Magnitude, measured properly.** The cross-machine ratio above (~12x) is *not*
the size of the effect — it compares one machine's fp64 to another's fp32 and
attributes the whole difference to precision. Re-running the same host after the
fix isolates it:

| RTX 5060, same kernel | ITERS | kernel time | throughput |
|---|---|---|---|
| double literals (fp64 path) | 60 | 39.30 ms | 48.9 G FMA/s |
| `f`-suffixed literals (fp32) | 8731 | 27.51 ms | **10,160 G FMA/s** |

**~208x on one machine**, and the fp32 figure — 20.3 TFLOP/s — is essentially the
card's fp32 peak, so the post-fix number is the honest one. The penalty exceeds
the nominal 1/64 fp64 rate because each iteration also pays two float↔double
conversions around the fp64 FMA. Two innocuous-looking constants cost two orders
of magnitude.

A prediction was made before this run (that ITERS would land near the Mac's ~700)
and it was wrong by an order of magnitude, for the reason above. Recorded because
the mistake is instructive: cross-machine ratios cannot isolate a
single-variable effect, and only the same-host before/after can.

Now that both hosts run fp32, the remaining ITERS gap is a real hardware
difference: 8731 iterations in 27.5 ms on the RTX 5060 against 702 in 38.5 ms on
the M1 Max, roughly 17x in throughput on *this dependent-chain microbenchmark*.
The Apple part sits near 11% of its fp32 peak here while the NVIDIA part sits
near 100%, which says the chain is latency-bound on Apple — a property of this
one synthetic kernel, not a general throughput comparison, and not something to
generalise from.

**Currently latent, not a live bug.** Instrumenting `IdpyKernel.__init__` across
the whole LBM suite found **no kernel built with a float constant** — the physics
code passes floating-point data as typed device arrays (`W_list`, `XI_list`),
never as `#define` macros. `test_overlap.py` was the first thing in the repo to
do it, and now emits `'0.99999f'` as an explicit string instead.

Two reasons to record it rather than let it sit:

1. **It is a portability-of-results issue, not just performance.** A float
   constant makes the same kernel compute fp64 intermediates on CUDA and fp32 on
   Apple. That is a silent cross-backend numerical divergence, which is exactly
   what `STRATEGY.md`'s "verified-identical results across backends" criterion
   forbids.
2. **It is the same weakness as F4.** The type model does not describe
   constants any more than it describes packed layouts — `CustomTypes` maps
   aliases to type strings and nothing knows what precision a literal should
   carry. A proper fix is type-aware constant emission, which is a metalanguage
   design decision (a kernel using `double` types legitimately wants double
   literals), not a one-line patch. Flagged for a decision rather than fixed
   unilaterally.

---

## 2e. Metal: drain-on-touch replaced by range-scoped waiting

This closes the last part of F2. `IdpyArrayMETAL.H2D/D2H/SetConst` used to open
with `_sync_tenet()` → `tenet.Finish()`, a full GPU drain, which made "write slot
k while the GPU reads slot j" impossible to express regardless of what the
hardware could do.

**The mechanism.** A Metal `Tenet` now keeps an ordered `_in_flight` list of
submitted command buffers, each with the byte spans it may have touched. A host
access to `[a, b)` scans newest-first for the latest entry that *overlaps*, waits
only on that one, and drops the prefix. Everything rests on a single property: a
single queue completes command buffers **in order**, so waiting on entry *i* also
completes `0..i-1`. That is why finding the latest overlapping entry suffices and
why no `MTLEvent` is needed — `wait_until_completed()` and `get_status()` per
command buffer are enough, exactly as §2b predicted.

**Safety is the default.** `touched = None` means "unknown, assume everything",
and every path that cannot describe what it touched lands there — batched
encodes, tenets predating the tracking, unrecognised arguments. The fallback is
the old drain, never a race. `Deploy` records whole-buffer spans for each
`IdpyArrayMETAL` argument by default, since a kernel may write anywhere in a
buffer it was handed; callers who know better pass an explicit
`touched={array: (start, stop)}`.

**Control experiment.** The overlap is genuinely caused by the range scoping,
not by something incidental — same measurement, only the declaration removed:

| declared touch span | tK | tC | tB | overlap |
|---|---|---|---|---|
| `{buf: (0, half)}` | 38.42 ms | 2.88 ms | 38.50 ms | **0.97** |
| omitted (whole buffer) | 38.62 ms | 2.91 ms | 41.16 ms | **0.13** |

Both rows are correct behaviour: the second is the conservative path doing
exactly what it should. The first is the capability that did not exist before.

Full Metal run in the standard harness: kernel 38.48 ms, copy 2.80 ms
(48.0 GB/s), both 38.54 ms against 41.28 ms serial → **overlap 0.98**, both
halves exact.

Note what "overlap" means here, since it differs from the discrete-GPU case:
there is no DMA engine involved. The host store into unified memory runs on the
CPU while the GPU runs the kernel — concurrency between processors, not between
a copy engine and compute. `async_` is accepted on the Metal primitives for
signature parity and is inert; once the range-scoped wait returns, the write is
an immediate host store.

**Phase 2b is now complete on CUDA, OpenCL and Metal.** CTypes remains, and is
trivially unified.

**Cross-check after the Metal work** (2026-08-05, `id`): the Metal rework touched
`IdpyCode.py`, which every backend imports, so CUDA and OpenCL were re-run there.
`test_residency` T1/T2/T3 exact on both; `test_overlap` 1.01 / 1.00 with
bandwidth unchanged (14.39 vs 14.38 GB/s CUDA, 8.93 vs 9.10 GB/s OpenCL —
transfer speed never involved the kernel); `idpy.test` 41 OK. No regression: the
Metal changes live inside `if idpy_langs_sys[METAL_T]` blocks or the Metal
module, and are invisible on a host without Metal.

---

## 2f. Phase 2: the residency policy, and the reuse question answered

`idpy/IdpyCode/IdpyResidency.py` is the policy layer over the Phase 2b
primitives: `BackingStore` (`MemMapStore` via `numpy.memmap` — mmap plus the OS
page cache — and `ArrayStore`), and `ResidentCache`, a fixed set of device slots
with LRU or FIFO eviction, dirty tracking, write-back and pinning.

**The file contains no per-backend branches.** It is written entirely against
`SubView` / `H2DSub` / `D2HSub` / `Sync`. That is the asymmetry the design is
built around, now demonstrated rather than asserted: the primitives underneath
are genuinely different per backend — staged async copy on CUDA, a second queue
on OpenCL, range-scoped waiting against in-flight command buffers on Metal,
plain numpy on CTypes — while the policy above them is one program.

### The open question about reuse is answered: halos

§7 asked whether the lattice case has any *eviction* story or only a streaming
one — if every block is touched once, no policy beats any other and the eviction
logic is never exercised. **It has reuse, and halos are where it comes from.**
Computing block *b* of a 3-point stencil needs *b-1*, *b*, *b+1*; the next step
needs *b*, *b+1*, *b+2*. Two of every three acquires are already resident.

That makes the traffic exactly predictable, which turns the reuse check into a
real test rather than an observation:

    misses = 3 + (n_blocks - 1) = n_blocks + 2      (3 cold, then one new block per step)
    hits   = 2 * (n_blocks - 1)

A cache that evicted the wrong block would still return correct numbers — the
data simply gets reloaded — so only the exact count catches a policy bug. The
ratio alone would not.

### Result

32 MiB lattice over a 4 MiB resident set (**8x**), 32 blocks of 1 MiB, 4 slots:

| check | result |
|---|---|
| P1 sweep vs whole-lattice numpy reference | `max|out-ref| = 0` |
| P2 reuse | **62 hits / 34 misses, exactly as predicted**; hit rate 0.646, 30 evictions |
| P3 pinning guard | raises rather than evicting a block still in use |
| P4 write-back | 32 blocks reached the file (read back from the file, not the cache) |
| P5 LRU vs FIFO | `max|lru-fifo| = 0` |

**Identical on all four backends across both machines** — including the hit/miss
counts, which is itself the evidence that the policy is backend-independent:

| backend | host | P1 | P2 counts |
|---|---|---|---|
| CTypes | M1 Max, `id` | exact | 62 / 34 |
| CUDA | `id` | exact | 62 / 34 |
| OpenCL | M1 Max, `id` | exact | 62 / 34 |
| Metal | M1 Max | exact | 62 / 34 |

The counts are a property of the policy rather than the hardware, so agreement
to the integer across four backends is a stronger statement than P1 alone: a
backend whose primitives diverged could still produce correct output by
reloading blocks it should have retained, and the count would give it away.

Two CUDA-specific risks were carried into that run and both cleared. `SubView`
had only ever been exercised transiently; `ResidentCache` holds four views of one
buffer for an entire sweep, relying on `base=` to keep the owning allocation
alive — pycuda's base chain holds. And `_WriteBack` calls `D2HSub` on a slot that
is itself a `SubView`, so byte offsets have to compose through two levels; the
sweep also reads a single element from the end of a view. Both compound paths are
correct.

Two honest notes:

- **"Larger than RAM" was substituted with "larger than the resident set".** A
  test suite cannot honestly arrange the former, and it is not the property that
  matters: what matters is that the dataset exceeds the device-resident working
  set, forcing eviction and write-back. On CTypes device memory *is* RAM, so the
  cache is exactly the binding constraint.
- **Compute runs host-side through the primitives**, which is what keeps the test
  free of per-backend branches. Handing slot views to a kernel is possible — they
  are ordinary Idpy arrays — but a halo spanning three separate slots is a layout
  problem belonging to the real lattice work, not to the policy layer.

### Placement deviation

The design sketch put this interface *on* `Tenet`. It is a free module taking
`tenet=` instead, matching `IdpyMemory.Array` / `Zeros` / `OnDevice` and keeping
`Tenet` — which every backend must implement — free of a dependency on policy
code. The capability is still per-tenet; only the spelling differs. Recorded
because it is a deliberate departure from §3.

---

## 2g. Phase 1: HostModule

`CTypesKernelModule` lifted into `idpy/Utils/HostModule.py` with the compiler as
a parameter. Nothing in that class was ever CPU-compute-specific — it hashes a
source string, caches the build, and hands back ctypes callables — so it is now
shared machinery and `CTYPES_T` is simply its first consumer.

Two axes that were hard-coded are parameters now:

- **`Toolchain`** — command + flags, source extension, library extension.
  `CToolchain()` reproduces the existing C build exactly; `SwiftToolchain()`
  drives `swiftc -emit-library`.
- **argtypes** — `GetKernelFunction` keeps the numpy-shaped ABI (every pointer
  becomes an `ndpointer`). `GetFunction` is the escape hatch: explicit ctypes
  argtypes, no numpy assumption. `CTypesTypes` gained `uintptr` / `handle` /
  `void` / `size_t`, and a pointer whose element type is opaque resolves to
  `c_void_p` instead of being wrapped — `ndpointer(c_void_p)` is meaningless,
  and a device pointer or queue handle has no host array behind it.

**Behaviour preserved deliberately**, down to the details: same cache directory,
same hash inputs (so warm caches stay valid), same option-string concatenation
without a separator, same `ndpointer` argtypes for kernels. The awkward bits were
inherited rather than tidied, because tidying them would have made this something
other than a lift — the space-splitting of the compile command is documented in
place instead.

### Acceptance

| check | result |
|---|---|
| H1 C toolchain | compiles, loads, computes — `max\|out-ref\| = 0` |
| H2 build cache | identical builds share an artifact and reuse it; changed flags produce a different library from the same source |
| H3 opaque argtypes | `void *` / `size_t` resolve to `c_void_p` / `c_size_t` and call correctly against a raw address |
| H4 raw entry point | `GetFunction` with explicit argtypes, no numpy |
| **H5 Swift shim** | **`swiftc` builds a `@_cdecl` library, ctypes loads and calls it** |

Plus the existing suites unchanged: core 40 OK, convolution 21 OK, shared 4
exact, residency 9 exact, policy 3 backends OK, Metal 10 OK, LBM with the same
pre-existing error.

**Verified on two different C compilers.** The M1 Max builds with
`clang -fPIC -shared -arch arm64 -std=c99`, `id` with
`gcc -fPIC -shared -std=c99`. H1–H4 pass identically on both, including H2's
flag-sensitivity check — which is where the inherited space-splitting of the
compile command meets a second compiler's argument handling, and the place a
toolchain abstraction would most plausibly leak. H5 skips on Linux
(`Available()` reports no `swiftc`) and the C path there also exercises
`CTypesKernelModule` through `idpy.test` and LBM. Two compilers is a weak
generalisation, but it is the difference between a parameter that is used and a
parameter that merely exists.

**H3 and H5 are the pair that matters.** Together they are the claim that Swift
is a *compiler choice* and not a language target: one facility builds both
`CTYPES_T`'s C kernels and a Swift shim exposing C entry points, so binding
`MTLIOCommandQueue` in Phase 3 needs no new entry in `idpy_langs_dict`. That was
the scoping question §8 flagged as the one worth getting right, and it now has a
working answer rather than an argued one.

### Placement

`idpy/Utils/` rather than `idpy/IdpyCode/`. `IdpyCode/__init__.py` imports every
backend package, so a backend importing back from it risks a cycle; `Utils` has
no import-time dependency on `IdpyCode`. Under the `STRATEGY.md` restructure both
land in `idpy.core` anyway, so this costs nothing later.

---

## 3. Standing constraints

- ~~No CUDA on the development machine.~~ **Both CUDA debts settled** on `id`
  (§2c). The constraint that remains is workflow, not capability: CUDA code is
  still *written* on a machine that cannot run it, so anything CUDA-only stays
  unverified until an `id` round-trip. Say so explicitly rather than implying it
  works.
- **8 GB VRAM per GPU on `id`.** Residency tests must not assume more; that
  ceiling is also what makes it a genuine larger-than-device-memory testbed.
- **No CI.** The four suites above are the entire safety net for Phase 1's
  refactor, and they must be run by hand — on two machines to cover all backends.
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
- **T3-overlap** (new, unblocked by §2c) — the two-stream variant that measures
  *demonstrated* overlap rather than correctness under concurrent issue: a
  non-default CUDA stream plus `_pinned_host_CUDA()`, or a second OpenCL queue.
  This is the honest version of the Phase 2b acceptance criterion and it feeds
  directly into P1's bandwidth measurement. `id` has two GPUs and real streams,
  so it can now be built and run.
