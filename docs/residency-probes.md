# Phase 0 findings — residency & host-capability layer

Companion to `idea.deploy-extension.md`. Records the Phase 0 answers and the
test baseline that Phase 1's acceptance criterion ("existing tests pass
unchanged") is measured against.

**Tree:** `master` @ `e49f997` **Date:** 2026-08-05
**Machine:** Apple M1 Max, macOS. Backends live: OpenCL, CTypes, Metal. **No CUDA.**

> **Module paths below are pre-Phase-0b and are left as they were run.** This is
> a findings record, and rewriting the commands would misrepresent what was
> executed against which tree. To re-run anything here, translate:
> `idpy.IdpyCode.X` → `idpy.core.X`, `idpy.Utils.X` → `idpy.core.utils.X`,
> `idpy.LBM.X` → `idpy.physics.lbm.X`. The old spellings do still work through
> the compatibility shims, so the commands are stale rather than broken.

---

## 1. Probe status

| id | question | method | answer |
|----|----------|--------|--------|
| F2 | can `IdpyMemory` express the residency op? | inspection | **no — settled** |
| F4 | can `IDPY_T` express sub-byte packed types? | inspection | **no — settled** |
| P1 | sustained storage->device bandwidth under overlap | measurement | **measured (§2k)**; CUDA pending |
| P3 | is one dynamic shared buffer enough? | built and measured | **yes — settled (§2l)** |

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

## 2h. Phase 4: insertion point (2) wired

`include_dirs`, `definitions_files` and `objects_files` were validated by
`IdpyKernel.__init__` and then discarded (§2). They are stored and used now.
The three do genuinely different things and reach different backends — that
asymmetry is the substance of the phase, not an implementation detail:

| mechanism | what it does | backends |
|---|---|---|
| `definitions_files` | source text injected into the compile unit | **all four** — every backend compiles from a source string |
| `include_dirs` | compiler search paths | CUDA (SourceModule's own parameter), OpenCL and CTypes (`-I`); **not expressible on Metal** |
| `objects_files` | native static linking | **CTypes only** — the only backend with a link step |

`definitions_files` entries may be paths, or objects exposing `Code(lang)`. That
last form is how the design's specific ask — passing an `IdpyKernel` into
another kernel's compile unit — is served: the injected kernel is emitted with
**`preamble=False`**, a new argument to `Code()` that suppresses headers, macros
and typedefs. Without it the duplicate `typedef float FType;` is a C99 error and
the mechanism would be unusable for precisely the case it exists for.

Consequence recorded in the code: the compile unit belongs to the top-level
kernel, so an injected kernel's own `definitions_files` are **not** pulled in
transitively. Declare everything the unit needs on the kernel that owns it.
Nesting would otherwise duplicate injections with no way to deduplicate across
independently-built sources.

### Refusals are the point

Where a mechanism has no meaning on a backend, `CheckLinkage` now raises
`NotImplementedError` from `Code()` — before anything reaches a compiler.
`objects_files` on CUDA/OpenCL/Metal, `include_dirs` on Metal. Silently
accepting an argument that cannot work is the behaviour being removed, and
replacing it with a different silence would have missed the point.

### Result

Verified on **all four backends** across both machines:

| check | CUDA | OpenCL | Metal | CTypes |
|---|---|---|---|---|
| L1 definitions from a path | exact | exact | exact | exact |
| L2 definitions from an `IdpyFunction` | exact | exact | exact | exact |
| L3 definitions from an `IdpyKernel` (donor present, one typedef) | exact | exact | exact | exact |
| L4 `include_dirs` | exact | exact | refused | exact |
| L5 `objects_files` | refused | refused | refused | **exact** |

L5 on CTypes is the end-to-end case: a native object compiled outside idpy
entirely, then linked into a generated kernel that calls into it. That is the
shape of every capability shim Phase 3 will need — code that is not generated,
not injectable as text, and available only as a compiled artifact.

Two CUDA-specific unknowns cleared on `id`. `SourceModule(include_dirs=...)`
exists and works on that pycuda — it could not be checked from the dev machine,
where pycuda is not importable. And L3 puts two `__global__` functions in one
`SourceModule`, with `get_function(name)` resolving the right one.

**One narrower claim than it looks.** pycuda wraps source in `extern "C" { ... }`
by default, so L4's `#include <idpy_hdr.h>` lands inside that wrapper. Harmless
for the macro-only header the test uses, but L4 passing does not establish that
a header carrying declarations would work on CUDA. Worth knowing before relying
on `include_dirs` for anything beyond macros there.

---

## 2i. Float precision: constants, sympy expressions, and a corrected claim

### Correction

An earlier entry recorded the double-literal trap as "latent, not live",
on the basis that instrumenting the LBM suite found no kernel built with a float
constant. **That measurement was invalid.** The LBM test errored at construction
(`Missing 'tau'`, cascading to `SC_G`, `psi_sym`, `psi_code`) and never reached
the kernel-building path at all — it measured a suite that was not running the
thing being measured.

Building a real `ShanChenMultiPhase`, the new warning fires immediately on five
live constants: `CM2 = 3.0`, `CM4 = 9.0`, `PI`, `SC_G = -3.6`, `OMEGA = 1.0`,
all emitted as double literals into kernels whose types are `float`. The trap was
live. The warning's first act was to catch a case previously cleared in error.

The stale test is now fixed, and the LBM suite passes for the first time — on
both machines. Verified on `id` (2026-08-06): constants OK, `idpy.test` 41 OK,
**LBM 5 OK**, convolution 21 OK, linkage 3 backends OK. That run matters more
than a routine re-check: with the test fixed, LBM now actually executes across
CUDA, OpenCL and CTypes there, where before it died at construction. Nothing
CUDA-specific was hiding behind the error.

### The sympy path is a second, larger instance

`_codify_sympy` is `str(expr)` after `.evalf()`, so sympy Rationals reach the
source as bare decimals — `0.333333333333333` for 1/3 — bypassing
`_format_c_macro_value` entirely. Four distinct double literals appear in **one**
population of D2Q9; with nine populations that is roughly forty per collision
kernel, in both the SRT and MRT rolled forms.

Not a problem: `**`. `str(sympy)` does emit `u_0**2`, but the live kernels
contain none — `LBMKernelsMeta.py:1033` substitutes powers away before emission.
The raw `CodifySingle*` output does contain it, so those functions depend on the
caller supplying substitution tuples.

### The real defect is inconsistency, and the fix is a trade-off

On a device with fp64 those literals make intermediates evaluate in double before
storing to float. On Apple there is no fp64, so the compiler demotes and
intermediates are fp32. **The same source computes differently per backend.**

Double intermediates with fp32 storage is a legitimate numerical choice — often a
desirable one. So homogenizing is not a pure win: it buys consistency and speed
at some accuracy on fp64 devices. The argument for it is that the alternative,
double intermediates everywhere, is *unachievable* on Apple and therefore cannot
be the consistent choice even in principle.

### `float_literals`: opt-in homogenization

`IdpyKernel(float_literals='FType')` rewrites every unsuffixed floating literal
in the assembled source to that type's precision, resolved through
`custom_types` at emission time so it follows the fp64 downcast.

A post-generation pass rather than a scope around emission, for a structural
reason: **kernel bodies are built in `__init__`** — LBM's sympy collision and
equilibrium expressions are already strings by the time `Code()` runs, so there
is no live expression left to print differently. Rewriting the assembled text is
the only chokepoint reaching all of it, and it also catches hand-written
metalanguage literals like `0.5 * F[d]`, which a sympy-aware printer never would.

`#define` lines are included — an LBM body multiplies by `CM2` far more often
than it writes a bare `0.5`, so excluding macros would not homogenize anything.
Constants with an explicit `constants_types` entry are exempt, so an intentional
double survives. Constants emitted as `-D` flags under `declare_macros='macro'`
are unreachable this way; `constants_types` is the only lever there.

Verified on the real `K_Collision_ShanChenGuoMultiPhase`: 4 bare doubles in the
body before, 0 after, with `CM2 3.0f`, `SC_G -3.6f`, `OMEGA 1.0f` in the macros
and `V 1088`, `Q 9`, `DIM 2` untouched.

**Default is None — output stays byte-identical and published results stay
reproducible until a kernel opts in.** Enabling it does change numerics on fp64
devices, which is why it is a per-kernel choice rather than a global switch.
Making the mixed case explicit per-literal is deferred to a second sweep.

---

## 2j. Phase 3: storage → device, CUDA lowering verified

`BackingStore.ReadBlockInto(block_id, view)` fills a device slot directly and
returns `False` to **decline**. Declining is not an error — it is how a store
says it has no direct path on this configuration, and the cache falls back to
the host route without caring why. The capability is therefore optional per
*store* rather than per backend, and the fallback stays the default.

`KvikIOStore` subclasses `MemMapStore` and reads through KvikIO (NVIDIA cuFile /
GPUDirect Storage). It is the right first row not for speed but because
**KvikIO degrades to a POSIX read when GPUDirect is absent**, so the path is
exercised and stays correct without GDS hardware, a compatible filesystem or the
nvidia-fs driver. Inheriting `MemMapStore` means the fallback is the same store
answering differently, not a second implementation that could drift.

### Verified on `id` (2026-08-06), kvikio-cu13 26.6.0

| backend | P1 | P2 | P6 read path |
|---|---|---|---|
| **CUDA** | exact | 62/34 exact | **34 direct / 0 staged** — cuFile |
| OpenCL | exact | 62/34 exact | 0 direct / 34 staged |
| CTypes | exact | 62/34 exact | 0 direct / 34 staged |

The direct storage→device path works: 34 of 34 block loads bypassed host memory
entirely, with the sweep still exact against the whole-lattice reference. The
risk carried into that run — whether KvikIO would accept an `IdpyArrayCUDA`
**SubView**, whose `gpudata` is borrowed with a `base` rather than freshly
allocated — resolved in favour of it working through
`__cuda_array_interface__`.

### The counters earned their place immediately

`direct_reads` / `staged_reads` exist because a direct path that silently
degraded to the host bounce would pass every correctness check unchanged — the
same failure mode as an `async` copy that is secretly synchronous, of which this
work has already produced two. P6 asserts `direct + staged == misses`.

They also caught a reporting bug on their first real run. `DirectPathName()`
returned `'kvikio/cuFile'` whenever kvikio was merely *importable*, so on a host
with it installed the CTypes and OpenCL rows read

    kvikio/cuFile: 0 direct / 34 staged

naming a path that was never taken. The counters were right; the label lied.
`DirectPathName()` is now keyed on a read having actually **succeeded**, so a
store that has only ever taken the staged route says so. Exactly the class of
label that would let a degraded fast path look engaged.

### Metal row: MTLIOCommandQueue via a Swift shim

The row that makes the storage claim **portable rather than merely present**.
Metal's storage API has no Python binding, pymetallic does not wrap it, and
Swift is the only language that can see it — so this is the case the whole
`HostModule` design was aimed at. Nothing is generated per kernel: it is fixed
host code, written once, which is why Swift stays a *compiler choice* and there
is still no `SWIFT_T` in `idpy_langs_dict`.

Three things had to be verified rather than assumed, and were, by probing before
building:

1. **Pointer bridging.** pymetallic exposes `_device_ptr` / `_buffer_ptr` as raw
   pointers; Swift reconstitutes them with
   `Unmanaged.fromOpaque(...).takeUnretainedValue()`. The probe created an
   `MTLIOCommandQueue` from pymetallic's own device.
2. **Sub-range targeting.** The load writes at a byte offset inside an existing
   buffer, so it fills a `SubView` of the cache rather than a whole allocation.
   This works *because* Phase 2b gave `IdpyArrayMETAL` an element offset against
   its parent Buffer — without that bookkeeping there is nothing to aim at.
   Verified in isolation first: reading the second block of a file into the
   middle of a buffer left the head untouched and matched exactly.
3. **The Swift spelling.** It is
   `load(_:offset:size:sourceHandle:sourceHandleOffset:)`. `loadBuffer(...)` is
   the Objective-C name and was obsoleted in Swift 3; the compiler says so
   plainly.

`CreateFileStore(path, array, block_elems, tenet=)` picks the lowering —
KvikIO/cuFile on CUDA, `MTLIOCommandQueue` on Metal, plain memmap elsewhere.
Every option subclasses `MemMapStore`, so an unmatched backend, a missing
binding or a failed open all land on the same staged path.

### Phase 3 status

| backend | mechanism | verified |
|---|---|---|
| **CUDA** | KvikIO / cuFile | **34 direct / 0 staged**, P1 exact (`id`) |
| **Metal** | `MTLIOCommandQueue` | **34 direct / 0 staged**, P1 exact (M1 Max) |
| OpenCL | — declines, stages | 0 direct / 34 staged |
| CTypes | — declines, stages | 0 direct / 34 staged |
| AMD | rocm-xio | not started; needs hardware |

Two backends now stream storage→device through **the same policy code** with
entirely different mechanisms underneath — cuFile on one, a Swift-compiled
Metal IO queue on the other. That is the design's central asymmetry carried all
the way to storage: the policy is one program, the lowerings are not.

### Layering lint

`scripts/check_layering.py` enforces `STRATEGY.md`'s "core never imports
physics" against the current layout via the §3 migration mapping. Static
analysis over import statements — no environment, no packages, no GPU — so it
runs as CI's first step in seconds, before dependencies install.

Two violations are grandfathered in a `KNOWN` allowlist, both function-local in
`idpy/Utils/IdpySymbolic.py`, so `idpy.Utils` still loads cleanly without
physics. Unpicking them is Phase 0b refactoring; the point is that nothing *new*
can be added. Verified by injection: a fresh core→physics import exits 1 naming
file and line.

It also surfaced two invalid escape sequences (`'\ '`) in
`idpy/IdpyCode/__init__.py` — accepted today with a `SyntaxWarning`, a
`SyntaxError` in some future Python. Replaced with the identical two-character
value spelled legally.

---

## 2k. P1 measured: the direct paths are correct, not faster (here)

`test_storage_bandwidth.py`. 256 MiB file through a 32 MiB cache, 8 MiB blocks,
min-of-3.

| backend | B1 staged | B2 direct | B2/B1 | overlap staged | overlap direct |
|---|---|---|---|---|---|
| OpenCL | 6.17 GB/s | 5.79 (no direct path) | 0.94× | 0.37 | 0.16 |
| **Metal** | 7.72 GB/s | **7.87** (MTLIOCommandQueue) | **1.02×** | 1.02 | 0.96 |
| CTypes | 6.94 GB/s | 7.37 (no direct path) | 1.06× | — serial | — |

### The OpenCL row is the noise floor, and it should be read first

OpenCL has no direct lowering, so **both** its columns are the same staged path
measured twice. They differ by **6% in bandwidth** and by **more than 2× in the
overlap ratio** (0.16 vs 0.37). That is run-to-run variation on an identical
quantity, and it calibrates everything else: bandwidth differences under ~10%
are noise, and overlap differences under ~0.2 are not resolvable at three
repeats.

Recorded because without it, Metal's 1.02× would read as a small win and
OpenCL's 0.16-vs-0.37 as a real effect. Neither is.

### What that leaves

**On this machine the direct path is correct and not faster.** Metal's
`MTLIOCommandQueue` delivers the same bandwidth as the staged route (1.02×,
inside the floor) and the same overlap (0.96 vs 1.02, both ≈ 1). That is the
expected result and it is worth saying plainly rather than hunting for a win:
Apple's unified memory means the staged path is *already* a host store into
shared storage with no bus to cross, and Phase 2b's range-scoped waiting already
lets it overlap with compute. There is nothing left for a DMA engine to remove
when the source is a warm page cache and the destination is host-visible.

Its value would appear where the staged path costs something real — a cold cache
to bypass, or a discrete GPU where staging means a genuine PCIe crossing. Neither
is true here.

**One effect does clear the floor: OpenCL overlaps badly.** ~0.2–0.4 against
Metal's ~1.0. `enqueue_copy` goes on a queue and partially serializes against
the kernel, while Metal's host store runs on the CPU concurrently with the GPU.
That is architectural, not noise.

### The number that matters for streamed CFD

**~7–8 GB/s through the cache machinery** — and this is a *warm page cache*, so
it is not the drive. A memcpy from RAM on an M1 Max should run at tens of GB/s,
so 7.7 GB/s is **the machinery, not the memory**: per-block Python bookkeeping
and memmap page faults at 8 MiB granularity.

Two consequences:

1. The earlier streamed-CFD estimate assumed ~7 GB/s and happens to land right —
   but for the wrong reason. That figure is a software ceiling, not a disk one.
2. **On a fast drive you would be software-limited before you were disk-limited.**
   Larger blocks would amortise the per-block overhead; that is the first thing
   to try if the streaming case is ever pursued seriously.

### The SWIFT_T gate this probe was built to guard has evaporated

P1 existed to decide whether Python could schedule fast enough, because failing
would have promoted Swift from a compiler choice to a language target. That
decision can no longer be reached: **H5** demonstrated Swift-as-compiler end to
end and the Metal storage row then used it in anger, and **F4** gated off the
workload whose scheduling was in question. Closing P1 is therefore not the same
as answering the question it was written for, and the record should not read as
though it were.

### CUDA: cuFile is 1.26x at plateau — settled by a size sweep

Cold, on `id`. Every point drops the page cache first, so every point reads the
drive:

| block | staged | cuFile | ratio |
|---|---|---|---|
| 256 KiB | 0.50 | **0.87** | 1.74x |
| 1 MiB | 0.84 | **1.32** | 1.57x |
| 4 MiB | 1.24 | **1.53** | 1.23x |
| 16 MiB | 1.18 | **1.52** | 1.29x |
| **64 MiB (plateau)** | **1.21** | **1.52** | **1.26x** |

**cuFile is faster at every size**, plateauing at 1.52 GB/s against staged's
1.21. It also exceeds B0's plain cold read (1.18–1.71 across runs, itself noisy),
which is consistent: B0 is a POSIX read into userspace carrying its own copy,
while GPUDirect skips the host. cuFile's 1.52, stable across three block sizes,
is the better estimate of what the drive can deliver.

The controls behave as they must: OpenCL and CTypes have no direct lowering, so
their two columns track each other to within 1–9% across the whole sweep.

### M1 Max: a different regime, and it favours Apple Silicon

macOS has neither `posix_fadvise` nor `O_DIRECT`, and `purge` needs sudo, so the
cold sweep cannot run there. `F_NOCACHE` does not evict pages that are already
resident — but setting it on the **write** keeps them out of the cache in the
first place, and reading back with it set then reaches the device. Verified:
~4.3 GB/s against ~11 GB/s for a cached read of the same file.

| | device read | vs `id` |
|---|---|---|
| **M1 Max internal SSD** | **~3.9 GB/s** (range 2.7–4.3, n=13) | **~2.6x faster** |
| `id` NVMe (cuFile plateau) | 1.52 GB/s | — |

The first figure recorded here was **4.32 GB/s** — the maximum of the range,
taken from a single sample. Repeating the measurement across two rounds and
three backends gave 2.66 to 4.32, a 1.6x spread. `DriveBandwidthNoCache` now
repeats internally and prints a median with its range, because a single sample
of a bandwidth figure has misled this document four separate times.

The two machines therefore sit in different regimes. On `id` the drive
(1.5 GB/s) is ~5x below the machinery ceiling, so storage dominates completely.
On the M1 Max the drive (~3.9) and the machinery (6.3–7.8 warm) are within ~2x
of each other, so both matter.

**Consequence for streamed CFD on Apple Silicon**, using ~400 GB/s of device
bandwidth:

| method | on `id` (1.52 GB/s) | **on M1 Max (~3.9 GB/s)** |
|---|---|---|
| conservative-form CFD | ~98x | **~34x** |
| D3Q27 LBM | ~295x | **~103x** |

**Streamed residency is ~2.6x more viable on Apple Silicon than on the NVIDIA
box** — the SSD is ~3x faster while GPU bandwidth is comparable. That is a point
in favour of `STRATEGY.md`'s central thesis which had not been measured before,
and it arrives from the direction the thesis did not claim: not FLOPS per dollar
or unified memory, but storage bandwidth relative to compute.

**One asymmetry worth recording:** Metal's warm `MTLIOCommandQueue` figure runs
7.1–7.7 GB/s, roughly **twice** the device rate of ~3.9, so **`MTLIOCommandQueue` does not
bypass the page cache the way cuFile does**. That conclusion strengthened on
repetition rather than weakening: across runs the warm figure stays ~2x the
device, well outside the spread of either. Apple's storage API is built around
efficient streaming and decompression rather than DMA-that-skips-the-host, and
the residency layer should not assume the two behave alike. The macOS sweep
therefore stays warm and is labelled as such rather than presented beside the
Linux cold numbers.

### Why this answer is trustworthy where three earlier ones were not

The same measurement previously produced **0.24x**, then **4.36x**, then
**1.12x**. Each was a single block size — one point on a curve — reported as
though it were the curve. The sweep was suggested by Matteo as standard practice
in bandwidth studies, and it is what turned a number that kept moving into
something that explains why it moved:

- 0.24x was a **warm-vs-cold** comparison, not a size effect (staged reading RAM).
- 4.36x and 1.12x were both **cold** and both at 8 MiB, differing only by
  cold-I/O tail — the very variance a plateau averages out.

What makes the plateau credible: it is consistent across 4, 16 and 64 MiB;
both routes are measured on the same curve under the same conditions; and the
two no-direct-path backends serve as controls that correctly show no difference.

### The knee, which was asserted and is now measured

Both routes rise steeply to ~4 MiB and flatten after. At 256 KiB the staged path
delivers ~40% of its plateau. "Larger blocks are the first lever" was claimed
earlier without evidence; the lever engages at **~4 MiB and is spent by 16**.
`ResidentCache`'s 8 MiB default sits just past the knee — defensible, with
perhaps a few percent available at 16 MiB.

### Overlap does not survive real I/O on this machine

B3 falls to ~0.0 for both routes cold, where warm runs showed 0.2–0.4. Once the
load is genuine disk I/O (4–18 ms) it does not hide behind compute here. Metal's
~1.0 stands apart and for a clear reason: a host store into unified memory runs
on the CPU while the GPU works, which is a different mechanism from queuing a
transfer.

### Consequence for streamed CFD

Using the measured cuFile plateau of **1.52 GB/s**:

| method | at the assumed 7 GB/s | **measured** |
|---|---|---|
| conservative-form CFD (18 planes state, 108 traffic) | 21x | **~98x** |
| D3Q27 LBM (27 planes state, 54 traffic) | 64x | **~295x** |

The ~3x ratio between methods is unchanged, since it depends on state-to-traffic
rather than on the drive. And the direct path is worth a real 1.26x of that —
modest, consistent, and no longer a claim resting on one sample.

### Three wrong readings, one cause

The same figure — B2/B1 = 0.24x — was recorded three times with three different
explanations, and every one was an artefact of the page cache:

1. *"cuFile is a 4x pessimization from compatibility mode."* Falsified by a
   one-line diagnostic: `is_compat_mode_preferred() -> False`.
2. *"The cold control fixes it."* It did not — `POSIX_FADV_DONTNEED` cannot evict
   pages held by a live mapping, and `MemMapStore` keeps the file mapped.
3. *"The ground-truth line will catch it."* It did not either — B0 itself read
   12 GB/s, because `measure()` still held the warm stores' mappings alive while
   B0 ran.

Each time, cuFile was the only path performing I/O while everything else was
served from RAM. The fix that finally worked was structural rather than
attentional: measure the drive **first, before any store exists**, and have
`RawColdBandwidth()` report warm and cold together with an explicit
`NOT EVICTED` verdict when `cold >= 0.5 * warm`. The harness now refuses to
present a cache figure as a drive figure, instead of relying on a reader to
notice the contradiction.

### One thing left unexplained

CUDA's cold staged path reaches 0.36 GB/s where OpenCL and CTypes reach ~1.25
through the same `MemMapStore`. The difference is downstream of the read, in
`H2DSub` — plausibly the pageable `memcpy_htod` combined with 4 KiB memmap
faults on cold pages — but that is a hypothesis, not a measurement, and this
document has already carried three of those. Recorded as open.

### Consequence for streamed CFD

The corrected figures stand, since the direct path achieves drive speed:

| method | as originally quoted | corrected |
|---|---|---|
| conservative-form CFD (18 planes state, 108 traffic) | 21x | **~92x** |
| D3Q27 LBM (27 planes state, 54 traffic) | 64x | **~275x** |

with the ~3x ratio between methods unchanged, since it depends on
state-to-traffic rather than on the drive.

What the fair comparison adds: **without the direct path, CUDA streaming would be
~4x worse still** (0.36 GB/s rather than 1.59). So the storage lowering is not a
refinement on this hardware — it is most of what makes streamed residency
arithmetically arguable at all.

### What B3 can and cannot resolve

The overlap estimator's noise floor is **±0.15**, established the same way as
the bandwidth floor: OpenCL has no direct lowering, so its two columns measure
one quantity twice, and they came back 0.31 and 0.44. Only large differences are
meaningful — Metal's ~1.0 against OpenCL's ~0.3 is real; anything closer is not.

Two harness defects were found and fixed while measuring, both of which had been
producing confident nonsense:

- `_sync()` poked `tenet.Finish()`, which **does not exist on the CUDA Tenet**,
  so the kernel leg timed 0.0 ms — an async launch with nothing waiting on it.
  Every `IdpyArray*` has carried `Sync()` since Phase 2b; that is the portable
  spelling, and the residency layer already used it.
- The filler kernel was calibrated *inside* the overlap routine, which runs once
  per route, so the two routes were timed against **different kernels** and the
  comparison silently stopped being controlled. Calibration now happens once per
  backend and the kernel is shared. A ratio outside `[-0.2, 1.2]` — impossible
  by construction — is now reported as unresolved rather than printed.

---

## 2l. P3 answered: one dynamic buffer is enough

The last open probe. `SetDynamicSharedMemory` allows one runtime-sized buffer
because CUDA exposes a single `extern __shared__` region, and the question was
whether that lowest-common-denominator constraint blocks the lattice kernels or
is merely an ergonomic wrinkle. If it blocked them, the constraint rather than
the residency layer would have been the real obstacle.

**It does not block them.** Answered by building the case that would break it: a
two-field 3-point stencil with periodic halos,

    out[i] = (a[i-1] + a[i] + a[i+1]) + 2*(b[i-1] + b[i] + b[i+1])

which needs two tiles by construction — each field carries its own `BLOCK+2`
halo window, and every output element reads slots written by other lanes.

| check | OpenCL | Metal |
|---|---|---|
| T1 two logical tiles inside **one** dynamic buffer, manual offsets | exact | exact |
| T2 two tiles from **static** shared memory | exact | exact |
| T3 guard: declaring two dynamic buffers raises | — raises `NotImplementedError` |

Two routes, both working. The dynamic case uses exactly the workaround the
`SetDynamicSharedMemory` docstring prescribes — one buffer, indexed manually for
multiple logical tiles — and the static case has no constraint to work around at
all, since a compile-time declaration is ordinary and a kernel may have as many
arrays as fit.

Capacity is not close to binding either: two tiles at `BLOCK=64` are ~0.5 KiB
against 48 KiB of CUDA shared memory and 32 KiB of Metal threadgroup memory. The
constraint is on the *number of declarations*, not on space, and one declaration
holds as many logical tiles as arithmetic can address.

### The probe has teeth

A negative control confirms it can fail: pointing both logical tiles at the same
base — the mis-addressing the manual-offset workaround exists to get right —
gives `max|out-ref| = 1532.25` rather than a quiet pass. The body is also
identical text between T1 and T2 apart from the tile bases, which isolates the
question to where the storage comes from rather than how it is addressed.

**Consequence:** the LCD constraint stands as documented and costs nothing for
lattice work. Phase 5 remains gated on F4, not on this.

**Phase 0 is complete.** F2 and F4 settled by inspection, both negative; P1
measured after correcting itself three times; P3 built and settled here.

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
