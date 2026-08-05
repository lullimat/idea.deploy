# idea.deploy — Residency & Host-Capability Layer: session bootstrap

**Branch:** `feature/host-module-and-residency` **Base:** `master` @ `e49f997` (merge of `feature/idpy-t-shared-memory`, PR #4) **Status:** design agreed, nothing implemented. Phase 0 is measurement, not code — and **half of it is already resolved by inspection** (see §4).

This file bootstraps a working session. Read it fully before proposing changes. The "Non-goals" section is load-bearing — this design drifted several times during scoping and those doors are deliberately closed.

**Revision 2026-08-05.** All §2 line anchors re-verified against `e49f997`; all still resolve. Two of the four Phase 0 probes (P2, P4) turned out to be answerable by reading the tree rather than by measurement, and **both answers are negative** — folded into §4 as findings F2/F4. Consequences: Phase 5 is now gated off by default, and the primary residency test case moved from the LLM workload to the lattice (§1, §5).

------

## 1. What this is for

Give `idea.deploy` a **portable residency capability**: the ability to hold a working set larger than device memory and stream it in/out under an explicit policy, expressed once and lowered to each backend.

The motivating demonstration is [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) (Apache-2.0, Swift+Metal), which runs a 26B-parameter MoE model in ~2 GB by streaming experts from SSD with a 16-slot LFU cache per layer. **The interesting part of that project is not its kernels — it's the host-side residency policy.** That policy is what we want, portably.

The durable payoff is not LLM inference. It is: *a lattice larger than device memory, on any vendor's silicon, from one source.*

**The lattice is therefore the primary test case, not the LLM.** This inverts the original plan, for a reason that only became visible on inspection: the two workloads want nearly disjoint machinery. The MoE case needs sub-byte packed types (F4 — not expressible), per-layer LFU expert caching, and possibly more than one threadgroup buffer (P3). The lattice case needs large contiguous halo-structured tiles, regular strided access, and fp32 — none of that machinery. Building the residency abstraction against the LLM first would shape it around access patterns the durable use case does not have, at the cost of core metalanguage work in service of a demonstration. The LLM case remains a well-documented, independently verifiable *secondary* validation, run only if P3/F4 are ever cleared.

### Target audience for the eventual artifact

Non-NVIDIA hardware vendors (AMD, Apple, Intel). The publishable claim is **memory footprint and portability**, explicitly *not* peak throughput. Headline metric should be *largest problem that fits and runs correctly on this device*, with a correctness column (cross-backend numerical agreement), not wall-clock rankings.

------

## 2. Verified current state

All of the following was checked against the tree at `e49f997`, and every line anchor re-verified on 2026-08-05. Re-verify before relying on any of it.

### Language/backend registry

```
idpy/IdpyCode/__init__.py:97
idpy_langs_dict = {'CUDA_T': CUDA_T, 'OCL_T': OCL_T, 'CTYPES_T': CTYPES_T, 'METAL_T': METAL_T}
```

The lang key is the **Python binding module name** (`pycuda`, `pyopencl`, `ctypes`, `pymetallic`), not a kernel language. A lang answers *"where does the kernel execute and what drives it."* Python is the host in every lang, including `CTYPES_T`.

`CTYPES_T` is a full peer, not a special case: own `Tenet`, own `KernelModule`, own `IdpyArrayCTYPES`, own entries in `KernQualif`/`AddrQualif` (`idpy/IdpyCode/IdpyConsts.py:69, 103, 120, 165`) where the kernel qualifier is empty — plain C functions. `IdpyKernel.Code()` branches on `CTYPES_T` in ~20 places, notably around `gthread_id_code`, rewriting the SIMT thread index into a loop. **The meta-concept already survives the SIMT → non-SIMT transition.** That generality is the asset this work builds on.

### The compile-and-load facility

`idpy/CTypes/CTypes.py:84` — `CTypesKernelModule`:

- md5-hashes source and compile string; caches `.c` and `.so` under `/tmp/idpy_ctypes_kernels` (`idpy/CTypes/__init__.py:55`)
- compiles via `subprocess.run` on a space-split command string
- `GetKernelFunction(name, custom_types)` builds `argtypes`, mapping every pointer param to `ndpointer(..., flags="C_CONTIGUOUS")`

Nothing in this class is CPU-compute-specific. **Two things block reuse as a general capability compiler:**

1. **Compiler is a module constant** — `idpy_ctypes_compiler_string_h` (`idpy/CTypes/__init__.py:46-52`) is `gcc`/`clang` with `-std=c99`. No path to `swiftc`.
2. **ABI is numpy-shaped** — `ndpointer` assumes host arrays. A cuFile shim needs a CUDA device pointer (integer handle from pycuda), a file descriptor, byte offsets, a stream handle. `CTypesTypes` has no opaque-handle/`uintptr` type.

### Reserved-but-unwired surface

`IdpyKernel.__init__` (`idpy/IdpyCode/IdpyCode.py:112`) accepts and **type-validates** `headers_files`, `include_dirs`, `definitions_files`, `objects_files` — but only `self.headers_files` is ever assigned (line 163) or consumed. The other three are validated and dropped. This is the static-linkage channel, carved out and not yet built.

Confirmed stronger than "unwired": `include_dirs`, `definitions_files` and `objects_files` occur in the whole of `IdpyCode.py` **only** in the signature (lines 129-130) and the validation (lines 136-141). They are never stored on `self`. A caller who passes `objects_files=[...]` today gets type-checked, then silently ignored — so Phase 4 is a **bug fix** as much as a feature, and until it lands the three parameters should arguably raise `NotImplementedError` rather than accept-and-drop.

### Metal backend

`idpy/Metal/Metal.py` — `Tenet` (line 107) carries `device`, `queue`, `mem_pool`, `allocator`, `last_command_buffer`. `MetalMemoryPool` (line 51) is a free-list keyed by `(shape, dtype)`.

`IdpyArrayMETAL` (`idpy/IdpyCode/IdpyMemory.py:271`) already documents true unified-memory semantics: `.host` is a NumPy view over the *same* storage as the Metal buffer; `D2H()` returns a shared view rather than a copy. **The zero-copy half of the UMA residency case is present already** — but only that half; see the next subsection before treating it as a head start.

### Memory layer: no sub-range, no async, and a drain on every host touch

This subsection records what was P2. It is the single most consequential fact in this document, so it belongs in the verified-state section rather than in a probe.

Across all four `IdpyArray*` classes in `idpy/IdpyCode/IdpyMemory.py` there is **no `__getitem__`, no slicing, no view, no `set_async`, no `memcpy_htod_async`**. The `async_` keyword is accepted and then discarded on every backend: `IdpyArrayCUDA.H2D` (line 62) forwards to the synchronous `super().set(ary=ary)`, dropping it. The underlying bindings support more than this — pycuda has `set_async` and contiguous slicing — but the idpy façade does not expose it.

On Metal the obstacle is not a missing feature but an actively conflicting design choice. `IdpyArrayMETAL.H2D`, `D2H` and `SetConst` each begin with `_sync_tenet()`, which calls `tenet.Finish()` — **a full GPU drain**. So the operation residency requires:

> write bytes into slot *k* of a device buffer, asynchronously, while the GPU reads slots *j≠k*

is **structurally excluded** on the Metal path, because the write path's first action is to serialize against all outstanding GPU work. Zero-copy was bought at the price of concurrency. Enabling residency streaming means replacing drain-on-touch with fence/event-scoped synchronization — a change to the Metal memory model, not an addition to it. Budget for it accordingly; it is a prerequisite of Phase 3's Metal row, not a detail of it.

### Known constraint

`IdpyKernel.SetDynamicSharedMemory` docstring: portability follows "the lowest common denominator (CUDA's single `extern __shared__` region), so at most one buffer is allowed." See probe **P3**.

------

## 3. The architecture

### Three insertion points, not two

| #    | mechanism                                           | where                                                  | status            |
| ---- | --------------------------------------------------- | ------------------------------------------------------ | ----------------- |
| 1    | code generated **per kernel object**                | `IdpyKernel.Code()`                                    | built             |
| 2    | static code linked into the **kernel compile unit** | `objects_files` / `definitions_files` / `include_dirs` | reserved, unwired |
| 3    | **host-side device-capability** extension           | `Tenet` (per-backend module)                           | does not exist    |

A kernel body never invokes an I/O command queue — the *host* does, around dispatch. So storage-streaming and residency belong at **(3)**, on `Tenet`. Channel (2) cannot reach it.

### The abstraction is a residency policy, not an I/O queue

Put on `Tenet` an interface meaning *"ensure these bytes are device-reachable; evict those."* It has two genuinely different lowerings, and this split is the publishable content:

- **Explicit staging** — discrete GPUs; real copies into device buffers, async, overlapping.
- **Page residency** — UMA (Apple, Grace/GB10, MI300A, Strix Halo); no copy exists, only which pages are resident. `mmap` + `madvise` + the OS page cache may do most of the work.

The kernels are backend-*syntactically* different. The residency policy is backend- *semantically* different. That asymmetry is the interesting result.

### Extract, do not overload

`CTypesKernelModule` should be lifted out of `idpy/CTypes/` into a shared **`HostModule`** facility with **the compiler as a parameter**. Then:

- `CTYPES_T` keeps meaning exactly one thing: *kernel executes as C on the CPU*.
- Every `Tenet` gets a compile-and-load facility for its capability layer.
- **The same facility compiles both** the C shims (cuFile, rocm-xio) **and** the `@_cdecl` Swift shim for `MTLIOCommandQueue`.

The last point matters: **Swift becomes a compiler choice, not a language target.** Do not add `SWIFT_T` to `idpy_langs_dict` to get a Metal binding — that would produce a `GetKind() == "cpu"` artifact driven by a CUDA/Metal Tenet, which is the conflation this design exists to avoid.

### Per-backend lowering

| backend | unified memory                                               | storage → device                             | binding work needed                                          |
| ------- | ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| CUDA    | `cudaMallocManaged`, `MemAdvise`/`MemPrefetchAsync`; system-allocated on Grace | cuFile / GPUDirect Storage                   | **none** — [KvikIO](https://github.com/rapidsai/kvikio) wraps cuFile in Python *with POSIX fallback* |
| AMD     | HIP managed memory; **OpenCL SVM** via pyopencl              | [rocm-xio](https://github.com/ROCm/rocm-xio) | ctypes binding to rocm-xio (C-only)                          |
| Metal   | native UMA (`IdpyArrayMETAL`) — **but serialized today, see F2** | `MTLIOCommandQueue` (Metal 4)                | `@_cdecl` Swift shim via `HostModule`                        |
| CTypes  | trivially unified                                            | `mmap` + page cache                          | none                                                         |

Three consequences worth internalizing:

- **AMD may need no new lang.** `OCL_T` already runs on AMD and OpenCL SVM *is* the unified-memory abstraction, exposed by pyopencl. Only the storage side needs new surface.
- **CUDA is the cheapest** because KvikIO degrades to POSIX when GDS is absent — giving a correctness path on machines without the hardware.
- **Metal is more expensive than this table suggests.** The "native UMA" cell is true of the hardware and only half true of the current binding: `IdpyArrayMETAL` drains the GPU on every host touch (§2, F2). The Metal row therefore carries Phase 2b's synchronization rework on top of the Swift shim. Start Phase 3 with CUDA, not with the home platform.

------

## 4. Phase 0 — findings and remaining probes

The original plan listed four probes. **Two of them (P2, P4) did not need measurement at all** — they were answerable by reading the tree, and both came back negative. They are recorded below as findings F2 and F4. One (P1) was measuring the wrong quantity and has been rewritten. Only P3 survives as a genuine probe.

The general lesson, now recorded in §8: *before designing a probe, check whether the tree already answers it.* An afternoon of reading replaced half of a phase.

### F2 (was P2) — `IdpyMemory` cannot express the residency op. **Settled: no.**

Full evidence in §2, "Memory layer". Summary: no sub-range views, no async path on any backend, and on Metal an unconditional GPU drain at the head of every host-side write. The required async partial write into a sub-range with stream semantics is absent on CUDA and structurally excluded on Metal.

**Consequence:** the memory layer is a *prerequisite* of Phase 2/3, not a passenger. Two work items, neither optional:

1. sub-range views + a real async path on `IdpyArray*` (CUDA first — pycuda already supports it underneath);
2. replace Metal's drain-on-touch with fence/event-scoped synchronization.

Item 2 is the larger risk and should be scheduled before any Metal residency lowering is attempted.

### F4 (was P4) — `IDPY_T` cannot express sub-byte packed types. **Settled: no.**

`CustomTypes` (`idpy/Utils/CustomTypes.py:32`) is a bare `{alias: c_type_string}` dictionary with `Push`/`Set`/`Pop`/`ToList`. There is no width, no packing, no layout, no accessor generation — it emits typedefs and nothing else. 4-bit affine weights at group 64 with group-wise scales/zero-points are not *hard* to express in this model; they are **outside it**.

**Consequence:** the outcome P4 was written to avoid is simply the actual outcome. Supporting the MoE kernel set means extending the metalanguage type system — core work on the crown-jewel asset in service of a demonstration. **Phase 5 is therefore gated off by default** (§5), and the primary residency test case moves to the lattice (§1). Reopening it requires a standalone decision to build a packed-layout type model *for its own sake*, justified by workloads beyond this one.

### P1 (rewritten) — sustained streaming bandwidth under overlap

*Original framing: per-layer Python planning latency, gating `SWIFT_T`. Dropped, because it was low-information — the document predicted a pass, and the non-goals already forbid `SWIFT_T` unless it fails. A probe that gates a door you have locked buys nothing.*

The real scheduling risk is not planning latency in isolation; it is **overlap**. Measure whether bounded parallel `pread` sustains enough bandwidth to keep the working set warm *while compute runs*, under the GIL, with F2's synchronization behaviour in the loop.

Build a stub with no model: real cache-planning logic, real `pread` against a file of representative size, and a concurrent dummy GPU workload. Report achieved streaming bandwidth with overlap enabled *and* disabled — the ratio is the answer. Include per-layer planning latency as a secondary number, since it is nearly free to collect and still gates `SWIFT_T`.

- **PASS** (overlap holds, planning ≲ budget) → dispatch stays in Python; only the binding is needed; `SWIFT_T` is not built.
- **FAIL on planning** → a specialized per-kernel Swift encoder becomes the fix, and `SWIFT_T` earns a place as a real lang (insertion point 1 applied to host code).
- **FAIL on overlap** → the fix is in the memory layer (F2), not in a new lang. Do not reach for `SWIFT_T` for this failure mode.

Failure mode if unmeasured: correct output at unusable rates. The memory claim survives and the demonstration dies.

### P3 — is one shared buffer enough? *(the one surviving probe)*

`SetDynamicSharedMemory` allows at most one dynamic shared buffer (CUDA LCD). Determine whether the target kernels need >1 threadgroup buffer. If yes, the LCD constraint — not the residency layer — is the real blocker, and this design needs revisiting.

Note the scope change: with the lattice now the primary test case, **P3 should be evaluated against the lattice kernels first** (tiled stencil/halo work), not against MoE attention. The MoE answer only matters if F4 is ever reopened.

**Deliverable for Phase 0:** a short findings file, `docs/residency-probes.md`, recording F2 and F4 as settled, the P1 measurement, the P3 answer, and the resulting go/no-go per phase.

------

## 5. Phases

Each phase is a separate PR onto `feature/host-module-and-residency`.

**Ordering decision (2026-08-05): `HostModule` lands first, in place under `idpy/`; the `src/` restructure comes later.** `STRATEGY.md` Phase 0 relocates `idpy/CTypes` to `src/idpy/core/backends/ctypes_backend.py`, and this document's Phase 1 lifts `CTypesKernelModule` out of the same directory. Neither document referenced the other. Resolved in favour of `HostModule` first: it is the smaller change, it has a recorded test baseline (`docs/residency-probes.md`), and it unblocks the residency work — whereas the `src/` move touches every `sys.path` hack and import in the tree with no CI to catch fallout. Relocating `HostModule` afterwards is a path-only edit. **Consequence:** Phase 1 targets a shared location *within the current layout* (e.g. `idpy/IdpyCode/HostModule.py`), not `src/`. `STRATEGY.md` should be updated to note that its Phase 0 now inherits this file.

**The project is Phases 1, 2, 2b and 4. Phases 3 and 5 are the evidence campaign.** This split is worth internalizing: 1, 2, 2b and 4 fix real coupling, a real dead API and a real gap in the memory layer, and they remain worth doing *even if the residency thesis and the LLM demonstration both evaporate*. They need no vendor hardware, no Swift, and no model. Phase 3 turns them into a published claim; Phase 5 is optional and currently gated off.

**Phase 1 — extract `HostModule`.** Pure refactor, no behavior change. Lift `CTypesKernelModule` to a shared location, parameterize the compiler (command + flags + source extension), keep the md5 cache. `CTYPES_T` becomes its first consumer with identical behavior. Add an opaque-handle/`uintptr` type to `CTypesTypes` and an argtype path that does not assume `ndpointer`. *Acceptance: existing tests pass unchanged; a trivial non-C (Swift) shim compiles and loads on macOS.*

**Phase 2 — residency policy interface + reference implementation.** Define the `Tenet` capability interface. Implement it for `CTYPES_T` via `mmap` + page cache — the simplest correct version, runs anywhere, and validates the eviction logic before any vendor machinery can hide a bug. *Acceptance: a working-set-larger-than-RAM lattice test passes on CPU.*

**Phase 2b — memory-layer prerequisites (new, from F2).** Sub-range views and a genuine async path on `IdpyArray*`, CUDA first; then replace Metal's `_sync_tenet()` drain-on-touch with fence/event-scoped synchronization. Split out as its own PR because it touches the existing memory model and carries regression risk for LBM. **Blocks the CUDA and Metal rows of Phase 3.** *Acceptance: an async partial write into a sub-range, concurrent with a kernel reading a disjoint sub-range, verified correct on CUDA and Metal; existing LBM tests unchanged.*

**Phase 3 — backend lowerings.** CUDA via KvikIO (start here; POSIX fallback means it works without GDS hardware). Metal via the `@_cdecl` shim on `HostModule`. AMD via pyopencl SVM, plus a rocm-xio ctypes binding if hardware access materializes. *Acceptance: same residency test passes per backend, same numbers.*

**Phase 4 — wire `objects_files`.** Implement insertion point (2), including passing `IdpyKernel` objects into other kernels' compile units. Independent of the above and valuable on its own; also removes the accept-and-drop behaviour documented in §2.

**Phase 5 — gated off by default; requires reopening F4.** The turbo-fieldfare kernel set, **reimplemented** in `IDPY_T` from the documented algorithms. Blocked on F4 (no packed sub-byte types) and conditional on P3. Do not start this without a standalone, separately justified decision to extend the metalanguage type system. *Acceptance, if ever run: cross-backend numerical agreement among idpy's own lowerings is the primary criterion; agreement with turbo-fieldfare's outputs is a secondary sanity check only. An oracle matched too closely becomes a specification of its quirks.*

------

## 6. Non-goals

Closed deliberately. Reopen only with an explicit reason recorded in the PR.

- **No source lifting / "stripping function"** over turbo-fieldfare's Metal or Swift. Lifting hand-tuned kernels is decompilation-into-a-DSL; the parts that resist lifting are exactly the tuned parts, which by design live below the portable line. And round-trip fidelity is trivially satisfiable by carrying the original text as a payload — it proves losslessness, not abstraction. **The property worth testing is cross-backend numerical agreement, not round-trip identity.**
- **No Apache-2.0 source into `idpy/`.** turbo-fieldfare is Apache-2.0 with NOTICE and attribution obligations. Reimplement from documentation. Keep any comparison harness in a separate repository. The core's clean provenance is a documented strategic asset.
- **No `SWIFT_T` in `idpy_langs_dict`** unless P1 fails *on planning latency*. A P1 failure on overlap points at the memory layer (F2), not at a new lang.
- **No port of the app, decode service, OpenAI server, repack installer, or CLI.** Those are Python-trivial and generate zero evidence for the portability claim while consuming all the time. A port makes you the maintainer of an inference runtime with a model-support treadmill; a *result* has no such obligation.
- **No competing on peak throughput.** Footprint is a generator-level property (layout, residency, tiling); speed is a backend-level property (warp scheduling, MMA intrinsics) that by design lives below the portable core. Conceding FLOPs is structural, not apologetic.

------

## 7. Open questions

- Does `pymetallic` expose enough of `MTLDevice`/`MTLHeap` to attach an I/O queue, or does the shim need to own device creation too? **Now coupled to Phase 2b:** the same investigation should establish whether pymetallic exposes command-buffer completion handlers or events fine-grained enough to replace `_sync_tenet()`. If it does not, the Swift shim may have to own synchronization as well as I/O, which enlarges its scope considerably.
- Does OpenCL SVM on AMD actually give fine-grained residency control, or only coarse-grained buffer sharing? (Determines whether AMD needs rocm-xio at all for the UMA case.)
- ~~Which workload is the honest demonstration on a 128 GB GB10 or MI300A, where a 26B model simply fits?~~ **Resolved** (§1): the lattice, and it is now the primary test case on every machine rather than a fallback for the large-memory ones. This removes the awkwardness of a demonstration whose premise dissolves on exactly the hardware most worth impressing.
- Does the lattice residency case have a natural *eviction* story, or only a streaming one? An LFU expert cache has obvious reuse structure; a sweep over a lattice larger than memory may be a pure streaming pattern with no meaningful cache policy to validate. If so, Phase 2's eviction logic needs a second test case with genuine reuse — blocked tiling, or multi-pass stencils — to be exercised at all.
- Hardware access for AMD/Intel rows. The empty rows in the published matrix are the outreach mechanism — "portable stack, reproducible harness, one command, I don't have your hardware" is a cheap ask for vendor devrel (credits or a loaner, no contract, no IP entanglement).

------

## 8. Working agreement for this session

- Verify claims against the tree before acting on them; this file's line anchors are from `e49f997` and will drift.
- **Before designing a probe, check whether the tree already answers it.** Two of the four Phase 0 probes were reading exercises misfiled as measurements, and both returned negative — meaning the design's two biggest risks were sitting in plain sight in the source while scheduled as future work. Measurement is for behaviour under load; inspection is for capability.
- **When a probe is expected to pass and gates something already ruled out, it is not a probe.** That was the original P1. Ask what would actually change based on the outcome.
- Phase 0 first, but it is now short: F2 and F4 are settled, so only P1 and P3 remain.
- Prefer small PRs with acceptance criteria over a long-running branch.
- When a design choice looks like "add a new lang," check first whether it is actually a *binding* (fixed code, written once, host-side) rather than a *target* (generated per kernel object). That test resolved most of the scoping questions behind this document.