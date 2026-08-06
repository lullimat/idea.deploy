# idpy architecture

Three diagrams, each showing something that is not obvious from the source tree.
Rendered by GitHub directly; no toolchain needed.

---

## 1. One kernel source, four lowerings

A kernel is written once against `IDPY_T`, the metalanguage. `IdpyKernel.Code(lang)`
emits real source for the target, and that source is available for inspection —
there is no hidden compilation step and nothing is rewritten behind your back.

```mermaid
flowchart TD
    K["IdpyKernel subclass<br/><i>body written once, IDPY_T</i>"]
    K --> C["Code(lang)"]

    C --> CU["CUDA C<br/><code>__shared__</code> · <code>__syncthreads()</code>"]
    C --> CL["OpenCL C<br/><code>__local</code> · <code>barrier(...)</code>"]
    C --> MT["Metal Shading Language<br/><code>threadgroup</code> · <code>threadgroup_barrier(...)</code>"]
    C --> CT["plain C<br/>thread index becomes a serial loop"]

    CU --> TCU["Tenet (pycuda)"]
    CL --> TCL["Tenet (pyopencl)"]
    MT --> TMT["Tenet (pymetallic)"]
    CT --> TCT["Tenet (ctypes)"]

    TCU & TCL & TMT & TCT --> D["Deploy(args)"]
```

The portable tokens are the point: `idpy_shared`, `idpy_sync`, `idpy_sync_global`
become each backend's native construct at emission. **CTypes is a full peer, not
a fallback** — it has its own `Tenet`, kernel module and array type, and
`Code()` rewrites the SIMT thread index into a loop for it. That the metalanguage
survives the SIMT → non-SIMT transition is why a CPU-only CI job is a meaningful
check on the generator.

---

## 2. Three insertion points, not two

Where code can enter a build. Most frameworks have the first two; the third is
what makes host-side capabilities — storage streaming, residency — expressible
at all.

```mermaid
flowchart LR
    subgraph one["1 · per kernel object"]
        A1["IdpyKernel.Code()"]
        A2["generated per instance<br/>constants, types, unrolled bodies"]
        A1 --- A2
    end

    subgraph two["2 · into the compile unit"]
        B1["definitions_files<br/>include_dirs · objects_files"]
        B2["static code linked beside<br/>the generated kernel"]
        B1 --- B2
    end

    subgraph three["3 · host-side capability"]
        C1["Tenet + HostModule"]
        C2["fixed host code, compiled once<br/>cuFile · MTLIOCommandQueue"]
        C1 --- C2
    end

    one --> BUILD["compiled kernel"]
    two --> BUILD
    three --> RUN["host drives the device<br/><i>around</i> dispatch"]
    BUILD --> RUN
```

A kernel body never issues an I/O command — the **host** does, around dispatch.
So storage streaming and residency belong at (3), and channel (2) cannot reach
them. This is also why Swift is a **compiler choice** rather than a language
target: the Metal storage shim is fixed host code built by `HostModule`, so
there is no `SWIFT_T` in `idpy_langs_dict`.

---

## 3. The residency layer: where the backends stop mattering

Holding a working set larger than device memory. The asymmetry is the design's
central claim.

```mermaid
flowchart TD
    POL["<b>ResidentCache</b> — policy<br/>LRU/FIFO · pinning · dirty tracking · write-back<br/><i>no per-backend branches</i>"]

    POL --> PRIM["<b>primitives</b><br/>SubView · H2DSub · D2HSub · Sync"]

    PRIM --> P1["CUDA<br/>staged async copy<br/>pinned host memory"]
    PRIM --> P2["OpenCL<br/>sibling command queue"]
    PRIM --> P3["Metal<br/>range-scoped waiting on<br/>in-flight command buffers"]
    PRIM --> P4["CTypes<br/>numpy; unified by construction"]

    POL --> ST["<b>BackingStore</b><br/>storage larger than the device"]
    ST --> S1["MemMapStore<br/>mmap + page cache"]
    ST --> S2["KvikIOStore<br/>cuFile / GPUDirect"]
    ST --> S3["MetalIOStore<br/>MTLIOCommandQueue via a Swift shim"]
```

**The policy layer contains no per-language branches at all.** The primitives
beneath it are genuinely different — a staged asynchronous copy on CUDA, a
second queue on OpenCL, waiting against in-flight command buffers on Metal,
plain numpy on CTypes — while the policy expressed on top is one program.

Measured rather than asserted: the same blocked stencil sweep reports **62 hits
and 34 misses on every backend**, matching the closed form `hits = 2(n-1)`,
`misses = n+2` exactly. Identical traffic across four backends is what
demonstrates the policy is backend-independent; identical *output* alone would
not have, since a diverging cache would still return correct numbers by
reloading.

`ReadBlockInto` returns `False` to **decline**, and declining is not an error —
it is how a store says it has no direct path here, after which the cache falls
back without caring why.

---

## Layering

> `idpy` core never imports `idpy` physics.

| layer | packages |
|---|---|
| core | `IdpyCode`, `CUDA`, `OpenCL`, `CTypes`, `Metal`, `Utils` |
| physics | `LBM`, `IdpyStencils`, `SpinNetworks`, `PRNGS` |

Enforced by `scripts/check_layering.py`, which runs as CI's first step — static
analysis over import statements, so it needs no environment and no GPU. Two
violations are grandfathered in a `KNOWN` allowlist; nothing new may be added.
