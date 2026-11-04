# DESIGN-D2D (pointer)

**Canonical design (Loom-owned path):**  
[`docs/planning/D2D-cross-device.md`](../../docs/planning/D2D-cross-device.md)

Rayleigh (NumPy Gate1) is the **Phase C consumer** (Option A orchestration under
`multitenet/`). Loom constraints are **RECEIVED**. Propose ≠ merge; Matteo gates.

Companion Grok Build prompt (Phase A only):  
[`docs/planning/D2D-cross-device-grok-build.md`](../../docs/planning/D2D-cross-device-grok-build.md)  
(local pointer: [`GROK_BUILD_PROMPT_D2D.md`](./GROK_BUILD_PROMPT_D2D.md))

Phase B face-strip design (plumbing only):  
[`docs/planning/D2D-phase-b-halo.md`](../../docs/planning/D2D-phase-b-halo.md)  
Grok Build brief: [`docs/planning/D2D-phase-b-grok-build.md`](../../docs/planning/D2D-phase-b-grok-build.md)

## Phase C — merged (PR #19 @ `573b3fb`; Ctx harness PR #20 @ `c946b0a`)

Multi-tenet `exchange_faces` (Rayleigh / LBM Option A): Loom **PASS-WITH-NITS**
RECEIVED; applied and **merged**. idHome 2-GPU product OVERALL_GATE=PASS;
stock CUDA smoke needs Ctx push (PR #20).

- Design: [`docs/planning/D2D-phase-c-exchange-faces.md`](../../docs/planning/D2D-phase-c-exchange-faces.md)
- Code: [`multitenet/`](./multitenet/) — `face_elem_range` (local copy),
  `bind_two_way_z_split`, `exchange_faces` (two sync `D2DSub`), tests
- `IdpyMemory` contracts **untouched** (ranges-only); no B2 / MemcpyPeer /
  aggregate stream API / silent macro face ship
- **Cut-halos are not collided** — only interiors collide; exchange fills
  neighbor cut-halos for pull
- Call-site order: **plant → collide → `exchange_faces` → stream**
- Populations only; C0 = two sequential sync `D2DSub` with per-leg streams

## Phase D — NumPy D0 merged (PR #21); CUDA D1 in flight

Dual-tenet CUDA cold chain (propose ≠ merge until landed):

- Design: [`docs/planning/D2D-phase-d1-cuda-dual-tenet.md`](../../docs/planning/D2D-phase-d1-cuda-dual-tenet.md)
- Driver: `step_two_way_cuda_cold` — D2H → plant+collide → H2D → exchange → D2H → stream → H2D
- Host plant/collide/stream; device exchange only; Ctx push (PR #20); no IdpyMemory edits

Dual-tenet LBM step wire (NumPy D0 first): plant → collide(owned) →
`exchange_faces` → stream **write-owned / read-halo**.

- Design: [`docs/planning/D2D-phase-d-dual-tenet-step.md`](../../docs/planning/D2D-phase-d-dual-tenet-step.md)
- Stream writes owned rows only (GPU0 z∈`[0,n0)`; GPU1 z∈`[depth,nz1)`); cut-halos
  are not stream destinations
- Native `(nz,nx,Q)` path A for D0; never Gate1 `[:,1:-1,:]` on multitenet
- CUDA D1 / IdpyMemory / Spin blocked until NumPy 1-step owned-gather bit-identical
- propose ≠ apply; Matteo gates
