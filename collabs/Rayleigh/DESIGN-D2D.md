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

## Phase D — NumPy D0 merged (PR #21); CUDA D1 in PR #23

Dual-tenet LBM step wire: plant → collide(owned) → `exchange_faces` → stream
**write-owned / read-halo**.

- Design (D0): [`docs/planning/D2D-phase-d-dual-tenet-step.md`](../../docs/planning/D2D-phase-d-dual-tenet-step.md)
- Stream writes owned rows only (GPU0 z∈`[0,n0)`; GPU1 z∈`[depth,nz1)`); cut-halos
  are not stream destinations
- Native `(nz,nx,Q)` path A; never Gate1 `[:,1:-1,:]` on multitenet
- NumPy D0 1-step owned-gather bit-identical (PR #21) — CUDA D1 gate cleared

### CUDA D1 (PR #23 — Loom PASS-WITH-NITS; Matteo merge gate)

- Design: [`docs/planning/D2D-phase-d1-cuda-dual-tenet.md`](../../docs/planning/D2D-phase-d1-cuda-dual-tenet.md)
- Driver: `step_two_way_cuda_cold` — D2H → plant+collide → H2D → exchange → D2H → stream → H2D
- Host plant/collide/stream; device exchange only; Ctx push (PR #20); no IdpyMemory edits
- idHome CUDA cold 1-step + N=2 soft owned-gather bit-identical (FYI log on idHome)
- propose ≠ merge; Matteo gates

## Phase E — dual-tenet science gate (in PR; propose ≠ merge)

Tiny buoyant NSF vs monolith on cold dual-tenet path (post-D1 @ `fc97779`).

- Design: [`docs/planning/D2D-phase-e-science-gate.md`](../../docs/planning/D2D-phase-e-science-gate.md)
  (Loom PASS-WITH-NITS must-fixes folded)
- Harness: `multitenet/test_science_two_way.py`
- Hard OVERALL_GATE: N=100 owned-gather bit-identical + NSF INTERIOR probe
  (`plant_sample_walls` → `hydro_from_lattice` → `slice(3,-3)`) match-monolith
  ≤1e-12 + buoyant `run_smoke` residual match-monolith
- Soft N=200 informational (does not fail gate); CUDA `<2` devices → SKIP
- Frozen E0: nx=32 nz=64 z_cut=32 depth=3 seed=42 nu=0.10 g=1.2e-3
- Not Half-A / Fig.4 campaign; no warm; no IdpyMemory
