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

## Phase D — NumPy D0 merged (PR #21); CUDA D1 merged (PR #23)

Dual-tenet LBM step wire: plant → collide(owned) → `exchange_faces` → stream
**write-owned / read-halo**.

- Design (D0): [`docs/planning/D2D-phase-d-dual-tenet-step.md`](../../docs/planning/D2D-phase-d-dual-tenet-step.md)
- Stream writes owned rows only (GPU0 z∈`[0,n0)`; GPU1 z∈`[depth,nz1)`); cut-halos
  are not stream destinations
- Native `(nz,nx,Q)` path A; never Gate1 `[:,1:-1,:]` on multitenet
- NumPy D0 1-step owned-gather bit-identical (PR #21) — CUDA D1 gate cleared

### CUDA D1 (PR #23 — merged)

- Design: [`docs/planning/D2D-phase-d1-cuda-dual-tenet.md`](../../docs/planning/D2D-phase-d1-cuda-dual-tenet.md)
- Driver: `step_two_way_cuda_cold` — D2H → plant+collide → H2D → exchange → D2H → stream → H2D
- Host plant/collide/stream; device exchange only; Ctx push (PR #20); no IdpyMemory edits
- idHome CUDA cold 1-step + N=2 soft owned-gather bit-identical (FYI log on idHome)

## Phase E — dual-tenet science gate (**merged** PR #24)

Tiny buoyant NSF vs monolith on cold dual-tenet path.

- Design: [`docs/planning/D2D-phase-e-science-gate.md`](../../docs/planning/D2D-phase-e-science-gate.md)
- Harness: `multitenet/test_science_two_way.py`
- Hard OVERALL_GATE: N=100 owned-gather bit-identical + NSF INTERIOR probe
  (`plant_sample_walls` → `hydro_from_lattice` → `slice(3,-3)`) match-monolith
  ≤1e-12 + buoyant `run_smoke` residual match-monolith
- Soft N=200 informational; CUDA `<2` → SKIP
- Frozen E0: nx=32 nz=64 z_cut=32 depth=3 seed=42 nu=0.10 g=1.2e-3
- idHome post-merge CUDA smoke: prior session log; re-verify on next idHome touch

## Half-A / Fig.4 multi-GPU (Loom PASS-WITH-NITS folded)

Cold D1 is not a Half-A campaign vehicle (~**12.12 GB** host traffic/step =
2×D2H+2×H2D of full `f` × ~226k steps). Staged ladder:

- Design: [`docs/planning/D2D-half-a-mgpu-fig4.md`](../../docs/planning/D2D-half-a-mgpu-fig4.md)
  (Loom defaults folded into §7)
- **F0** mid-size cold science ramp **256×512** — **in PR #25** @ `72f384d`
  (`multitenet/test_science_mid_two_way.py`; propose ≠ merge)
  Frozen: z_cut=256 depth=3 seed=42 nu=0.10 g=1.2e-3 tu/td=0.95/1.05
  **width=8.0 amp=2.0 k=[1,4]** (E0/SMOKE mid; not Table I-A)
- **F1-short** host NumPy dual-tenet Half-A bit-id → **F1-campaign** banded NSF (Table I-A)
- **F2** device-warm multi-GPU after Forge lands collide/stream kernels; warm ≠ OVERALL_GATE default
- No Full-A; no Guo dual-tenet fork; no IdpyMemory
