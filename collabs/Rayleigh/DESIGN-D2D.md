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

## Phase C — applied (PR pending merge)

Multi-tenet `exchange_faces` (Rayleigh / LBM Option A): Loom **PASS-WITH-NITS**
RECEIVED (2026-11-05 Asia/Hong_Kong); **Matteo apply greenlight used** — applied
via Grok Build on Gödel; **propose ≠ merge**.

- Design: [`docs/planning/D2D-phase-c-exchange-faces.md`](../../docs/planning/D2D-phase-c-exchange-faces.md)
- Grok Build brief: [`docs/planning/D2D-phase-c-grok-build.md`](../../docs/planning/D2D-phase-c-grok-build.md)
- Code: [`multitenet/`](./multitenet/) — `face_elem_range` (local copy),
  `bind_two_way_z_split`, `exchange_faces` (two sync `D2DSub`), tests
- `IdpyMemory` contracts **untouched** (ranges-only); no B2 / MemcpyPeer /
  aggregate stream API / silent macro face ship
