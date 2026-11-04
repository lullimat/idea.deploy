"""Phase E science gate: tiny buoyant dual-tenet NSF vs monolith.

Run from ``collabs/Rayleigh``::

    PYTHONPATH=../..:. python -m multitenet.test_science_two_way

Hard OVERALL_GATE = N=100 owned-gather bit-identical + NSF INTERIOR probe
match-monolith (abs <= 1e-12) + buoyant residual match-monolith.
Soft N=200 does not own the gate. CUDA with <2 devices → SKIP (not FAIL).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_RAYLEIGH = _HERE.parent
if str(_RAYLEIGH) not in sys.path:
    sys.path.insert(0, str(_RAYLEIGH))

from rt_lbm.config import RTConfig
from rt_lbm.diagnostics import nsf_fields, volume_means
from rt_lbm.equilibrium import moments
from rt_lbm.initial import initial_populations
from rt_lbm.step import (
    hydro_from_lattice,
    plant_sample_walls,
    prepare_boundaries,
    step as gate1_step,
)

from multitenet.bind import (
    MIN_DEPTH,
    bind_two_way_z_split,
    split_global_to_subdomains,
    transpose_gate1_to_multitenet,
    transpose_multitenet_to_gate1,
)
from multitenet.step_two_way import (
    CudaTenetCtx,
    gather_owned,
    split_bounds_for_two_way,
    step_two_way_cuda_cold,
    step_two_way_numpy,
)

# Scagliarini Appendix B: three nodes in from mesh edge (run_smoke).
INTERIOR = slice(3, -3)

# Frozen E0 geometry (Loom soft nit: named).
E0_NX = 32
E0_NZ = 64
E0_Z_CUT = 32
E0_DEPTH = MIN_DEPTH  # 3
E0_SEED = 42
N_HARD = 100
N_SOFT = 200
MATCH_ATOL = 1e-12
# Probe times for np.gradient residuals ending at hard N.
SAMPLE_HARD = (80, 90, 100)
SAMPLE_SOFT = (160, 180, 200)


def e0_cfg() -> RTConfig:
    """Frozen tiny buoyant scalars (SMOKE ballpark, smaller box)."""
    return RTConfig(
        name="phase-e-e0",
        nx=E0_NX,
        nz=E0_NZ,
        tu=0.95,
        td=1.05,
        nu=0.10,
        g=1.2e-3,
        nsteps=0,
        sample_every=1,
        interface_width=4.0,
        perturbation_amp=1.0,
        kmin=1,
        kmax=2,
        seed=E0_SEED,
    )


def _mean_interior(field: np.ndarray) -> float:
    return float(np.mean(field[:, INTERIOR]))


def probe_nsf(f_gate1: np.ndarray, cfg: RTConfig, bounds: dict, step_index: int) -> dict:
    """Locked run_smoke.sample_state contract (plant walls → hydro → INTERIOR)."""
    f = f_gate1.copy()
    plant_sample_walls(f, bounds)
    _rho, ux, uz, temperature = moments(f)
    ux_h, uz_h, t_h = hydro_from_lattice(ux, uz, temperature, cfg.g)
    lat = nsf_fields(ux_h, uz_h, t_h, cfg.nu, cfg.g, cfg.tm, kind="lattice")
    row = {
        "t": float(step_index),
        "half_mean_u2": 0.5 * _mean_interior(ux_h ** 2 + uz_h ** 2),
        "eps_nu": _mean_interior(lat["eps_nu_density"]),
        "buoyancy_energy": _mean_interior(lat["buoyancy_energy_density"]),
        "half_mean_w2": 0.5 * _mean_interior(lat["omega_sq"]),
        "eps_omega": _mean_interior(lat["eps_omega_density"]),
        "buoyancy_enstrophy": _mean_interior(lat["buoyancy_enstrophy_density"]),
    }
    vol = volume_means({k: v[:, INTERIOR] for k, v in lat.items()})
    for k, v in vol.items():
        row["vol_%s" % k] = float(v)
    return row


def _attach_residuals(rows: List[dict]) -> None:
    """Buoyant run_smoke._residuals normalization (lattice energy/enstrophy)."""
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    half_u = np.array([r["half_mean_u2"] for r in rows], dtype=np.float64)
    half_w = np.array([r["half_mean_w2"] for r in rows], dtype=np.float64)
    ddt_u = np.gradient(half_u, t)
    ddt_w = np.gradient(half_w, t)
    for i, row in enumerate(rows):
        row["ddt_half_u2"] = float(ddt_u[i])
        row["ddt_half_w2"] = float(ddt_w[i])
        rhs_e = -row["eps_nu"] + row["buoyancy_energy"]
        rhs_w = -row["eps_omega"] + row["buoyancy_enstrophy"]
        row["residual_energy"] = abs(row["ddt_half_u2"] - rhs_e) / max(
            abs(row["buoyancy_energy"]), 1e-30
        )
        row["residual_enstrophy"] = abs(row["ddt_half_w2"] - rhs_w) / max(
            abs(row["buoyancy_enstrophy"]), 1e-30
        )


def _match_rows(a: dict, b: dict, keys: List[str], label: str) -> None:
    for k in keys:
        da = float(a[k])
        db = float(b[k])
        if abs(da - db) > MATCH_ATOL:
            raise AssertionError(
                "%s key %s dual=%r mono=%r |diff|=%g > %g"
                % (label, k, da, db, abs(da - db), MATCH_ATOL)
            )


def _compare_keys(rows_a: List[dict], rows_b: List[dict]) -> List[str]:
    skip = {"t"}
    keys = sorted(k for k in rows_a[0] if k not in skip and not k.startswith("_"))
    return keys


def _run_numpy_pair(
    n_steps: int, sample_times: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[dict]]:
    cfg = e0_cfg()
    f0_g1, temperature_h, rho, _eta = initial_populations(cfg)
    bounds = prepare_boundaries(cfg, rho, temperature_h)
    bounds0, bounds1 = split_bounds_for_two_way(bounds)
    bind = bind_two_way_z_split(
        Nz=E0_NZ, nx=cfg.nx, Q=f0_g1.shape[-1], z_cut=E0_Z_CUT, depth=E0_DEPTH
    )

    f_mono = f0_g1.copy()
    f_mt = transpose_gate1_to_multitenet(f0_g1.copy())
    h0, h1 = split_global_to_subdomains(f_mt, bind)
    # mutable subdomain buffers
    s0 = h0.copy()
    s1 = h1.copy()

    mono_rows: List[dict] = []
    dual_rows: List[dict] = []
    sample_set = set(sample_times)

    for n in range(1, n_steps + 1):
        f_mono = gate1_step(f_mono, cfg, bounds)
        step_two_way_numpy(s0, s1, bind, cfg, bounds0, bounds1)
        if n in sample_set:
            owned = gather_owned(s0, s1, bind)
            dual_g1 = transpose_multitenet_to_gate1(owned)
            mono_rows.append(probe_nsf(f_mono, cfg, bounds, n))
            dual_rows.append(probe_nsf(dual_g1, cfg, bounds, n))

    owned_final = gather_owned(s0, s1, bind)
    mono_mt = transpose_gate1_to_multitenet(f_mono)
    _attach_residuals(mono_rows)
    _attach_residuals(dual_rows)
    return owned_final, mono_mt, dual_rows, mono_rows


def test_numpy_hard_n100():
    print("[RUN ] Phase E NumPy hard N=%d owned gather + NSF + residuals" % N_HARD)
    owned, mono_mt, dual_rows, mono_rows = _run_numpy_pair(N_HARD, SAMPLE_HARD)
    if not np.array_equal(owned, mono_mt):
        raise AssertionError(
            "NumPy N=%d owned != monolith (max|diff|=%g)"
            % (N_HARD, np.max(np.abs(owned - mono_mt)))
        )
    keys = _compare_keys(dual_rows, mono_rows)
    for dr, mr in zip(dual_rows, mono_rows):
        _match_rows(dr, mr, keys, "NumPy NSF@t=%s" % dr["t"])
    # residual match-monolith at last sample
    _match_rows(
        dual_rows[-1],
        mono_rows[-1],
        ["residual_energy", "residual_enstrophy"],
        "NumPy residual@N=%d" % N_HARD,
    )
    print(
        "[PASS] Phase E NumPy hard N=%d (f bit-identical; NSF+residual match ≤%g)"
        % (N_HARD, MATCH_ATOL)
    )
    print(
        "[INFO] mono residual_energy=%.3e residual_enstrophy=%.3e"
        % (mono_rows[-1]["residual_energy"], mono_rows[-1]["residual_enstrophy"])
    )


def test_numpy_soft_n200():
    print("[RUN ] Phase E NumPy soft N=%d (does not own OVERALL_GATE)" % N_SOFT)
    try:
        owned, mono_mt, dual_rows, mono_rows = _run_numpy_pair(N_SOFT, SAMPLE_SOFT)
        if not np.array_equal(owned, mono_mt):
            raise AssertionError(
                "soft owned mismatch max|diff|=%g"
                % np.max(np.abs(owned - mono_mt))
            )
        keys = _compare_keys(dual_rows, mono_rows)
        for dr, mr in zip(dual_rows, mono_rows):
            _match_rows(dr, mr, keys, "soft NSF@t=%s" % dr["t"])
        _match_rows(
            dual_rows[-1],
            mono_rows[-1],
            ["residual_energy", "residual_enstrophy"],
            "soft residual",
        )
        print("[PASS] Phase E NumPy soft N=%d" % N_SOFT)
    except Exception as e:
        print("[SOFT-FAIL] Phase E NumPy soft N=%d: %s" % (N_SOFT, e))
        print("[INFO] soft-fail does not fail OVERALL_GATE")


def _cuda_device_count() -> int:
    try:
        from idpy.IdpyCode import CUDA_T, idpy_langs_sys

        if not idpy_langs_sys.get(CUDA_T):
            return 0
        import pycuda.driver as drv

        drv.init()
        return int(drv.Device.count())
    except Exception:
        return 0


def _run_cuda_pair(
    n_steps: int, sample_times: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[dict]]:
    from idpy.IdpyCode import CUDA_T, GetTenet, IdpyMemory

    cfg = e0_cfg()
    f0_g1, temperature_h, rho, _eta = initial_populations(cfg)
    bounds = prepare_boundaries(cfg, rho, temperature_h)
    bounds0, bounds1 = split_bounds_for_two_way(bounds)
    bind = bind_two_way_z_split(
        Nz=E0_NZ, nx=cfg.nx, Q=f0_g1.shape[-1], z_cut=E0_Z_CUT, depth=E0_DEPTH
    )

    f_mono = f0_g1.copy()
    f_mt = transpose_gate1_to_multitenet(f0_g1.copy())
    h0, h1 = split_global_to_subdomains(f_mt, bind)

    tenet0 = tenet1 = None
    mono_rows: List[dict] = []
    dual_rows: List[dict] = []
    sample_set = set(sample_times)
    try:
        tenet0 = GetTenet({"lang": CUDA_T, "device": 0})
        tenet1 = GetTenet({"lang": CUDA_T, "device": 1})
        n0 = int(np.prod(bind.shape0))
        n1 = int(np.prod(bind.shape1))
        with CudaTenetCtx(tenet0):
            f0 = IdpyMemory.Zeros(shape=(n0,), dtype=np.float64, tenet=tenet0)
            staging_a = IdpyMemory.PinnedHost(
                shape=(bind.n_face,), dtype=np.float64, tenet=tenet0
            )
        with CudaTenetCtx(tenet1):
            f1 = IdpyMemory.Zeros(shape=(n1,), dtype=np.float64, tenet=tenet1)
            staging_b = IdpyMemory.PinnedHost(
                shape=(bind.n_face,), dtype=np.float64, tenet=tenet1
            )
        with CudaTenetCtx(tenet0):
            f0.H2D(h0.reshape(-1))
        with CudaTenetCtx(tenet1):
            f1.H2D(h1.reshape(-1))

        for n in range(1, n_steps + 1):
            f_mono = gate1_step(f_mono, cfg, bounds)
            step_two_way_cuda_cold(
                f0, f1, bind, cfg, bounds0, bounds1, staging_a, staging_b
            )
            if n in sample_set:
                with CudaTenetCtx(tenet0):
                    got0 = np.asarray(f0.D2H()).reshape(bind.shape0)
                with CudaTenetCtx(tenet1):
                    got1 = np.asarray(f1.D2H()).reshape(bind.shape1)
                owned = gather_owned(got0, got1, bind)
                dual_g1 = transpose_multitenet_to_gate1(owned)
                mono_rows.append(probe_nsf(f_mono, cfg, bounds, n))
                dual_rows.append(probe_nsf(dual_g1, cfg, bounds, n))

        with CudaTenetCtx(tenet0):
            got0 = np.asarray(f0.D2H()).reshape(bind.shape0)
        with CudaTenetCtx(tenet1):
            got1 = np.asarray(f1.D2H()).reshape(bind.shape1)
        owned_final = gather_owned(got0, got1, bind)
        mono_mt = transpose_gate1_to_multitenet(f_mono)
        _attach_residuals(mono_rows)
        _attach_residuals(dual_rows)
        return owned_final, mono_mt, dual_rows, mono_rows
    finally:
        for t in (tenet0, tenet1):
            if t is None:
                continue
            try:
                with CudaTenetCtx(t):
                    t.mem_pool.free_held()
                    t.mem_pool.stop_holding()
                t.cuda_context.detach()
            except Exception as e:
                print("[WARN] end_tenet: %s" % e)


def test_cuda_hard_n100():
    print("[RUN ] Phase E CUDA hard N=%d owned gather + NSF + residuals" % N_HARD)
    n_dev = _cuda_device_count()
    if n_dev < 2:
        print(
            "[SKIP] Phase E CUDA hard: need >=2 CUDA devices, found %d (not FAIL)"
            % n_dev
        )
        return
    owned, mono_mt, dual_rows, mono_rows = _run_cuda_pair(N_HARD, SAMPLE_HARD)
    if not np.array_equal(owned, mono_mt):
        raise AssertionError(
            "CUDA N=%d owned != monolith (max|diff|=%g)"
            % (N_HARD, np.max(np.abs(owned - mono_mt)))
        )
    keys = _compare_keys(dual_rows, mono_rows)
    for dr, mr in zip(dual_rows, mono_rows):
        _match_rows(dr, mr, keys, "CUDA NSF@t=%s" % dr["t"])
    _match_rows(
        dual_rows[-1],
        mono_rows[-1],
        ["residual_energy", "residual_enstrophy"],
        "CUDA residual@N=%d" % N_HARD,
    )
    print(
        "[PASS] Phase E CUDA hard N=%d (f bit-identical; NSF+residual match ≤%g)"
        % (N_HARD, MATCH_ATOL)
    )


def test_cuda_soft_n200():
    print("[RUN ] Phase E CUDA soft N=%d (does not own OVERALL_GATE)" % N_SOFT)
    n_dev = _cuda_device_count()
    if n_dev < 2:
        print(
            "[SKIP] Phase E CUDA soft: need >=2 CUDA devices, found %d (not FAIL)"
            % n_dev
        )
        return
    try:
        owned, mono_mt, dual_rows, mono_rows = _run_cuda_pair(N_SOFT, SAMPLE_SOFT)
        if not np.array_equal(owned, mono_mt):
            raise AssertionError(
                "soft owned mismatch max|diff|=%g"
                % np.max(np.abs(owned - mono_mt))
            )
        keys = _compare_keys(dual_rows, mono_rows)
        for dr, mr in zip(dual_rows, mono_rows):
            _match_rows(dr, mr, keys, "CUDA soft NSF@t=%s" % dr["t"])
        _match_rows(
            dual_rows[-1],
            mono_rows[-1],
            ["residual_energy", "residual_enstrophy"],
            "CUDA soft residual",
        )
        print("[PASS] Phase E CUDA soft N=%d" % N_SOFT)
    except Exception as e:
        print("[SOFT-FAIL] Phase E CUDA soft N=%d: %s" % (N_SOFT, e))
        print("[INFO] soft-fail does not fail OVERALL_GATE")


def main():
    print("=== Phase E dual-tenet science gate (E0) ===")
    print(
        "cfg: nx=%d nz=%d z_cut=%d depth=%d seed=%d nu=0.10 g=1.2e-3"
        % (E0_NX, E0_NZ, E0_Z_CUT, E0_DEPTH, E0_SEED)
    )
    test_numpy_hard_n100()
    test_numpy_soft_n200()
    test_cuda_hard_n100()
    test_cuda_soft_n200()
    print("=== Phase E E0 done (hard owns OVERALL_GATE; soft informational) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
