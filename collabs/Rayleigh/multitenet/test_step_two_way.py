"""Phase D NumPy D0 acceptance: dual-tenet step vs monolith Gate1.

Run from ``collabs/Rayleigh``::

    PYTHONPATH=../..:. python -m multitenet.test_step_two_way
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_RAYLEIGH = _HERE.parent
if str(_RAYLEIGH) not in sys.path:
    sys.path.insert(0, str(_RAYLEIGH))

from rt_lbm.config import RTConfig
from rt_lbm.initial import initial_populations
from rt_lbm.step import prepare_boundaries, step as gate1_step
from rt_lbm.stencil import CX  # noqa: F401 — stencil import side-effect OK

from multitenet.bind import (
    MIN_DEPTH,
    bind_two_way_z_split,
    split_global_to_subdomains,
    transpose_gate1_to_multitenet,
    transpose_multitenet_to_gate1,
)
from multitenet.step_two_way import (
    gather_owned,
    split_bounds_for_two_way,
    step_two_way_numpy,
)


def _tiny_cfg(nx: int = 16, nz: int = 32, seed: int = 3) -> RTConfig:
    return RTConfig(
        name="phase-d-d0",
        nx=nx,
        nz=nz,
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
        seed=seed,
    )


def _run_compare(n_steps: int, *, nz: int, z_cut: int, depth: int, seed: int = 3):
    cfg = _tiny_cfg(nx=16, nz=nz, seed=seed)
    f_g1, temperature_h, rho, _eta = initial_populations(cfg)
    bounds = prepare_boundaries(cfg, rho, temperature_h)
    bounds0, bounds1 = split_bounds_for_two_way(bounds)

    bind = bind_two_way_z_split(Nz=nz, nx=cfg.nx, Q=f_g1.shape[-1], z_cut=z_cut, depth=depth)

    # Monolith reference in Gate1 layout
    f_mono = f_g1.copy()
    for _ in range(n_steps):
        f_mono = gate1_step(f_mono, cfg, bounds)
    mono_mt = transpose_gate1_to_multitenet(f_mono)

    # Dual-tenet
    f_mt0 = transpose_gate1_to_multitenet(f_g1.copy())
    f0, f1 = split_global_to_subdomains(f_mt0, bind)
    for _ in range(n_steps):
        step_two_way_numpy(f0, f1, bind, cfg, bounds0, bounds1)
    owned = gather_owned(f0, f1, bind)

    if not np.array_equal(owned, mono_mt):
        diff = np.max(np.abs(owned.astype(np.float64) - mono_mt.astype(np.float64)))
        raise AssertionError(
            "owned gather != monolith after %d steps (max|diff|=%g) "
            "nz=%d z_cut=%d depth=%d" % (n_steps, diff, nz, z_cut, depth)
        )
    # Halo asserts: after a step, cut-halos should match neighbor post-collide
    # faces from the *previous* exchange — already enforced inside step.
    # Extra: halo slots exist and are finite
    assert np.isfinite(f0[bind.n0 :]).all()
    assert np.isfinite(f1[: bind.depth]).all()
    return bind


def test_numpy_1_step_owned_gather():
    print("[RUN ] NumPy 1-step owned gather vs monolith")
    _run_compare(1, nz=32, z_cut=16, depth=MIN_DEPTH)
    print("[PASS] NumPy 1-step owned gather vs monolith")


def test_numpy_n_step_owned_gather():
    print("[RUN ] NumPy N-step owned gather (N=2,8)")
    for n in (2, 8):
        _run_compare(n, nz=32, z_cut=16, depth=MIN_DEPTH, seed=3 + n)
        print("[INFO] N=%d PASS" % n)
    print("[PASS] NumPy N-step owned gather (N=2,8)")


def test_depth_eq_z_cut_edge():
    print("[RUN ] tiny-domain depth==z_cut owned gather")
    # n0 == depth == 3; GPU0 collide z∈[1,3)
    _run_compare(1, nz=24, z_cut=3, depth=3, seed=11)
    _run_compare(2, nz=24, z_cut=3, depth=3, seed=11)
    # depth == Nz - z_cut
    _run_compare(1, nz=24, z_cut=21, depth=3, seed=12)
    print("[PASS] tiny-domain depth==z_cut owned gather")


def test_call_order_doc():
    print("[RUN ] call-order docstring lock")
    import inspect
    from multitenet import step_two_way as st

    doc = inspect.getdoc(st.step_two_way_numpy) or ""
    assert "plant" in doc and "collide" in doc and "exchange" in doc and "stream" in doc
    src = inspect.getsource(st.step_two_way_numpy)
    # Order in source: plant, collide, exchange, stream
    i_plant = src.index("plant_local_wall")
    i_coll = src.index("collide_owned_interior")
    i_ex = src.index("exchange_faces_numpy")
    i_st = src.index("stream_write_owned")
    assert i_plant < i_coll < i_ex < i_st
    print("[PASS] call-order docstring lock")


def test_never_gate1_slice_helper():
    print("[RUN ] helpers reject Gate1-style misuse docs")
    import inspect
    from multitenet import step_two_way as st

    src = inspect.getsource(st)
    # Executable Gate1 monolith collide slice must not appear
    assert "[:, 1:-1, :]" not in src and "[:,1:-1,:]" not in src
    assert "collide_interior(" not in src
    print("[PASS] helpers reject Gate1-style misuse docs")


def main():
    print("=== Phase D NumPy D0 dual-tenet step acceptance ===")
    test_call_order_doc()
    test_never_gate1_slice_helper()
    test_numpy_1_step_owned_gather()
    test_numpy_n_step_owned_gather()
    test_depth_eq_z_cut_edge()
    print("=== Phase D D0 ALL PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
