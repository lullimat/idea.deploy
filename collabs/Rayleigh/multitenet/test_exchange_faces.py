"""Phase C acceptance: Rayleigh ``exchange_faces`` (Option A).

Canonical invocation (from idea.deploy root)::

    cd collabs/Rayleigh
    PYTHONPATH=../..:../../py-env/idpy-env/lib/python3.11/site-packages \
      ../../py-env/idpy-env/bin/python -m multitenet.test_exchange_faces

If ``idpy`` is already importable in the active env, ``PYTHONPATH=../..`` is enough::

    cd collabs/Rayleigh
    PYTHONPATH=../.. python -m multitenet.test_exchange_faces

CUDA 2-GPU smoke skips cleanly when <2 CUDA devices (Gödel). No Fig.4 NSF.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Allow ``python -m multitenet.test_exchange_faces`` from collabs/Rayleigh
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAYLEIGH = os.path.dirname(_HERE)
if _RAYLEIGH not in sys.path:
    sys.path.insert(0, _RAYLEIGH)

from multitenet import (  # noqa: E402
    MIN_DEPTH,
    assert_range_disjointness,
    bind_two_way_z_split,
    exchange_faces_from_bind,
    exchange_faces_numpy,
    face_elem_range,
    monolith_face_halo_reference,
    split_global_to_subdomains,
    transpose_gate1_to_multitenet,
    transpose_multitenet_to_gate1,
)


def _make_global(Nz=32, nx=16, Q=37, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        rng.standard_normal((Nz, nx, Q), dtype=np.float64)
    )


def test_face_elem_range_local_copy():
    print("[RUN ] face_elem_range Rayleigh-local copy")
    start, stop = face_elem_range(
        (8, 4, 3), axis=0, face_index=2, depth=3, order="C"
    )
    assert (start, stop) == (2 * 4 * 3, 5 * 4 * 3)
    try:
        face_elem_range((4, 8, 3), axis=1, face_index=2, depth=3, order="C")
        raise AssertionError("non-contig z-slab must raise")
    except ValueError as e:
        assert "non-contiguous" in str(e).lower(), str(e)
    # Must not import from test_d2d (__init__ shadows submodule name)
    import importlib
    import inspect

    fer_mod = importlib.import_module("multitenet.face_elem_range")
    src = open(inspect.getfile(fer_mod)).read()
    import_lines = [
        line for line in src.splitlines()
        if line.lstrip().startswith(("import ", "from "))
    ]
    assert not any("test_d2d" in line for line in import_lines), import_lines
    assert fer_mod.face_elem_range is face_elem_range
    print("[PASS] face_elem_range Rayleigh-local copy")


def test_bind_guards():
    print("[RUN ] bind guards depth<=z_cut and depth<=Nz-z_cut")
    # Happy path
    b = bind_two_way_z_split(Nz=32, nx=8, Q=37, z_cut=16, depth=3)
    assert b.n0 == 16 and b.n1 == 16
    assert b.nz0 == 19 and b.nz1 == 19
    assert b.n_face == 3 * 8 * 37
    assert_range_disjointness(b)

    try:
        bind_two_way_z_split(Nz=32, nx=8, Q=37, z_cut=2, depth=3)
        raise AssertionError("depth > z_cut must raise")
    except ValueError as e:
        assert "z_cut" in str(e), str(e)

    try:
        bind_two_way_z_split(Nz=32, nx=8, Q=37, z_cut=30, depth=3)
        raise AssertionError("depth > Nz-z_cut must raise")
    except ValueError as e:
        assert "Nz-z_cut" in str(e) or "Nz - z_cut" in str(e), str(e)

    try:
        bind_two_way_z_split(Nz=32, nx=8, Q=37, depth=2)
        raise AssertionError("depth < MIN_DEPTH must raise")
    except ValueError as e:
        assert "MIN_DEPTH" in str(e) or "depth" in str(e).lower(), str(e)

    print("[PASS] bind guards depth<=z_cut and depth<=Nz-z_cut")


def test_layout_transpose_bind_only():
    print("[RUN ] Gate1 (nx,nz,Q) transpose at multi-tenet bind only")
    nx, nz, Q = 12, 20, 37
    f_gate1 = np.arange(nx * nz * Q, dtype=np.float64).reshape(nx, nz, Q)
    f_mt = transpose_gate1_to_multitenet(f_gate1)
    assert f_mt.shape == (nz, nx, Q)
    assert f_mt.flags.c_contiguous
    # Round-trip
    back = transpose_multitenet_to_gate1(f_mt)
    assert np.array_equal(back, f_gate1)
    # Contiguous z-face on mt layout
    start, stop = face_elem_range(f_mt.shape, axis=0, face_index=5, depth=3)
    assert stop - start == 3 * nx * Q
    print("[PASS] Gate1 (nx,nz,Q) transpose at multi-tenet bind only")


def test_numpy_bit_identical_vs_monolith():
    print("[RUN ] NumPy 2-subdomain bit-identical vs monolith face copy")
    Nz, nx, Q = 40, 16, 37
    depth = MIN_DEPTH
    bind = bind_two_way_z_split(Nz=Nz, nx=nx, Q=Q, depth=depth)
    f_global = _make_global(Nz, nx, Q, seed=42)

    f0_ref, f1_ref = monolith_face_halo_reference(f_global, bind)
    f0, f1 = split_global_to_subdomains(f_global, bind)

    # Interiors before exchange must match ref interiors; halos still zero/empty
    assert np.array_equal(f0[: bind.n0], f0_ref[: bind.n0])
    assert np.array_equal(f1[bind.depth :], f1_ref[bind.depth :])

    interior0_before = f0[: bind.n0].copy()
    interior1_before = f1[bind.depth :].copy()

    exchange_faces_numpy(f0, f1, bind)

    # Bit-identical halos + interiors vs monolith reference
    assert np.array_equal(f0, f0_ref), "GPU0 buffer != monolith reference"
    assert np.array_equal(f1, f1_ref), "GPU1 buffer != monolith reference"

    # Interiors stable during exchange
    assert np.array_equal(f0[: bind.n0], interior0_before)
    assert np.array_equal(f1[bind.depth :], interior1_before)

    # Document cut-halos not collided / populations only / call order
    assert bind.cut_halos_collided is False
    assert bind.populations_only is True
    assert "exchange_faces" in bind.call_site_order
    assert bind.call_site_order == "plant→collide→exchange_faces→stream"

    print("[PASS] NumPy 2-subdomain bit-identical vs monolith face copy")


def test_inplace_timing_invariants():
    print("[RUN ] in-place timing invariants / non-overlap")
    bind = bind_two_way_z_split(Nz=48, nx=8, Q=37, z_cut=24, depth=3)
    assert_range_disjointness(bind)

    # Explicit empty intersection both ways
    s0, e0 = bind.src_range_0to1
    d0, f0 = bind.dst_range_1to0
    assert e0 <= d0 or f0 <= s0
    s1, e1 = bind.src_range_1to0
    d1, f1 = bind.dst_range_0to1
    assert e1 <= d1 or f1 <= s1

    f_global = _make_global(bind.Nz, bind.nx, bind.Q, seed=7)
    f0, f1 = split_global_to_subdomains(f_global, bind)
    # Poison halos, keep interiors
    f0[bind.n0 :] = -999.0
    f1[: bind.depth] = -999.0
    interior0 = f0[: bind.n0].copy()
    interior1 = f1[bind.depth :].copy()
    exchange_faces_numpy(f0, f1, bind)
    assert np.array_equal(f0[: bind.n0], interior0), "GPU0 interior mutated"
    assert np.array_equal(f1[bind.depth :], interior1), "GPU1 interior mutated"
    # Halos filled with neighbor faces (not poison)
    assert not np.any(f0[bind.n0 :] == -999.0)
    assert not np.any(f1[: bind.depth] == -999.0)
    print("[PASS] in-place timing invariants / non-overlap")


def test_ctypes_d2dsub_exchange_smoke():
    """Same-host two-buffer exchange via real D2DSub (CTypes) — plumbing smoke."""
    print("[RUN ] CTypes D2DSub exchange_faces smoke")
    try:
        from idpy.IdpyCode import (
            CTYPES_T,
            GetTenet,
            IdpyMemory,
            idpy_langs_sys,
        )
    except Exception as e:
        print("[SKIP] CTypes D2DSub exchange: idpy unavailable (%s)" % e)
        return
    if not idpy_langs_sys.get(CTYPES_T):
        print("[SKIP] CTypes D2DSub exchange: CTypes unavailable")
        return

    bind = bind_two_way_z_split(Nz=24, nx=8, Q=37, depth=3)
    f_global = _make_global(bind.Nz, bind.nx, bind.Q, seed=3)
    f0_h, f1_h = split_global_to_subdomains(f_global, bind)
    f0_ref, f1_ref = monolith_face_halo_reference(f_global, bind)

    tenet = GetTenet({"lang": CTYPES_T, "device": 0})
    n0 = int(np.prod(bind.shape0))
    n1 = int(np.prod(bind.shape1))
    f0 = IdpyMemory.Zeros(shape=(n0,), dtype=np.float64, tenet=tenet)
    f1 = IdpyMemory.Zeros(shape=(n1,), dtype=np.float64, tenet=tenet)
    f0.H2D(f0_h.reshape(-1))
    f1.H2D(f1_h.reshape(-1))

    staging_a, staging_b = (
        IdpyMemory.PinnedHost(shape=(bind.n_face,), dtype=np.float64, tenet=tenet),
        IdpyMemory.PinnedHost(shape=(bind.n_face,), dtype=np.float64, tenet=tenet),
    )
    ret = exchange_faces_from_bind(
        f0, f1, bind, mode="auto", staging_a=staging_a, staging_b=staging_b
    )
    assert ret is None, "C0 sync exchange_faces must return None"

    got0 = np.asarray(f0.D2H()).reshape(bind.shape0)
    got1 = np.asarray(f1.D2H()).reshape(bind.shape1)
    assert np.array_equal(got0, f0_ref)
    assert np.array_equal(got1, f1_ref)
    print("[PASS] CTypes D2DSub exchange_faces smoke")


def _cuda_device_count():
    try:
        from idpy.IdpyCode import CUDA_T, idpy_langs_sys

        if not idpy_langs_sys.get(CUDA_T):
            return 0
        import pycuda.driver as drv

        drv.init()
        return int(drv.Device.count())
    except Exception:
        try:
            import cupy

            return int(cupy.cuda.runtime.getDeviceCount())
        except Exception:
            return 0


def test_cuda_two_gpu_smoke():
    print("[RUN ] CUDA 2-GPU exchange_faces smoke")
    n_dev = _cuda_device_count()
    if n_dev < 2:
        print(
            "[SKIP] CUDA 2-GPU exchange_faces smoke: need >=2 CUDA devices, "
            "found %d (Gödel / single-GPU OK)" % n_dev
        )
        return

    import pycuda.driver as cuda
    from idpy.IdpyCode import CUDA_T, GetTenet, IdpyMemory

    class _Ctx:
        """Push owning tenet context for multi-GPU Zeros/H2D/D2H/PinnedHost.

        Same pattern as ID-home Phase B D2DSub smoke: GetTenet leaves each
        device context current only briefly; cross-device alloc without push
        hits cuMemcpyHtoD illegal memory access.
        """

        def __init__(self, tenet):
            self.tenet = tenet

        def __enter__(self):
            self.tenet.cuda_context.push()
            return self

        def __exit__(self, *a):
            cuda.Context.pop()
            return False

    def _zeros(n, dtype, tenet):
        with _Ctx(tenet):
            return IdpyMemory.Zeros(shape=(n,), dtype=dtype, tenet=tenet)

    def _h2d(ary, host):
        with _Ctx(ary.tenet):
            ary.H2D(host)

    def _d2h(ary):
        with _Ctx(ary.tenet):
            return np.asarray(ary.D2H())

    def _pinned(n, dtype, tenet):
        with _Ctx(tenet):
            return IdpyMemory.PinnedHost(shape=(n,), dtype=dtype, tenet=tenet)

    def _end_tenet(t):
        if t is None:
            return
        try:
            with _Ctx(t):
                t.mem_pool.free_held()
                t.mem_pool.stop_holding()
            t.cuda_context.detach()
        except Exception as e:
            print("[WARN] end_tenet: %s" % e)

    bind = bind_two_way_z_split(Nz=24, nx=8, Q=37, depth=3)
    f_global = _make_global(bind.Nz, bind.nx, bind.Q, seed=11)
    f0_h, f1_h = split_global_to_subdomains(f_global, bind)
    f0_ref, f1_ref = monolith_face_halo_reference(f_global, bind)

    tenet0 = tenet1 = None
    try:
        tenet0 = GetTenet({"lang": CUDA_T, "device": 0})
        tenet1 = GetTenet({"lang": CUDA_T, "device": 1})
        n0 = int(np.prod(bind.shape0))
        n1 = int(np.prod(bind.shape1))
        f0 = _zeros(n0, np.float64, tenet0)
        f1 = _zeros(n1, np.float64, tenet1)
        _h2d(f0, f0_h.reshape(-1))
        _h2d(f1, f1_h.reshape(-1))
        staging_a = _pinned(bind.n_face, np.float64, tenet0)
        staging_b = _pinned(bind.n_face, np.float64, tenet1)

        t0 = time.perf_counter()
        exchange_faces_from_bind(
            f0, f1, bind, mode="auto", staging_a=staging_a, staging_b=staging_b
        )
        dt = time.perf_counter() - t0
        bytes_moved = 2 * bind.n_face * 8  # both legs, f64
        gbps = (bytes_moved / dt) / 1e9 if dt > 0 else float("inf")
        print(
            "[INFO] CUDA exchange soft BW: %.2f GB/s for 2×%.2f KiB faces (informational)"
            % (gbps, bind.n_face * 8 / 1024.0)
        )

        # Peer refuse honesty on one leg
        try:
            IdpyMemory.D2DSub(
                f0,
                f1,
                src_start=bind.src_range_0to1[0],
                src_stop=bind.src_range_0to1[1],
                dst_start=bind.dst_range_0to1[0],
                mode="peer",
            )
            # If peer somehow works, OK; else must have raised
        except Exception as e:
            assert (
                "peer" in str(e).lower()
                or "Peer" in str(e)
                or "access" in str(e).lower()
            ), str(e)
            print("[INFO] peer refuse OK: %s" % e)

        got0 = _d2h(f0).reshape(bind.shape0)
        got1 = _d2h(f1).reshape(bind.shape1)
        assert np.array_equal(got0, f0_ref)
        assert np.array_equal(got1, f1_ref)
        print("[PASS] CUDA 2-GPU exchange_faces smoke")
    finally:
        _end_tenet(tenet0)
        _end_tenet(tenet1)


def test_no_aggregate_stream_api():
    print("[RUN ] no aggregate stream API on exchange_faces")
    import inspect
    from multitenet import exchange as ex

    sig = inspect.signature(ex.exchange_faces)
    params = set(sig.parameters)
    # Per-leg only
    assert "stream_src_a" in params and "stream_dst_a" in params
    assert "stream_src_b" in params and "stream_dst_b" in params
    assert "streams" not in params
    assert "stream" not in params
    print("[PASS] no aggregate stream API on exchange_faces")


def main():
    print("=== Phase C exchange_faces acceptance (Rayleigh Option A) ===")
    test_face_elem_range_local_copy()
    test_bind_guards()
    test_layout_transpose_bind_only()
    test_numpy_bit_identical_vs_monolith()
    test_inplace_timing_invariants()
    test_no_aggregate_stream_api()
    test_ctypes_d2dsub_exchange_smoke()
    test_cuda_two_gpu_smoke()
    print("=== Phase C acceptance done ===")


if __name__ == "__main__":
    try:
        main()
        print("ALL PASS (or SKIP where expected)")
        sys.exit(0)
    except Exception as e:
        print("[FAIL] %s" % e)
        raise
