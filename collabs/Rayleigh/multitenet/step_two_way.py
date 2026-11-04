"""Phase D NumPy D0 — dual-tenet LBM step (plant → collide → exchange → stream).

Layout: ``(nz_local, nx, Q)`` C-order (path A). Stream is **write-owned /
read-halo** (Loom must-fix). Cut-halos are not collided and are not stream
destinations. Never call Gate1 collide_interior unchanged on multitenet buffers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from rt_lbm.config import RTConfig
from rt_lbm.equilibrium import equilibrium, moments
from rt_lbm.stencil import CX, CZ, D

from .bind import TwoWayZSplitBind
from .exchange import exchange_faces_numpy


def collide_owned_interior(
    f: np.ndarray, z_lo: int, z_hi: int, cfg: RTConfig
) -> None:
    """BGK on local z ∈ ``[z_lo, z_hi)`` in place. Layout ``(nz, nx, Q)``."""
    if z_lo < 0 or z_hi > f.shape[0] or z_lo >= z_hi:
        raise ValueError(
            "collide_owned_interior: bad range [%d, %d) for nz=%d"
            % (z_lo, z_hi, f.shape[0])
        )
    interior = f[z_lo:z_hi, :, :]
    rho, ux, uz, temperature = moments(interior)
    tau = cfg.tau_lb
    g2 = cfg.g * cfg.g
    ux_eq = ux
    uz_eq = uz - tau * cfg.g
    t_eq = temperature + tau * (1.0 - tau) * g2 / D
    feq = equilibrium(rho, ux_eq, uz_eq, t_eq)
    omega = 1.0 / tau
    f[z_lo:z_hi, :, :] = (1.0 - omega) * interior + omega * feq


def stream_write_owned(
    f: np.ndarray,
    z_write_lo: int,
    z_write_hi: int,
    *,
    ghost_bot=None,
    ghost_top=None,
) -> None:
    """Pull-stream into owned rows only (write-owned / read-halo).

    Reads may use cut-halos already in ``f`` and optional local wall ghosts.
    Does **not** write cut-halo slots. Layout ``(nz, nx, Q)``; ghosts are
    Gate1-shaped ``(nx, Q)`` rows (list of 3).
    """
    nz, nx, nq = f.shape
    if z_write_lo < 0 or z_write_hi > nz or z_write_lo >= z_write_hi:
        raise ValueError(
            "stream_write_owned: bad write range [%d, %d) for nz=%d"
            % (z_write_lo, z_write_hi, nz)
        )
    owned = np.empty((z_write_hi - z_write_lo, nx, nq), dtype=f.dtype)
    cx = CX.astype(int)
    cz = CZ.astype(int)
    for q, (vx, vz) in enumerate(zip(cx, cz)):
        for iz, zdst in enumerate(range(z_write_lo, z_write_hi)):
            z_src = int(zdst - vz)
            if 0 <= z_src < nz:
                row = f[z_src, :, q]
            elif z_src < 0:
                if ghost_bot is None:
                    raise ValueError(
                        "stream_write_owned: z_src=%d < 0 needs ghost_bot" % z_src
                    )
                k = -z_src - 1
                row = ghost_bot[k][:, q]
            else:
                if ghost_top is None:
                    raise ValueError(
                        "stream_write_owned: z_src=%d >= nz=%d needs ghost_top"
                        % (z_src, nz)
                    )
                kk = z_src - nz
                row = ghost_top[kk][:, q]
            owned[iz, :, q] = np.roll(row, int(vx), axis=0)
    f[z_write_lo:z_write_hi, :, :] = owned


def plant_local_wall(f: np.ndarray, *, wall_row: np.ndarray, z_index: int) -> None:
    """Plant one Dirichlet wall row into ``f`` at local ``z_index``."""
    if wall_row.shape != (f.shape[1], f.shape[2]):
        raise ValueError(
            "plant_local_wall: wall_row shape %s != (nx,Q)=(%d,%d)"
            % (wall_row.shape, f.shape[1], f.shape[2])
        )
    f[z_index, :, :] = wall_row


def split_bounds_for_two_way(bounds: Dict) -> Tuple[Dict, Dict]:
    """Split Gate1 ``prepare_boundaries`` dict into per-tenet bounds0/bounds1."""
    bounds0 = {
        "wall_bot": bounds["wall_bot"],
        "pre_bot": bounds["pre_bot"],
        "ghost_bot": bounds["ghost_bot"],
    }
    bounds1 = {
        "wall_top": bounds["wall_top"],
        "pre_top": bounds["pre_top"],
        "ghost_top": bounds["ghost_top"],
    }
    return bounds0, bounds1


def gather_owned(
    f0: np.ndarray, f1: np.ndarray, bind: TwoWayZSplitBind
) -> np.ndarray:
    """Concatenate owned rows → global ``(Nz, nx, Q)`` (no halo slots)."""
    out = np.empty((bind.Nz, bind.nx, bind.Q), dtype=f0.dtype)
    out[: bind.n0] = f0[: bind.n0]
    out[bind.z_cut :] = f1[bind.depth :]
    return out


def step_two_way_numpy(
    f0: np.ndarray,
    f1: np.ndarray,
    bind: TwoWayZSplitBind,
    cfg: RTConfig,
    bounds0: Dict,
    bounds1: Dict,
    *,
    debug_assert: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """One dual-tenet step: plant → collide → exchange_faces → stream.

    Mutates ``f0``/``f1`` in place; returns ``(f0, f1)`` for chaining.
    Call-site order locked. Populations only; cut-halos not collided.

    ``debug_assert`` (default True for NumPy D0 tests): halo snapshot checks.
    Hot path / CUDA callers should pass ``debug_assert=False``.
    """
    if f0.shape != bind.shape0 or f1.shape != bind.shape1:
        raise ValueError(
            "step_two_way_numpy: shape mismatch f0=%s want %s; f1=%s want %s"
            % (f0.shape, bind.shape0, f1.shape, bind.shape1)
        )

    # --- plant (local walls into f) ---
    plant_local_wall(f0, wall_row=bounds0["wall_bot"], z_index=0)
    plant_local_wall(f1, wall_row=bounds1["wall_top"], z_index=bind.nz1 - 1)

    halo0_before = f0[bind.n0 :].copy() if debug_assert else None
    halo1_before = f1[: bind.depth].copy() if debug_assert else None

    # --- collide owned interiors only ---
    collide_owned_interior(f0, 1, bind.n0, cfg)
    collide_owned_interior(f1, bind.depth, bind.nz1 - 1, cfg)

    if debug_assert:
        if not np.array_equal(f0[bind.n0 :], halo0_before):
            raise RuntimeError("step_two_way_numpy: collide wrote GPU0 cut-halo")
        if not np.array_equal(f1[: bind.depth], halo1_before):
            raise RuntimeError("step_two_way_numpy: collide wrote GPU1 cut-halo")

    # --- exchange cut faces ---
    exchange_faces_numpy(f0, f1, bind)

    halo0_ex = f0[bind.n0 :].copy() if debug_assert else None
    halo1_ex = f1[: bind.depth].copy() if debug_assert else None

    # --- stream write-owned / read-halo ---
    stream_write_owned(
        f0, 0, bind.n0, ghost_bot=bounds0["ghost_bot"], ghost_top=None
    )
    stream_write_owned(
        f1, bind.depth, bind.nz1, ghost_bot=None, ghost_top=bounds1["ghost_top"]
    )

    if debug_assert:
        if not np.array_equal(f0[bind.n0 :], halo0_ex):
            raise RuntimeError("step_two_way_numpy: stream wrote GPU0 cut-halo")
        if not np.array_equal(f1[: bind.depth], halo1_ex):
            raise RuntimeError("step_two_way_numpy: stream wrote GPU1 cut-halo")

    return f0, f1


class CudaTenetCtx:
    """PR #20 Ctx: push owning tenet context for multi-GPU alloc/H2D/D2H."""

    def __init__(self, tenet):
        self.tenet = tenet

    def __enter__(self):
        import pycuda.driver as cuda  # noqa: F401 — used in __exit__

        self.tenet.cuda_context.push()
        return self

    def __exit__(self, *a):
        import pycuda.driver as cuda

        cuda.Context.pop()
        return False


def _cuda_h2d(ary, host) -> None:
    with CudaTenetCtx(ary.tenet):
        ary.H2D(np.ascontiguousarray(host).reshape(-1))


def _cuda_d2h(ary, shape) -> np.ndarray:
    with CudaTenetCtx(ary.tenet):
        return np.asarray(ary.D2H()).reshape(shape)


def step_two_way_cuda_cold(
    f0_dev,
    f1_dev,
    bind: TwoWayZSplitBind,
    cfg: RTConfig,
    bounds0: Dict,
    bounds1: Dict,
    staging_a,
    staging_b,
    *,
    mode: str = "auto",
) -> Tuple[object, object]:
    """CUDA D1 cold round-trip (OVERALL_GATE default).

    Locked barriers every step (Loom must-fix)::

        D2H → plant+collide (host) → H2D → exchange_faces_from_bind
            → D2H → stream_write_owned (host) → H2D

    Plant/collide/stream stay host-only; exchange is device-only.
    """
    from .exchange import exchange_faces_from_bind

    # 1) D2H full buffers
    h0 = _cuda_d2h(f0_dev, bind.shape0)
    h1 = _cuda_d2h(f1_dev, bind.shape1)

    # 2) plant + collide on host (no hot-path halo copies)
    plant_local_wall(h0, wall_row=bounds0["wall_bot"], z_index=0)
    plant_local_wall(h1, wall_row=bounds1["wall_top"], z_index=bind.nz1 - 1)
    collide_owned_interior(h0, 1, bind.n0, cfg)
    collide_owned_interior(h1, bind.depth, bind.nz1 - 1, cfg)

    # 3) H2D before exchange (barrier)
    _cuda_h2d(f0_dev, h0)
    _cuda_h2d(f1_dev, h1)

    # 4) device exchange C0
    exchange_faces_from_bind(
        f0_dev,
        f1_dev,
        bind,
        mode=mode,
        staging_a=staging_a,
        staging_b=staging_b,
    )

    # 5) D2H before stream (barrier; full buffers)
    h0 = _cuda_d2h(f0_dev, bind.shape0)
    h1 = _cuda_d2h(f1_dev, bind.shape1)

    # 6) stream write-owned on host
    stream_write_owned(
        h0, 0, bind.n0, ghost_bot=bounds0["ghost_bot"], ghost_top=None
    )
    stream_write_owned(
        h1, bind.depth, bind.nz1, ghost_bot=None, ghost_top=bounds1["ghost_top"]
    )

    # 7) H2D post-stream
    _cuda_h2d(f0_dev, h0)
    _cuda_h2d(f1_dev, h1)

    return f0_dev, f1_dev
