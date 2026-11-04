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
) -> Tuple[np.ndarray, np.ndarray]:
    """One dual-tenet step: plant → collide → exchange_faces → stream.

    Mutates ``f0``/``f1`` in place; returns ``(f0, f1)`` for chaining.
    Call-site order locked. Populations only; cut-halos not collided.
    """
    if f0.shape != bind.shape0 or f1.shape != bind.shape1:
        raise ValueError(
            "step_two_way_numpy: shape mismatch f0=%s want %s; f1=%s want %s"
            % (f0.shape, bind.shape0, f1.shape, bind.shape1)
        )

    # --- plant (local walls into f) ---
    plant_local_wall(f0, wall_row=bounds0["wall_bot"], z_index=0)
    plant_local_wall(f1, wall_row=bounds1["wall_top"], z_index=bind.nz1 - 1)

    # Snapshot halos to assert collide does not write them
    halo0_before = f0[bind.n0 :].copy()
    halo1_before = f1[: bind.depth].copy()

    # --- collide owned interiors only ---
    # GPU0: z ∈ [1, n0); GPU1: z ∈ [depth, nz1-1)
    collide_owned_interior(f0, 1, bind.n0, cfg)
    collide_owned_interior(f1, bind.depth, bind.nz1 - 1, cfg)

    if not np.array_equal(f0[bind.n0 :], halo0_before):
        raise RuntimeError("step_two_way_numpy: collide wrote GPU0 cut-halo")
    if not np.array_equal(f1[: bind.depth], halo1_before):
        raise RuntimeError("step_two_way_numpy: collide wrote GPU1 cut-halo")

    # --- exchange cut faces ---
    exchange_faces_numpy(f0, f1, bind)

    # Snapshot halos: stream must not write them
    halo0_ex = f0[bind.n0 :].copy()
    halo1_ex = f1[: bind.depth].copy()

    # --- stream write-owned / read-halo ---
    stream_write_owned(
        f0, 0, bind.n0, ghost_bot=bounds0["ghost_bot"], ghost_top=None
    )
    stream_write_owned(
        f1, bind.depth, bind.nz1, ghost_bot=None, ghost_top=bounds1["ghost_top"]
    )

    if not np.array_equal(f0[bind.n0 :], halo0_ex):
        raise RuntimeError("step_two_way_numpy: stream wrote GPU0 cut-halo")
    if not np.array_equal(f1[: bind.depth], halo1_ex):
        raise RuntimeError("step_two_way_numpy: stream wrote GPU1 cut-halo")

    return f0, f1
