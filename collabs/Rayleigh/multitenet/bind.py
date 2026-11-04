"""Phase C multi-tenet 2-way z-split bind (Rayleigh Option A).

Authoritative sizing for consumer bind (supersedes Phase B §4.1 illustrative
formulas). Layout for multi-tenet subdomains: C-order ``(nz_local, nx, Q)``.
Gate1 single-GPU keeps ``(nx, nz, Q)``; optional transpose only at multi-tenet
bind — never silent inside D2DSub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .face_elem_range import face_elem_range

# D2Q37 |c|_∞ = 3; matches Gate1 three ghost rows. Spin stencil ≤3 is a flag only.
MIN_DEPTH = 3


@dataclass(frozen=True)
class TwoWayZSplitBind:
    """Bind result for a 2-way z-split population field (populations only)."""

    Nz: int
    nx: int
    Q: int
    z_cut: int
    depth: int
    n0: int  # GPU0 interior rows
    n1: int  # GPU1 interior rows
    nz0: int  # GPU0 nz_local = n0 + depth (interior + top halo)
    nz1: int  # GPU1 nz_local = n1 + depth (interior + bottom halo)
    n_face: int
    shape0: Tuple[int, int, int]  # (nz0, nx, Q)
    shape1: Tuple[int, int, int]  # (nz1, nx, Q)
    # Flat dtype-element ranges on each parent (C-order contiguous z-faces)
    src_range_0to1: Tuple[int, int]  # GPU0 top interior → send
    dst_range_0to1: Tuple[int, int]  # GPU1 bottom halo → recv
    src_range_1to0: Tuple[int, int]  # GPU1 bottom interior → send
    dst_range_1to0: Tuple[int, int]  # GPU0 top halo → recv
    # Call-site order reminder (RECEIVED): plant → collide → exchange → stream
    call_site_order: str = "plant→collide→exchange_faces→stream"
    populations_only: bool = True
    cut_halos_collided: bool = False  # cut-halos are NOT collided


def _validate_bind_guards(*, depth: int, z_cut: int, Nz: int) -> None:
    if depth < MIN_DEPTH:
        raise ValueError(
            "bind 2-way z-split: depth=%d < MIN_DEPTH=%d (D2Q37 |c|_∞=3)"
            % (depth, MIN_DEPTH)
        )
    if z_cut < 0 or z_cut > Nz:
        raise ValueError(
            "bind 2-way z-split: z_cut=%d out of range for Nz=%d" % (z_cut, Nz)
        )
    if depth > z_cut:
        raise ValueError(
            "bind 2-way z-split: depth=%d > z_cut=%d — halo would eat past "
            "GPU0 owned interior (require depth <= z_cut)" % (depth, z_cut)
        )
    if depth > (Nz - z_cut):
        raise ValueError(
            "bind 2-way z-split: depth=%d > Nz-z_cut=%d — halo would eat past "
            "GPU1 owned interior (require depth <= Nz - z_cut)"
            % (depth, Nz - z_cut)
        )


def bind_two_way_z_split(
    *,
    Nz: int,
    nx: int,
    Q: int,
    z_cut: Optional[int] = None,
    depth: int = MIN_DEPTH,
) -> TwoWayZSplitBind:
    """Compute 2-way z-split sizes, shapes, and face ranges.

    Parameters
    ----------
    Nz, nx, Q :
        Global Gate1-like sizes. ``Nz`` is global fluid+wall rows along z.
    z_cut :
        Split index; GPU0 owns ``[0, z_cut)``, GPU1 owns ``[z_cut, Nz)``.
        Default ``Nz // 2``.
    depth :
        Halo / face depth; must be ``>= 3`` and satisfy bind guards
        ``depth <= z_cut`` and ``depth <= Nz - z_cut``.
    """
    Nz = int(Nz)
    nx = int(nx)
    Q = int(Q)
    depth = int(depth)
    if z_cut is None:
        z_cut = Nz // 2
    z_cut = int(z_cut)

    _validate_bind_guards(depth=depth, z_cut=z_cut, Nz=Nz)

    n0 = z_cut
    n1 = Nz - z_cut
    nz0 = n0 + depth
    nz1 = n1 + depth
    n_face = depth * nx * Q
    shape0 = (nz0, nx, Q)
    shape1 = (nz1, nx, Q)

    # GPU0: interior z_local[0:n0]; top halo z_local[n0:n0+depth]
    src_range_0to1 = face_elem_range(
        shape0, axis=0, face_index=n0 - depth, depth=depth
    )
    dst_range_1to0 = face_elem_range(
        shape0, axis=0, face_index=n0, depth=depth
    )
    # GPU1: bottom halo z_local[0:depth]; interior z_local[depth:depth+n1]
    src_range_1to0 = face_elem_range(
        shape1, axis=0, face_index=depth, depth=depth
    )
    dst_range_0to1 = face_elem_range(
        shape1, axis=0, face_index=0, depth=depth
    )

    # Sanity: equal face sizes both legs
    for label, rng in (
        ("src_0to1", src_range_0to1),
        ("dst_0to1", dst_range_0to1),
        ("src_1to0", src_range_1to0),
        ("dst_1to0", dst_range_1to0),
    ):
        if rng[1] - rng[0] != n_face:
            raise RuntimeError(
                "bind: %s span %d != n_face %d" % (label, rng[1] - rng[0], n_face)
            )

    return TwoWayZSplitBind(
        Nz=Nz,
        nx=nx,
        Q=Q,
        z_cut=z_cut,
        depth=depth,
        n0=n0,
        n1=n1,
        nz0=nz0,
        nz1=nz1,
        n_face=n_face,
        shape0=shape0,
        shape1=shape1,
        src_range_0to1=src_range_0to1,
        dst_range_0to1=dst_range_0to1,
        src_range_1to0=src_range_1to0,
        dst_range_1to0=dst_range_1to0,
    )


def assert_range_disjointness(bind: TwoWayZSplitBind) -> None:
    """C0 in-place safety: src faces ∩ dst halos empty both ways on each buffer."""
    # GPU0: src top-interior vs dst top-halo
    s0, e0 = bind.src_range_0to1
    d0, f0 = bind.dst_range_1to0
    if not (e0 <= d0 or f0 <= s0):
        raise AssertionError(
            "GPU0 src face ∩ dst halo nonempty: src[%d,%d) dst[%d,%d)"
            % (s0, e0, d0, f0)
        )
    # GPU1: src bottom-interior vs dst bottom-halo
    s1, e1 = bind.src_range_1to0
    d1, f1 = bind.dst_range_0to1
    if not (e1 <= d1 or f1 <= s1):
        raise AssertionError(
            "GPU1 src face ∩ dst halo nonempty: src[%d,%d) dst[%d,%d)"
            % (s1, e1, d1, f1)
        )


def transpose_gate1_to_multitenet(f_gate1: np.ndarray) -> np.ndarray:
    """Explicit ``(nx, nz, Q)`` → ``(nz, nx, Q)`` at multi-tenet bind only.

    Gate1 single-GPU keeps ``(nx, nz, Q)``. Never call this inside D2DSub.
    """
    f_gate1 = np.asarray(f_gate1)
    if f_gate1.ndim != 3:
        raise ValueError(
            "transpose_gate1_to_multitenet: expected (nx, nz, Q), got shape %s"
            % (f_gate1.shape,)
        )
    # (nx, nz, Q) → (nz, nx, Q)
    return np.ascontiguousarray(np.transpose(f_gate1, (1, 0, 2)))


def transpose_multitenet_to_gate1(f_mt: np.ndarray) -> np.ndarray:
    """Inverse of :func:`transpose_gate1_to_multitenet`."""
    f_mt = np.asarray(f_mt)
    if f_mt.ndim != 3:
        raise ValueError(
            "transpose_multitenet_to_gate1: expected (nz, nx, Q), got shape %s"
            % (f_mt.shape,)
        )
    return np.ascontiguousarray(np.transpose(f_mt, (1, 0, 2)))


def split_global_to_subdomains(
    f_global_mt: np.ndarray, bind: TwoWayZSplitBind
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a global ``(Nz, nx, Q)`` multitenet-layout field into subdomain buffers.

    GPU0 buffer: interior ``[0, z_cut)`` + zeroed top halo.
    GPU1 buffer: zeroed bottom halo + interior ``[z_cut, Nz)``.
    Cut-halos are **not** pre-filled (exchange fills them).
    """
    f_global_mt = np.ascontiguousarray(f_global_mt)
    if f_global_mt.shape != (bind.Nz, bind.nx, bind.Q):
        raise ValueError(
            "split_global_to_subdomains: expected shape %s, got %s"
            % ((bind.Nz, bind.nx, bind.Q), f_global_mt.shape)
        )
    f0 = np.zeros(bind.shape0, dtype=f_global_mt.dtype)
    f1 = np.zeros(bind.shape1, dtype=f_global_mt.dtype)
    f0[: bind.n0] = f_global_mt[: bind.z_cut]
    f1[bind.depth :] = f_global_mt[bind.z_cut :]
    return f0, f1


def allocate_prefer_src_staging(
    bind: TwoWayZSplitBind,
    *,
    dtype=np.float64,
    tenet_src_a=None,
    tenet_src_b=None,
):
    """Allocate two prefer-src staging buffers of ``n_face`` (C0 default).

    When ``tenet_src_a`` / ``tenet_src_b`` are provided, allocate
    ``IdpyMemory.PinnedHost`` on that tenet. Otherwise return NumPy pageable
    buffers (sync-only OK per Phase A/B staging rules).
    """
    n = bind.n_face
    if tenet_src_a is None and tenet_src_b is None:
        return (
            np.empty(n, dtype=dtype),
            np.empty(n, dtype=dtype),
        )
    from idpy.IdpyCode import IdpyMemory

    staging_a = IdpyMemory.PinnedHost(
        shape=(n,), dtype=dtype, tenet=tenet_src_a
    )
    staging_b = IdpyMemory.PinnedHost(
        shape=(n,), dtype=dtype, tenet=tenet_src_b
    )
    return staging_a, staging_b
