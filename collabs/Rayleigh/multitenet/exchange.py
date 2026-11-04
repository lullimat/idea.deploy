"""Phase C ``exchange_faces`` — two sequential sync D2DSub legs (C0).

Bidirectional cut-face exchange each LBM step:

* GPU0 top interior → GPU1 bottom halo
* GPU1 bottom interior → GPU0 top halo

Call-site order (RECEIVED): plant → collide → **exchange_faces** → stream.
Populations only — no silent macro face ship. Cut-halos are not collided.
Per-leg ``stream_src`` / ``stream_dst`` only — no aggregate stream API.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .bind import TwoWayZSplitBind, assert_range_disjointness, split_global_to_subdomains


def exchange_faces_numpy(
    f0: np.ndarray,
    f1: np.ndarray,
    bind: TwoWayZSplitBind,
) -> None:
    """In-place NumPy reference exchange using bind face ranges (C0 semantics).

    Writes halo slots only; interiors stable. Used for bit-identical tests and
    as the host reference path when CUDA cross-device is unavailable.
    """
    assert_range_disjointness(bind)
    if f0.shape != bind.shape0 or f1.shape != bind.shape1:
        raise ValueError(
            "exchange_faces_numpy: shape mismatch f0=%s (want %s) f1=%s (want %s)"
            % (f0.shape, bind.shape0, f1.shape, bind.shape1)
        )
    if not f0.flags.c_contiguous or not f1.flags.c_contiguous:
        raise ValueError("exchange_faces_numpy: f0/f1 must be C-contiguous")

    a = f0.reshape(-1)
    b = f1.reshape(-1)
    s0, e0 = bind.src_range_0to1
    d0, e_d0 = bind.dst_range_0to1
    s1, e1 = bind.src_range_1to0
    d1, e_d1 = bind.dst_range_1to0

    # Snapshot both src faces first so sequential legs cannot race if ranges
    # ever overlapped (they must not — assert_range_disjointness).
    face_01 = a[s0:e0].copy()
    face_10 = b[s1:e1].copy()
    b[d0:e_d0] = face_01
    a[d1:e_d1] = face_10


def exchange_faces(
    f0,
    f1,
    *,
    src_range_0to1: Tuple[int, int],
    dst_range_0to1: Tuple[int, int],
    src_range_1to0: Tuple[int, int],
    dst_range_1to0: Tuple[int, int],
    mode: str = "auto",
    staging_a=None,
    staging_b=None,
    stream_src_a=None,
    stream_dst_a=None,
    stream_src_b=None,
    stream_dst_b=None,
    async_: bool = False,
):
    """Bidirectional cut-face exchange via two D2DSub calls (C0 default sync).

    Parameters
    ----------
    f0, f1 :
        IdpyMemory device arrays for GPU0 / GPU1 subdomains
        (flat-addressable; layout ``(nz_local, nx, Q)`` C-order).
    src_range_*/dst_range_* :
        Flat dtype-element ``(start, stop)`` from Rayleigh-local
        ``face_elem_range`` / bind helper.
    staging_a, staging_b :
        Prefer-src staging for legs 0→1 and 1→0 (allocated once at bind).
        Required when ``async_=True`` (inherited D2DSub contract).
    stream_src_a/dst_a, stream_src_b/dst_b :
        Per-leg streams passed through to each ``D2DSub`` — **no** aggregate
        stream API on this helper.
    async_ :
        C0 default ``False`` (sequential sync both legs → ``None``).
        C1 async is optional and not required for Phase C accept.

    Returns
    -------
    None
        when ``async_=False`` (C0).
    tuple
        ``(ret_leg_a, ret_leg_b)`` when ``async_=True`` — each leg's
        ``{'stream_src', 'stream_dst'}`` from D2DSub (no new aggregate type).
    """
    from idpy.IdpyCode import IdpyMemory

    s0, e0 = src_range_0to1
    d0, _ = dst_range_0to1
    s1, e1 = src_range_1to0
    d1, _ = dst_range_1to0

    # Leg 0→1: GPU0 top interior → GPU1 bottom halo
    ret_a = IdpyMemory.D2DSub(
        f0,
        f1,
        src_start=s0,
        src_stop=e0,
        dst_start=d0,
        mode=mode,
        staging=staging_a,
        stream_src=stream_src_a,
        stream_dst=stream_dst_a,
        async_=async_,
    )
    # Leg 1→0: GPU1 bottom interior → GPU0 top halo
    ret_b = IdpyMemory.D2DSub(
        f1,
        f0,
        src_start=s1,
        src_stop=e1,
        dst_start=d1,
        mode=mode,
        staging=staging_b,
        stream_src=stream_src_b,
        stream_dst=stream_dst_b,
        async_=async_,
    )
    if not async_:
        return None
    return ret_a, ret_b


def exchange_faces_from_bind(
    f0,
    f1,
    bind: TwoWayZSplitBind,
    *,
    mode: str = "auto",
    staging_a=None,
    staging_b=None,
    stream_src_a=None,
    stream_dst_a=None,
    stream_src_b=None,
    stream_dst_b=None,
    async_: bool = False,
):
    """Convenience wrapper: :func:`exchange_faces` with ranges from ``bind``."""
    assert_range_disjointness(bind)
    return exchange_faces(
        f0,
        f1,
        src_range_0to1=bind.src_range_0to1,
        dst_range_0to1=bind.dst_range_0to1,
        src_range_1to0=bind.src_range_1to0,
        dst_range_1to0=bind.dst_range_1to0,
        mode=mode,
        staging_a=staging_a,
        staging_b=staging_b,
        stream_src_a=stream_src_a,
        stream_dst_a=stream_dst_a,
        stream_src_b=stream_src_b,
        stream_dst_b=stream_dst_b,
        async_=async_,
    )


def monolith_face_halo_reference(
    f_global_mt: np.ndarray, bind: TwoWayZSplitBind
) -> Tuple[np.ndarray, np.ndarray]:
    """Monolith reference: copy interior faces into neighbor halo slots.

    Returns ``(f0_ref, f1_ref)`` subdomain buffers with cut-halos filled as
    exchange would, without calling D2DSub.
    """
    f0, f1 = split_global_to_subdomains(f_global_mt, bind)
    # GPU0 top interior rows → GPU1 bottom halo
    f1[: bind.depth] = f_global_mt[bind.z_cut - bind.depth : bind.z_cut]
    # GPU1 bottom interior rows → GPU0 top halo
    f0[bind.n0 :] = f_global_mt[bind.z_cut : bind.z_cut + bind.depth]
    return f0, f1
