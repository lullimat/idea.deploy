"""Rayleigh multi-tenet Phase C orchestration (Option A).

Composes two Phase B ``IdpyMemory.D2DSub`` calls for 2-way z-split cut-face
exchange. Does **not** edit IdpyMemory contracts; ``face_elem_range`` is a
Rayleigh-local copy (not imported from ``test_d2d``).

Call-site order: plant → collide → exchange_faces → stream.
Populations only; cut-halos are not collided.
"""

from .bind import (
    MIN_DEPTH,
    TwoWayZSplitBind,
    allocate_prefer_src_staging,
    assert_range_disjointness,
    bind_two_way_z_split,
    split_global_to_subdomains,
    transpose_gate1_to_multitenet,
    transpose_multitenet_to_gate1,
)
from .exchange import (
    exchange_faces,
    exchange_faces_from_bind,
    exchange_faces_numpy,
    monolith_face_halo_reference,
)
from .face_elem_range import face_elem_range

__all__ = [
    "MIN_DEPTH",
    "TwoWayZSplitBind",
    "allocate_prefer_src_staging",
    "assert_range_disjointness",
    "bind_two_way_z_split",
    "exchange_faces",
    "exchange_faces_from_bind",
    "exchange_faces_numpy",
    "face_elem_range",
    "monolith_face_halo_reference",
    "split_global_to_subdomains",
    "transpose_gate1_to_multitenet",
    "transpose_multitenet_to_gate1",
]
