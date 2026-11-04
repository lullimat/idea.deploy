"""Rayleigh-local copy of face_elem_range (Phase C).

Copied from idpy.IdpyCode.test_d2d — do **not** import from test_d2d.
Stays out of hot IdpyMemory API (Phase B LOCKED / Phase C Option A).
"""


def face_elem_range(shape, *, axis, face_index, depth, order="C"):
    """
    If shape[axis] slab [face_index : face_index+depth] is one contiguous flat
    range in the given order, return (start, stop) in flat dtype elements.
    Else raise.
    """
    shape = tuple(int(s) for s in shape)
    axis = int(axis)
    face_index = int(face_index)
    depth = int(depth)
    if order not in ("C", "F"):
        raise ValueError("face_elem_range: order must be 'C' or 'F'")
    if axis < 0 or axis >= len(shape):
        raise ValueError("face_elem_range: axis out of range")
    if depth <= 0:
        raise ValueError("face_elem_range: depth must be > 0")
    if face_index < 0 or face_index + depth > shape[axis]:
        raise ValueError(
            "face_elem_range: face [%d, %d) out of bounds for axis size %d"
            % (face_index, face_index + depth, shape[axis])
        )

    # Contiguous slab iff the split axis is the slowest-varying spatial axis
    # for that order (axis 0 for C, last for F) — then a face is one [start,stop).
    if order == "C":
        contiguous_axis = 0
    else:
        contiguous_axis = len(shape) - 1
    if axis != contiguous_axis:
        raise ValueError(
            "face_elem_range: slab along axis %d is non-contiguous in %s-order "
            "shape %s; use layout with split axis slowest among spatial "
            "(e.g. C-order (nz_local, nx, Q)) or a Phase-C pack — B0 has no "
            "silent gather" % (axis, order, shape)
        )

    strides = [1] * len(shape)
    if order == "C":
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
    else:
        for i in range(1, len(shape)):
            strides[i] = strides[i - 1] * shape[i - 1]

    start = face_index * strides[axis]
    stop = (face_index + depth) * strides[axis]
    return start, stop
