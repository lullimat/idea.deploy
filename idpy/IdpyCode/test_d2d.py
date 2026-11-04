__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2026 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
__credits__ = ["Matteo Lulli"]
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
__version__ = "0.1"
__maintainer__ = "Matteo Lulli"
__email__ = "matteo.lulli@gmail.com"
__status__ = "Development"

"""
Phase A acceptance: IdpyMemory.D2D redesign (host-mediated cross-device).

Run directly:
    python -m idpy.IdpyCode.test_d2d
"""

import numpy as np

from idpy.IdpyCode import (
    CUDA_T, OCL_T, CTYPES_T, METAL_T,
    idpy_langs_sys, idpy_langs_human_dict, GetTenet,
)
from idpy.IdpyCode import IdpyMemory
from idpy.Utils.TestExit import report_exit as _report_exit


_D2D_LANGS = (CUDA_T, OCL_T, METAL_T, CTYPES_T)


def _lang_ok(lang):
    return lang in idpy_langs_sys and idpy_langs_sys[lang]


def _tenet(lang):
    # Metal/CTypes Tenet are not context managers; match test_residency style.
    return GetTenet({"lang": lang, "device": 0})


def _ones(n, dtype, tenet):
    return IdpyMemory.Const(shape=(n,), dtype=dtype, const=1., tenet=tenet)


def _zeros(n, dtype, tenet):
    return IdpyMemory.Zeros(shape=(n,), dtype=dtype, tenet=tenet)


def _alloc_pair(tenet, n, dtype=np.float32):
    host = np.arange(n, dtype=dtype)
    src = _ones(n, dtype, tenet)
    dst = _zeros(n, dtype, tenet)
    src.H2D(host)
    return host, src, dst


def _assert_equal(host, dst, msg):
    got = np.asarray(dst.D2H())
    if not np.allclose(got, host):
        raise AssertionError(
            msg + ": expected %s... got %s..." % (host[:4], got[:4])
        )


def _sync_stream(ary, stream):
    if stream is None:
        ary.Sync()
        return
    sync = getattr(stream, "synchronize", None)
    if callable(sync):
        sync()
        return
    finish = getattr(stream, "finish", None)
    if callable(finish):
        finish()
        return
    ary.Sync(idpy_stream=stream)


def test_same_device_device_mode(lang):
    if not _lang_ok(lang):
        print("[SKIP] same-device device mode: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] same-device device mode (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    host, src, dst = _alloc_pair(tenet, 64)
    ret = IdpyMemory.D2D(src, dst, mode="device")
    assert ret is None, "sync device mode must return None"
    _assert_equal(host, dst, "device mode copy")
    print("[PASS] same-device device mode (%s)" % idpy_langs_human_dict[lang])


def test_same_device_host_mode(lang):
    if not _lang_ok(lang):
        print("[SKIP] same-device host mode: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] same-device host mode (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    host, src, dst = _alloc_pair(tenet, 64)
    ret = IdpyMemory.D2D(src, dst, mode="host")
    assert ret is None
    _assert_equal(host, dst, "host mode copy")
    print("[PASS] same-device host mode (%s)" % idpy_langs_human_dict[lang])


def test_same_device_auto(lang):
    if not _lang_ok(lang):
        print("[SKIP] same-device auto: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] same-device auto (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    host, src, dst = _alloc_pair(tenet, 64)
    ret = IdpyMemory.D2D(src, dst, mode="auto")
    assert ret is None
    _assert_equal(host, dst, "auto same-device copy")
    print("[PASS] same-device auto (%s)" % idpy_langs_human_dict[lang])


def test_peer_requires_capability(lang):
    if not _lang_ok(lang):
        print("[SKIP] peer capability: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] peer mode refuse (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    host, src, dst = _alloc_pair(tenet, 32)
    try:
        IdpyMemory.D2D(src, dst, mode="peer")
        raise AssertionError("peer mode must refuse here")
    except (RuntimeError, ValueError) as e:
        msg = str(e).lower()
        assert ("peer" in msg or "cuda" in msg or "same" in msg
                or "canaccesspeer" in msg), msg
    print("[PASS] peer mode refuse (%s)" % idpy_langs_human_dict[lang])


def test_async_requires_staging(lang):
    if not _lang_ok(lang):
        print("[SKIP] async staging required: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] async_=True requires staging= (%s)"
          % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    host, src, dst = _alloc_pair(tenet, 32)
    try:
        IdpyMemory.D2D(src, dst, mode="host", async_=True)
        raise AssertionError("async without staging must raise")
    except (ValueError, RuntimeError, TypeError) as e:
        assert "staging" in str(e).lower(), str(e)
    print("[PASS] async_=True requires staging= (%s)"
          % idpy_langs_human_dict[lang])


def test_async_host_loom(lang):
    """Loom (A): wait D2H complete before H2D; return only stream_dst pending."""
    if not _lang_ok(lang):
        print("[SKIP] async host loom: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    if lang == CTYPES_T:
        print("[SKIP] async host loom: CTypes has no streams")
        return
    print("[RUN ] async host loom (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    n = 128
    host, src, dst = _alloc_pair(tenet, n)
    staging = IdpyMemory.PinnedHost(shape=(n,), dtype=np.float32, tenet=tenet)
    stream_src = IdpyMemory.SiblingStream(tenet)
    stream_dst = IdpyMemory.SiblingStream(tenet)
    streams = IdpyMemory.D2D(
        src, dst, mode="host", async_=True, staging=staging,
        stream_src=stream_src, stream_dst=stream_dst,
    )
    assert streams is not None and isinstance(streams, dict)
    assert "stream_dst" in streams and "stream_src" in streams
    _sync_stream(dst, streams["stream_dst"])
    _assert_equal(host, dst, "async host loom copy")
    print("[PASS] async host loom (%s)" % idpy_langs_human_dict[lang])


def test_element_count_mismatch(lang):
    if not _lang_ok(lang):
        print("[SKIP] element-count: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] element-count mismatch (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    src = _ones(64, np.float32, tenet)
    dst = _zeros(32, np.float32, tenet)
    try:
        IdpyMemory.D2D(src, dst, mode="host")
        raise AssertionError("mismatched sizes must raise")
    except (ValueError, RuntimeError, AssertionError) as e:
        msg = str(e).lower()
        assert ("size" in msg or "element" in msg or "length" in msg
                or "shape" in msg or "count" in msg), msg
    print("[PASS] element-count mismatch (%s)" % idpy_langs_human_dict[lang])


def test_different_lang_refuse():
    available = [L for L in (OCL_T, METAL_T, CTYPES_T, CUDA_T) if _lang_ok(L)]
    if len(available) < 2:
        print("[SKIP] different-lang refuse: need >=2 backends")
        return
    lang_a, lang_b = available[0], available[1]
    print("[RUN ] different-lang refuse (%s vs %s)"
          % (idpy_langs_human_dict[lang_a], idpy_langs_human_dict[lang_b]))
    tenet_a = GetTenet({"lang": lang_a, "device": 0})
    tenet_b = GetTenet({"lang": lang_b, "device": 0})
    src = _ones(16, np.float32, tenet_a)
    dst = _zeros(16, np.float32, tenet_b)
    try:
        IdpyMemory.D2D(src, dst, mode="host")
        raise AssertionError("different langs must refuse")
    except ValueError as e:
        assert "lang" in str(e).lower(), str(e)
    print("[PASS] different-lang refuse")


def test_dtype_mismatch(lang):
    if not _lang_ok(lang):
        print("[SKIP] dtype mismatch: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] dtype mismatch (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    # float32 vs int32 -- Apple OpenCL often has no FP64.
    src = _ones(16, np.float32, tenet)
    dst = _zeros(16, np.int32, tenet)
    try:
        IdpyMemory.D2D(src, dst, mode="host")
        raise AssertionError("dtype mismatch must raise")
    except ValueError as e:
        assert "dtype" in str(e).lower(), str(e)
    print("[PASS] dtype mismatch (%s)" % idpy_langs_human_dict[lang])


def test_delayed_producer_loom(lang):
    """Delayed-producer: host path bit-identical after producer fill."""
    if not _lang_ok(lang):
        print("[SKIP] delayed-producer: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    if lang == CTYPES_T:
        print("[SKIP] delayed-producer: CTypes has no streams")
        return
    print("[RUN ] delayed-producer loom (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    n = 256
    dtype = np.float32
    src = _zeros(n, dtype, tenet)
    dst = _zeros(n, dtype, tenet)
    host = np.linspace(1.0, 2.0, n, dtype=dtype)
    src.H2D(host)
    staging = IdpyMemory.PinnedHost(shape=(n,), dtype=dtype, tenet=tenet)
    stream_src = IdpyMemory.SiblingStream(tenet)
    stream_dst = IdpyMemory.SiblingStream(tenet)
    streams = IdpyMemory.D2D(
        src, dst, mode="host", async_=True, staging=staging,
        stream_src=stream_src, stream_dst=stream_dst,
    )
    _sync_stream(dst, streams["stream_dst"])
    _assert_equal(host, dst, "delayed-producer")
    print("[PASS] delayed-producer loom (%s)" % idpy_langs_human_dict[lang])


def test_sync_return_none(lang):
    if not _lang_ok(lang):
        print("[SKIP] sync return None: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] sync return None (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    host, src, dst = _alloc_pair(tenet, 32)
    assert IdpyMemory.D2D(src, dst, mode="host") is None
    assert IdpyMemory.D2D(src, dst, mode="device") is None
    assert IdpyMemory.D2D(src, dst, mode="auto") is None
    print("[PASS] sync return None (%s)" % idpy_langs_human_dict[lang])


def test_multi_gpu_skip_note():
    print("[SKIP] multi-GPU peer / distinct CUDA context: no multi-GPU on this Mac")


# ---------------------------------------------------------------------------
# Phase B — D2DSub (Loom PASS-WITH-NITS)
# ---------------------------------------------------------------------------

def face_elem_range(shape, *, axis, face_index, depth, order="C"):
    """
    Test util (not hot public API): if shape[axis] slab
    [face_index : face_index+depth] is one contiguous flat range in the given
    order, return (start, stop) in flat dtype elements. Else raise.
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


def _fill_pattern(n, dtype=np.float32):
    return (np.arange(n, dtype=dtype) + 0.25).astype(dtype)


def _assert_strip_equal(expected, got, msg):
    expected = np.asarray(expected).ravel()
    got = np.asarray(got).ravel()
    if expected.shape != got.shape or not np.array_equal(expected, got):
        raise AssertionError(
            "%s: expected %s... got %s..." % (msg, expected[:4], got[:4])
        )


def test_d2dsub_same_device_strip_both_ways(lang):
    """Bit-identical strip both ways with explicit non-zero dst_start."""
    if not _lang_ok(lang):
        print("[SKIP] D2DSub strip both ways: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] D2DSub strip both ways + nonzero dst_start (%s)"
          % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    n = 128
    dtype = np.float32
    host = _fill_pattern(n, dtype)
    src = _zeros(n, dtype, tenet)
    dst = _zeros(n, dtype, tenet)
    src.H2D(host)

    src_start, src_stop = 16, 48
    dst_start = 64
    n_strip = src_stop - src_start
    ret = IdpyMemory.D2DSub(
        src, dst, src_start=src_start, src_stop=src_stop, dst_start=dst_start,
        mode="auto",
    )
    assert ret is None
    got = np.asarray(dst.D2H())
    _assert_strip_equal(
        host[src_start:src_stop], got[dst_start:dst_start + n_strip],
        "0-way strip auto",
    )
    # untouched outside dst strip
    assert np.all(got[:dst_start] == 0), "prefix must stay zero"
    assert np.all(got[dst_start + n_strip:] == 0), "suffix must stay zero"

    # reverse direction into a fresh dst with different nonzero dst_start
    dst2 = _zeros(n, dtype, tenet)
    src2 = _zeros(n, dtype, tenet)
    src2.H2D(host)
    src_start2, src_stop2 = 8, 40
    dst_start2 = 20
    IdpyMemory.D2DSub(
        src2, dst2, src_start=src_start2, src_stop=src_stop2,
        dst_start=dst_start2, mode="host",
    )
    got2 = np.asarray(dst2.D2H())
    _assert_strip_equal(
        host[src_start2:src_stop2],
        got2[dst_start2:dst_start2 + (src_stop2 - src_start2)],
        "1-way strip host",
    )
    print("[PASS] D2DSub strip both ways + nonzero dst_start (%s)"
          % idpy_langs_human_dict[lang])


def test_d2dsub_device_mode_strip(lang):
    if not _lang_ok(lang):
        print("[SKIP] D2DSub device strip: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] D2DSub device strip (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    n, dtype = 96, np.float32
    host = _fill_pattern(n, dtype)
    src = _zeros(n, dtype, tenet)
    dst = _zeros(n, dtype, tenet)
    src.H2D(host)
    IdpyMemory.D2DSub(
        src, dst, src_start=10, src_stop=50, dst_start=30, mode="device",
    )
    got = np.asarray(dst.D2H())
    _assert_strip_equal(host[10:50], got[30:70], "device strip")
    print("[PASS] D2DSub device strip (%s)" % idpy_langs_human_dict[lang])


def test_d2dsub_peer_refuse(lang):
    if not _lang_ok(lang):
        print("[SKIP] D2DSub peer refuse: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] D2DSub peer refuse (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    host, src, dst = _alloc_pair(tenet, 64)
    try:
        IdpyMemory.D2DSub(
            src, dst, src_start=0, src_stop=32, dst_start=0, mode="peer",
        )
        raise AssertionError("peer mode must refuse here")
    except (RuntimeError, ValueError) as e:
        msg = str(e).lower()
        assert ("peer" in msg or "cuda" in msg or "same" in msg
                or "canaccesspeer" in msg), msg
    print("[PASS] D2DSub peer refuse (%s)" % idpy_langs_human_dict[lang])


def test_d2dsub_async_requires_staging(lang):
    if not _lang_ok(lang):
        print("[SKIP] D2DSub async staging: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] D2DSub async_=True requires staging= (%s)"
          % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    host, src, dst = _alloc_pair(tenet, 64)
    try:
        IdpyMemory.D2DSub(
            src, dst, src_start=0, src_stop=32, dst_start=0,
            mode="host", async_=True,
        )
        raise AssertionError("async without staging must raise")
    except (ValueError, RuntimeError, TypeError) as e:
        assert "staging" in str(e).lower(), str(e)
    print("[PASS] D2DSub async_=True requires staging= (%s)"
          % idpy_langs_human_dict[lang])


def test_d2dsub_async_host_loom(lang):
    if not _lang_ok(lang):
        print("[SKIP] D2DSub async host loom: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    if lang == CTYPES_T:
        print("[SKIP] D2DSub async host loom: CTypes has no streams")
        return
    print("[RUN ] D2DSub async host loom (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    n, dtype = 128, np.float32
    host = _fill_pattern(n, dtype)
    src = _zeros(n, dtype, tenet)
    dst = _zeros(n, dtype, tenet)
    src.H2D(host)
    src_start, src_stop, dst_start = 16, 80, 8
    n_strip = src_stop - src_start
    staging = IdpyMemory.PinnedHost(shape=(n_strip,), dtype=dtype, tenet=tenet)
    stream_src = IdpyMemory.SiblingStream(tenet)
    stream_dst = IdpyMemory.SiblingStream(tenet)
    streams = IdpyMemory.D2DSub(
        src, dst, src_start=src_start, src_stop=src_stop, dst_start=dst_start,
        mode="host", async_=True, staging=staging,
        stream_src=stream_src, stream_dst=stream_dst,
    )
    assert streams is not None and isinstance(streams, dict)
    assert "stream_dst" in streams and "stream_src" in streams
    _sync_stream(dst, streams["stream_dst"])
    got = np.asarray(dst.D2H())
    _assert_strip_equal(
        host[src_start:src_stop], got[dst_start:dst_start + n_strip],
        "async D2DSub loom",
    )
    print("[PASS] D2DSub async host loom (%s)" % idpy_langs_human_dict[lang])


def test_d2dsub_range_validation(lang):
    if not _lang_ok(lang):
        print("[SKIP] D2DSub range validation: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] D2DSub range validation (%s)" % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    src = _ones(32, np.float32, tenet)
    dst = _zeros(32, np.float32, tenet)
    try:
        IdpyMemory.D2DSub(
            src, dst, src_start=0, src_stop=40, dst_start=0, mode="host",
        )
        raise AssertionError("OOB src must raise")
    except ValueError as e:
        assert "range" in str(e).lower() or "bound" in str(e).lower(), str(e)
    try:
        IdpyMemory.D2DSub(
            src, dst, src_start=0, src_stop=16, dst_start=20, mode="host",
        )
        raise AssertionError("OOB dst must raise")
    except ValueError as e:
        assert "range" in str(e).lower() or "bound" in str(e).lower(), str(e)
    # unequal full sizes OK for Sub if strip fits
    dst_small = _zeros(24, np.float32, tenet)
    IdpyMemory.D2DSub(
        src, dst_small, src_start=0, src_stop=16, dst_start=4, mode="host",
    )
    got = np.asarray(dst_small.D2H())
    host = np.asarray(src.D2H())
    _assert_strip_equal(host[0:16], got[4:20], "unequal parent sizes strip")
    print("[PASS] D2DSub range validation (%s)" % idpy_langs_human_dict[lang])


def test_d2dsub_overlap_device_refuse(lang):
    if not _lang_ok(lang):
        print("[SKIP] D2DSub overlap refuse: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] D2DSub overlap device refuse / auto->host (%s)"
          % idpy_langs_human_dict[lang])
    tenet = _tenet(lang)
    n, dtype = 64, np.float32
    host = _fill_pattern(n, dtype)
    ary = _zeros(n, dtype, tenet)
    ary.H2D(host)
    try:
        IdpyMemory.D2DSub(
            ary, ary, src_start=0, src_stop=20, dst_start=10, mode="device",
        )
        raise AssertionError("overlapping device memcpy must refuse")
    except ValueError as e:
        assert "overlap" in str(e).lower(), str(e)
    # auto forces host and remains bit-identical
    IdpyMemory.D2DSub(
        ary, ary, src_start=0, src_stop=20, dst_start=10, mode="auto",
    )
    got = np.asarray(ary.D2H())
    expect = host.copy()
    expect[10:30] = host[0:20]
    _assert_strip_equal(expect, got, "overlap auto->host")
    print("[PASS] D2DSub overlap device refuse / auto->host (%s)"
          % idpy_langs_human_dict[lang])


def test_d2dsub_soft_bw_log(lang):
    if not _lang_ok(lang):
        print("[SKIP] D2DSub soft BW: %s unavailable"
              % idpy_langs_human_dict[lang])
        return
    print("[RUN ] D2DSub soft BW log (%s)" % idpy_langs_human_dict[lang])
    import time
    tenet = _tenet(lang)
    # ~4 MiB strip of float32
    n = 1 << 20
    dtype = np.float32
    host = np.random.RandomState(0).randn(n).astype(dtype)
    src = _zeros(n, dtype, tenet)
    dst = _zeros(n, dtype, tenet)
    src.H2D(host)
    # warmup
    IdpyMemory.D2DSub(src, dst, src_start=0, src_stop=n, dst_start=0,
                      mode="host")
    t0 = time.perf_counter()
    IdpyMemory.D2DSub(src, dst, src_start=0, src_stop=n, dst_start=0,
                      mode="host")
    dt = max(time.perf_counter() - t0, 1e-9)
    nbytes = n * int(np.dtype(dtype).itemsize)
    gbps = (nbytes / dt) / 1e9
    # Informational only — never hard-fail CI on strip BW.
    print("[INFO] D2DSub soft BW (%s): %.2f GB/s for %.2f MiB host strip "
          "(informational; not a CI gate)"
          % (idpy_langs_human_dict[lang], gbps, nbytes / (1024 * 1024)))
    _assert_strip_equal(host, np.asarray(dst.D2H()), "soft BW correctness")
    print("[PASS] D2DSub soft BW log (%s)" % idpy_langs_human_dict[lang])


def test_face_elem_range_util():
    print("[RUN ] face_elem_range test util")
    # C-order (nz, nx, Q): z-slab contiguous
    start, stop = face_elem_range((8, 4, 3), axis=0, face_index=2, depth=3,
                                  order="C")
    assert (start, stop) == (2 * 4 * 3, 5 * 4 * 3)
    # C-order (nx, nz, Q): z-slab non-contiguous
    try:
        face_elem_range((4, 8, 3), axis=1, face_index=2, depth=3, order="C")
        raise AssertionError("non-contig z-slab must raise")
    except ValueError as e:
        assert "non-contiguous" in str(e).lower(), str(e)
    print("[PASS] face_elem_range test util")


def test_d2dsub_multi_gpu_skip_note():
    print("[SKIP] D2DSub multi-GPU cross-device strip: no multi-GPU on this Mac")



def main():
    print("=== IdpyMemory.D2D Phase A acceptance ===")
    for lang in _D2D_LANGS:
        test_same_device_device_mode(lang)
        test_same_device_host_mode(lang)
        test_same_device_auto(lang)
        test_peer_requires_capability(lang)
        test_async_requires_staging(lang)
        test_async_host_loom(lang)
        test_element_count_mismatch(lang)
        test_dtype_mismatch(lang)
        test_delayed_producer_loom(lang)
        test_sync_return_none(lang)
    test_different_lang_refuse()
    test_multi_gpu_skip_note()
    print("=== Phase A acceptance done ===")

    print("=== IdpyMemory.D2DSub Phase B acceptance ===")
    for lang in _D2D_LANGS:
        test_d2dsub_same_device_strip_both_ways(lang)
        test_d2dsub_device_mode_strip(lang)
        test_d2dsub_peer_refuse(lang)
        test_d2dsub_async_requires_staging(lang)
        test_d2dsub_async_host_loom(lang)
        test_d2dsub_range_validation(lang)
        test_d2dsub_overlap_device_refuse(lang)
        test_d2dsub_soft_bw_log(lang)
    test_face_elem_range_util()
    test_d2dsub_multi_gpu_skip_note()
    print("=== Phase B acceptance done ===")


if __name__ == "__main__":
    try:
        main()
        _report_exit(True)
    except Exception as e:
        print("[FAIL] %s" % e)
        raise


