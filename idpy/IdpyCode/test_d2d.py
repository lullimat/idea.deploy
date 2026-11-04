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


if __name__ == "__main__":
    try:
        main()
        _report_exit(True)
    except Exception as e:
        print("[FAIL] %s" % e)
        raise
