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

'''
T3-overlap: does an async partial write actually OVERLAP with compute?

test_residency.py T3 establishes that an async H2DSub issued with a kernel in
flight produces correct results. It deliberately does not claim more: on a
single in-order queue (OpenCL) or the default stream (CUDA), the runtime is
entitled to order the copy after the kernel, and correctness would look
identical either way. This module answers the question T3 leaves open, by
measuring rather than asserting.

Kept separate from test_residency.py on purpose: that suite is exact-correctness
and must never go flaky. This one is a measurement -- its numbers are
machine-dependent and its thresholds are judgement calls.

Method
------
Three timings, each the minimum over repeats (min is the robust estimator for
timing -- noise only ever adds):

    tK   kernel alone on stream A
    tC   async H2DSub alone on stream B
    tB   both issued back-to-back, then both synced

    overlap = (tK + tC - tB) / min(tK, tC)

which is 1.0 for perfect overlap (tB == max(tK, tC)) and 0.0 for full
serialization (tB == tK + tC).

Two things make the measurement honest rather than flattering:

- **A compute-bound kernel.** A memory-bound kernel contends with the transfer
  for bandwidth, so poor overlap would be reported even when the two genuinely
  run concurrently. The filler kernel is a long dependent FMA chain: high
  arithmetic intensity, negligible memory traffic.
- **Calibration.** ITERS is tuned so tK reaches max(tC, target_ms) -- long
  enough to be measurable well above timer noise, and never shorter than the
  transfer it is meant to hide. When the transfer is much faster than that floor
  (unified memory, typically) the two are not balanced, which is fine: the ratio
  divides by min(tK, tC), so it then reads as "what fraction of the copy was
  hidden" and stays fully discriminating. What would break it is tC dropping to
  timer noise -- watch the reported copy time, not just the ratio.

Correctness is verified in the overlapped run too, and exactly:

- the transferred half must equal the payload bit-for-bit -- that half is what
  the residency primitive owns;
- the computed half must equal the output of the *same kernel run alone on the
  same device*, so the comparison is against a deterministic device reference
  rather than a host re-implementation. That sidesteps FMA-contraction
  differences, which are a property of the filler kernel and not of anything
  under test here.

Run directly:
    python -m idpy.IdpyCode.test_overlap
'''

from collections import OrderedDict
from time import perf_counter

import numpy as np

from idpy.IdpyCode import (
    IDPY_T, CUDA_T, OCL_T, CTYPES_T, METAL_T,
    idpy_langs_sys, idpy_langs_human_dict, GetTenet,
)
from idpy.IdpyCode.IdpyCode import IdpyKernel
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode.IdpyUnroll import _codify_comment
from idpy.Utils.CustomTypes import CustomTypes

_OVERLAP_LANGS = (CUDA_T, OCL_T, METAL_T)

# Reported verdict bands. Judgement calls, stated rather than hidden.
_OVERLAP_GOOD = 0.50
_OVERLAP_PARTIAL = 0.15


class K_ComputeBound(IdpyKernel):
    '''
    A long dependent FMA chain per lane: buf[i] = fma^ITERS(buf[i]).

    Deliberately compute-bound -- one load and one store per lane, ITERS worth
    of arithmetic in between -- so that it does not compete with a concurrent
    transfer for memory bandwidth. The chain is a data dependency, so unrolling
    does not reduce the work and the loop cannot be folded to a closed form.
    '''

    def __init__(self, iters=1024, custom_types=None, optimizer_flag=None):
        if custom_types is None:
            custom_types = CustomTypes({'FType': 'float'}).Push()

        constants = OrderedDict()
        constants['ITERS'] = int(iters)
        '''
        np.float32, not bare Python floats.

        A Python float reaches the generated source as a bare literal --
        '#define CHAIN_A 0.99999' -- which is a *double* in C. The surrounding
        'float v' then promotes and the whole chain evaluates in fp64 wherever
        fp64 exists; on an RTX 5060 that measured ~208x slower than the fp32
        path. np.float32 pins the literal to fp32 on every backend, which is
        what this benchmark wants: the filler kernel's cost must depend on the
        hardware, not on which hardware happens to have fp64 units.

        Pinning rather than declaring constants_types={'CHAIN_A': 'FType'} is
        deliberate here. These are algorithmic constants of the benchmark, not
        physics values that should track the kernel's working precision.

        |A| < 1 keeps the chain bounded; B keeps it from decaying to zero.
        '''
        constants['CHAIN_A'] = np.float32(0.99999)
        constants['CHAIN_B'] = np.float32(0.00001)

        IdpyKernel.__init__(
            self, custom_types=custom_types, constants=constants,
            optimizer_flag=optimizer_flag,
        )
        self.SetCodeFlags('g_tid')

        self.params = {'FType * buf': ['global', 'restrict']}

        body = ""
        body += _codify_comment("compute-bound filler: dependent FMA chain")
        body += "FType v = buf[g_tid];\n"
        body += "for(int i = 0; i < ITERS; i++){\n"
        body += "v = v * CHAIN_A + CHAIN_B;\n"
        body += "}\n"
        body += "buf[g_tid] = v;\n"
        self.kernels[IDPY_T] = "\n" + body

    def dump_code(self, lang=IDPY_T):
        return self.Code(lang)


def _sync(stream):
    '''Backend-agnostic wait: CUDA streams synchronize(), OpenCL queues finish().'''
    if stream is None:
        return
    if hasattr(stream, 'synchronize'):
        stream.synchronize()
    elif hasattr(stream, 'finish'):
        stream.finish()


def _tenet_params(lang):
    params = {'lang': lang}
    if lang == OCL_T:
        params['cl_kind'] = 'gpu'
    return params


class _Harness:
    '''Owns the buffers, streams and kernel for one backend measurement.'''

    def __init__(self, tenet, lang, copy_mb=128, block_size=256):
        self.tenet, self.lang = tenet, lang
        self.block_size = block_size

        # buffer split in halves: kernel owns [0, half), transfer owns [half, n)
        self.half = (int(copy_mb) * (1 << 20)) // np.dtype(np.float32).itemsize
        self.n = 2 * self.half
        self.copy_bytes = self.half * np.dtype(np.float32).itemsize

        self.host0 = np.linspace(
            0.1, 1.0, self.n, dtype=np.float32
        )
        self.buf = IdpyMemory.OnDevice(self.host0, tenet=tenet)

        # page-locked on CUDA -- without it an async H2D cannot overlap at all
        self.payload = IdpyMemory.PinnedHost(
            (self.half,), np.float32, tenet=tenet
        )
        self.payload[:] = np.arange(self.half, dtype=np.float32) * 0.5

        self.stream_k = IdpyMemory.SiblingStream(tenet)
        self.stream_c = IdpyMemory.SiblingStream(tenet)
        if lang != METAL_T and (self.stream_k is None or self.stream_c is None):
            raise RuntimeError(
                "backend exposes no sibling streams; overlap cannot be measured"
            )

        self.idea = None

        '''
        'idpy_stream' does not mean the same thing on both backends:

          CUDA   -- Deploy passes it straight through as the launch stream, so
                    the kernel goes on stream_k and the tenet is untouched.
          OpenCL -- Deploy uses it as a 'wait_for' EVENT LIST and enqueues on
                    the queue bound at kernel instantiation. To put the kernel
                    on a different queue we must instantiate against that queue
                    and pass no wait-list.

        Cross-queue access to one buffer inside a single context is legal; the
        halves are disjoint and both queues are drained before reading.
        '''
        if lang == OCL_T:
            self.kernel_tenet, self.deploy_stream = self.stream_k, None
        else:
            self.kernel_tenet, self.deploy_stream = self.tenet, self.stream_k

    def build_kernel(self, iters):
        kern = K_ComputeBound(iters=iters)
        grid = ((self.half + self.block_size - 1) // self.block_size, 1, 1)
        block = (self.block_size, 1, 1)
        self.idea = kern(tenet=self.kernel_tenet, grid=grid, block=block)

    def deploy(self):
        '''
        Metal has no sibling streams: concurrency there is host-vs-GPU, not
        stream-vs-stream. The declared 'touched' span is what lets the host
        write to the upper half without waiting on this kernel -- omit it and
        the whole buffer is assumed touched, which is safe but serializes.
        '''
        if self.lang == METAL_T:
            return self.idea.Deploy(
                [self.buf], touched={self.buf: (0, self.half)},
            )
        return self.idea.Deploy([self.buf], idpy_stream=self.deploy_stream)

    def sync_kernel(self):
        if self.lang == METAL_T:
            self.tenet.Finish()
        else:
            _sync(self.stream_k)

    def sync_copy(self):
        # On Metal the write is a completed host store; nothing to wait for.
        if self.lang != METAL_T:
            _sync(self.stream_c)

    def write_slot(self):
        return self.buf.H2DSub(self.payload, start=self.half,
                               async_=True, idpy_stream=self.stream_c)

    def reset(self):
        self.buf.H2D(self.host0)
        self.sync_kernel()
        self.sync_copy()

    def time_kernel(self):
        self.reset()
        t0 = perf_counter()
        self.deploy()
        self.sync_kernel()
        return perf_counter() - t0

    def time_copy(self):
        self.reset()
        t0 = perf_counter()
        self.write_slot()
        self.sync_copy()
        return perf_counter() - t0

    def time_both(self):
        self.reset()
        t0 = perf_counter()
        self.deploy()
        self.write_slot()
        self.sync_kernel()
        self.sync_copy()
        return perf_counter() - t0

    def kernel_reference(self):
        '''Lower half after the kernel alone -- the deterministic device oracle.'''
        self.reset()
        self.deploy()
        self.sync_kernel()
        return self.buf.D2H()[:self.half].copy()

    def close(self):
        if hasattr(self.tenet, 'End'):
            self.tenet.End()


def _best_of(fn, repeats):
    return min(fn() for _ in range(repeats))


def measure(lang, copy_mb=128, repeats=3, target_ms=40.0, iters0=256):
    tenet = GetTenet(_tenet_params(lang))
    h = _Harness(tenet, lang, copy_mb=copy_mb)
    out = OrderedDict()
    try:
        # --- calibrate: pick ITERS so the kernel roughly matches the transfer
        h.build_kernel(iters0)
        h.time_kernel()                      # warm up: compile + first touch
        t_k0 = _best_of(h.time_kernel, 2)
        t_c = _best_of(h.time_copy, repeats)

        target = max(t_c, target_ms * 1e-3)
        iters = int(max(1, min(1 << 22, round(iters0 * target / max(t_k0, 1e-9)))))
        h.build_kernel(iters)
        h.time_kernel()                      # warm up the recompiled kernel

        # --- measure
        t_k = _best_of(h.time_kernel, repeats)
        t_c = _best_of(h.time_copy, repeats)
        t_b = _best_of(h.time_both, repeats)

        # --- correctness of the overlapped run, against a device oracle
        ref_lower = h.kernel_reference()
        h.reset()
        h.deploy()
        h.write_slot()
        h.sync_kernel()
        h.sync_copy()
        got = h.buf.D2H()
        err_upper = float(np.max(np.abs(got[h.half:] - h.payload)))
        err_lower = float(np.max(np.abs(got[:h.half] - ref_lower)))

        overlap = (t_k + t_c - t_b) / max(min(t_k, t_c), 1e-12)

        out['iters'] = iters
        out['copy_MB'] = h.copy_bytes / (1 << 20)
        out['t_kernel_ms'] = t_k * 1e3
        out['t_copy_ms'] = t_c * 1e3
        out['t_both_ms'] = t_b * 1e3
        out['bandwidth_GBs'] = h.copy_bytes / t_c / 1e9
        out['overlap'] = overlap
        out['err_transferred'] = err_upper
        out['err_computed'] = err_lower
    finally:
        h.close()
    return out


def _verdict(overlap):
    if overlap >= _OVERLAP_GOOD:
        return "OVERLAP"
    if overlap >= _OVERLAP_PARTIAL:
        return "partial"
    return "SERIALIZED"


def main(copy_mb=128, repeats=3):
    print("=== T3-overlap: is the async partial write concurrent with compute? ===\n")
    for lang in _OVERLAP_LANGS:
        human = idpy_langs_human_dict[lang]
        if not idpy_langs_sys[lang]:
            print(f"  [skip] {human}: backend not available on this machine\n")
            continue
        try:
            r = measure(lang, copy_mb=copy_mb, repeats=repeats)
        except Exception as exc:
            print(f"  [err ] {human}: {type(exc).__name__}: {exc}\n")
            continue

        ok = (r['err_transferred'] == 0.0 and r['err_computed'] == 0.0)
        print(f"  {human}: ITERS={r['iters']}, transfer={r['copy_MB']:.0f} MB")
        print(f"    kernel alone   {r['t_kernel_ms']:8.2f} ms")
        print(f"    copy alone     {r['t_copy_ms']:8.2f} ms"
              f"   ({r['bandwidth_GBs']:.2f} GB/s)")
        print(f"    both together  {r['t_both_ms']:8.2f} ms"
              f"   (serial would be {r['t_kernel_ms'] + r['t_copy_ms']:.2f})")
        print(f"    overlap        {r['overlap']:8.2f}   -> {_verdict(r['overlap'])}")
        print(f"    correctness    transferred half {r['err_transferred']:g}, "
              f"computed half {r['err_computed']:g}"
              f"   -> {'OK' if ok else 'FAIL'}")
        print()

    print(
        "overlap = (tK + tC - tB) / min(tK, tC):  1.0 fully concurrent, "
        "0.0 fully serialized.\n"
        "The copy-alone bandwidth is the number P1 needs; overlap says whether "
        "it can be\nsustained while compute runs. Metal is absent here until "
        "IdpyArrayMETAL's\ndrain-on-touch is replaced (F2) -- until then it has "
        "nothing to overlap with."
    )


if __name__ == '__main__':
    main()
